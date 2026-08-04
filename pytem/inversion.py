"""
inversion.py - Inversion utilities for 1-D layered-earth TEM.

Contains:
  - getR               : roughness (smoothness) matrix
  - getJ_fd            : finite-difference Jacobian
  - getJ_ana           : analytical Jacobian (CUDA/Numba/NumPy)
  - dbdt_to_apprho     : dB/dt -> apparent resistivity
  - getRMS, getAlpha, getAlphas : inversion helpers
  - _gn_solve          : Gauss-Newton normal equations solver
  - _backtrack         : step-halving bound enforcement
  - _alpha_search      : log-spaced regularisation search with parabola backtrack
  - invert             : regularised Gauss-Newton inversion loop
"""

import time as _time_mod
from collections import OrderedDict

import numpy as np

from .transform_weights import MU0, HANKEL_FILTERS, FOURIER_FILTERS, EULER_PARAMS
from .backends import HAS_CUDA
from .kernels_numba import HAS_NUMBA
from .forward import (fwd_circle_central, fwd_square_central,
                      fwd_circle_offset, fwd_square_offset,
                      _precompute_filter_dlf, _precompute_filter_euler)

if HAS_NUMBA:
    from .kernels_jacobian import (
        _te_rte_grad_jit,
        _tem_circular_grad_jit, _tem_square_grad_jit,
        _tem_circular_grad_euler_jit, _tem_square_grad_euler_jit,
    )

if HAS_CUDA:
    import cupy as cp
    from .kernels_jacobian import (
        _te_reflection_coeff_grad_gpu,
        _tem_circular_grad_gpu, _tem_square_grad_gpu,
        _tem_circular_grad_euler_gpu, _tem_square_grad_euler_gpu,
    )


_JANA_CONTEXT_CACHE = OrderedDict()
_JANA_CONTEXT_CACHE_MAX = 32


def _arr_cache_key(a):
    a = np.asarray(a)
    return (a.shape, a.dtype.str, a.tobytes())


def _cache_put(cache, key, value, max_size):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > max_size:
        cache.popitem(last=False)


def _get_jana_context(thicknesses, times, tx_size,
                      geometry, rx_x, rx_y, n_quad, use_symmetry,
                      transform, hankel_filter, fourier_filter, euler_order,
                      tx_height, rx_height, system_filter):
    """Build or fetch model-independent analytical-Jacobian context.

    This caches geometry/filter setup that does not depend on resistivity,
    which is reused across inversion iterations and benchmark repeats.
    """
    times = np.asarray(times, dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    altitude = float(tx_height) + float(rx_height)
    a = float(tx_size)

    key = (
        _arr_cache_key(thicknesses),
        _arr_cache_key(times),
        a, geometry, float(rx_x), float(rx_y), int(n_quad), bool(use_symmetry),
        transform, hankel_filter, fourier_filter, int(euler_order),
        float(tx_height), float(rx_height),
        None if system_filter is None else id(system_filter),
    )

    cached = _JANA_CONTEXT_CACHE.get(key)
    if cached is not None:
        _JANA_CONTEXT_CACHE.move_to_end(key)
        return cached

    h_base, h_j0, h_j1 = HANKEL_FILTERS[hankel_filter]
    _use_euler = (transform == 'euler')
    e_eta = e_A = f_base = f_sin = None
    if _use_euler:
        e_eta, e_A = EULER_PARAMS[euler_order]
    else:
        f_base, f_sin, _ = FOURIER_FILTERS[fourier_filter]

    _is_circle = geometry in ('circle_central', 'circle_offset')
    lam = lam_kern = dist_q = area_w = quad_scale = None

    if geometry == 'circle_central':
        lam = h_base / a
        lam_kern = lam * h_j1
    elif geometry == 'circle_offset':
        from scipy.special import j0 as _j0

        lam = h_base / a
        lam_kern = lam * _j0(lam * float(rx_x)) * h_j1
    elif geometry == 'square_central':
        side = a
        hs = side / 2.0
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        x_pts = hs / 2.0 * (1.0 + gl_nodes)
        w_pts = gl_weights * hs / 2.0
        if use_symmetry:
            dist_q_l, area_w_l = [], []
            for _i in range(n_quad):
                for _jj in range(_i, n_quad):
                    w = w_pts[_i] * w_pts[_jj]
                    if _i != _jj:
                        w *= 2.0
                    dist_q_l.append(np.sqrt(x_pts[_i] ** 2 + x_pts[_jj] ** 2))
                    area_w_l.append(w)
            dist_q = np.array(dist_q_l)
            area_w = np.array(area_w_l)
        else:
            _xx, _yy = np.meshgrid(x_pts, x_pts)
            _wx, _wy = np.meshgrid(w_pts, w_pts)
            dist_q = np.sqrt(_xx.ravel() ** 2 + _yy.ravel() ** 2)
            area_w = (_wx * _wy).ravel()
        quad_scale = 4.0
    elif geometry == 'square_offset':
        side = a
        hs = side / 2.0
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        x_pts = hs * gl_nodes
        wx = hs * gl_weights
        if float(rx_y) == 0.0:
            keep = gl_nodes >= 0.0
            y_pts = hs * gl_nodes[keep]
            wy = hs * gl_weights[keep] * np.where(gl_nodes[keep] > 0.0, 2.0, 1.0)
            _xx, _yy = np.meshgrid(x_pts, y_pts, indexing='xy')
            _wx, _wy = np.meshgrid(wx, wy, indexing='xy')
            dist_q = np.sqrt((_xx.ravel() - float(rx_x)) ** 2 + _yy.ravel() ** 2)
        else:
            _xx, _yy = np.meshgrid(x_pts, x_pts, indexing='xy')
            _wx, _wy = np.meshgrid(wx, wx, indexing='xy')
            dist_q = np.sqrt((_xx.ravel() - float(rx_x)) ** 2 + (_yy.ravel() - float(rx_y)) ** 2)
        dist_q = np.maximum(dist_q, 1e-6)
        area_w = (_wx * _wy).ravel()
        quad_scale = 1.0
    else:
        raise ValueError(
            f"Unknown geometry '{geometry}'. Choose from: "
            "'circle_central', 'circle_offset', 'square_central', 'square_offset'."
        )

    if _is_circle and altitude != 0.0:
        lam_kern = lam_kern * np.exp(-lam * altitude)

    if _use_euler:
        n_eval = len(e_eta)
        filter_weights = (
            _precompute_filter_euler(system_filter, times, e_eta, e_A)
            if system_filter is not None
            else np.ones((len(times), n_eval), dtype=np.complex128)
        )
    else:
        n_f = len(f_base)
        filter_weights = (
            _precompute_filter_dlf(system_filter, times, f_base)
            if system_filter is not None
            else np.ones((len(times), n_f), dtype=np.complex128)
        )

    ctx = {
        'h_base': h_base,
        'h_j0': h_j0,
        'h_j1': h_j1,
        '_use_euler': _use_euler,
        'e_eta': e_eta,
        'e_A': e_A,
        'f_base': f_base,
        'f_sin': f_sin,
        '_is_circle': _is_circle,
        'lam': lam,
        'lam_kern': lam_kern,
        'dist_q': dist_q,
        'area_w': area_w,
        'quad_scale': quad_scale,
        'filter_weights': filter_weights,
        'altitude': altitude,
    }
    _cache_put(_JANA_CONTEXT_CACHE, key, ctx, _JANA_CONTEXT_CACHE_MAX)
    return ctx


# ============================================================
# NumPy / CuPy batch function (array-level, xp-agnostic)
# ============================================================

def _te_grad_batch(lam, omegas, thick, rho, xp):
    """Batched TE gradient for M omegas. Works with numpy or cupy.
    lam: (K,) xp, omegas: (M,) xp, thick: cpu list/array, rho: (N,) xp
    Returns r_TE (M,K), dr_TE (N,M,K)
    """
    n_lay = len(rho)
    K = len(lam)
    M = len(omegas)
    sigma = 1.0 / xp.asarray(rho, dtype=xp.complex128)
    thick_f = [float(t) for t in np.asarray(thick)]
    sval = 1j * omegas
    lam2 = lam ** 2

    Gamma = xp.sqrt(lam2[None, None, :] + sval[None, :, None] * MU0 * sigma[:, None, None])
    dG = -sval[None, :, None] * MU0 * sigma[:, None, None] / (2.0 * Gamma)

    # Upward recursion (paper Eq. 2), batched over M frequencies. gamma_N = 0
    # and gamma_j = (psi_j + gamma_{j+1} E_{j+1}) / (1 + psi_j gamma_{j+1} E_{j+1}).
    psi_st = xp.empty((n_lay, M, K), dtype=xp.complex128)
    e_st = xp.empty((n_lay, M, K), dtype=xp.complex128)
    gb_st = xp.empty((n_lay, M, K), dtype=xp.complex128)

    gamma = xp.zeros((M, K), dtype=xp.complex128)          # gamma_N = 0
    for j in range(n_lay - 1, -1, -1):
        G_above = lam[None, :] if j == 0 else Gamma[j - 1]
        Gj = Gamma[j]
        ps = (G_above - Gj) / (G_above + Gj)
        E = (xp.exp(-2.0 * Gj * thick_f[j]) if j < n_lay - 1
             else xp.zeros((M, K), dtype=xp.complex128))
        gb_st[j] = gamma
        psi_st[j] = ps
        e_st[j] = E
        gamma = (ps + gamma * E) / (1.0 + ps * gamma * E)
    r_TE = gamma

    # Adjoint backward pass; dr accumulates d r_TE / d Gamma_j (scaled by dG at end).
    dr = xp.zeros((n_lay, M, K), dtype=xp.complex128)
    ladj = xp.ones((M, K), dtype=xp.complex128)            # d r_TE / d gamma_0
    for j in range(n_lay):
        G_above = lam[None, :] if j == 0 else Gamma[j - 1]
        Gj = Gamma[j]
        ps = psi_st[j]; E = e_st[j]; g = gb_st[j]
        den2 = (1.0 + ps * g * E) ** 2
        dg_dpsi = (1.0 - (g * E) ** 2) / den2
        dg_dE = g * (1.0 - ps ** 2) / den2
        dg_dgb = E * (1.0 - ps ** 2) / den2
        dE_dGj = (-2.0 * thick_f[j] * E) if j < n_lay - 1 else 0.0
        dr[j] += ladj * (dg_dpsi * (-2.0 * G_above / (G_above + Gj) ** 2) + dg_dE * dE_dGj)
        if j >= 1:
            dr[j - 1] += ladj * dg_dpsi * (2.0 * Gj / (G_above + Gj) ** 2)
        ladj = ladj * dg_dgb

    dr *= dG
    return r_TE, dr

# ============================================================
# Numba JIT kernels live in kernels_jacobian.py (imported above).
# The functions imported are:
#   _te_rte_grad_jit              - single-omega adjoint recursion
#   _tem_circular_grad_jit/euler  - circle DLF / Euler (with filter_weights)
#   _tem_square_grad_jit/euler    - square DLF / Euler (with filter_weights)
# ============================================================

def getJ_ana(thicknesses, log_resistivities, tx_size, times,
             geometry='circle_central',
             rx_x=0.0, rx_y=0.0, n_quad=5, use_symmetry=True,
             use_numba=True, use_cuda=True,
             system_filter=None,
             tx_height=0.0, rx_height=0.0,
             transform='dlf', hankel_filter='key_101',
             fourier_filter='key_81', euler_order=11,
             jacobian_mode='log'):

    """Analytical Jacobian  d(ln(-dBdt_i)) / d(ln rho_j)  for all loop geometries.

    Uses the adjoint upward recursion: a single forward+backward pass per
    quadrature frequency yields gradients for all N layers simultaneously.
    This avoids one separate forward solve per layer, but the runtime still
    grows with N because the recursion and the returned Jacobian both carry
    one sensitivity value per layer.

    Transform modes
    ---------------
    'dlf'   - Digital Linear Filter Fourier transform (default).
    'euler' - Euler-Stehfest inverse Laplace transform.  All backends
              (NumPy / Numba / CUDA) share the same adjoint recursion.

    Supported geometries
    --------------------
    'circle_central'  - Rx at centre of circular Tx loop (default)
    'circle_offset'   - Rx at radial offset rx_x from circular Tx
    'square_central'  - Rx at centre of square Tx loop
    'square_offset'   - Rx at (rx_x, rx_y) offset from square Tx centre

    Backend strategy
    ----------------
    NumPy   : All n_f frequencies batched into a single _te_grad_batch call
              per gate time so the inner upward recursion runs in NumPy's C
              layer.  A Python for-loop over frequencies would add n_f
              function-call overheads; batching eliminates them entirely.
    Numba   : Scalar loops compiled to SIMD machine code via JIT; prange
              over gate times achieves CPU parallelism.  Tight scalar loops
              beat NumPy broadcasting inside JIT because large intermediate
              arrays exceed L2 cache for typical problem sizes.
    GPU     : Full (n_t, n_f, K) tensor batched in one CuPy operation to
              saturate GPU occupancy.  Per-frequency launches would leave
              most warps idle.

    System filter
    -------------
    system_filter is applied in the frequency domain before the imaginary
    (DLF) or real (Euler) part is taken.  Since H(omega) is independent of
    resistivity, it multiplies the gradient kernel identically:
        d/d(ln rho_j) [H * K] = H * dK/d(ln rho_j)

    Parameters
    ----------
    thicknesses       : (N-1,) layer thicknesses [m]
    log_resistivities : (N,)   ln(rho_j)
    tx_size           : float  Tx loop dimension [m]: radius for circle
                               geometries, side length for square geometries
    times             : (n_t,) gate times [s]
    geometry          : str    loop geometry (default 'circle_central')
    rx_x              : float  receiver radial / x-offset [m] (default 0.0)
    rx_y              : float  receiver y-offset for square_offset [m]
    n_quad            : int    Gauss-Legendre order for square geometries
    use_symmetry      : bool   exploit x<->y symmetry for square_central
    use_numba         : bool   Numba JIT backend (default True)
    use_cuda          : bool   CuPy GPU backend  (default True)
    system_filter     : callable or None  H(omega) -> complex (default None)
    tx_height         : float  Tx elevation above ground [m] (default 0.0)
    rx_height         : float  Rx elevation above ground [m] (default 0.0)
    transform         : 'dlf' or 'euler'
    hankel_filter     : str   (default 'key_101')
    fourier_filter    : str   (default 'key_81') - DLF only
    euler_order       : int   8, 11, 15, or 19 (default 11) - Euler only
    jacobian_mode     : str
        'log'      -> return d(ln(-dBdt))/d(ln rho) (default, legacy behavior)
        'absolute' -> return d(-dBdt)/d(ln rho)

    Returns
    -------
    J : (n_t, N) float64
    """
    thicknesses   = np.asarray(thicknesses, dtype=float)
    resistivities = np.exp(np.asarray(log_resistivities, dtype=float))
    times         = np.asarray(times, dtype=float)
    n_lay         = len(resistivities)
    n_t           = len(times)
    ctx = _get_jana_context(
        thicknesses=thicknesses,
        times=times,
        tx_size=tx_size,
        geometry=geometry,
        rx_x=rx_x,
        rx_y=rx_y,
        n_quad=n_quad,
        use_symmetry=use_symmetry,
        transform=transform,
        hankel_filter=hankel_filter,
        fourier_filter=fourier_filter,
        euler_order=euler_order,
        tx_height=tx_height,
        rx_height=rx_height,
        system_filter=system_filter,
    )

    h_base = ctx['h_base']
    h_j0 = ctx['h_j0']
    _use_euler = ctx['_use_euler']
    e_eta = ctx['e_eta']
    e_A = ctx['e_A']
    f_base = ctx['f_base']
    f_sin = ctx['f_sin']
    _is_circle = ctx['_is_circle']
    lam = ctx['lam']
    lam_kern = ctx['lam_kern']
    dist_q = ctx['dist_q']
    area_w = ctx['area_w']
    quad_scale = ctx['quad_scale']
    filter_weights = ctx['filter_weights']
    altitude = ctx['altitude']

    # Circle Tx radius (used by the GPU circle kernels, which recompute lam = h_base / a).
    a = float(tx_size)

    # ---- Backend dispatch: CUDA > Numba > NumPy ----
    _use_nb  = HAS_NUMBA and use_numba
    _use_gpu = HAS_CUDA  and use_cuda and not _use_nb

    if _use_nb:
        # Numba path - scalar loops JIT-compiled to SIMD + prange over gates.
        # filter_weights passed as complex128 array; Numba handles arithmetic natively.
        if _use_euler:
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_euler_jit(
                    times, thicknesses, resistivities, lam, lam_kern, MU0,
                    e_eta, e_A, filter_weights)
            else:
                dbdt, J_raw = _tem_square_grad_euler_jit(
                    times, thicknesses, resistivities,
                    dist_q, area_w, float(quad_scale),
                    h_base, h_j0, MU0, e_eta, e_A, filter_weights, altitude)
        else:
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_jit(
                    times, thicknesses, resistivities, lam, lam_kern, MU0,
                    f_base, f_sin, filter_weights)
            else:
                dbdt, J_raw = _tem_square_grad_jit(
                    times, thicknesses, resistivities,
                    dist_q, area_w, float(quad_scale),
                    h_base, h_j0, MU0, f_base, f_sin, filter_weights, altitude)

    elif _use_gpu:
        # GPU path - full (n_t, n_f, K) tensor batched in one CuPy operation.
        # d_filter_weights transferred to GPU; CuPy handles complex128 natively.
        d_h_base         = cp.asarray(h_base)
        d_filter_weights = cp.asarray(filter_weights)
        if _use_euler:
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_euler_gpu(
                    times, thicknesses, resistivities, a, lam_kern,
                    d_h_base, e_eta, e_A, d_filter_weights)
            else:
                d_h_j0 = cp.asarray(h_j0)
                dbdt, J_raw = _tem_square_grad_euler_gpu(
                    times, thicknesses, resistivities,
                    dist_q, area_w, float(quad_scale),
                    d_h_base, d_h_j0, e_eta, e_A, d_filter_weights, altitude)
        else:
            d_f_base = cp.asarray(f_base)
            d_f_sin  = cp.asarray(f_sin)
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_gpu(
                    times, thicknesses, resistivities, a, lam_kern,
                    d_h_base, d_f_base, d_f_sin, d_filter_weights)
            else:
                d_h_j0 = cp.asarray(h_j0)
                dbdt, J_raw = _tem_square_grad_gpu(
                    times, thicknesses, resistivities,
                    dist_q, area_w, float(quad_scale),
                    d_h_base, d_h_j0, d_f_base, d_f_sin, d_filter_weights, altitude)

    else:
        # NumPy path - all n_f frequencies batched into one _te_grad_batch call
        # per gate time.  _te_grad_batch(lam, omega_arr, thick, rho, np) returns
        # r_TE (M, K) and dr_TE (N, M, K), keeping inner loops in NumPy's C layer.
        dbdt  = np.zeros(n_t)
        J_raw = np.zeros((n_t, n_lay))

        if _use_euler:
            k_arr   = np.arange(len(e_eta), dtype=float)
            signs_k = (-1.0)**k_arr * e_eta                 # (n_eval,)
            for i, t in enumerate(times):
                c         = e_A / (2.0 * t)
                h_step    = np.pi / t
                omega_arr = k_arr * h_step - c * 1j          # (n_eval,) complex
                fw        = filter_weights[i]                 # (n_eval,) complex
                if _is_circle:
                    r_TE, dr_TE = _te_grad_batch(lam, omega_arr, thicknesses, resistivities, np)
                    # r_TE: (n_eval, K),  dr_TE: (N, n_eval, K)
                    hz_c  = 0.5 * (r_TE  * lam_kern[None, :]).sum(-1)         # (n_eval,)
                    dhz_c = 0.5 * (dr_TE * lam_kern[None, None, :]).sum(-1)   # (N, n_eval)
                else:
                    hz_c  = np.zeros(len(omega_arr), dtype=complex)
                    dhz_c = np.zeros((n_lay, len(omega_arr)), dtype=complex)
                    for q in range(len(dist_q)):
                        rq     = dist_q[q];  wq = area_w[q]
                        lam_q  = h_base / rq
                        kern_q = lam_q**2 * h_j0 / (rq * 4.0 * np.pi)
                        if altitude != 0.0:
                            kern_q = kern_q * np.exp(-lam_q * altitude)
                        r_q, dr_q = _te_grad_batch(lam_q, omega_arr, thicknesses, resistivities, np)
                        hz_c  += wq * (r_q  * kern_q[None, :]).sum(-1)
                        dhz_c += wq * (dr_q * kern_q[None, None, :]).sum(-1)
                    hz_c  *= quad_scale
                    dhz_c *= quad_scale
                # Apply filter, then Euler dot product
                hz_acc  = MU0 * np.dot(signs_k, (hz_c  * fw).real)         # scalar
                dhz_acc = MU0 * ((dhz_c * fw[None, :]).real @ signs_k)     # (N,)
                prefac      = np.exp(e_A / 2.0) / t
                dbdt[i]     = -prefac * hz_acc
                J_raw[i, :] = -prefac * dhz_acc

        else:
            for i, t in enumerate(times):
                omega_arr = f_base / t      # (n_f,) - all frequencies at once
                fw        = filter_weights[i]  # (n_f,) complex
                if _is_circle:
                    r_TE, dr_TE = _te_grad_batch(lam, omega_arr, thicknesses, resistivities, np)
                    # r_TE: (n_f, K),  dr_TE: (N, n_f, K)
                    hz_c  = 0.5 * (r_TE  * lam_kern[None, :]).sum(-1)         # (n_f,)
                    dhz_c = 0.5 * (dr_TE * lam_kern[None, None, :]).sum(-1)   # (N, n_f)
                else:
                    hz_c  = np.zeros(len(omega_arr), dtype=complex)
                    dhz_c = np.zeros((n_lay, len(omega_arr)), dtype=complex)
                    for q in range(len(dist_q)):
                        rq     = dist_q[q];  wq = area_w[q]
                        lam_q  = h_base / rq
                        kern_q = lam_q**2 * h_j0 / (rq * 4.0 * np.pi)
                        if altitude != 0.0:
                            kern_q = kern_q * np.exp(-lam_q * altitude)
                        r_q, dr_q = _te_grad_batch(lam_q, omega_arr, thicknesses, resistivities, np)
                        hz_c  += wq * (r_q  * kern_q[None, :]).sum(-1)
                        dhz_c += wq * (dr_q * kern_q[None, None, :]).sum(-1)
                    hz_c  *= quad_scale
                    dhz_c *= quad_scale
                # Apply filter, then Fourier dot product
                hz_im  = MU0 * (hz_c  * fw).imag           # (n_f,) real
                dhz_im = MU0 * (dhz_c * fw[None, :]).imag  # (N, n_f) real
                dbdt[i]     = np.dot(hz_im, f_sin) / t
                J_raw[i, :] = (dhz_im @ f_sin) / t

    # DLF: normalise by 2/pi.  Euler kernels incorporate exp(A/2)/t and the
    # step-off sign (-1) internally, so no further scaling is needed.
    if not _use_euler:
        scale  = 2.0 / np.pi
        dbdt  *= scale
        J_raw *= scale

    if jacobian_mode == 'absolute':
        # If response is r = -dBdt, and J_raw = d(dBdt)/d(ln rho),
        # then dr/d(ln rho) = -J_raw.
        J_abs = -J_raw
        np.nan_to_num(J_abs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return J_abs

    if jacobian_mode != 'log':
        raise ValueError("jacobian_mode must be 'log' or 'absolute'")

    f0 = -dbdt
    if np.any(f0 <= 0):
        print(f"WARNING: {(f0 <= 0).sum()} non-positive dbdt values (zeroed in J)")
    J = np.zeros((n_t, n_lay))
    valid = f0 > 0
    J[valid, :] = -J_raw[valid, :] / f0[valid, None]
    np.nan_to_num(J, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return J



def dbdt_to_apprho(obs_data, tx_area, times):
    """Convert dB/dt to apparent resistivity.

    Parameters
    ----------
    obs_data : array_like
        Observed dB/dt values [T/s].
    tx_area  : float
        Transmitter moment (current x area) [A.m^2].
    times    : array_like
        Gate centre times [s].

    Returns
    -------
    rho_a : ndarray
        Apparent resistivity [Ohm.m].
    """
    M = tx_area
    term = (2 * MU0 * M) / (5 * times * obs_data)
    app_rho = (MU0 / (4 * np.pi * times)) * (term ** (2 / 3))
    return app_rho


def getRMS(obs_data, mod_data, obs_noise):
    """Root-mean-square misfit normalised by noise."""
    total_points = obs_data.size
    data_residual = (mod_data - obs_data) ** 2 / (obs_noise) ** 2
    rms = np.sqrt(np.sum(data_residual) / total_points)
    return rms


def getAlpha(alpha_start, step, alpha_step=1/9):
    """Log-spaced regularisation parameter for a given cooling step."""
    log_alpha = np.log10(alpha_start) - alpha_step * step
    alpha = 10 ** log_alpha
    return alpha


def getAlphas(alpha, thicknesses):
    """Depth-weighted regularisation vector.

    Works with N-1 finite-layer thicknesses (standard pytem convention) and
    returns N alpha factors (one per model parameter including the half-space).
    The half-space midpoint is extrapolated one layer-thickness below the base
    of the last finite layer.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    # Midpoints of every finite layer
    tops       = np.cumsum(np.concatenate(([0.0], thicknesses[:-1])))
    mid_finite = tops + thicknesses / 2.0
    # Extrapolate one midpoint for the half-space
    hs_mid     = tops[-1] + thicknesses[-1] + thicknesses[-1] / 2.0
    midpoints  = np.append(mid_finite, hs_mid)          # N midpoints total
    del_z      = np.diff(midpoints)                      # N-1 spacings
    alpha_factor = np.empty(len(del_z) + 1, dtype=del_z.dtype)  # N values
    alpha_factor[0]    = 1.0 / del_z[0]
    alpha_factor[1:-1] = 1.0 / del_z[:-1] + 1.0 / del_z[1:]
    alpha_factor[-1]   = 1.0 / del_z[-1]
    return alpha * alpha_factor


def getR(resistivities, damp=1e-4):
    """First-order roughness (smoothness) matrix with optional damping."""
    n_params = resistivities.size
    D = np.zeros((n_params - 1, n_params))
    for k in range(n_params - 1):
        D[k, k] = -1.0
        D[k, k + 1] = 1.0
    R = D.T @ D + damp * np.eye(n_params)
    return R


def getJ_fd(thicknesses, log_resistivities, tx_size, times,
            use_numba=False, use_cuda=True, eps=None, fwd=fwd_circle_central,
            tx_height=0.0, rx_height=0.0,
            transform='dlf', hankel_filter='key_101', fourier_filter='key_81',
            euler_order=11):
    """Finite-difference Jacobian d(log(-dBdt))/d(ln rho).

    Note: the Euler transform forward is only ~8-digit accurate, so finite
    differencing it amplifies round-off into large relative errors. For the
    'euler' transform prefer the analytical Jacobian ``getJ_ana`` (or use
    ``transform='dlf'``). When ``eps`` is left unset it defaults to 1e-4 for
    'dlf' and a larger 1e-2 for 'euler' to reduce that amplification.
    """
    if eps is None:
        eps = 1e-2 if transform == 'euler' else 1e-4
        if transform == 'euler':
            print("WARNING: FD Jacobian on the Euler transform is unreliable "
                  "(round-off amplification); using eps=1e-2. Prefer getJ_ana "
                  "or transform='dlf' for accurate sensitivities.")
    elif transform == 'euler':
        print("WARNING: FD Jacobian on the Euler transform is unreliable "
              "(round-off amplification). Prefer getJ_ana or transform='dlf'.")

    fwd_kw = dict(use_numba=use_numba, use_cuda=use_cuda, transform=transform,
                  tx_height=tx_height, rx_height=rx_height,
                  hankel_filter=hankel_filter, fourier_filter=fourier_filter,
                  euler_order=euler_order)

    if fwd is fwd_square_central:
        f0 = -fwd(thicknesses=thicknesses, resistivities=np.exp(log_resistivities),
                  tx_side=tx_size, times=times, **fwd_kw)
    else:
        f0 = -fwd(thicknesses=thicknesses, resistivities=np.exp(log_resistivities),
                  tx_radius=tx_size, times=times, **fwd_kw)

    bad_f0 = f0 <= 0
    if np.any(bad_f0):
        print(f"WARNING: f0 has {bad_f0.sum()} non-positive values at gate indices {np.where(bad_f0)[0]} (zeroed in J)")

    J = np.zeros((f0.size, log_resistivities.size))
    bad_count = 0
    for i in range(log_resistivities.size):
        perturbed = log_resistivities.copy()
        step = eps * max(1.0, abs(log_resistivities[i]))
        perturbed[i] += step
        if fwd is fwd_square_central:
            fi = -fwd(thicknesses=thicknesses, resistivities=np.exp(perturbed),
                      tx_side=tx_size, times=times, **fwd_kw)
        else:
            fi = -fwd(thicknesses=thicknesses, resistivities=np.exp(perturbed),
                      tx_radius=tx_size, times=times, **fwd_kw)

        valid = (f0 > 0) & (fi > 0)
        if not np.all(valid):
            bad_count += 1
        J[valid, i] = (np.log(fi[valid]) - np.log(f0[valid])) / step

    if bad_count:
        print(f"WARNING: {bad_count}/{log_resistivities.size} perturbed models had non-positive values (zeroed in J)")

    return J

# ============================================================
# Regularised Gauss-Newton inversion helpers
# ============================================================

def _gn_solve(Jw, dw, R, alpha_vector, m):
    """Solve the weighted, regularised Gauss-Newton normal equations.

    Solves  (Jw^T Jw + diag(alpha) R) dm = Jw^T dw - diag(alpha) R m
    via least-squares (robust to mild rank deficiency).

    Parameters
    ----------
    Jw           : (n_d, N)  noise-weighted Jacobian
    dw           : (n_d,)    noise-weighted log-space residuals
    R            : (N, N)    roughness matrix from getR()
    alpha_vector : (N,)      per-layer regularisation weights from getAlphas()
    m            : (N,)      current log-resistivity model

    Returns
    -------
    dm : (N,) model update
    """
    AR  = np.diag(alpha_vector) @ R
    lhs = Jw.T @ Jw + AR
    rhs = Jw.T @ dw - AR @ m
    dm, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=1e-10)
    return dm


def _backtrack(m, delta, ln_rho_min, ln_rho_max):
    """Halve the step length until the trial model is within bounds.

    Parameters
    ----------
    m           : (N,) current log-resistivity model
    delta       : (N,) proposed step
    ln_rho_min  : float  lower bound in log-resistivity space
    ln_rho_max  : float  upper bound in log-resistivity space

    Returns
    -------
    trial : (N,) new model (clipped to bounds as a last resort)
    step  : float  accepted step length in [0, 1]
    """
    step = 1.0
    for _ in range(10):
        trial = m + step * delta
        if np.all(trial >= ln_rho_min) and np.all(trial <= ln_rho_max):
            return trial, step
        step *= 0.5
    return np.clip(m + step * delta, ln_rho_min, ln_rho_max), step


def _alpha_search(alpha_start, alpha_steps, Jw, dw, R, m,
                  thicknesses, fwd_fn, obs_data, w,
                  ln_rho_min, ln_rho_max, alpha_step=1/9, rms_current=np.inf,
                  plot=False):
    """Log-spaced alpha search with parabola backtrack to RMS = 1.

    Tests ``alpha_steps`` regularisation strengths starting from
    ``alpha_start`` on a log-spaced ladder defined by ``getAlpha``, evaluates
    the RMS for each, fits a polynomial, and locates alpha* where RMS = 1.

    Parameters
    ----------
    alpha_start  : float      largest (strongest) regularisation to try
    alpha_steps  : int        number of alpha values on the ladder
    Jw           : (n_d, N)   noise-weighted Jacobian
    dw           : (n_d,)     noise-weighted log-space residuals
    R            : (N, N)     roughness matrix
    m            : (N,)       current log-resistivity model
    thicknesses  : (N,)       layer thicknesses for getAlphas() depth weighting
    fwd_fn       : callable   log_rho -> (n_t,) positive forward response
    obs_data     : (n_t,)     observed data (positive)
    w            : (n_t,)     noise weights (1 / noise_log)
    ln_rho_min   : float
    ln_rho_max   : float
    alpha_step   : float      log10 step size between consecutive alpha values
                              (default 1/9, i.e. ~10 steps per decade)
    rms_current  : float      RMS of the current model before any update;
                              the 'RMS increased' early stop only fires once
                              at least one alpha has improved on this value
    plot         : bool       show alpha-RMS diagnostic figure (default False)

    Returns
    -------
    alpha_hist : list of float    tested + parabola alpha values
    rms_hist   : list of float    corresponding RMS values
    delta_hist : list of ndarray  corresponding model deltas (m_trial - m)
    mod_hist   : list of ndarray  corresponding forward responses for each trial
    """
    alpha_hist, rms_hist, delta_hist, mod_hist = [], [], [], []

    for i in range(alpha_steps):
        alpha = getAlpha(alpha_start, step=i, alpha_step=alpha_step)
        avs   = getAlphas(alpha, thicknesses)
        delta = _gn_solve(Jw, dw, R, avs, m)
        trial, step = _backtrack(m, delta, ln_rho_min, ln_rho_max)
        mod   = fwd_fn(trial)
        valid = (obs_data > 0) & (mod > 0)
        d_res = np.log(obs_data[valid]) - np.log(mod[valid])
        rms   = np.sqrt(np.mean((w[valid] * d_res) ** 2))
        print(f"    Alpha = {alpha:.2f},  RMS = {rms:.2f}"
              + (f"  (step = {step:.2f})" if step < 1.0 else ""))
        alpha_hist.append(alpha)
        rms_hist.append(rms)
        delta_hist.append(trial - m)
        mod_hist.append(mod)

        if rms < 1.0:
            print("    RMS below 1 - stopping for parabola fit.")
            break

        if len(rms_hist) > 1 and rms > rms_hist[-2] and min(rms_hist[:-1]) < rms_current:
            print("    RMS increased - stopping alpha search early.")
            break

    # Polynomial backtrack to find alpha* where RMS = 1 (only when below 1 is reached)
    x_data = np.log10(np.array(alpha_hist))
    y_data = np.array(rms_hist)
    deg    = min(2, len(x_data) - 1)
    parabola_alpha = None
    coeffs         = None

    if deg >= 1 and np.min(y_data) < 1.0:
        coeffs         = np.polyfit(x_data, y_data, deg)
        root_c         = coeffs.copy()
        root_c[-1]    -= 1.0
        roots          = np.roots(root_c)
        x_lo, x_hi     = x_data.min() - 1.0, x_data.max() + 1.0
        real_roots      = roots[np.abs(roots.imag) < 1e-10].real
        valid_roots     = real_roots[(real_roots >= x_lo) & (real_roots <= x_hi)]

        if valid_roots.size > 0:
            parabola_x     = float(valid_roots.max())
            parabola_alpha = 10.0 ** parabola_x
            avs_par        = getAlphas(parabola_alpha, thicknesses)
            delta_par      = _gn_solve(Jw, dw, R, avs_par, m)
            trial_par, step_par = _backtrack(m, delta_par, ln_rho_min, ln_rho_max)
            mod_par        = fwd_fn(trial_par)
            valid_par      = (obs_data > 0) & (mod_par > 0)
            d_par          = np.log(obs_data[valid_par]) - np.log(mod_par[valid_par])
            rms_par        = np.sqrt(np.mean((w[valid_par] * d_par) ** 2))
            print(f"    Fitted Alpha = {parabola_alpha:.3f}"
                  f", Actual RMS = {rms_par:.3f}"
                  + (f"  (step = {step_par:.2f})" if step_par < 1.0 else ""))
            alpha_hist.append(parabola_alpha)
            rms_hist.append(rms_par)
            delta_hist.append(trial_par - m)
            mod_hist.append(mod_par)

    if plot and coeffs is not None:
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(5, 3.5))
        x_fit = np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 300)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '-', color='C0', lw=1.5,
                label=f'Degree-{deg} fit')
        ax.plot(x_data, y_data, 'o', color='C1', zorder=5,
                label='Tested alphas')
        ax.axhline(1.0, color='k', ls='--', lw=1, label='RMS = 1 target')
        if parabola_alpha is not None:
            ax.axvline(np.log10(parabola_alpha), color='C2', ls='--', lw=1,
                       label=f'$\\alpha^* = {parabola_alpha:.3g}$')
            ax.plot(np.log10(parabola_alpha), 1.0, '*', color='C2',
                    markersize=12, zorder=6)
        ax.set_xlabel('$\\log_{{10}}(\\alpha)$')
        ax.set_ylabel('RMS')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _plt.show()

    return alpha_hist, rms_hist, delta_hist, mod_hist


# ============================================================
# Main inversion function
# ============================================================

def invert(obs_data, thicknesses, log_resistivities, tx_size, times,
           alpha_start=None, alpha_steps=5, alpha_step=1/9, maxit=20, eps=1e-4,
           noise_std=0.02, use_numba=True, use_cuda=True,
           calc_sens=False, store_J=False,
           transform='euler', hankel_filter='key_101', fourier_filter='key_81',
           euler_order=11, rho_min=1e-1, rho_max=1e5, max_noise_frac=0.10,
           plot_alpha=False, analytical_j=True,
           system_filter=None,
           waveform_times=None, waveform_currents=None, n_step=200,
           waveform_n_quad=5,
           geometry='circle_central', n_quad=5,
           rx_x=0.0, rx_y=0.0, tx_height=0.0, rx_height=0.0,
           circle_warmstart=False):
    """Regularised Gauss-Newton inversion for 1-D layered-earth TEM.

    Minimises  phi(m) = ||W (ln d_obs - ln d_pred(m))||^2 + alpha * m^T R m
    using iterative Gauss-Newton updates with a log-spaced alpha search and
    parabola backtrack to target RMS = 1.

    All optimisation is performed in log-resistivity space, so the forward
    function is always evaluated with ``resistivities = exp(m)``.

    System filter
    -------------
    Pass ``system_filter`` (a callable H(omega) -> complex array) to apply a
    frequency-domain instrument response inside the forward model and the
    analytical Jacobian.  When ``analytical_j=False`` the filter is applied
    inside the geometry's forward function automatically; when
    ``analytical_j=True`` it is passed to ``getJ_ana``.

    Waveform convolution
    --------------------
    When ``waveform_times`` and ``waveform_currents`` are supplied, the step-off
    response is evaluated only at the deduplicated quadrature times produced by
    ``waveform.setup_waveform`` and combined with the piecewise-linear waveform
    through a precomputed sparse weight matrix.  Both finite-difference and
    analytical Jacobians support this mode: because the convolution operator is
    linear, the analytical path convolves the per-layer gradients with the same
    weight matrix (chain rule), avoiding N+1 separate convolution passes.

    Parameters
    ----------
    obs_data            : (n_t,) observed dB/dt [T/s], positive values
    thicknesses         : (N,)   layer thicknesses [m]
    log_resistivities   : (N,)   initial ln(rho) [ln(Ohm.m)]
    tx_size             : float  Tx loop dimension [m]: radius for circle
                          geometries, side length for square geometries
    times               : (n_t,) gate centre times [s]
    alpha_start         : float or None  starting regularisation strength
                          (auto-estimated from JTd if None)
    alpha_steps         : int    number of alpha values per outer iteration
    alpha_step          : float  log10 step size between consecutive alpha values
                          (default 1/9 ~ 10 steps per decade; increase for
                          larger jumps, e.g. 1/4 ~ 4 steps per decade)
    maxit               : int    maximum Gauss-Newton iterations
    eps                 : float  finite-difference step for FD Jacobian
    noise_std           : float or (n_t,)  fractional noise standard deviation
    use_numba           : bool   enable Numba JIT backend
    use_cuda            : bool   enable CuPy GPU backend
    calc_sens           : bool   compute parameter sensitivity at convergence
    store_J             : bool   store Jacobian at each iteration
    transform           : 'dlf' or 'euler'
    hankel_filter       : str    (default 'key_101')
    fourier_filter      : str    (default 'key_81')
    euler_order         : int    (default 11)
    rho_min             : float  lower resistivity bound [Ohm.m] (default 0.1)
    rho_max             : float  upper resistivity bound [Ohm.m] (default 1e5)
    max_noise_frac      : float  noise floor as a fraction of peak data
    plot_alpha          : bool   show alpha-RMS plot at each iteration
    analytical_j        : bool   use analytical Jacobian (ignored when waveform)
    system_filter       : callable or None  H(omega) -> complex
    waveform_times      : array-like or None  waveform break points [s]
    waveform_currents   : array-like or None  current at break points [A]
    n_step              : int    unused (kept for backwards compatibility)
    waveform_n_quad     : int    GL quadrature order for waveform convolution (default 5)
    tx_height           : float  Tx elevation above ground [m] (default 0.0)
    rx_height           : float  Rx elevation above ground [m] (default 0.0)

    Returns
    -------
    result : dict with keys
        'log_resistivities' : (N,)         final ln(rho)
        'resistivities'     : (N,)         final rho [Ohm.m]
        'thicknesses'       : (N,)         layer thicknesses [m]
        'model_history'     : list of (N,) all models (initial + each iteration)
        'rms_history'       : list of float  RMS after each iteration
        'J_history'         : list of (n_t, N) or None
        'sensitivity'       : (N,) or None   column-norm of final J
        'times'             : (n_t,) gate times [s]
        'obs_data'          : (n_t,) observed data
        'n_iter'            : int   number of completed iterations
    """
    # ---- Parse inputs ----
    obs_data    = np.asarray(obs_data,          dtype=float)
    thicknesses = np.asarray(thicknesses,       dtype=float)
    m           = np.asarray(log_resistivities, dtype=float).copy()
    times       = np.asarray(times,             dtype=float)

    # ---- Backend/use-path logging ----
    fwd_backend = 'cuda' if (HAS_CUDA and use_cuda) else ('numba' if (HAS_NUMBA and use_numba) else 'numpy')
    if analytical_j:
        if HAS_NUMBA and use_numba:
            jac_backend = 'numba'
        elif HAS_CUDA and use_cuda:
            jac_backend = 'cuda'
        else:
            jac_backend = 'numpy'
    else:
        jac_backend = 'finite_difference'
    print(
        '[invert] '
        f'geometry={geometry}, transform={transform}, '
        f'fwd_backend={fwd_backend}, jac_backend={jac_backend}, '
        f'analytical_j={analytical_j}, waveform={waveform_times is not None and waveform_currents is not None}'
    )

    # ---- Circle warm-start ----
    if circle_warmstart and geometry.startswith('square'):
        r_circ    = float(tx_size) / np.sqrt(np.pi)
        circ_geom = 'circle_central' if geometry == 'square_central' else 'circle_offset'
        print(f'[circle_warmstart] Running circle pre-inversion '
              f'(geometry={circ_geom}, r={r_circ:.3f} m)...')
        ws = invert(
            obs_data=obs_data, thicknesses=thicknesses,
            log_resistivities=m, tx_size=r_circ, times=times,
            alpha_start=alpha_start, alpha_steps=alpha_steps, alpha_step=alpha_step,
            maxit=maxit,
            eps=eps, noise_std=noise_std, use_numba=use_numba, use_cuda=use_cuda,
            calc_sens=False, store_J=False,
            transform=transform, hankel_filter=hankel_filter,
            fourier_filter=fourier_filter, euler_order=euler_order,
            rho_min=rho_min, rho_max=rho_max, max_noise_frac=max_noise_frac,
            plot_alpha=plot_alpha, analytical_j=analytical_j,
            system_filter=system_filter,
            waveform_times=waveform_times, waveform_currents=waveform_currents,
            waveform_n_quad=waveform_n_quad,
            n_step=n_step, geometry=circ_geom, n_quad=1,
            rx_x=rx_x, rx_y=rx_y, tx_height=tx_height,
            rx_height=rx_height, circle_warmstart=False,
        )
        print(f'[circle_warmstart] Circle converged (RMS={ws["rms_history"][-1]:.3f}). '
              f'Running up to {maxit} square refinement steps...')
        return invert(
            obs_data=obs_data, thicknesses=thicknesses,
            log_resistivities=ws['log_resistivities'], tx_size=tx_size,
            times=times,
            alpha_start=None, alpha_steps=alpha_steps, alpha_step=alpha_step,
            maxit=maxit, eps=eps,
            noise_std=noise_std, use_numba=use_numba, use_cuda=use_cuda,
            calc_sens=calc_sens, store_J=store_J,
            transform=transform, hankel_filter=hankel_filter,
            fourier_filter=fourier_filter, euler_order=euler_order,
            rho_min=rho_min, rho_max=rho_max, max_noise_frac=max_noise_frac,
            plot_alpha=plot_alpha, analytical_j=analytical_j,
            system_filter=system_filter,
            waveform_times=waveform_times, waveform_currents=waveform_currents,
            waveform_n_quad=waveform_n_quad,
            n_step=n_step, geometry=geometry, n_quad=n_quad,
            rx_x=rx_x, rx_y=rx_y, tx_height=tx_height,
            rx_height=rx_height, circle_warmstart=False,
        )

    ln_rho_min = np.log(float(rho_min))
    ln_rho_max = np.log(float(rho_max))

    # ---- Noise weighting ----
    if np.isscalar(noise_std):
        noise_abs = noise_std * np.abs(obs_data)
    else:
        noise_abs = np.asarray(noise_std, dtype=float)
    # Floor: noise cannot be smaller than max_noise_frac * peak |obs|
    noise_abs = np.maximum(noise_abs, max_noise_frac * np.abs(obs_data).max())
    # Log-space noise: sigma_log_i = sigma_i / |d_i|
    noise_log = noise_abs / np.abs(obs_data)
    w         = 1.0 / noise_log   # weight per datum in log space

    # ---- Waveform convolution setup ----
    _use_waveform = (waveform_times is not None and waveform_currents is not None)
    if _use_waveform:
        from .waveform import setup_waveform as _setup_waveform
        _wf_t  = np.asarray(waveform_times,    dtype=float)
        _wf_I  = np.asarray(waveform_currents, dtype=float)
        _wf_comp_times, _wf_apply = _setup_waveform(
            times, _wf_t, _wf_I, n_quad=waveform_n_quad
        )


    _fwd_kw = dict(
        use_numba=use_numba,
        use_cuda=use_cuda,
        system_filter=system_filter,
        transform=transform,
        tx_height=tx_height,
        rx_height=rx_height,
        hankel_filter=hankel_filter,
        fourier_filter=fourier_filter,
        euler_order=euler_order,
    )

    # ---- Geometry dispatch helper ----
    def _call_fwd(thick, res, t):
        if geometry == 'circle_central':
            return -fwd_circle_central(thick, res, float(tx_size), t, **_fwd_kw)
        elif geometry == 'circle_offset':
            return -fwd_circle_offset(thick, res, float(tx_size),
                                      float(rx_x), t, **_fwd_kw)
        elif geometry == 'square_central':
            return -fwd_square_central(thick, res, float(tx_size), t,
                                       n_quad=n_quad, **_fwd_kw)
        else:  # square_offset
            return -fwd_square_offset(thick, res, float(tx_size),
                                      float(rx_x), float(rx_y), t,
                                      n_quad=n_quad, **_fwd_kw)

    # ---- Forward model closure ----
    def _forward_response(log_rho):
        res = np.exp(log_rho)
        if _use_waveform:
            step_resp = _call_fwd(thicknesses, res, _wf_comp_times)
            return _wf_apply(step_resp)
        return _call_fwd(thicknesses, res, times)

    # ---- Jacobian closure ----
    def _build_jacobian(log_rho):
        if analytical_j and not _use_waveform:
            # Pure analytical Jacobian - no waveform.  getJ_ana takes tx_size
            # directly (radius for circle, side for square), matching invert.
            return getJ_ana(
                thicknesses=thicknesses,
                log_resistivities=log_rho,
                tx_size=float(tx_size),
                times=times,
                geometry=geometry,
                rx_x=float(rx_x),
                rx_y=float(rx_y),
                n_quad=n_quad,
                use_numba=use_numba,
                use_cuda=False,
                system_filter=system_filter,
                tx_height=tx_height,
                rx_height=rx_height,
                transform=transform,
                hankel_filter=hankel_filter,
                fourier_filter=fourier_filter,
                euler_order=euler_order,
            )

        if analytical_j and _use_waveform:
            # Analytical Jacobian with waveform convolution.
            #
            # G_i = conv(F, w)_i is linear in F, so:
            #   dG_i/d(ln rho_j) = conv(dF/d(ln rho_j), w)_i
            #
            # getJ_ana returns J_anal[k,j] = d ln(F(t_k)) / d ln(rho_j)
            # Absolute gradient: dF(t_k)/d(ln rho_j) = J_anal[k,j] * F(t_k)
            # Log-space Jacobian: J_conv[i,j] = dG_i/d(ln rho_j) / G_i
            #
            # Both the step response and the analytical Jacobian are evaluated
            # only at _wf_comp_times (the precomputed deduplicated quadrature
            # times), not on a dense grid.  This is the empymod pattern.
            res       = np.exp(log_rho)
            step_resp = _call_fwd(thicknesses, res, _wf_comp_times)  # (n_unique,)
            J_anal = getJ_ana(
                thicknesses=thicknesses,
                log_resistivities=log_rho,
                tx_size=float(tx_size),
                times=_wf_comp_times,
                geometry=geometry,
                rx_x=float(rx_x),
                rx_y=float(rx_y),
                n_quad=n_quad,
                use_numba=use_numba,
                use_cuda=False,
                system_filter=system_filter,
                tx_height=tx_height,
                rx_height=rx_height,
                transform=transform,
                hankel_filter=hankel_filter,
                fourier_filter=fourier_filter,
                euler_order=euler_order,
            )  # (n_unique, N)
            G = _wf_apply(step_resp)                      # (n_gates,)
            # dF matrix: (n_unique, N) - absolute gradients at comp times
            dF = J_anal * step_resp[:, None]              # broadcast
            dG = _wf_apply(dF)                            # (n_gates, N)
            J_conv = np.zeros((len(times), log_rho.size))
            valid_g = G > 0
            J_conv[valid_g, :] = dG[valid_g, :] / G[valid_g, None]
            return J_conv

        # Finite-difference Jacobian: waveform convolution is included
        # automatically because _forward_response handles it.
        f0 = _forward_response(log_rho)
        J  = np.zeros((f0.size, log_rho.size))
        for i in range(log_rho.size):
            pert    = log_rho.copy()
            h       = eps * max(1.0, abs(log_rho[i]))
            pert[i] += h
            fi      = _forward_response(pert)
            valid   = (f0 > 0) & (fi > 0)
            J[valid, i] = (np.log(fi[valid]) - np.log(f0[valid])) / h
        return J

    # ---- Print average apparent resistivity of observed data ----
    tx_area = (np.pi * float(tx_size) ** 2 if geometry.startswith('circle')
               else float(tx_size) ** 2)
    app_rho_obs = dbdt_to_apprho(obs_data, tx_area, times)
    valid_rho = app_rho_obs[(obs_data > 0) & np.isfinite(app_rho_obs)]
    if valid_rho.size > 0:
        print(f"Observed data: Mean Apparent Resistivity = {np.mean(valid_rho):.1f} Ohm.m "
              f"(over {valid_rho.size}/{len(times)} valid gates)")

    # ---- Initial alpha heuristic ----
    #print("Building initial Jacobian...")
    t0 = _time_mod.time()
    J0 = _build_jacobian(m)
    d0 = _forward_response(m)
    #print(f"  done ({_time_mod.time() - t0:.1f} s)")

    valid0  = (obs_data > 0) & (d0 > 0)
    res0    = np.zeros(len(obs_data))
    res0[valid0] = np.log(obs_data[valid0]) - np.log(d0[valid0])
    Jw0     = J0 * w[:, None]
    dw0     = res0 * w

    if alpha_start is None:
        alpha_start = float(np.linalg.norm(Jw0.T @ dw0, np.inf) + 1e-30)
    print(f"Alpha start = {alpha_start:.3f}")

    # ---- Gauss-Newton loop ----
    rms_history   = []
    model_history = [m.copy()]
    J_history     = [J0.copy()] if store_J else []
    J_cur         = J0

    t_loop = _time_mod.time()
    d_pred = d0  # reuse the forward response already computed above

    for it in range(maxit):
        valid  = (obs_data > 0) & (d_pred > 0)

        if not np.any(valid):
            print("WARNING: No valid data - stopping.")
            break

        res_log         = np.zeros(len(obs_data))
        res_log[valid]  = np.log(obs_data[valid]) - np.log(d_pred[valid])
        rms = np.sqrt(np.mean((w[valid] * res_log[valid]) ** 2))
        rms_history.append(rms)

        elapsed = _time_mod.time() - t_loop
        print(f"Iteration {it + 1:>3d}:  RMS = {rms:.2f}")

        if rms <= 1.0:
            print("  RMS <= 1 - converged.")
            break

        if it > 0:
            J_cur = _build_jacobian(m)
            if store_J:
                J_history.append(J_cur.copy())

        R  = getR(m)
        Jw = J_cur * w[:, None]
        dw = res_log * w

        alpha_h, rms_h, delta_h, mod_h = _alpha_search(
            alpha_start, alpha_steps,
            Jw, dw, R, m,
            thicknesses, _forward_response, obs_data, w,
            ln_rho_min, ln_rho_max,
            alpha_step=alpha_step,
            rms_current=rms,
            plot=plot_alpha,
        )

        # Select best model update.
        # When any search RMS dipped below 1 (overshoot), _alpha_search appended
        # the parabola-adjusted model as the last entry.  Always prefer that entry
        # so the update lands as close to RMS = 1 as possible - even if the
        # parabola RMS ended up slightly above 1 (which the "below" filter would
        # otherwise reject).
        rms_arr   = np.array(rms_h)
        overshoot = bool(np.any(rms_arr < 1.0))
        if overshoot:
            best_idx          = len(rms_h) - 1
            raw_overshoot_rms = float(rms_arr[rms_arr < 1.0].min())
            #print(f"  Overshoot (RMS = {raw_overshoot_rms:.3f})")
        else:
            below = rms_arr[rms_arr <= 1.0]
            if below.size > 0:
                best_idx = int(np.where(rms_arr == below.max())[0][-1])
            else:
                best_idx = int(np.argmin(rms_arr))

        # Stop if no alpha value improved the fit.
        if rms_h[best_idx] >= rms:
            print("  No improvement found - stopping.")
            break

        # Cool alpha_start to one step above the best alpha found so far.
        # Starting exactly at the best means the next search only explores
        # weaker regularisation; shifting up by one step ensures the optimum
        # remains within the search window even if it drifts upward.
        alpha_start = alpha_h[best_idx] * (10.0 ** alpha_step)

        m = np.clip(m + delta_h[best_idx], ln_rho_min, ln_rho_max)
        d_pred = mod_h[best_idx]
        model_history.append(m.copy())

        # The parabola step is terminal - stop immediately after applying it.
        if overshoot:
            rms_history.append(rms_h[best_idx])
            #print(f"  Parabola applied - stopping.")
            break

    # ---- Optional sensitivity ----
    sensitivity = None
    if calc_sens:
        Jf          = _build_jacobian(m)
        sensitivity = np.sqrt(np.sum(Jf ** 2, axis=0))

    return {
        'log_resistivities': m,
        'resistivities':     np.exp(m),
        'thicknesses':       thicknesses,
        'model_history':     model_history,
        'rms_history':       rms_history,
        'J_history':         J_history if store_J else None,
        'sensitivity':       sensitivity,
        'times':             times,
        'obs_data':          obs_data,
        'n_iter':            len(rms_history),
    }

