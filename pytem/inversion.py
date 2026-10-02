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
  - _alpha_search      : log-spaced regularisation-strength ladder search
  - invert             : regularised Gauss-Newton inversion loop
  - invert_joint       : joint inversion of gate-averaged datasets (L2 or L1)
  - invert_sci         : spatially constrained inversion of many soundings
"""

import time as _time_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

import numpy as np

from .transform_weights import MU0, HANKEL_FILTERS, FOURIER_FILTERS, EULER_PARAMS
from .backends import HAS_CUDA
from .kernels_numba import HAS_NUMBA, KERNEL_MODES
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
             jacobian_mode='log', kernel='exact'):

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
    kernel            : str, default 'exact'
        Numba recursion: 'exact', 'fast_sqrt' or 'vectorized' (see kernels_numba)

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
                    e_eta, e_A, filter_weights, KERNEL_MODES[kernel])
            else:
                dbdt, J_raw = _tem_square_grad_euler_jit(
                    times, thicknesses, resistivities,
                    dist_q, area_w, float(quad_scale),
                    h_base, h_j0, MU0, e_eta, e_A, filter_weights, altitude, KERNEL_MODES[kernel])
        else:
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_jit(
                    times, thicknesses, resistivities, lam, lam_kern, MU0,
                    f_base, f_sin, filter_weights, KERNEL_MODES[kernel])
            else:
                dbdt, J_raw = _tem_square_grad_jit(
                    times, thicknesses, resistivities,
                    dist_q, area_w, float(quad_scale),
                    h_base, h_j0, MU0, f_base, f_sin, filter_weights, altitude, KERNEL_MODES[kernel])

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


def getR(resistivities, damp=1e-4, weights=None):
    """First-order roughness (smoothness) matrix with optional damping.

    Pass ``weights`` (see :func:`_irls_weights`) to reweight each first
    difference row, turning the default L2 (smooth) penalty into an IRLS
    approximation of an L1 (blocky / minimum-support) penalty.
    """
    n_params = resistivities.size
    D = np.zeros((n_params - 1, n_params))
    for k in range(n_params - 1):
        D[k, k] = -1.0
        D[k, k + 1] = 1.0
    DTD = D.T @ np.diag(weights) @ D if weights is not None else D.T @ D
    R = DTD + damp * np.eye(n_params)
    return R


def _irls_weights(m, beta=1e-2):
    """IRLS row-weights approximating an L1 (blocky) roughness penalty.

    Minimising ``sum(|D m|)`` (sparse/blocky layer-to-layer contrasts) is
    approximated by iteratively reweighted least squares: each first
    difference is weighted by ``1 / sqrt((D m)_k^2 + beta^2)``, so large
    contrasts are penalised less than small ones -- unlike the fixed L2
    penalty, which penalises every contrast equally regardless of size and
    therefore always smooths across sharp boundaries. ``beta`` is a small
    stabilising floor (in log-resistivity units) below which contrasts are
    treated as smooth; smaller beta -> blockier results but a harder,
    more nonlinear problem.
    """
    Dm = np.diff(np.asarray(m, dtype=float))
    return 1.0 / np.sqrt(Dm ** 2 + beta ** 2)


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


def _trial_log_rms(obs_data, mod, w):
    """Score all fitted gates; invalid trials must not win by dropping gates."""
    if (np.size(mod) == 0 or np.any(~np.isfinite(mod)) or np.any(mod <= 0)
            or np.any(~np.isfinite(obs_data)) or np.any(obs_data <= 0)
            or np.any(~np.isfinite(w)) or np.any(w <= 0)):
        return np.inf
    residual = np.log(obs_data) - np.log(mod)
    return float(np.sqrt(np.mean((w * residual) ** 2)))


def _backtrack_rms(m, delta, ln_rho_min, ln_rho_max, fwd_fn, obs_data, w,
                    rms_current, max_halving=6):
    """Bounds backtrack (see :func:`_backtrack`), *then* shrink the step
    further if it makes the fit worse than the current model.

    ``_backtrack`` only guards against the trial model leaving
    ``[ln_rho_min, ln_rho_max]``; a full Gauss-Newton step can still be
    accepted even when the linearisation is poor (e.g. near a sharp
    resistivity contrast) and the resulting RMS is far worse than before it.
    Used per trial with ``step_backtrack=True``, or as a fallback after a
    failed alpha ladder with ``step_backtrack='auto'``.
    Halves the step length (like ``_backtrack``, but keyed on the RMS
    rather than the bounds) until it no longer increases the RMS, or
    ``max_halving`` halvings are exhausted (falls back to the smallest step
    tried).
    """
    step = 1.0
    trial = mod = rms = None
    for i in range(max_halving + 1):
        trial, actual_step = _backtrack(m, step * delta, ln_rho_min, ln_rho_max)
        mod = fwd_fn(trial)
        rms = _trial_log_rms(obs_data, mod, w)
        if np.isfinite(rms) and (rms < rms_current or i == max_halving):
            return trial, step * actual_step, mod, rms
        if i == max_halving:
            return trial, step * actual_step, mod, rms
        step *= 0.5
    return trial, actual_step, mod, rms


def _fitted_alpha(alphas, rms):
    """Alpha at which the trials' RMS crosses 1, or None without a bracket.

    A quadratic through all trials in log10(alpha) (a line for two trials);
    the largest root inside the bracketing pair is used, else the linear
    interpolation of that pair.
    """
    x, y = np.log10(alphas), np.asarray(rms, dtype=float)
    crossing = np.flatnonzero((y[:-1] - 1.0) * (y[1:] - 1.0) <= 0.0)
    if crossing.size == 0 or not np.all(np.isfinite(y)):
        return None
    k = crossing[0]
    roots = np.roots(np.polyfit(x, y - 1.0, min(2, x.size - 1)))
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= min(x[k:k + 2])) & (roots <= max(x[k:k + 2]))]
    if roots.size:
        return float(10.0 ** roots.max())
    if y[k + 1] == y[k]:
        return None
    return float(10.0 ** (x[k] + np.clip((1.0 - y[k]) / (y[k + 1] - y[k]), 0.0, 1.0) * (x[k + 1] - x[k])))


def _step_functions(thicknesses, t_step, tx_size, geometry, rx_x, rx_y, n_quad, use_numba, use_cuda,
                    transform, system_filter, kernel, tx_height, rx_height):
    """Step response on ``t_step`` and its Jacobian d(step)/d(ln rho), as functions of ln(rho)."""
    common = dict(use_numba=use_numba, use_cuda=use_cuda, transform=transform, system_filter=system_filter,
                  kernel=kernel, tx_height=tx_height, rx_height=rx_height)

    def step(log_rho):
        if geometry == 'square_offset':
            return -fwd_square_offset(thicknesses, np.exp(log_rho), tx_size, rx_x, rx_y, t_step,
                                      current=1.0, signal=-1, n_quad=n_quad, **common)
        return -fwd_circle_offset(thicknesses, np.exp(log_rho), tx_size, rx_x, t_step,
                                  current=1.0, signal=-1, **common)

    def jacobian(log_rho):
        return getJ_ana(thicknesses=thicknesses, log_resistivities=log_rho, tx_size=tx_size, times=t_step,
                        geometry=geometry, rx_x=rx_x, rx_y=rx_y, n_quad=n_quad, jacobian_mode='absolute', **common)

    return step, jacobian


def _alpha_search(alpha_start, alpha_steps, Jw, dw, R, m,
                  thicknesses, fwd_fn, obs_data, w,
                  ln_rho_min, ln_rho_max, alpha_step=1/9, rms_current=np.inf,
                  plot=False, verbose=True, step_backtrack=False, fit_target=False):
    """Log-spaced regularisation-strength ladder search.

    Tests ``alpha_steps`` regularisation strengths starting from
    ``alpha_start`` on a log-spaced ladder defined by ``getAlpha`` and
    evaluates the RMS for each, stopping early once a trial reaches RMS < 1
    or the RMS stops improving. The caller picks the accepted trial (see
    ``invert``/``invert_joint``): the strongest regularisation that still
    reaches RMS <= 1, or the lowest-RMS trial if none did. Trailing the
    ladder with a weaker step is deliberately incremental - true
    convergence (RMS <= 1 on the accepted, freshly re-evaluated model) is
    checked again at the top of the next outer iteration, so a trial that
    undershoots RMS = 1 by a wide margin simply gets refined further next
    time rather than being accepted outright.

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
    verbose      : bool       print the per-alpha RMS trace (default True)
    step_backtrack : bool or 'auto'
        False uses bounds-only steps. True backtracks each alpha trial.
        'auto' first tries the normal ladder. Only if no trial improves RMS,
        try up to six halvings of each of the two best bounds-safe steps,
        stopping at the first improvement. No repeated full-step evaluation.
    fit_target   : bool       when a trial undershoots RMS 1 (below 0.995),
                              run one more at the alpha interpolated to RMS 1
                              (see ``_fitted_alpha``), so the smoothest model
                              that fits can be kept

    Returns
    -------
    alpha_hist : list of float    tested alpha values
    rms_hist   : list of float    corresponding RMS values
    delta_hist : list of ndarray  corresponding model deltas (m_trial - m)
    mod_hist   : list of ndarray  corresponding forward responses for each trial
    """
    if step_backtrack not in (False, True, 'auto'):
        raise ValueError("step_backtrack must be False, True, or 'auto'.")
    if alpha_steps < 1:
        raise ValueError('alpha_steps must be at least 1.')
    alpha_hist, rms_hist, delta_hist, mod_hist = [], [], [], []

    def run_trial(alpha):
        avs   = getAlphas(alpha, thicknesses)
        delta = _gn_solve(Jw, dw, R, avs, m)
        if step_backtrack == True:
            trial, step, mod, rms = _backtrack_rms(
                m, delta, ln_rho_min, ln_rho_max, fwd_fn, obs_data, w,
                rms_current=rms_current)
        else:
            trial, step = _backtrack(m, delta, ln_rho_min, ln_rho_max)
            mod   = fwd_fn(trial)
            rms   = _trial_log_rms(obs_data, mod, w)
        if verbose:
            print(f"    Alpha = {alpha:.2f},  RMS = {rms:.2f}"
                  + (f"  (step = {step:.2f})" if step < 1.0 else ""))
        alpha_hist.append(alpha)
        rms_hist.append(rms)
        delta_hist.append(trial - m)
        mod_hist.append(mod)
        return rms

    for i in range(alpha_steps):
        rms = run_trial(getAlpha(alpha_start, step=i, alpha_step=alpha_step))
        if rms < 1.0:
            if verbose:
                print("    RMS below 1 - stopping ladder.")
            fitted = _fitted_alpha(alpha_hist, rms_hist) if fit_target and rms < 0.995 else None
            if fitted is not None:
                run_trial(fitted)
            break

        if len(rms_hist) > 1 and rms > rms_hist[-2] and min(rms_hist[:-1]) < rms_current:
            if verbose:
                print("    RMS increased - stopping alpha search early.")
            break

    if step_backtrack == 'auto' and min(rms_hist) >= rms_current:
        candidates = np.argsort(rms_hist, kind='stable')[:2]
        for candidate in candidates:
            trial, step, mod, rms = _backtrack_rms(
                m, 0.5 * delta_hist[candidate], ln_rho_min, ln_rho_max,
                fwd_fn, obs_data, w, rms_current, max_halving=5)
            alpha_hist.append(alpha_hist[candidate])
            rms_hist.append(rms)
            delta_hist.append(trial - m)
            mod_hist.append(mod)
            if verbose:
                print(f"    Auto backtrack: RMS = {rms:.3f}, step = {0.5 * step:.4f}")
            if rms < rms_current:
                break

    if plot:
        import matplotlib.pyplot as _plt
        x_data = np.log10(np.array(alpha_hist))
        y_data = np.array(rms_hist)
        fig, ax = _plt.subplots(figsize=(5, 3.5))
        ax.plot(x_data, y_data, 'o-', color='C1', zorder=5,
                label='Tested alphas')
        ax.axhline(1.0, color='k', ls='--', lw=1, label='RMS = 1 target')
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
           circle_warmstart=False, regularization='l2', l1_beta=1.0,
           step_backtrack=False):
    """Regularised Gauss-Newton inversion for 1-D layered-earth TEM.

    Minimises  phi(m) = ||W (ln d_obs - ln d_pred(m))||^2 + alpha * m^T R m
    using iterative Gauss-Newton updates with a log-spaced alpha ladder
    search targeting RMS = 1.

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
    regularization      : 'l2' (default, smooth Occam-style) or 'l1' (IRLS
                          approximation of a blocky/sharp-boundary penalty,
                          see :func:`_irls_weights`)
    l1_beta             : float  starting IRLS stabilising floor for
                          regularization='l1' (default 1.0, deliberately
                          large/stable); cooled down each iteration alongside
                          alpha (floored at 1e-4) so the model progressively
                          sharpens instead of fighting a fixed stabilisation
                          level for the whole inversion
    step_backtrack      : bool   opt-in RMS-based step-length backtracking
                          (default False, preserves original behaviour): the
                          plain bounds-only backtrack (`_backtrack`) can still
                          accept a Gauss-Newton step that makes the RMS much
                          worse (e.g. near a sharp resistivity contrast, where
                          the linearisation is poor); with this on, the step
                          is halved until it no longer increases the RMS
                          (see :func:`_backtrack_rms`)

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
            regularization=regularization, l1_beta=l1_beta,
            step_backtrack=step_backtrack,
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
            regularization=regularization, l1_beta=l1_beta,
            step_backtrack=step_backtrack,
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
    # L1 IRLS stabiliser is cooled alongside alpha: starting large (behaving
    # like L2, for a stable first step) and shrinking each iteration lets the
    # model progressively sharpen instead of being stuck fighting a fixed
    # stabilisation level for the whole inversion.
    l1_beta_it    = l1_beta

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

        R  = getR(m, weights=_irls_weights(m, beta=l1_beta_it) if regularization == 'l1' else None)
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
            step_backtrack=step_backtrack,
        )

        # Select the strongest regularisation that still reaches RMS <= 1;
        # if none did, fall back to the lowest-RMS trial on the ladder.
        rms_arr = np.array(rms_h)
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
        if regularization == 'l1':
            l1_beta_it = max(l1_beta_it * (10.0 ** -alpha_step), 1e-4)

        m = np.clip(m + delta_h[best_idx], ln_rho_min, ln_rho_max)
        d_pred = mod_h[best_idx]
        model_history.append(m.copy())

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


def compute_doi(weighted_log_jacobian, thicknesses, threshold=0.8,
                conservative_threshold=1.2):
    """Estimate sensitivity-based depth of investigation (DOI) in metres.

    Uses the cumulative-sensitivity approach of Christiansen & Auken (2012).
    ``weighted_log_jacobian`` must contain only data rows, with entries
    d ln(predicted data) / d ln(resistivity) divided by relative data error.
    Do not include regularization rows or normalize the columns.
    ``thicknesses`` contains the N-1 finite layer thicknesses for N columns.

    Absolute column sums are accumulated from the bottom upward. DOI is
    interpolated at layer tops where this cumulative sensitivity crosses
    the threshold. Defaults 0.8 and 1.2 give standard and more conservative
    estimates; they are adjustable, not calibrated uncertainty bounds.

    Returns a dict with ``standard``, ``conservative`` (metres),
    ``standard_capped``, ``conservative_capped``, ``sensitivity``,
    ``cumulative``, ``layer_tops``, and the two thresholds. A capped depth is
    the top of the bottom half-space, not a resolved DOI within it. A
    single half-space therefore has a depth cap of zero. A DOI of zero
    without a cap means total sensitivity does not exceed the threshold.
    This diagnostic does not establish that an inversion has converged.
    """
    weighted_log_jacobian = np.asarray(weighted_log_jacobian, dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    if (weighted_log_jacobian.ndim != 2
            or min(weighted_log_jacobian.shape) == 0
            or not np.all(np.isfinite(weighted_log_jacobian))):
        raise ValueError('weighted_log_jacobian must be a finite, nonempty 2D array.')
    if (thicknesses.ndim != 1
            or thicknesses.size != weighted_log_jacobian.shape[1] - 1
            or not np.all(np.isfinite(thicknesses))
            or np.any(thicknesses <= 0)):
        raise ValueError('Provide N-1 finite positive thicknesses for N model layers.')
    if (not np.isfinite(threshold) or not np.isfinite(conservative_threshold)
            or not 0 < threshold <= conservative_threshold):
        raise ValueError('Require 0 < threshold <= conservative_threshold, both finite.')

    layer_tops = np.r_[0.0, np.cumsum(thicknesses)]
    sensitivity = np.abs(weighted_log_jacobian).sum(axis=0)
    cumulative = np.cumsum(sensitivity[::-1])[::-1]

    def crossing(level):
        if cumulative[0] <= level:
            return 0.0, False
        if cumulative[-1] >= level:
            return float(layer_tops[-1]), True
        return float(np.interp(level, cumulative[::-1], layer_tops[::-1])), False

    standard, standard_capped = crossing(threshold)
    conservative, conservative_capped = crossing(conservative_threshold)
    return {
        'standard': standard, 'conservative': conservative,
        'standard_capped': standard_capped,
        'conservative_capped': conservative_capped,
        'sensitivity': sensitivity, 'cumulative': cumulative,
        'layer_tops': layer_tops, 'threshold': float(threshold),
        'conservative_threshold': float(conservative_threshold),
    }


def invert_joint(fit_systems, thicknesses, rho_start, t_step, tx_size, geometry,
                 rx_x=0.0, rx_y=0.0, rho_min=0.1, rho_max=1e5,
                 alpha_steps=5, alpha_step=1 / 9, maxit=15, n_quad=5,
                 use_numba=True, use_cuda=False, transform='dlf', verbose=True,
                 step_backtrack='auto', calc_doi=False, doi_threshold=0.8,
                 doi_conservative_threshold=1.2, doi_refinement=1,
                 system_filter=None, kernel='exact', tx_height=0.0, rx_height=0.0,
                 norm='l2', half_space_start=False, adaptive_alpha=False):
    """
    Joint Gauss-Newton inversion of one or more pre-gated datasets (e.g. a
    station's LM and HM soundings) sharing one layered-earth model.

    Unlike ``invert()``, which convolves the waveform at instantaneous gate
    centre times (``waveform.setup_waveform``), this accepts one or more
    precomputed gate-averaging matrices from ``setup_waveform_matrix`` (its
    ``.matrix`` attribute) -- so both the transmitter waveform *and* the
    finite receiver gate width are accounted for. This matters most for
    early-time gates, where the gate width is a large fraction of the gate
    centre time.

    All ``fit_systems`` matrices must be built on the same ``t_step`` step-
    time grid (build it once and reuse for every dataset/station -- see
    ``setup_waveform_matrix``).

    Parameters
    ----------
    fit_systems : list of dict, each with keys
        'M'     : (n_gates, n_step) gate-averaging matrix for this dataset
        'obs'   : (n_gates,) observed data, positive
        'noise' : (n_gates,) absolute noise standard deviation
    thicknesses : (N-1,) layer thicknesses [m]
    rho_start   : (N,) initial resistivities [Ohm.m]
    t_step      : (n_step,) shared step-response time grid used to build
                 every ``M`` matrix in ``fit_systems``
    tx_size, geometry, rx_x, rx_y : forward-model geometry, as in
                 ``fwd_circle_offset`` / ``fwd_square_offset``
    rho_min, rho_max : resistivity bounds [Ohm.m]
    alpha_steps, alpha_step, maxit : Gauss-Newton / alpha-search controls
    n_quad      : square-loop quadrature order (ignored for circle geometries)
    use_numba, use_cuda : backend flags
    transform   : 'dlf' or 'euler'
    verbose     : print the per-iteration alpha-search trace (default True).
                 Set to False for clean output when running many stations in
                 parallel (e.g. from multiple threads), since the printed
                 trace is not thread-safe to redirect/capture per call.
    step_backtrack : bool or 'auto', default 'auto'
        Automatically try shorter steps only if the full alpha ladder fails
        to improve RMS. True backtracks each trial; False disables RMS
        backtracking. All modes retain bounds checks. See ``_alpha_search``.
    calc_doi : bool, default False
        Evaluate one extra Jacobian at the final model to compute DOI with
        the same gates, noise weights, geometry, and transform as the fit.
    doi_threshold, doi_conservative_threshold : float
        Cumulative-sensitivity thresholds passed to ``compute_doi``.
    doi_refinement : int, default 1
        Subdivide each finite inversion layer into this many equal layers for
        the DOI Jacobian only. Larger values provide finer DOI sampling
        without changing the inversion model or forward response.
    system_filter : callable or None
        Shared receiver transfer function H(omega) for all fit systems.
        Applied to both the step response and analytical Jacobian, including
        DOI. Defaults to no filtering. Gate matrices must not already include
        this receiver filter.
    kernel : 'exact', 'fast_sqrt' or 'vectorized'
        Numba recursion variant for forwards and Jacobians (see kernels_numba);
        'vectorized' is ~2x faster with rounding-level differences.
    tx_height, rx_height : float
        Transmitter and receiver heights above the ground [m].
    norm : 'l2' or 'l1'
        'l2' penalises squared steps between layers (smooth); 'l1' their
        absolute value (blocky), by reweighting the roughness from the
        current model every iteration (IRLS, epsilon 0.05).
    half_space_start : bool
        Start from the best of 33 half-spaces (1 Ohm m to 10 kOhm m) in
        place of ``rho_start``.
    adaptive_alpha : bool
        When an alpha trial undershoots RMS 1, run one more at the alpha
        interpolated to RMS 1 and keep the smoothest model that fits
        (RMS up to 1.005) in place of the first trial below 1.

    The inversion also stops ('stagnated') when the RMS improved by less
    than 1 % over the last 5 iterations.

    Returns
    -------
    dict with keys:
        'resistivities', 'log_resistivities' : final model
        'rms'          : final weighted log-RMS misfit
        'rms_history'  : RMS of the start model and after every iteration
        'n_iter'       : number of alpha searches attempted (at most maxit)
        'converged'    : bool, True if rms <= 1.0 was reached
        'termination_reason' : 'converged', 'stalled', 'stagnated',
             'max_iterations' or 'invalid_response'
        'alpha_final'  : float or None, the regularisation strength applied
             for the last accepted model update (None if no update was made)
        'observed'     : (n_d,) concatenated observed data (all fit_systems, in order)
        'predicted'    : (n_d,) concatenated final modelled data, same order/shape as 'observed'
        'n_gates'      : list of int, number of gates contributed by each fit_systems entry
        'doi'          : ``compute_doi`` result plus 'valid' and 'reason', or
                 None when calc_doi=False. Invalid predictions/weights
                 give NaN depths and valid=False, preserving the fit.
    """
    if step_backtrack not in (False, True, 'auto'):
        raise ValueError("step_backtrack must be False, True, or 'auto'.")
    if alpha_steps < 1 or maxit < 0:
        raise ValueError('alpha_steps must be positive and maxit nonnegative.')
    if (not isinstance(doi_refinement, (int, np.integer))
            or isinstance(doi_refinement, bool) or doi_refinement < 1):
        raise ValueError('doi_refinement must be a positive integer.')
    thicknesses = np.asarray(thicknesses, dtype=float)
    t_step = np.asarray(t_step, dtype=float)
    # Only evaluate step times with a nonzero weight in some gate: the padded
    # ends of the shared grid usually fall outside every gate (exact).
    used = np.any(np.vstack([np.asarray(f['M']) for f in fit_systems]) != 0, axis=0)
    t_step = t_step[used]
    fit_systems = [dict(f, M=np.asarray(f['M'])[:, used]) for f in fit_systems]
    observed = np.concatenate([np.asarray(f['obs'], dtype=float) for f in fit_systems])
    noise_abs = np.concatenate([np.asarray(f['noise'], dtype=float) for f in fit_systems])
    weights = observed / noise_abs

    step, jacobian_step = _step_functions(thicknesses, t_step, tx_size, geometry, rx_x, rx_y, n_quad, use_numba,
                                          use_cuda, transform, system_filter, kernel, tx_height, rx_height)

    def predict(log_rho):
        response = step(log_rho)
        return np.concatenate([f['M'] @ response for f in fit_systems])

    def weighted_log_rms(pred):
        residual = np.log(observed) - np.log(np.maximum(pred, 1e-300))
        return np.sqrt(np.mean((weights * residual) ** 2)), residual

    def jacobian(log_rho, pred):
        jac_abs = jacobian_step(log_rho)
        gate_jac = np.vstack([f['M'] @ jac_abs for f in fit_systems])
        result = np.zeros_like(gate_jac)
        positive = pred > 0
        result[positive] = gate_jac[positive] / pred[positive, None]
        return result

    lower, upper = np.log(float(rho_min)), np.log(float(rho_max))
    model = np.log(np.asarray(rho_start, dtype=float))
    roughness = getR(model)
    target = 1.005 if adaptive_alpha else 1.0
    if half_space_start:
        model = min((np.full(model.size, np.clip(np.log(10.0) * g / 8.0, lower, upper)) for g in range(33)),
                    key=lambda trial: _trial_log_rms(observed, predict(trial), weights))
    pred = predict(model)
    rms, resid = weighted_log_rms(pred)
    rms_history = [rms]
    weighted_jac = jacobian(model, pred) * weights[:, None]
    alpha = float(np.linalg.norm(weighted_jac.T @ (weights * resid), np.inf) + 1e-30)
    n_iter = 0
    termination_reason = 'max_iterations'
    alpha_applied = None  # the fitted alpha actually applied to the final model update

    # pred/rms/resid always describe the current model: the accepted trial's
    # response is reused, and the initial Jacobian serves the first iteration.
    for iteration in range(maxit):
        if not np.isfinite(_trial_log_rms(observed, pred, weights)):
            termination_reason = 'invalid_response'
            break
        if rms <= target:
            break
        n_iter += 1
        if weighted_jac is None:
            weighted_jac = jacobian(model, pred) * weights[:, None]
        if norm == 'l1':  # IRLS weights from the current model
            roughness = getR(model, weights=_irls_weights(model, 0.05))
        alpha_h, rms_h, delta_h, mod_h = _alpha_search(
            alpha, alpha_steps, weighted_jac, weights * resid, roughness, model,
            thicknesses, predict, observed, weights, lower, upper,
            alpha_step=alpha_step, rms_current=rms, plot=False, verbose=verbose,
            step_backtrack=step_backtrack, fit_target=adaptive_alpha)
        # Select the strongest regularisation that still reaches the target RMS;
        # if none did, fall back to the lowest-RMS trial on the ladder.
        rms_arr = np.asarray(rms_h)
        acceptable = np.flatnonzero(rms_arr <= target)
        best = (int(acceptable[np.argmax(np.asarray(alpha_h)[acceptable])])
                if acceptable.size else int(np.argmin(rms_arr)))
        if not np.isfinite(rms_h[best]) or rms_h[best] >= rms:
            termination_reason = 'stalled'
            break
        model = np.clip(model + delta_h[best], lower, upper)
        alpha_applied = alpha_h[best]
        alpha = alpha_h[best] * 10.0 ** alpha_step
        previous_rms = rms
        pred = mod_h[best]
        rms, resid = weighted_log_rms(pred)
        weighted_jac = None
        rms_history.append(rms)
        if rms_h[best] > target and previous_rms - rms_h[best] <= 1e-8 * max(1.0, previous_rms):
            termination_reason = 'stalled'
            break
        if len(rms_history) > 5 and rms_history[-6] - rms < 0.01 * rms_history[-6]:
            termination_reason = 'stagnated'
            break

    final_pred = pred
    final_rms, _ = weighted_log_rms(final_pred)
    valid_response = np.isfinite(_trial_log_rms(observed, final_pred, weights))
    converged = bool(valid_response and final_rms <= target)
    if converged:
        termination_reason = 'converged'
    elif not valid_response:
        termination_reason = 'invalid_response'
    doi = None
    if calc_doi:
        invalid_predictions = int(np.count_nonzero(~np.isfinite(final_pred) | (final_pred <= 0)))
        invalid_weights = int(np.count_nonzero(~np.isfinite(weights) | (weights <= 0)))
        if final_pred.size == 0 or invalid_predictions or invalid_weights:
            reason = (f'DOI unavailable: {final_pred.size} gates, '
                      f'{invalid_predictions} nonfinite/nonpositive predictions, '
                      f'{invalid_weights} nonfinite/nonpositive data weights.')
            doi = {
                'standard': np.nan, 'conservative': np.nan,
                'standard_capped': False, 'conservative_capped': False,
                'sensitivity': np.full(model.size, np.nan),
                'cumulative': np.full(model.size, np.nan),
                'layer_tops': np.r_[0.0, np.cumsum(thicknesses)],
                'threshold': float(doi_threshold),
                'conservative_threshold': float(doi_conservative_threshold),
                'valid': False, 'reason': reason,
            }
        else:
            doi_thicknesses = np.repeat(thicknesses / doi_refinement,
                                        doi_refinement)
            doi_model = np.concatenate([
                np.repeat(model[:-1], doi_refinement), model[-1:]])
            doi_jac_abs = getJ_ana(
                thicknesses=doi_thicknesses, log_resistivities=doi_model,
                tx_size=tx_size, times=t_step, geometry=geometry, rx_x=rx_x,
                rx_y=rx_y, n_quad=n_quad, use_numba=use_numba,
                use_cuda=use_cuda, transform=transform,
                jacobian_mode='absolute', system_filter=system_filter, kernel=kernel,
                tx_height=tx_height, rx_height=rx_height)
            doi_gate_jac = np.vstack([f['M'] @ doi_jac_abs for f in fit_systems])
            doi_jac = np.zeros_like(doi_gate_jac)
            positive = final_pred > 0
            doi_jac[positive] = doi_gate_jac[positive] / final_pred[positive, None]
            doi = compute_doi(doi_jac * weights[:, None], doi_thicknesses,
                              threshold=doi_threshold,
                              conservative_threshold=doi_conservative_threshold)
            doi['refinement'] = doi_refinement
            doi.update(valid=True, reason='')
    return {
        'log_resistivities': model,
        'resistivities': np.exp(model),
        'rms': final_rms,
        'rms_history': rms_history,
        'n_iter': n_iter,
        'converged': converged,
        'termination_reason': termination_reason,
        'alpha_final': alpha_applied,
        'observed': observed,
        'predicted': final_pred,
        'n_gates': [len(f['obs']) for f in fit_systems],
        'doi': doi,
    }


def invert_stations(station_ids, prepare_station, invert_kwargs=None,
                    max_workers=1, progress_callback=None):
    """Run the joint inversion for a collection of prepared stations.

    ``prepare_station`` owns data-format-specific work. It receives one
    station ID and returns a mapping with a required ``fit_systems`` entry and
    optional metadata entries. The fit systems are passed to
    :func:`invert_joint`; metadata is copied onto that station's result.
    """
    station_ids = list(station_ids)
    invert_kwargs = dict(invert_kwargs or {})
    if max_workers < 1:
        raise ValueError('max_workers must be positive.')
    if 'fit_systems' in invert_kwargs:
        raise ValueError("invert_kwargs must not contain 'fit_systems'.")

    def run_station(station):
        prepared = prepare_station(station)
        if not isinstance(prepared, dict) or 'fit_systems' not in prepared:
            raise TypeError("prepare_station must return a dict with 'fit_systems'.")
        station_t0 = _time_mod.perf_counter()
        result = invert_joint(prepared['fit_systems'], **invert_kwargs)
        result = dict(result)
        result['station'] = station
        result.update({key: value for key, value in prepared.items()
                       if key != 'fit_systems'})
        result['elapsed_s'] = _time_mod.perf_counter() - station_t0
        return result

    results = {}
    total = len(station_ids)
    if max_workers == 1:
        for completed, station in enumerate(station_ids, start=1):
            result = run_station(station)
            results[station] = result
            if progress_callback is not None:
                progress_callback(completed, total, result)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_station, station): station
                       for station in station_ids}
            for completed, future in enumerate(as_completed(futures), start=1):
                station = futures[future]
                result = future.result()
                results[station] = result
                if progress_callback is not None:
                    progress_callback(completed, total, result)

    return {station: results[station] for station in station_ids}




# ============================================================
# Spatially constrained inversion
# ============================================================

def _neighbour_edges(x, y):
    """Pairs of neighbouring soundings: the Delaunay edges.

    Three far vertices are triangulated along with the soundings (as in a
    Bowyer-Watson construction), which leaves out the long, thin triangles
    along the hull that would tie distant soundings of a line together. A
    sounding left without an edge is joined to its nearest neighbour.
    """
    from itertools import combinations
    from scipy.spatial import Delaunay
    n = len(x)
    points = np.c_[x - np.mean(x) + 1e-7 * np.arange(n), y - np.mean(y) - 1.3e-7 * (np.arange(n) % 7)]  # jitter separates coincident points
    extent = max(1.0, np.abs(points).max()) if n else 1.0
    simplices = Delaunay(np.vstack([points, [[-40 * extent, -40 * extent], [40 * extent, -40 * extent], [0.0, 40 * extent]]])).simplices
    edges = {tuple(sorted(pair)) for simplex in simplices.tolist() for pair in combinations(simplex, 2) if max(pair) < n}
    for i in set(range(n)) - {i for edge in edges for i in edge}:
        if n > 1:
            distance = np.hypot(*(points - points[i]).T)
            distance[i] = np.inf
            edges.add(tuple(sorted((i, int(np.argmin(distance))))))
    return sorted(edges)


def invert_sci(stations, thicknesses, t_step, tx_size, geometry, x, y, elevation=None,
               rx_x=0.0, rx_y=0.0, rho_start=None, rho_min=0.1, rho_max=1e5,
               vertical_factor=3.0, lateral_factor=1.5, reference_distance=100.0, distance_power=0.5,
               elevation_constraints=True, adaptive=False, maxit=30, stop_at_half_fit=True,
               n_quad=5, use_numba=True, use_cuda=False, transform='dlf', system_filter=None,
               kernel='exact', tx_height=0.0, rx_height=0.0, verbose=True):
    """
    Spatially constrained inversion (fast SCI, after Lupus): all soundings in
    one system, with constraints between adjacent layers and between map
    neighbours, and Marquardt damping set by a step-length limit.

    Every constraint is "the difference in ln(rho) is 0 +/- sigma" with
    sigma = ln(factor): ``vertical_factor`` between adjacent layers, and
    ``lateral_factor`` between Delaunay neighbours ``reference_distance``
    apart, loosening as (distance / reference_distance)^``distance_power``.
    The data misfit is in ln(data), as in :func:`invert_joint`.

    Parameters
    ----------
    stations : list of ``fit_systems`` lists, one per sounding (see
        :func:`invert_joint`); all share ``t_step`` and the layer grid.
    x, y, elevation : (n_soundings,) positions and ground elevations [m].
    rho_start : None (each sounding starts from its best half-space), or
        (N,) / (n_soundings, N) resistivities.
    elevation_constraints : compare each layer with the neighbour's layers
        it overlaps at the same elevation (weighted by the overlap) instead
        of the same layer at the same depth.
    adaptive : SCI adaptive, the tightest constraints that fit. Both factors
        are raised to the powers 0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8, 4;
        each run starts from the previous models and runs to convergence,
        and the first run with a total RMS <= 1 is kept.
    maxit : iteration limit of one run.
    stop_at_half_fit : stop a run once the median RMS reaches 1, before the
        soundings that already fit are over-fitted (not used when adaptive).
    Other arguments: as :func:`invert_joint`.

    A sounding above RMS 1 that improves by less than 1 % over 3 iterations
    is abandoned: its model is frozen and it leaves the system.

    Returns
    -------
    dict with 'resistivities' (n_soundings, N), 'rms' (n_soundings,),
    'rms_history', 'predicted' and 'observed' (lists, one per sounding),
    'converged' (n_soundings,), 'abandoned' (iteration, 0 = never),
    'n_iter', 'termination_reason', 'vertical_factor', 'lateral_factor'
    (the factors of the run kept) and 'runs' (one dict per SCI run).
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    thicknesses = np.asarray(thicknesses, dtype=float)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    S, L = len(stations), thicknesses.size + 1
    z = np.zeros(S) if elevation is None else np.asarray(elevation, dtype=float)
    M = [np.vstack([f['M'] for f in station]) for station in stations]
    used = np.any(np.vstack(M) != 0, axis=0)
    M = [matrix[:, used] for matrix in M]
    obs = [np.concatenate([np.asarray(f['obs'], dtype=float) for f in station]) for station in stations]
    w = [o / np.concatenate([np.asarray(f['noise'], dtype=float) for f in station]) for o, station in zip(obs, stations)]
    step, jacobian_step = _step_functions(thicknesses, np.asarray(t_step, dtype=float)[used], tx_size, geometry, rx_x, rx_y,
                                          n_quad, use_numba, use_cuda, transform, system_filter, kernel,
                                          tx_height, rx_height)
    lower, upper = np.log(float(rho_min)), np.log(float(rho_max))
    median = lambda values: np.sort(values)[len(values) // 2] if len(values) else 0.0

    def residual(s, g):  # ln(d / g) / sigma, continued linearly below g = d / 1000
        floor = 1e-3 * obs[s]
        return w[s] * np.where(g >= floor, np.log(obs[s] / np.maximum(g, floor)), np.log(1e3) + (floor - g) / floor)

    rms = lambda s, g: float(np.sqrt(np.mean(residual(s, g) ** 2)))

    # Constraint rows sum(c_k m_k) = 0 +/- sigma: vertical ones between adjacent layers, lateral ones
    # between neighbours. ``unit`` is sigma without ln(factor); a and b are the soundings involved.
    tops = np.concatenate([[0.0], np.cumsum(thicknesses)])
    own_bottom, other_bottom = np.append(tops[1:], tops[-1] + thicknesses[-1]), np.append(tops[1:], np.inf)
    rows = [([s * L + j, s * L + j + 1], [-1.0, 1.0], 1.0, False, s, s) for s in range(S) for j in range(L - 1)]
    for a, b in _neighbour_edges(x, y):
        unit = (max(1.0, np.hypot(x[a] - x[b], y[a] - y[b])) / reference_distance) ** distance_power
        if not elevation_constraints:
            rows += [([a * L + j, b * L + j], [-1.0, 1.0], unit, True, a, b) for j in range(L)]
            continue
        for p, q in ((a, b), (b, a)):  # from each side, so the pair stays symmetric
            shift = z[q] - z[p]
            overlap = np.clip(np.minimum(own_bottom[:, None] + shift, other_bottom) - np.maximum(tops[:, None] + shift, tops), 0.0, None)
            for j in np.flatnonzero(overlap.sum(axis=1) > 0):  # none above the neighbour's surface
                k = np.flatnonzero(overlap[j])
                rows.append(([p * L + j, *(q * L + k)], [-1.0, *(overlap[j, k] / overlap[j].sum())], unit * np.sqrt(2.0), True, p, q))
    index = np.repeat(np.arange(len(rows)), [len(r[0]) for r in rows])
    C = sparse.csr_matrix((np.concatenate([r[1] for r in rows]), (index, np.concatenate([r[0] for r in rows]))), shape=(len(rows), S * L))
    unit, lateral, row_a, row_b = (np.array([r[k] for r in rows]) for k in (2, 3, 4, 5))

    if rho_start is None:  # each sounding from its best half-space; one step response serves them all
        values = np.clip(np.log(10.0) * np.arange(33) / 8.0, lower, upper)
        responses = [step(np.full(L, value)) for value in values]
        model = np.array([np.full(L, values[np.argmin([rms(s, M[s] @ r) for r in responses])]) for s in range(S)])
    else:
        model = np.log(np.broadcast_to(np.asarray(rho_start, dtype=float), (S, L))).copy()

    def run(vertical, lateral_factor_, model, half_fit):
        sigma = np.where(lateral, np.log(lateral_factor_), np.log(vertical)) * unit
        active, abandoned = np.ones(S, dtype=bool), np.zeros(S, dtype=int)
        pred = [M[s] @ step(model[s]) for s in range(S)]
        history = [[rms(s, pred[s])] for s in range(S)]
        measure = median([h[-1] for h in history])
        step_max, previous_level, n_iter, termination = 3.0, 1, 0, 'max_iterations'
        for _ in range(maxit):
            if not np.isfinite(measure) or not active.any():
                termination = 'invalid_response' if active.any() else 'every_sounding_abandoned'
                break
            weighted = sparse.diags((active[row_a] & active[row_b]) / sigma) @ C
            CtC = (weighted.T @ weighted).tocsr()
            blocks, rhs = [], np.zeros((S, L))
            for s in range(S):  # Gauss-Newton term and gradient of each sounding
                if active[s]:
                    A = (w[s] / np.maximum(pred[s], 1e-3 * obs[s]))[:, None] * (M[s] @ jacobian_step(model[s]))
                    rhs[s] = A.T @ residual(s, pred[s])
                blocks.append(A.T @ A if active[s] else np.zeros((L, L)))
            rhs = np.where(active[:, None], rhs - (CtC @ model.ravel()).reshape(S, L), 0.0).ravel()
            G = (sparse.block_diag(blocks) + CtC).tocsc()
            mean_diagonal = G.diagonal().mean()
            # Marquardt ladder (x3 per level), from three levels below the last accepted one: the first
            # step within the step limit that lowers the median RMS enough is accepted.
            for level in range(max(1, previous_level - 3), 41):
                delta = spsolve(G + 1e-4 * mean_diagonal * 3.0 ** (level - 1) * sparse.identity(S * L, format='csc'), rhs)
                if not np.all(np.isfinite(delta)) or np.abs(delta).max() > step_max:
                    continue
                candidate = np.clip(model + delta.reshape(S, L), lower, upper)
                candidate_pred = [M[s] @ step(candidate[s]) if active[s] else pred[s] for s in range(S)]
                for s in np.flatnonzero(active):  # a sounding that overshoots backs off towards its model
                    limit = np.sqrt(2.0) * history[s][-1]
                    for _ in range(4):
                        if rms(s, candidate_pred[s]) <= limit:
                            break
                        candidate[s] = 0.5 * (model[s] + candidate[s])
                        candidate_pred[s] = M[s] @ step(candidate[s])
                    if not rms(s, candidate_pred[s]) <= limit:
                        candidate[s], candidate_pred[s] = model[s], pred[s]
                change = (measure - median([rms(s, candidate_pred[s]) for s in np.flatnonzero(active)])) / measure
                if np.isfinite(change) and change >= 0.007:
                    break
                if step_max > 1.1:
                    step_max = max(step_max / 1.8, 1.1)
                elif np.isfinite(change) and change > 0.0:  # improving, but by less than the threshold
                    termination = 'minimum_step'
                    break
            else:
                termination = 'no_improvement'
                break
            n_iter, previous_level, model, pred, step_max = n_iter + 1, level, candidate, candidate_pred, step_max * 1.2
            for s in range(S):
                history[s].append(rms(s, pred[s]) if active[s] else history[s][-1])
            accepted = median([history[s][-1] for s in np.flatnonzero(active)])
            for s in np.flatnonzero(active):  # abandon a misfitting sounding that stopped improving
                h = history[s]
                if h[-1] > 1.005 and len(h) >= 4 and h[-4] - h[-1] < 0.01 * h[-4]:
                    active[s], abandoned[s] = False, n_iter
            measure = median([history[s][-1] for s in np.flatnonzero(active)])
            if verbose:
                print(f"  iteration {n_iter}: median RMS {accepted:.2f}, {active.sum()} of {S} soundings active")
            if half_fit and accepted <= 1.0:
                termination = 'median_rms_below_1'
            if termination != 'max_iterations':
                break
        return model, pred, history, abandoned, n_iter, termination

    runs = []
    for scale in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0) if adaptive else (1.0,):
        vertical, lateral_ = vertical_factor ** scale, lateral_factor ** scale
        if verbose:
            print(f"SCI: vertical factor {vertical:.3g}, lateral factor {lateral_:.3g}")
        model, pred, history, abandoned, n_iter, termination = run(vertical, lateral_, model, stop_at_half_fit and not adaptive)
        final = np.array([h[-1] for h in history])
        gates = np.array([o.size for o in obs])
        total = float(np.sqrt(np.sum(final ** 2 * gates) / gates.sum()))
        runs.append({'vertical_factor': vertical, 'lateral_factor': lateral_, 'total_rms': total,
                     'median_rms': float(np.median(final)), 'n_iter': n_iter, 'termination_reason': termination})
        if total <= 1.0:  # total, so that soundings that fit poorly count
            break
    return {'resistivities': np.exp(model), 'rms': final, 'rms_history': history, 'predicted': pred, 'observed': obs,
            'converged': final <= 1.005, 'abandoned': abandoned, 'n_iter': n_iter, 'termination_reason': termination,
            'vertical_factor': vertical, 'lateral_factor': lateral_, 'runs': runs}
