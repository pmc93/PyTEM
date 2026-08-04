"""
forward.py - TEM forward modelling functions and analytical solutions.

Internal helpers:
    _rx_scale, _resolve_filters, _resolve_gpu_filters,
    _apply_signal_scaling, _precompute_filter_dlf, _precompute_filter_euler,
    _vmd_greenfct, _build_circular_geometry, _build_square_geometry,
    _run_circular, _run_square

Forward functions:
    fwd_circle_central  - central-loop circular Tx, Rx at loop centre
    fwd_circle_offset   - circular Tx, Rx at radial offset
    fwd_square_central  - central square-loop Tx, Rx at loop centre
    fwd_square_offset   - square Tx, Rx at arbitrary (rx_x, rx_y)

Analytical solutions:
    fwd_analytical_central         - Ward & Hohmann eq 4.69a
    fwd_analytical_offset  - Ward & Hohmann eq 4.97

Adding a new geometry
---------------------
1. Write _build_<name>_geometry(...) to produce geometry arrays (distances
   and quadrature weights) describing source-receiver relationships.
2. Call _run_circular (J1 Hankel kernel) or _run_square (J0 VMD area integral).
3. Write a public fwd_<name>: cast inputs -> resolve filters -> precompute filter
   -> build geometry -> run kernel -> quadrature scale if needed
   -> _apply_signal_scaling -> scale by rx_fac.
"""

import numpy as np
from scipy.special import j0, erf
from collections import OrderedDict

from .transform_weights import MU0, HANKEL_FILTERS, FOURIER_FILTERS, EULER_PARAMS
from .backends import HAS_CUDA, GPU_HANKEL, GPU_FOURIER
from .recursion import te_reflection_coeff
from .kernels_numba import (
    HAS_NUMBA,
    _tem_circular_jit, _tem_square_jit,
    _tem_circular_euler_jit, _tem_square_euler_jit,
)

if HAS_CUDA:
    import cupy as cp
    from .kernels_gpu import (
        _tem_circular_gpu, _tem_square_gpu,
        _tem_circular_euler_gpu, _tem_square_euler_gpu,
    )


_FW_CACHE_MAX = 64
_FILTER_WEIGHTS_CACHE = OrderedDict()
_CIRC_GEOM_CACHE = OrderedDict()
_SQUARE_GEOM_CACHE = OrderedDict()


def _arr_cache_key(a):
    a = np.asarray(a)
    return (a.shape, a.dtype.str, a.tobytes())


def _cache_put(cache, key, value, max_size):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > max_size:
        cache.popitem(last=False)


# ============================================================================
# Helper functions
# ============================================================================

def _rx_scale(rx_area=1.0, rx_turns=1):
    """Receiver effective-area scaling factor.

    The voltage induced in a multi-turn coil scales with the total effective
    area N * A (turns times the area of one turn).  The kernels return dB/dt
    per unit effective area, so the final response is multiplied by this factor.
    """
    return float(rx_area) * float(rx_turns)


def _resolve_filters(hankel_filter, fourier_filter, transform, euler_order=11):
    """Look up the digital-filter / quadrature coefficient arrays.

    The forward model evaluates a Hankel transform (space -> wavenumber) and an
    inverse time transform (frequency or Laplace -> time).  Both are weighted
    sums over precomputed abscissae stored in module-level registries.

    Parameters
    ----------
    hankel_filter  : str  key into HANKEL_FILTERS (e.g. 'key_101', 'key_201')
    fourier_filter : str  key into FOURIER_FILTERS (e.g. 'key_81', 'key_101')
    transform      : str  'dlf' for the sine/cosine digital linear filter, or
                          'euler' for the Euler-Stehfest inverse Laplace scheme
    euler_order    : int  number of Euler terms when transform == 'euler'

    Returns
    -------
    tuple : (h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A)
        h_base, h_j0, h_j1 are the Hankel abscissae and J0 / J1 weights.
        For 'dlf' the Fourier triplet (f_base, f_sin, f_cos) is filled and the
        Euler slots are None; for 'euler' the Euler parameters (e_eta, e_A) are
        filled and the Fourier slots are None.
    """
    h_base, h_j0, h_j1 = HANKEL_FILTERS[hankel_filter]
    if transform == 'dlf':
        f_base, f_sin, f_cos = FOURIER_FILTERS[fourier_filter]
        return h_base, h_j0, h_j1, f_base, f_sin, f_cos, None, None
    else:  # 'euler'
        e_eta, e_A = EULER_PARAMS[euler_order]
        return h_base, h_j0, h_j1, None, None, None, e_eta, e_A


def _resolve_gpu_filters(hankel_filter, fourier_filter, transform):
    """Look up the same filter arrays as _resolve_filters, but as CuPy arrays.

    Used by the CUDA backend so the weights already live in device memory.
    Only the DLF (sine/cosine) path is GPU-accelerated, so the Fourier triplet
    is returned as None when transform != 'dlf'.

    Returns
    -------
    tuple : (d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos) device arrays.
    """
    d_h_base, d_h_j0, d_h_j1 = GPU_HANKEL[hankel_filter]
    if transform == 'dlf':
        d_f_base, d_f_sin, d_f_cos = GPU_FOURIER[fourier_filter]
        return d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos
    else:
        return d_h_base, d_h_j0, d_h_j1, None, None, None


def _apply_signal_scaling(dbdt, current, signal, transform):
    """Scale the raw transform output by current and the waveform convention.

    The kernels return a unit-current response in the natural sign of the
    chosen transform.  This applies the Tx current amplitude and the
    step-off / step-on / impulse sign and normalisation, in place.

    Parameters
    ----------
    dbdt      : ndarray  raw response, modified in place and returned
    current   : float    Tx current [A]
    signal    : int      -1 = step-off, +1 = step-on, 0 = impulse
    transform : str      'dlf' (applies the 2/pi sine-transform factor) or
                         'euler' (sign only)
    """
    if transform == 'euler':
        dbdt *= current * float(signal)
    else:  # dlf
        if signal == 1:
            dbdt *= -current * 2.0 / np.pi
        else:
            dbdt *= current * 2.0 / np.pi
    return dbdt


def _precompute_filter_dlf(system_filter, times, f_base):
    """Pre-evaluate the system filter at every DLF Fourier frequency.

    For the sine/cosine digital linear filter, each gate time t maps to a fixed
    set of angular frequencies omega = f_base / t.  Evaluating the (model-
    independent) system filter H(omega) once on this (n_t, n_f) grid lets the
    kernels reuse it instead of recomputing it on every forward call.

    Returns
    -------
    ndarray, shape (n_t, n_f), complex : H(omega) at each gate and abscissa.
    """
    n_t = len(times)
    n_f = len(f_base)
    fw = np.empty((n_t, n_f), dtype=np.complex128)
    for i in range(n_t):
        omega_row = f_base / times[i]
        fw[i] = system_filter(omega_row)
    return fw


def _precompute_filter_euler(system_filter, times, e_eta, e_A):
    """Pre-evaluate the system filter at every Euler-Stehfest Bromwich point.

    The Euler inverse Laplace transform samples the response on a vertical line
    in the complex s-plane, s = c + i*k*pi/t for k = 0..n_euler-1 with
    c = e_A / (2t).  Each Laplace point maps to omega = s / i, where the system
    filter is evaluated once per gate time.

    Returns
    -------
    ndarray, shape (n_t, n_euler), complex : H(omega) at each Bromwich point.
    """
    n_t = len(times)
    n_euler = len(e_eta)
    fw = np.empty((n_t, n_euler), dtype=np.complex128)
    for i in range(n_t):
        t = times[i]
        c = e_A / (2.0 * t)
        h_step = np.pi / t
        ks = np.arange(n_euler, dtype=float)
        s_vals = c + ks * h_step * 1j
        omega_vals = s_vals / 1j
        fw[i] = system_filter(omega_vals)
    return fw


def _vmd_greenfct(dist, omega, thicknesses, resistivities, h_base, h_j0,
                  altitude=0.0):
    """Vertical-magnetic-dipole Green's function via a J0 Hankel transform.

    Evaluates the vertical magnetic field at horizontal distance ``dist`` from a
    unit VMD over a layered earth, for one angular frequency.  The Hankel
    integral over wavenumber lam is approximated by the J0 digital linear
    filter: G = sum( r_TE(lam) * lam^2 * J0_weights ) / dist / (4 pi), where
    r_TE is the transverse-electric reflection coefficient of the layer stack.

    Parameters
    ----------
    dist          : float   source-element to receiver distance [m]
    omega         : complex angular frequency [rad/s]
    thicknesses   : ndarray layer thicknesses [m]
    resistivities : ndarray layer resistivities [Ohm.m]
    h_base, h_j0  : ndarray Hankel abscissae and J0 filter weights
    altitude      : float   total Tx+Rx elevation [m]; adds an
                            exp(-lam*altitude) upward-continuation factor to
                            each wavenumber (0.0 = on ground).

    Returns
    -------
    complex : the Green's function G(dist, omega).
    """
    lam = h_base / dist
    r_te = te_reflection_coeff(lam, omega, thicknesses, resistivities)
    weight = r_te * lam**2
    if altitude != 0.0:
        weight = weight * np.exp(-lam * altitude)
    return np.dot(weight, h_j0) / dist / (4.0 * np.pi)


def _filter_weights(system_filter, times, transform, f_base, e_eta, e_A):
    """Pre-evaluate the system filter at every transform frequency.

    Returns an (n_t, n_eval) complex array, or None when no filter is set.
    The result depends only on the gate times and the filter, not on the
    earth model, so callers can compute it once and reuse it.
    """
    if system_filter is None:
        return None
    key = (
        None if system_filter is None else id(system_filter),
        transform,
        _arr_cache_key(times),
        None if f_base is None else _arr_cache_key(f_base),
        None if e_eta is None else _arr_cache_key(e_eta),
        None if e_A is None else float(e_A),
    )
    cached = _FILTER_WEIGHTS_CACHE.get(key)
    if cached is not None:
        _FILTER_WEIGHTS_CACHE.move_to_end(key)
        return cached
    if transform == 'dlf':
        fw = _precompute_filter_dlf(system_filter, times, f_base)
    else:
        fw = _precompute_filter_euler(system_filter, times, e_eta, e_A)
    _cache_put(_FILTER_WEIGHTS_CACHE, key, fw, _FW_CACHE_MAX)
    return fw


def _integrate_python(hz_sec, times, signal, transform,
                      f_base, f_sin, f_cos, e_eta, e_A):
    """Reference pure-Python time-domain integration shared by all geometries.

    hz_sec(omega) returns the (already filtered) secondary field at one
    angular frequency.  This routine handles the DLF step-off/step-on case
    (signal = +-1), the DLF impulse case (signal = 0), and the Euler-Stehfest
    inverse Laplace transform, so the circular and square dispatchers only
    have to supply their own hz_sec closure.
    """
    dbdt = np.zeros(len(times))
    if transform == 'euler':
        n_eval = len(e_eta)
        ks = np.arange(n_eval, dtype=float)
        for i, t in enumerate(times):
            c = e_A / (2.0 * t)
            vals = np.array([np.real(MU0 * hz_sec((c + k * np.pi / t * 1j) / 1j))
                             for k in range(n_eval)])
            dbdt[i] = np.exp(e_A / 2.0) / t * np.dot(e_eta * (-1.0)**ks, vals)
    else:
        for i, t in enumerate(times):
            omega_pts = f_base / t
            if signal in (-1, 1):
                spectral = np.array([MU0 * np.imag(hz_sec(w)) for w in omega_pts])
                dbdt[i] = np.dot(spectral, f_sin) / t
            else:  # impulse (signal=0)
                spectral = np.array([np.real(-MU0 * 1j * w * hz_sec(w))
                                     for w in omega_pts])
                dbdt[i] = np.dot(spectral, f_cos) / t
    return dbdt



# ============================================================================
# Geometry builders
# ============================================================================

def _build_circular_geometry(tx_radius, h_base, rx_offset=0.0, altitude=0.0):
    """
    Pre-compute the per-wavenumber extra weights for a circular Tx loop.

    The circular-loop kernel evaluates a J1 Hankel integral at wavenumbers
    lam = h_base / tx_radius.  This routine returns the extra multiplicative
    weight applied to each lam to encode the receiver position and altitude.

    Central receiver (rx_offset = 0): weights are ones (the plain J1 loop
        integral, valid at the loop centre).
    Offset receiver: weights are J0(lam * rx_offset), the additional Bessel
        factor that moves the observation point off the loop axis.

    Parameters
    ----------
    tx_radius : float    Tx loop radius [m]
    h_base    : ndarray  Hankel abscissae
    rx_offset : float    radial Tx-centre to Rx distance [m] (0.0 = centre)
    altitude  : float    total elevation of Tx plus Rx above the ground [m].
        A raised or airborne system continues the field upward through the
        non-conductive air half-space, multiplying every wavenumber by
        exp(-lam * altitude).  altitude = 0.0 reproduces the on-ground response.

    Returns
    -------
    ndarray, shape (len(h_base),) : the per-wavenumber weights.
    """
    key = (float(tx_radius), float(rx_offset), float(altitude), _arr_cache_key(h_base))
    cached = _CIRC_GEOM_CACHE.get(key)
    if cached is not None:
        _CIRC_GEOM_CACHE.move_to_end(key)
        return cached

    lam = h_base / float(tx_radius)
    if rx_offset == 0.0:
        weights = np.ones(len(lam), dtype=float)
    else:
        weights = j0(lam * float(rx_offset))
    if altitude != 0.0:
        weights = weights * np.exp(-lam * float(altitude))

    _cache_put(_CIRC_GEOM_CACHE, key, weights, _FW_CACHE_MAX)
    return weights


def _build_square_geometry(tx_side, n_quad, rx_x=0.0, rx_y=0.0,
                           use_symmetry=True):
    """
    Build the Gauss-Legendre quadrature nodes and weights for a square Tx loop.

    The square loop is modelled as a continuous area distribution of vertical
    magnetic dipoles.  This routine returns, for each quadrature point, the
    horizontal distance from that point to the receiver and the corresponding
    area weight; the square kernel then sums area_w * G(dist) over all points.

    Parameters
    ----------
    tx_side      : float  square Tx side length L [m]
    n_quad       : int    Gauss-Legendre points per axis
    rx_x, rx_y   : float  receiver coordinates [m] (offset case only)
    use_symmetry : bool   see the two modes below

    Modes
    -----
    Central (use_symmetry=True):
        Integrates over one quadrant [0, L/2]^2, exploiting the x<->y symmetry
        of a centred receiver.  Off-diagonal node pairs (i != j) carry a factor
        of 2 to cover their mirror image.  The caller must multiply the final
        dbdt by 4 to account for all four quadrants.

    Offset (use_symmetry=False):
        Integrates over the full square [-L/2, L/2]^2 because an off-centre
        receiver breaks the symmetry.  Distances are clipped at 1e-6 m to avoid
        the 1/dist singularity when the receiver sits on a quadrature node.

    Returns
    -------
    (dist_q, area_w) : ndarrays of equal length
        dist_q : horizontal source-to-receiver distance at each node [m]
        area_w : Gauss-Legendre area weight at each node
    """
    key = (float(tx_side), int(n_quad), float(rx_x), float(rx_y), bool(use_symmetry))
    cached = _SQUARE_GEOM_CACHE.get(key)
    if cached is not None:
        _SQUARE_GEOM_CACHE.move_to_end(key)
        return cached

    L = float(tx_side)
    hs = L / 2.0
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    if use_symmetry:
        x_pts = hs / 2.0 * (1.0 + gl_nodes)
        w_pts = gl_weights * hs / 2.0
        dist_q, area_w = [], []
        for i in range(n_quad):
            for jj in range(i, n_quad):
                w = w_pts[i] * w_pts[jj] * (2.0 if i != jj else 1.0)
                dist_q.append(np.sqrt(x_pts[i]**2 + x_pts[jj]**2))
                area_w.append(w)
        out = np.asarray(dist_q, dtype=float), np.asarray(area_w, dtype=float)
        _cache_put(_SQUARE_GEOM_CACHE, key, out, _FW_CACHE_MAX)
        return out
    else:
        x_pts = hs * gl_nodes
        wx = hs * gl_weights
        if rx_y == 0.0:
            # Rx on the x-axis: the loop is mirror-symmetric about y=0, so a
            # source at (x_i, +y_j) and its mirror (x_i, -y_j) are equidistant
            # from the receiver. The GL nodes/weights are symmetric about 0, so
            # we keep only y_j >= 0 and double the off-axis weights. This halves
            # the number of point sources with no change in the result.
            keep = gl_nodes >= 0.0
            y_pts = hs * gl_nodes[keep]
            wy = hs * gl_weights[keep] * np.where(gl_nodes[keep] > 0.0, 2.0, 1.0)
            xx, yy = np.meshgrid(x_pts, y_pts, indexing='xy')
            ww_x, ww_y = np.meshgrid(wx, wy, indexing='xy')
            dist_q = np.sqrt((xx.ravel() - rx_x)**2 + yy.ravel()**2)
            dist_q = np.maximum(dist_q, 1e-6)
            out = dist_q, (ww_x * ww_y).ravel()
            _cache_put(_SQUARE_GEOM_CACHE, key, out, _FW_CACHE_MAX)
            return out
        xx, yy = np.meshgrid(x_pts, x_pts, indexing='xy')
        ww_x, ww_y = np.meshgrid(wx, wx, indexing='xy')
        dist_q = np.sqrt((xx.ravel() - rx_x)**2 + (yy.ravel() - rx_y)**2)
        dist_q = np.maximum(dist_q, 1e-6)
        out = dist_q, (ww_x * ww_y).ravel()
        _cache_put(_SQUARE_GEOM_CACHE, key, out, _FW_CACHE_MAX)
        return out


# ============================================================================
# Kernel dispatchers
# ============================================================================

def _run_circular(times, thicknesses, resistivities,
                  tx_radius, extra_weights,
                  h_base, h_j1, f_base, f_sin, f_cos, e_eta, e_A,
                  filter_weights, system_filter, signal, transform,
                  use_numba, use_cuda, hankel_filter, fourier_filter):
    """
    Dispatch the circular-loop kernel to the best available backend.

    Returns dbdt before _apply_signal_scaling and rx_fac scaling.
    Backends tried in order: CUDA > Numba > pure Python.
    The pure Python path also handles the impulse response (signal=0).
    """
    a = float(tx_radius)

    # PATH 1: CUDA GPU
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        d_extra = cp.asarray(extra_weights)
        if transform == 'euler':
            return _tem_circular_euler_gpu(
                times, thicknesses, resistivities, a,
                d_extra, d_h_base, d_h_j1,
                e_eta, e_A, filter_weights=filter_weights)
        else:
            return _tem_circular_gpu(
                times, thicknesses, resistivities, a,
                d_extra, d_h_base, d_h_j1, d_f_base, d_f_sin,
                filter_weights=filter_weights)

    # PATH 2: Numba JIT
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
        fw = filter_weights if filter_weights is not None else \
            np.ones((len(times), n_eval), dtype=np.complex128)
        if transform == 'euler':
            return _tem_circular_euler_jit(
                times, thicknesses, resistivities,
                a, extra_weights, MU0, h_base, h_j1, e_eta, e_A, fw)
        else:
            return _tem_circular_jit(
                times, thicknesses, resistivities,
                a, extra_weights, MU0, h_base, h_j1, f_base, f_sin, fw)

    # PATH 3: Pure Python (also handles impulse, signal=0)
    lam = h_base / a

    def _hz_sec(omega):
        r_te = te_reflection_coeff(lam, omega, thicknesses, resistivities)
        hz = 0.5 * np.dot(r_te * lam * extra_weights, h_j1)
        if system_filter is not None:
            hz *= system_filter(omega)
        return hz

    return _integrate_python(_hz_sec, times, signal, transform,
                             f_base, f_sin, f_cos, e_eta, e_A)


def _run_square(times, thicknesses, resistivities,
                dist_q, area_w,
                h_base, h_j0, f_base, f_sin, f_cos, e_eta, e_A,
                filter_weights, system_filter, signal, transform,
                use_numba, use_cuda, hankel_filter, fourier_filter,
                altitude=0.0):
    """
    Dispatch the square-loop (VMD area integral) kernel to the best backend.

    Returns dbdt before any quadrature scale factor, _apply_signal_scaling,
    and rx_fac. For fwd_square_central, multiply the result by 4 before
    calling _apply_signal_scaling to account for the one-quadrant integration.

    altitude : total Tx+Rx elevation [m] (0.0 = on ground).
    """
    # PATH 1: CUDA GPU
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        if transform == 'euler':
            return _tem_square_euler_gpu(
                times, thicknesses, resistivities,
                dist_q, area_w, d_h_base, d_h_j0,
                e_eta, e_A, filter_weights=filter_weights, altitude=altitude)
        else:
            return _tem_square_gpu(
                times, thicknesses, resistivities,
                dist_q, area_w, d_h_base, d_h_j0,
                d_f_base, d_f_sin, filter_weights=filter_weights, altitude=altitude)

    # PATH 2: Numba JIT
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
        fw = filter_weights if filter_weights is not None else \
            np.ones((len(times), n_eval), dtype=np.complex128)
        if transform == 'euler':
            return _tem_square_euler_jit(
                times, thicknesses, resistivities,
                dist_q, area_w, MU0,
                h_base, h_j0, e_eta, e_A, fw, altitude)
        else:
            return _tem_square_jit(
                times, thicknesses, resistivities,
                dist_q, area_w, MU0,
                h_base, h_j0, f_base, f_sin, fw, altitude)

    # PATH 3: Pure Python (also handles impulse, signal=0)
    def _hz_sec(omega):
        total = 0j
        for q in range(len(dist_q)):
            total += area_w[q] * _vmd_greenfct(
                dist_q[q], omega, thicknesses, resistivities, h_base, h_j0,
                altitude=altitude)
        if system_filter is not None:
            total *= system_filter(omega)
        return total

    return _integrate_python(_hz_sec, times, signal, transform,
                             f_base, f_sin, f_cos, e_eta, e_A)


# ============================================================================
# Public forward functions
# ============================================================================

def fwd_circle_central(thicknesses, resistivities, tx_radius, times,
                       current=1.0, signal=-1,
                       system_filter=None, use_numba=True, use_cuda=True,
                       rx_area=1.0, rx_turns=1,
                       tx_height=0.0, rx_height=0.0,
                       hankel_filter='key_101', fourier_filter='key_81',
                       transform='dlf', euler_order=11):
    """
    Central-loop TEM forward response for a 1-D layered earth.

    Computes dB/dt [V/m^2] at the centre of a circular Tx loop.
    Backends tried in order: CUDA > Numba > pure Python.
    The pure Python path also supports the impulse response (signal=0).

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)  layer thicknesses [m]
    resistivities : array-like, shape (N,)    layer resistivities [Ohm.m]
    tx_radius     : float                     Tx loop radius [m]
    times         : array-like, shape (n_t,)  gate times [s]
    current       : float, default 1.0        Tx current [A]
    signal        : int, default -1           -1=step-off, +1=step-on, 0=impulse
    system_filter : callable or None          H(omega) -> complex
    use_numba     : bool, default True
    use_cuda      : bool, default True
    rx_area       : float, default 1.0        Rx area [m^2]
    rx_turns      : int, default 1            Rx turns
    tx_height     : float, default 0.0        Tx elevation above ground [m]
    rx_height     : float, default 0.0        Rx elevation above ground [m]
        Tx and Rx heights are summed into a single air gap altitude =
        tx_height + rx_height.  The field travels down through tx_height to
        reach the earth and back up through rx_height to the receiver, so the
        two non-conductive air paths combine into one exp(-lam*altitude)
        upward-continuation factor (exponents of the same base add).
    hankel_filter : str, default 'key_101'
    fourier_filter: str, default 'key_81'
    transform     : str, default 'dlf'        'dlf' or 'euler'
    euler_order   : int, default 11

    Returns
    -------
    dbdt : ndarray, shape (n_t,)  dB/dt [V/m^2]
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)
    altitude = float(tx_height) + float(rx_height)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    filter_weights = _filter_weights(system_filter, times, transform, f_base, e_eta, e_A)

    extra_weights = _build_circular_geometry(tx_radius, h_base, altitude=altitude)
    dbdt = _run_circular(times, thicknesses, resistivities,
                         tx_radius, extra_weights,
                         h_base, h_j1, f_base, f_sin, f_cos, e_eta, e_A,
                         filter_weights, system_filter, signal, transform,
                         use_numba, use_cuda, hankel_filter, fourier_filter)
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


def fwd_circle_offset(thicknesses, resistivities, tx_radius, rx_offset,
                      times, current=1.0, signal=-1,
                      system_filter=None, use_numba=True, use_cuda=True,
                      rx_area=1.0, rx_turns=1,
                      tx_height=0.0, rx_height=0.0,
                      hankel_filter='key_101', fourier_filter='key_81',
                      transform='dlf', euler_order=11):
    """
    Offset-loop TEM forward response for a 1-D layered earth.

    Computes dB/dt [V/m^2] at a receiver displaced by rx_offset [m]
    from the centre of a circular Tx loop.

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)
    resistivities : array-like, shape (N,)
    tx_radius     : float   Tx loop radius [m]
    rx_offset     : float   Radial Tx-centre to Rx distance [m]
    times         : array-like, shape (n_t,)
    tx_height     : float, default 0.0  Tx elevation above ground [m]
    rx_height     : float, default 0.0  Rx elevation above ground [m]
    current, signal, system_filter, use_numba, use_cuda,
    rx_area, rx_turns, hankel_filter, fourier_filter,
    transform, euler_order : see fwd_circle_central

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)
    altitude = float(tx_height) + float(rx_height)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    filter_weights = _filter_weights(system_filter, times, transform, f_base, e_eta, e_A)

    extra_weights = _build_circular_geometry(tx_radius, h_base,
                                             rx_offset=float(rx_offset),
                                             altitude=altitude)
    dbdt = _run_circular(times, thicknesses, resistivities,
                         tx_radius, extra_weights,
                         h_base, h_j1, f_base, f_sin, f_cos, e_eta, e_A,
                         filter_weights, system_filter, signal, transform,
                         use_numba, use_cuda, hankel_filter, fourier_filter)
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


def fwd_square_central(thicknesses, resistivities, tx_side, times,
                       current=1.0, signal=-1,
                       system_filter=None, n_quad=5, use_numba=True,
                       use_cuda=True, use_symmetry=True,
                       rx_area=1.0, rx_turns=1,
                       tx_height=0.0, rx_height=0.0,
                       hankel_filter='key_101', fourier_filter='key_81',
                       transform='dlf', euler_order=11):
    """
    Central square-loop TEM forward response for a 1-D layered earth.

    Models the square Tx loop as an area distribution of VMDs integrated by
    2-D Gauss-Legendre quadrature over one quadrant (exploiting x<->y symmetry).

    Parameters
    ----------
    thicknesses, resistivities, times, current, signal,
    system_filter, use_numba, use_cuda, rx_area, rx_turns,
    hankel_filter, fourier_filter, transform, euler_order : see fwd_circle_central
    tx_side   : float  square Tx side length [m]
    n_quad        : int    GL quadrature points per axis (default 5)
    use_symmetry  : bool   exploit x<->y symmetry within quadrant (default True)
    tx_height     : float, default 0.0  Tx elevation above ground [m]
    rx_height     : float, default 0.0  Rx elevation above ground [m]

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)
    altitude = float(tx_height) + float(rx_height)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    filter_weights = _filter_weights(system_filter, times, transform, f_base, e_eta, e_A)

    dist_q, area_w = _build_square_geometry(
        tx_side, n_quad, use_symmetry=use_symmetry)
    dbdt = _run_square(times, thicknesses, resistivities,
                       dist_q, area_w,
                       h_base, h_j0, f_base, f_sin, f_cos, e_eta, e_A,
                       filter_weights, system_filter, signal, transform,
                       use_numba, use_cuda, hankel_filter, fourier_filter,
                       altitude=altitude)

    # One-quadrant integration: multiply by 4 to recover the full loop response.
    dbdt *= 4.0
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


def fwd_square_offset(thicknesses, resistivities, tx_side,
                      rx_x, rx_y, times,
                      current=1.0, signal=-1,
                      system_filter=None, n_quad=11, use_numba=True,
                      use_cuda=True, rx_area=1.0, rx_turns=1,
                      tx_height=0.0, rx_height=0.0,
                      hankel_filter='key_101', fourier_filter='key_81',
                      transform='dlf', euler_order=11):
    """
    Square-loop TEM response at an arbitrary receiver position (rx_x, rx_y).

    Integrates over the full square [-L/2, L/2]^2 without symmetry reduction.

    Parameters
    ----------
    thicknesses, resistivities, times, current, signal,
    system_filter, use_numba, use_cuda, rx_area, rx_turns,
    hankel_filter, fourier_filter, transform, euler_order : see fwd_circle_central
    tx_side : float  square Tx side length [m]
    rx_x        : float  Rx x-coordinate [m]
    rx_y        : float  Rx y-coordinate [m]
    n_quad      : int    GL quadrature points per axis (default 11)
    tx_height   : float, default 0.0  Tx elevation above ground [m]
    rx_height   : float, default 0.0  Rx elevation above ground [m]

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)
    altitude = float(tx_height) + float(rx_height)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    filter_weights = _filter_weights(system_filter, times, transform, f_base, e_eta, e_A)

    dist_q, area_w = _build_square_geometry(
        tx_side, n_quad, rx_x=float(rx_x), rx_y=float(rx_y),
        use_symmetry=False)
    dbdt = _run_square(times, thicknesses, resistivities,
                       dist_q, area_w,
                       h_base, h_j0, f_base, f_sin, f_cos, e_eta, e_A,
                       filter_weights, system_filter, signal, transform,
                       use_numba, use_cuda, hankel_filter, fourier_filter,
                       altitude=altitude)
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


# ============================================================================
# Analytical half-space solutions
# ============================================================================

def fwd_analytical_central(resistivity, tx_radius, times, current=1.0):
    """Analytical central-loop dB/dt over a homogeneous half-space.

    Closed-form reference solution (Ward & Hohmann, 1988, eq 4.69a) for the
    vertical field decay at the centre of a circular loop on a uniform earth.
    Useful for validating the layered forward model in the single-layer limit.

    Parameters
    ----------
    resistivity : float    half-space resistivity [Ohm.m]
    tx_radius   : float    Tx loop radius [m]
    times       : array-like  gate times [s]
    current     : float, default 1.0  Tx current [A]

    Returns
    -------
    dbdt : ndarray, shape (n_t,)  dB/dt [V/m^2]
    """
    times = np.asarray(times, dtype=float)
    sigma = 1.0 / resistivity
    a = float(tx_radius)

    theta = np.sqrt(MU0 * sigma / (4.0 * times))
    theta_a = theta * a
    dbdt = (-current / (sigma * a**3)) * (
        3.0 * erf(theta_a)
        - (2.0 / np.sqrt(np.pi)) * theta_a * (3.0 + 2.0 * theta_a**2)
        * np.exp(-theta_a**2)
    )
    return dbdt


def fwd_analytical_offset(resistivity, tx_radius, rx_offset, times,
                                   current=1.0):
    """Analytical offset dB/dt over a homogeneous half-space (far field).

    Closed-form reference solution (Ward & Hohmann, 1988, eq 4.97, differen-
    tiated with respect to time) for the vertical field at a radial offset from
    a vertical magnetic dipole on a uniform earth.  The loop is treated as a
    point dipole of moment M = current * pi * tx_radius^2, so the result is
    valid only in the far field where rx_offset >> tx_radius.

    Parameters
    ----------
    resistivity : float    half-space resistivity [Ohm.m]
    tx_radius   : float    Tx loop radius [m] (sets the dipole moment)
    rx_offset   : float    radial Tx-centre to Rx distance [m]
    times       : array-like  gate times [s]
    current     : float, default 1.0  Tx current [A]

    Returns
    -------
    dbdt : ndarray, shape (n_t,)  dB/dt [V/m^2]
    """
    times = np.asarray(times, dtype=float)
    sigma = 1.0 / resistivity
    r = float(rx_offset)
    M = current * np.pi * float(tx_radius)**2

    theta = np.sqrt(MU0 * sigma / (4.0 * times))
    u = theta * r
    dbdt = (MU0 * M / (8.0 * np.pi * r**3 * times)) * (
        (9.0 / u**2) * erf(u)
        - (1.0 / np.sqrt(np.pi)) * (18.0 / u + 12.0 * u + 8.0 * u**3)
        * np.exp(-u**2)
    )
    return dbdt
