"""
waveform.py - Waveform convolution for piecewise-linear transmitter waveforms.

Contains:
  setup_waveform         - precompute quadrature structure (empymod-style)
  convolve_waveform      - public API (dispatches to Numba JIT when available)
  setup_waveform_matrix  - convolution-free quintic B-spline matrix operator
  quintic_b_spline       - compact-support quintic interpolation kernel
  _log_interp_scalar     - Numba JIT log-time interpolation
  _convolve_waveform_jit - Numba JIT inner loops
"""

import numpy as np
from .kernels_numba import HAS_NUMBA

if HAS_NUMBA:
    import numba as nb
    _NB_OPTS = {'nogil': True, 'cache': True}

    @nb.njit(**_NB_OPTS)
    def _log_interp_scalar(log_t, log_st, sr):
        """Linear interpolation of step_response at a single log(t) value."""
        n = len(log_st)
        if log_t <= log_st[0]:
            return sr[0]
        if log_t >= log_st[n - 1]:
            return sr[n - 1]
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if log_st[mid] <= log_t:
                lo = mid
            else:
                hi = mid
        frac = (log_t - log_st[lo]) / (log_st[hi] - log_st[lo])
        return sr[lo] + frac * (sr[hi] - sr[lo])

    @nb.njit(**_NB_OPTS)
    def _convolve_waveform_jit(gate_times, wf_t, wf_I, log_st, sr,
                               gl_nodes, gl_weights):
        """Numba-compiled waveform convolution core."""
        n_gates = len(gate_times)
        n_seg = len(wf_t) - 1
        n_quad = len(gl_nodes)
        result = np.zeros(n_gates)

        for seg in range(n_seg):
            dt_seg = wf_t[seg + 1] - wf_t[seg]
            if abs(dt_seg) < 1e-30:
                continue
            slope = (wf_I[seg + 1] - wf_I[seg]) / dt_seg
            if abs(slope) < 1e-30:
                continue

            mid = 0.5 * (wf_t[seg + 1] + wf_t[seg])
            half = 0.5 * dt_seg

            for j in range(n_gates):
                tg = gate_times[j]
                accum = 0.0
                for q in range(n_quad):
                    tau = mid + half * gl_nodes[q]
                    t_eval = tg - tau
                    if t_eval <= 0.0:
                        continue
                    val = _log_interp_scalar(np.log(t_eval), log_st, sr)
                    accum += gl_weights[q] * val
                result[j] += -slope * half * accum

        return result


def setup_waveform(gate_times, waveform_times, waveform_currents, n_quad=8):
    """
    Precompute the quadrature structure for waveform convolution.

    Follows the empymod pattern: deduplicate all GL quadrature times across
    every (gate, segment) pair once at setup, so the forward model only needs
    to evaluate the step response at the returned ``comp_times`` array on each
    inversion iteration.

    Parameters
    ----------
    gate_times        : array-like  Measurement gate centre times [s].
    waveform_times    : array-like  Break points of the piecewise-linear waveform [s].
    waveform_currents : array-like  Current at each break point [A].
    n_quad            : int         Gauss-Legendre order per segment (default 8).

    Returns
    -------
    comp_times : ndarray, shape (n_unique,)
        Deduplicated set of times at which the step response must be evaluated.
    apply_waveform : callable
        ``apply_waveform(step_resp)`` where *step_resp* is a 1-D array of
        shape ``(n_unique,)`` or a 2-D array ``(n_unique, N)`` (e.g. Jacobian
        columns).  Returns the convolved result of shape ``(n_gates,)`` or
        ``(n_gates, N)``.
    """
    gate_times = np.asarray(gate_times, dtype=float)
    wf_t = np.asarray(waveform_times, dtype=float)
    wf_I = np.asarray(waveform_currents, dtype=float)

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    dt = np.diff(wf_t)
    dIdt = np.diff(wf_I) / dt

    act = np.abs(dIdt) > 1e-30
    t0 = wf_t[:-1][act]
    t1 = wf_t[1:][act]
    slope = dIdt[act]
    n_gates = gate_times.size
    n_active = t0.size

    if n_active == 0:
        # No ramp segments: waveform is constant - return zeros.
        def apply_waveform(step_resp):
            sr = np.asarray(step_resp)
            if sr.ndim == 1:
                return np.zeros(n_gates)
            return np.zeros((n_gates, sr.shape[1]))
        return np.array([1.0]), apply_waveform

    # Delays from each gate to segment endpoints: (n_gates, n_active)
    ta = gate_times[:, None] - t0[None, :]   # gate - segment_start
    tb = gate_times[:, None] - t1[None, :]   # gate - segment_end

    valid = ta > 0.0                          # gate must be after segment start
    tb = np.where(tb < 0.0, 0.0, tb)         # clamp: gate within segment -> tb=0

    # GL quadrature delay times: (n_gates, n_active, n_quad)
    comp_time = (0.5 * (tb - ta))[:, :, None] * gl_nodes \
              + (0.5 * (ta + tb))[:, :, None]

    # Per-(gate,segment) weight = Jacobian_of_interval_transform * slope
    seg_w = np.where(valid, 0.5 * (tb - ta) * slope[None, :], 0.0)  # (n_gates, n_active)

    # Mask out invalid (gate before segment start, or non-positive delay)
    valid3 = valid[:, :, None] & (comp_time > 0.0)   # (n_gates, n_active, n_quad)

    # Replace invalid times with zero so np.unique works cleanly
    comp_time_safe = np.where(valid3, comp_time, 0.0)

    # Deduplicate - same approach as empymod
    comp_times_flat, map_time = np.unique(comp_time_safe[valid3], return_inverse=True)

    # Clamp to a minimum time so the forward model is never called at
    # unphysically early times (empymod uses the same guard).
    _min_t = gate_times.min() * 1e-2
    comp_times_flat = np.where(comp_times_flat < _min_t, _min_t, comp_times_flat)

    # Build a weight matrix W of shape (n_gates, n_unique) once at setup.
    # apply_waveform then reduces to a single matmul: W @ step_resp.
    j_idx, k_idx, q_idx = np.where(valid3)
    entry_weights = seg_w[j_idx, k_idx] * gl_weights[q_idx]  # (n_valid,)
    W = np.zeros((n_gates, len(comp_times_flat)))
    np.add.at(W, (j_idx, map_time), entry_weights)

    def apply_waveform(step_resp):
        """Compute convolved response from step response at comp_times."""
        sr = np.asarray(step_resp)
        return W @ sr

    return comp_times_flat, apply_waveform


def convolve_waveform(step_times, step_response, waveform_times,
                      waveform_currents, gate_times, n_quad=8):
    """
    Convolve a step response with a piecewise-linear transmitter waveform.

    Computes V(t) = -integral (dI/dtau) * S(t - tau) dtau using
    Gauss-Legendre quadrature per waveform segment.

    Parameters
    ----------
    step_times       : array-like  Times at which the step response is known [s].
    step_response    : array-like  Step response values.
    waveform_times   : array-like  Break points of the piecewise-linear waveform [s].
    waveform_currents: array-like  Current at each break point [A].
    gate_times       : array-like  Output measurement gate centre times [s].
    n_quad           : int         Gauss-Legendre order per segment (default 8).

    Returns
    -------
    result : ndarray, shape (n_gates,)
    """
    gate_times = np.asarray(gate_times, dtype=float)
    wf_t = np.asarray(waveform_times, dtype=float)
    wf_I = np.asarray(waveform_currents, dtype=float)

    log_st = np.log(np.asarray(step_times, dtype=float))
    sr = np.asarray(step_response, dtype=float)

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    if HAS_NUMBA:
        return _convolve_waveform_jit(gate_times, wf_t, wf_I,
                                      log_st, sr, gl_nodes, gl_weights)

    # Vectorised NumPy fallback - empymod-style (Key, DIPOLE1D).
    # Reference: empymod.utils.check_waveform
    #
    # Strategy: pre-compute all GL quadrature delay times for every
    # (gate, segment) pair at once, deduplicate with np.unique, do a
    # single batched np.interp, then reassemble the weighted sum.

    dt = np.diff(wf_t)
    dIdt = np.diff(wf_I) / dt

    # Keep only segments with a non-zero current ramp.
    act = np.abs(dIdt) > 1e-30
    t0 = wf_t[:-1][act]       # segment start times
    t1 = wf_t[1:][act]        # segment end times
    slope = dIdt[act]

    if t0.size == 0:
        return np.zeros(gate_times.size)

    # Delays from each gate time to the segment endpoints: (n_gates, n_seg)
    #   ta = gate - t0  >=0 when gate is after segment start
    #   tb = gate - t1  clipped to 0 when gate is within the segment
    ta = gate_times[:, None] - t0[None, :]
    tb = gate_times[:, None] - t1[None, :]

    # A segment only contributes to gate j if the segment has already started.
    valid = ta > 0.0                          # (n_gates, n_seg)

    # Truncate: if gate is still within the segment, cap the upper delay at 0.
    tb = np.where(tb < 0.0, 0.0, tb)

    # GL quadrature delay points for every (gate, segment, node): (n_gates, n_seg, n_quad)
    # Change of interval from [-1,1] to [ta, tb]:
    #   delay = (tb-ta)/2 * g_x + (ta+tb)/2
    comp_time = (0.5 * (tb - ta))[:, :, None] * gl_nodes \
              + (0.5 * (ta + tb))[:, :, None]

    # Per-(gate,segment) integration weight: Jacobian * slope.
    # Jacobian = (tb-ta)/2  (<0 since ta>tb), so the sign already gives -slope*half.
    seg_w = np.where(valid, 0.5 * (tb - ta) * slope[None, :], 0.0)

    # Mask invalid entries (gate before segment, or non-positive delay).
    valid3 = valid[:, :, None] & (comp_time > 0.0)   # (n_gates, n_seg, n_quad)

    # Flatten, deduplicate, interpolate in one vectorised pass.
    t_flat = comp_time.ravel()
    v_flat = valid3.ravel()

    resp_flat = np.zeros(t_flat.size)
    if v_flat.any():
        t_good = t_flat[v_flat]
        uniq_t, inv = np.unique(t_good, return_inverse=True)
        resp_flat[v_flat] = np.interp(np.log(uniq_t), log_st, sr)[inv]

    resp = resp_flat.reshape(comp_time.shape)   # (n_gates, n_seg, n_quad)

    # Weighted sum: result[j] = sum_{k,q}  seg_w[j,k] * gl_weights[q] * resp[j,k,q]
    return np.einsum('jk,q,jkq->j', seg_w, gl_weights, resp)


# ---------------------------------------------------------------------------
# Convolution-free quintic B-spline matrix method
# ---------------------------------------------------------------------------
#
# The waveform + gating convolution is written as a single fixed matrix M so
# that every forward call reduces to one matmul  d = M @ B_step, where B_step
# is the step response sampled on a fixed log-time grid. M depends only on the
# timing (waveform breakpoints, gate times, step grid) and never on the earth
# model, so it is built once and reused across every inversion iteration.
#
# Interpolation kernel: the compactly-supported quintic B-spline,
# phi(x) = 0 for |x| > 2.

def _quintic_b_spline_S1(x):
    return 1 - (9 / 4) * x ** 2 * (1 - (2 / 9) * x * (1 + (5 / 2) * x * (1 - (2 / 5) * x)))


def _quintic_b_spline_S2(x):
    return -(1 / 2) * x * (1 - (11 / 6) * x * (1 - (2 / 11) * x * (1 + (5 / 2) * x * (1 - (2 / 5) * x))))


def quintic_b_spline(x):
    """Quintic B-spline interpolation kernel, vectorised. Support on [-2, 2]."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = (x >= -2) & (x <= -1); out[m] = _quintic_b_spline_S2(-x[m] - 1)
    m = (x > -1) & (x <= 0);   out[m] = _quintic_b_spline_S1(-x[m])
    m = (x > 0) & (x <= 1);    out[m] = _quintic_b_spline_S1(x[m])
    m = (x > 1) & (x <= 2);    out[m] = _quintic_b_spline_S2(x[m] - 1)
    return out



def _trapezoid_weights(t):
    w = np.zeros_like(t, dtype=float)
    w[1:-1] = (t[2:] - t[:-2]) / 2
    w[0] = (t[1] - t[0]) / 2
    w[-1] = (t[-1] - t[-2]) / 2
    return w


def _simpson_weights(t):
    """Composite Simpson weights on a uniform grid; trapezoid fallback otherwise."""
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return _trapezoid_weights(t)
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-10, atol=0.0):
        return _trapezoid_weights(t)
    h = dt[0]
    n = len(t)
    w = np.zeros(n, dtype=float)
    n_simp = n if (n - 1) % 2 == 0 else n - 1
    w[0] += h / 3
    w[n_simp - 1] += h / 3
    w[1:n_simp - 1:2] += 4 * h / 3
    w[2:n_simp - 1:2] += 2 * h / 3
    if n_simp < n:
        w[n_simp - 1] += h / 2
        w[n_simp] += h / 2
    return w


def _waveform_slope_kernel(waveform_times, waveform_currents, n_per_segment=81):
    """Sample -dI/dtau over each piecewise-linear segment into (tk, Wk) weights.

    Returns delta-representation times and weights approximating the convolution
    kernel -dI/dtau via Simpson quadrature per segment.
    """
    wf_t = np.asarray(waveform_times, dtype=float)
    wf_I = np.asarray(waveform_currents, dtype=float)
    tk_all, wk_all = [], []
    for t0, t1, I0, I1 in zip(wf_t[:-1], wf_t[1:], wf_I[:-1], wf_I[1:]):
        dt = t1 - t0
        if abs(dt) < 1e-30:
            continue
        slope = (I1 - I0) / dt
        if abs(slope) < 1e-30:
            continue
        tau = np.linspace(t0, t1, n_per_segment)
        tk_all.append(tau)
        wk_all.append(-slope * _simpson_weights(tau))
    if not tk_all:
        return np.array([0.0]), np.array([0.0])
    return np.concatenate(tk_all), np.concatenate(wk_all)


def _build_waveform_matrix(tk, Wk, t_in, t_out):
    """Compact-support assembly of the waveform matrix using quintic B-spline.

    M[o, i] = sum_k W_k * phi( (log(t_out_o - t_k) - log(t_in_i)) / dlogt )

    Only the nonzero spline columns (|x| <= 2) are evaluated, so cost is
    independent of the number of waveform samples once t_in is fixed.
    """
    t_in = np.asarray(t_in, dtype=float)
    t_out = np.asarray(t_out, dtype=float)
    dlogt = np.log(t_in[1]) - np.log(t_in[0])
    log_t0 = np.log(t_in[0])
    n_in = len(t_in)
    M = np.zeros((len(t_out), n_in))
    offsets = np.arange(-2, 4)
    for wk, tkk in zip(Wk, tk):
        shifted = t_out - tkk
        rows = np.flatnonzero(shifted > 0)
        if rows.size == 0:
            continue
        positions = (np.log(shifted[rows]) - log_t0) / dlogt
        base_cols = np.floor(positions).astype(int)
        cols = base_cols[:, None] + offsets[None, :]
        x = positions[:, None] - cols
        inside = (cols >= 0) & (cols < n_in) & (x >= -2.0) & (x <= 2.0)
        if np.any(inside):
            row_idx = np.broadcast_to(rows[:, None], cols.shape)[inside]
            col_idx = cols[inside]
            np.add.at(M, (row_idx, col_idx), wk * quintic_b_spline(x[inside]))
    return M


def _compose_poly(poly, offset, scale):
    from numpy.polynomial import Polynomial
    z = Polynomial([offset, scale])
    out = Polynomial([0.0])
    for power, coef in enumerate(poly.coef):
        out = out + coef * (z ** power)
    return out


def _phi_segments():
    from numpy.polynomial import Polynomial
    X = Polynomial([0.0, 1.0])
    s1 = 1 - (9 / 4) * X ** 2 * (1 - (2 / 9) * X * (1 + (5 / 2) * X * (1 - (2 / 5) * X)))
    s2 = -(1 / 2) * X * (1 - (11 / 6) * X * (1 - (2 / 11) * X * (1 + (5 / 2) * X * (1 - (2 / 5) * X))))
    return [
        (-2.0, -1.0, _compose_poly(s2, -1.0, -1.0)),
        (-1.0, 0.0, _compose_poly(s1, 0.0, -1.0)),
        (0.0, 1.0, _compose_poly(s1, 0.0, 1.0)),
        (1.0, 2.0, _compose_poly(s2, -1.0, 1.0)),
    ]


def _int_u_power_exp(power, a, lo, hi):
    """Integral of u**power * exp(a*u) from lo to hi (closed form)."""
    import math
    if abs(a) < 1e-14:
        return (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)

    def anti(u):
        total = 0.0
        fact_n = math.factorial(power)
        for m in range(power + 1):
            coef = ((-1) ** m) * fact_n / math.factorial(power - m) / (a ** (m + 1))
            total += coef * (u ** (power - m))
        return np.exp(a * u) * total

    return anti(hi) - anti(lo)


def _int_poly_exp(poly, a, lo, hi):
    total = 0.0
    for power, coef in enumerate(poly.coef):
        if coef != 0:
            total += coef * _int_u_power_exp(power, a, lo, hi)
    return total


# Fixed Gauss-Legendre nodes/weights for stable gate-integral quadrature.
_GATE_GL_NODES, _GATE_GL_WEIGHTS = np.polynomial.legendre.leggauss(12)


def _int_poly_exp_gl(poly, a, lo, hi):
    """Integral of poly(u) * exp(a*u) over [lo, hi] via Gauss-Legendre quadrature.

    Numerically stable for all ``a`` (unlike the closed form, which divides by
    ``a**(power+1)`` and loses precision when the log-grid spacing ``a`` is
    small, e.g. for narrow early-time gates on a fine dense grid).
    """
    half = 0.5 * (hi - lo)
    mid = 0.5 * (hi + lo)
    u = mid + half * _GATE_GL_NODES
    vals = poly(u) * np.exp(a * u)
    return half * np.dot(_GATE_GL_WEIGHTS, vals)


def _build_gating_matrix(gate_open, gate_close, t_grid):
    """Boxcar-gate average of the quintic B-spline basis on a log grid.

    G[g, i] = (1 / (b - a)) * integral_a^b phi((log t - log t_i)/dlogt) dt,
    integrated per gate [a, b] with stable Gauss-Legendre quadrature.
    """
    gate_open = np.asarray(gate_open, dtype=float)
    gate_close = np.asarray(gate_close, dtype=float)
    if gate_open.shape != gate_close.shape:
        raise ValueError('gate_open and gate_close must have the same shape')

    t_grid = np.asarray(t_grid, dtype=float)
    dlogt = np.log(t_grid[1]) - np.log(t_grid[0])
    segments = _phi_segments()
    G = np.zeros((gate_open.size, len(t_grid)))
    for g, (a, b) in enumerate(zip(gate_open, gate_close)):
        width = b - a
        if width <= 0:
            raise ValueError('all gate close times must exceed gate open times')
        for i, ti in enumerate(t_grid):
            u0 = np.log(a / ti) / dlogt
            u1 = np.log(b / ti) / dlogt
            if u1 < -2.0 or u0 > 2.0:
                continue
            basis_integral = 0.0
            for seg_lo, seg_hi, seg_poly in segments:
                lo = max(u0, seg_lo)
                hi = min(u1, seg_hi)
                if lo < hi:
                    basis_integral += _int_poly_exp_gl(seg_poly, dlogt, lo, hi)
            G[g, i] = ti * dlogt * basis_integral / width
    return G


def setup_waveform_matrix(gate_times, waveform_times, waveform_currents,
                          gate_open=None, gate_close=None, t_step=None,
                          n_step=600, n_per_segment=81, n_dense_gate=400,
                          pad_decades=2.0):
    """
    Convolution-free waveform operator based on a quintic B-spline matrix.

    Builds a single fixed matrix ``M`` so the waveform (and optional receiver
    gate integration) reduces to one matmul ``d = M @ step_resp``. This matches
    the :func:`setup_waveform` interface and is a drop-in replacement, but the
    returned ``comp_times`` is a fixed log-spaced grid and ``apply_waveform`` is
    a single matrix multiply. ``M`` never depends on the earth model, so it is
    built once and reused across all inversion iterations.

    Parameters
    ----------
    gate_times        : array-like  Measurement gate centre times [s].
    waveform_times    : array-like  Break points of the piecewise-linear waveform [s].
    waveform_currents : array-like  Current at each break point [A].
    gate_open, gate_close : array-like or None
        Gate open/close times [s]. If both are given, ``M`` includes the
        analytic boxcar gate average. If ``None``, the response is evaluated at
        the gate centre times (waveform only).
    t_step : array-like or None
        Fixed log-spaced grid on which the step response is sampled. If ``None``
        it is built automatically from the gate/waveform time span with
        ``n_step`` points and ``pad_decades`` of padding on each side.
    n_step : int         Number of step-response grid points when ``t_step`` is None.
    n_per_segment : int  Simpson samples per waveform segment for the kernel.
    n_dense_gate : int   Dense grid size for gate averaging (gate mode only).
    pad_decades : float  Decades of padding added around the time span.

    Returns
    -------
    comp_times : ndarray
        Times at which the step response must be evaluated (the ``t_step`` grid).
    apply_waveform : callable
        ``apply_waveform(step_resp)`` returning the convolved gate response.
        Accepts a 1-D array ``(n_step,)`` or a 2-D array ``(n_step, N)`` (e.g.
        Jacobian columns), returning ``(n_gates,)`` or ``(n_gates, N)``.
    """
    gate_times = np.asarray(gate_times, dtype=float)
    wf_t = np.asarray(waveform_times, dtype=float)
    wf_I = np.asarray(waveform_currents, dtype=float)

    use_gates = gate_open is not None and gate_close is not None
    if use_gates:
        gate_open = np.asarray(gate_open, dtype=float)
        gate_close = np.asarray(gate_close, dtype=float)
        t_lo = min(gate_open.min(), gate_times.min())
        t_hi = max(gate_close.max(), gate_times.max())
    else:
        t_lo = gate_times.min()
        t_hi = gate_times.max()

    if t_step is None:
        t_min = max(1e-9, t_lo * 10.0 ** (-pad_decades))
        t_max = t_hi * 10.0 ** pad_decades
        t_step = np.logspace(np.log10(t_min), np.log10(t_max), n_step)
    else:
        t_step = np.asarray(t_step, dtype=float)

    tk, Wk = _waveform_slope_kernel(wf_t, wf_I, n_per_segment=n_per_segment)

    if use_gates:
        t_dense = np.logspace(np.log10(gate_open.min()),
                              np.log10(gate_close.max()), n_dense_gate)
        M_wave = _build_waveform_matrix(tk, Wk, t_step, t_dense)
        G = _build_gating_matrix(gate_open, gate_close, t_dense)
        M = G @ M_wave
    else:
        M = _build_waveform_matrix(tk, Wk, t_step, gate_times)

    def apply_waveform(step_resp):
        return M @ np.asarray(step_resp)

    apply_waveform.matrix = M
    return t_step, apply_waveform
