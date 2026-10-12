"""
VES (Vertical Electrical Sounding) — 1-D DC resistivity modelling and inversion.

Public API
----------
forward(ab2, resistivities, thicknesses, filter_set='gs7') -> ndarray
jacobian(ab2, resistivities, thicknesses, filter_set='gs7') -> ndarray
invert(ab2, rhoap_obs, resistivities, thicknesses, ...)     -> dict
fit_system(ab2, rhoap_obs, thicknesses, noise=0.03)        -> dict for pytem.invert_joint
"""

import numpy as np
from numpy.linalg import inv, svd as _svd

# ── Guptasarma & Singh (1997) linear filter coefficients ─────────────────────
# Keys: 'gs7', 'gs11', 'gs22'  (also accept old 'guptasarma_N' names)
_FILTERS = {
    "gs7": (
        np.array([-0.17445, 0.09672, 0.36789, 0.63906, 0.91023, 1.1814, 1.45257]),
        np.array([0.1732, 0.2945, 2.147, -2.1733, 0.6646, -0.1215, 0.0155]),
    ),
    "gs11": (
        np.array([-0.420625, -0.20265625, 0.0153125, 0.23328125, 0.45125,
                   0.66921875, 0.8871875, 1.10515625, 1.323125, 1.54109375, 1.7590625]),
        np.array([0.041873, -0.022258, 0.38766, 0.647103, 1.84873, -2.96084,
                   1.358412, -0.37759, 0.097107, -0.024243, 0.004046]),
    ),
    "gs22": (
        np.array([-0.980685, -0.771995, -0.563305, -0.354615, -0.145925,
                   0.062765, 0.271455, 0.480145, 0.688835, 0.897525, 1.106215,
                   1.314905, 1.523595, 1.732285, 1.940975, 2.149665, 2.358355,
                   2.567045, 2.775735]),
        np.array([0.00097112, -0.00102152, 0.00906965, 0.01404316, 0.09012,
                   0.30171582, 0.99627084, 1.3690832, -2.99681171, 1.65463068,
                   -0.59399277, 0.22329813, -0.10119309, 0.05186135, -0.02748647,
                   0.01384932, -0.00599074, 0.00190463, -0.0003216]),
    ),
}
_FILTER_ALIASES = {"guptasarma_7": "gs7", "guptasarma_11": "gs11", "guptasarma_22": "gs22"}


def _get_filter(filter_set):
    key = _FILTER_ALIASES.get(filter_set, filter_set)
    if key not in _FILTERS:
        raise ValueError(
            f"Unknown filter '{filter_set}'. Choose from: "
            f"{list(_FILTERS)} or {list(_FILTER_ALIASES)}."
        )
    return _FILTERS[key]


# ── Core recursion ────────────────────────────────────────────────────────────

def _dc_recursion(lam, rho, h):
    """
    DC Wait upward recursion for wavenumber *lam*.

    Returns
    -------
    T1 : float
        Surface kernel value T(lambda).
    T_arr : ndarray (n,)
        Kernel at every layer interface (T_arr[n-1] = rho_n = half-space seed).
    tau, sech2, B : ndarray (n-1,)
        Intermediate values needed for the analytical Jacobian.
    """
    n = len(rho)
    T_arr = np.empty(n)
    tau   = np.empty(n - 1)
    sech2 = np.empty(n - 1)
    B     = np.empty(n - 1)

    T_arr[n - 1] = rho[n - 1]                           # half-space seed

    for j in range(n - 2, -1, -1):                      # upward through layers
        lh       = lam * h[j]
        tau[j]   = np.tanh(lh)
        sech2[j] = 1.0 - tau[j] ** 2                    # sech^2 = 1 - tanh^2
        B[j]     = rho[j] + T_arr[j + 1] * tau[j]
        T_arr[j] = rho[j] * (T_arr[j + 1] + rho[j] * tau[j]) / B[j]

    return T_arr[0], T_arr, tau, sech2, B


def _analytic_jac_at_lambda(lam, rho, T_arr, tau, sech2, B):
    """
    Analytical dT1/d(rho_j) and dT1/d(h_j) for one wavenumber.

    Uses the chain rule P_j = dT1/dT_j propagated downward through the
    recursion, together with the closed-form local derivatives.

    Returns
    -------
    dT1_drho : ndarray (n,)
    dT1_dh   : ndarray (n-1,)
    """
    n = len(rho)
    dT1_drho = np.empty(n)
    dT1_dh   = np.empty(n - 1)

    P = 1.0  # P[0] = dT1/dT1 = 1; propagated downward

    for j in range(n - 1):
        Tj1 = T_arr[j + 1]

        # dT_j/d(rho_j)  [T_{j+1} treated as fixed]
        dT1_drho[j] = P * ((Tj1**2 + rho[j]**2) * tau[j] + 2*rho[j]*Tj1*tau[j]**2) / B[j]**2

        # dT_j/d(h_j)
        dT1_dh[j] = P * (rho[j] * lam * sech2[j] * (rho[j]**2 - Tj1**2) / B[j]**2)

        # Propagate P downward:  P_{j+1} = P_j * dT_j/dT_{j+1}
        P = P * (rho[j]**2 * sech2[j] / B[j]**2)

    dT1_drho[n - 1] = P  # half-space: dT_n/d(rho_n) = 1, scaled by P

    return dT1_drho, dT1_dh


# ── Public API ────────────────────────────────────────────────────────────────

def forward(ab2, resistivities, thicknesses, filter_set="gs7"):
    """
    Compute the VES apparent resistivity sounding curve.

    Parameters
    ----------
    ab2 : array_like (M,)
        Electrode half-spacings [m].
    resistivities : array_like (N,)
        Layer resistivities [Ohm·m]; last entry is the half-space.
    thicknesses : array_like (N-1,)
        Layer thicknesses [m].
    filter_set : str
        Linear filter set: 'gs7', 'gs11', or 'gs22'.

    Returns
    -------
    rhoap : ndarray (M,)
        Apparent resistivity [Ohm·m].
    """
    ab2  = np.asarray(ab2,          dtype=float)
    rho  = np.asarray(resistivities, dtype=float)
    h    = np.asarray(thicknesses,   dtype=float)
    a_f, phi_f = _get_filter(filter_set)

    rhoap = np.empty(len(ab2))
    for i, r in enumerate(ab2):
        val = 0.0
        for k in range(len(a_f)):
            lam = 10.0 ** (a_f[k] - np.log10(r))
            T1, _, _, _, _ = _dc_recursion(lam, rho, h)
            val += phi_f[k] * T1
        rhoap[i] = val
    return rhoap


def jacobian(ab2, resistivities, thicknesses, filter_set="gs7"):
    """
    Analytical Jacobian of the sounding curve w.r.t. model parameters.

    The model vector is [rho_1, ..., rho_N, h_1, ..., h_{N-1}] (length 2N-1).
    Derivatives are computed via closed-form differentiation through the DC
    Wait recursion — no finite differences required.

    Parameters
    ----------
    ab2 : array_like (M,)
    resistivities : array_like (N,)
    thicknesses : array_like (N-1,)
    filter_set : str

    Returns
    -------
    J : ndarray (M, 2N-1)
        J[:, :N]  = d(rhoap)/d(rho_j)  for all N layers
        J[:, N:]  = d(rhoap)/d(h_j)    for layers 1..N-1
    """
    ab2  = np.asarray(ab2,          dtype=float)
    rho  = np.asarray(resistivities, dtype=float)
    h    = np.asarray(thicknesses,   dtype=float)
    a_f, phi_f = _get_filter(filter_set)

    N = len(rho)
    J = np.zeros((len(ab2), 2 * N - 1))

    for i, r in enumerate(ab2):
        drhoap_drho = np.zeros(N)
        drhoap_dh   = np.zeros(N - 1)
        for k in range(len(a_f)):
            lam = 10.0 ** (a_f[k] - np.log10(r))
            _, T_arr, tau, sech2, B = _dc_recursion(lam, rho, h)
            dT1_drho, dT1_dh = _analytic_jac_at_lambda(lam, rho, T_arr, tau, sech2, B)
            drhoap_drho += phi_f[k] * dT1_drho
            drhoap_dh   += phi_f[k] * dT1_dh
        J[i, :N] = drhoap_drho
        J[i, N:] = drhoap_dh

    return J


def forward_ip(ab2, resistivities, thicknesses, chargeabilities, filter_set="gs7"):
    """
    Compute the VES apparent chargeability using the Seigel (1959) approximation.

    The DC-limit of the Cole-Cole / Pelton model reduces the resistivity of
    each layer from rho_0 to rho_0*(1 - m).  The apparent chargeability is then:

        M_app(r) = [rhoap(rho_0) - rhoap(rho_0*(1-m))] / rhoap(rho_0)

    Parameters
    ----------
    ab2 : array_like (M,)
        Electrode half-spacings [m].
    resistivities : array_like (N,)
        DC layer resistivities [Ohm·m].
    thicknesses : array_like (N-1,)
        Layer thicknesses [m].
    chargeabilities : array_like (N,)
        Chargeability of each layer (0–1).  Layers with m=0 have no IP effect.
    filter_set : str
        Linear filter set: 'gs7', 'gs11', or 'gs22'.

    Returns
    -------
    m_app : ndarray (M,)
        Apparent chargeability (dimensionless, 0–1).
    rhoap_dc : ndarray (M,)
        Standard apparent resistivity (no IP).
    rhoap_ip : ndarray (M,)
        Apparent resistivity computed with polarised (DC-limit) resistivities.
    """
    rho = np.asarray(resistivities,   dtype=float)
    m   = np.asarray(chargeabilities, dtype=float)
    rhoap_dc  = forward(ab2, rho,           thicknesses, filter_set)
    rhoap_ip  = forward(ab2, rho * (1 - m), thicknesses, filter_set)
    m_app     = (rhoap_dc - rhoap_ip) / rhoap_dc
    return m_app, rhoap_dc, rhoap_ip


def _ves_depth_weights(thicknesses, alpha):
    """Depth-weighted alpha vector (same convention as pytem getAlphas)."""
    h = np.asarray(thicknesses, dtype=float)
    N = len(h) + 1                              # number of layers
    tops        = np.concatenate([[0.0], np.cumsum(h[:-1])])
    mids_finite = tops + h / 2.0
    hs_mid      = tops[-1] + h[-1] + h[-1] / 2.0
    midpoints   = np.append(mids_finite, hs_mid)
    del_z       = np.diff(midpoints)
    av          = np.empty(N)
    av[0]       = 1.0 / del_z[0]
    av[1:-1]    = 1.0 / del_z[:-1] + 1.0 / del_z[1:]
    av[-1]      = 1.0 / del_z[-1]
    return alpha * av


def _ves_build_R(N):
    """First-order roughness matrix (N x N)."""
    if N < 2:
        return np.zeros((N, N))
    D = np.zeros((N - 1, N))
    for k in range(N - 1):
        D[k, k]     = -1.0
        D[k, k + 1] =  1.0
    return D.T @ D


def _ves_gn_solve(J, res, alpha_vec, R, ln_rho):
    """Solve the regularised Gauss-Newton normal equations.

    Solves  (J^T J + diag(alpha) R) dm = J^T res - diag(alpha) R m.
    """
    AR  = np.diag(alpha_vec) @ R
    lhs = J.T @ J + AR
    rhs = J.T @ res - AR @ ln_rho
    dm, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=1e-10)
    return dm


def _ves_alpha_search(ab2, rhoap_obs, J_log, res_log, noise_frac,
                      h, ln_rho, alpha_start, filter_set,
                      alpha_steps=8, alpha_step=1.0/3.0,
                      rms_norm_current=np.inf):
    """Log-spaced alpha search targeting normalised RMS = 1.

    Tries ``alpha_steps`` alpha values starting from ``alpha_start`` on a
    log-spaced ladder (each step divides by 10^alpha_step).  For each alpha
    the GN step is computed and the *actual* forward model is evaluated so
    that the RMS reflects the true nonlinear misfit.

    After the ladder, a polynomial is fit to the (log alpha, norm_RMS) pairs
    and the root at norm_RMS = 1 is located (parabola backtrack).

    Parameters
    ----------
    norm_RMS = RMS / noise_frac;  target = 1.0

    Returns
    -------
    delta : (N,) update to ln_rho (i.e. trial - ln_rho)
    """
    N          = len(ln_rho)
    R          = _ves_build_R(N)
    ln_rho_min = np.log(1e-2)
    ln_rho_max = np.log(1e7)

    alpha_hist, rms_hist, delta_hist = [], [], []

    for i in range(alpha_steps):
        alpha = alpha_start * 10.0 ** (-alpha_step * i)
        av    = _ves_depth_weights(h, alpha)
        dm    = _ves_gn_solve(J_log, res_log, av, R, ln_rho)
        trial = np.clip(ln_rho + dm, ln_rho_min, ln_rho_max)
        delta = trial - ln_rho

        # actual forward evaluation
        rhoap_pred = forward(ab2, np.exp(trial), h, filter_set)
        valid      = (rhoap_pred > 0) & (rhoap_obs > 0)
        res_new    = np.zeros(len(ab2))
        res_new[valid] = np.log(rhoap_obs[valid]) - np.log(rhoap_pred[valid])
        rms_norm   = float(np.sqrt(np.mean(res_new ** 2))) / noise_frac

        alpha_hist.append(alpha)
        rms_hist.append(rms_norm)
        delta_hist.append(delta)

        if rms_norm < 1.0:
            break
        if len(rms_hist) > 1 and rms_hist[-1] > rms_hist[-2] and min(rms_hist[:-1]) < rms_norm_current:
            break

    # ── polynomial backtrack to find alpha* where norm_RMS = 1 ───────────────
    x   = np.log10(np.array(alpha_hist))
    y   = np.array(rms_hist)
    deg = min(2, len(x) - 1)

    best_delta = delta_hist[-1]

    if deg >= 1 and np.min(y) < 1.0:
        coeffs       = np.polyfit(x, y, deg)
        root_c       = coeffs.copy()
        root_c[-1]  -= 1.0
        roots        = np.roots(root_c)
        x_lo, x_hi   = x.min() - 1.0, x.max() + 1.0
        real_roots   = roots[np.abs(roots.imag) < 1e-10].real
        valid_roots  = real_roots[(real_roots >= x_lo) & (real_roots <= x_hi)]

        if valid_roots.size > 0:
            pa    = 10.0 ** float(valid_roots.max())
            av_p  = _ves_depth_weights(h, pa)
            dm_p  = _ves_gn_solve(J_log, res_log, av_p, R, ln_rho)
            trial_p = np.clip(ln_rho + dm_p, ln_rho_min, ln_rho_max)

            # evaluate actual RMS for parabola step
            rhoap_p = forward(ab2, np.exp(trial_p), h, filter_set)
            valid_p = (rhoap_p > 0) & (rhoap_obs > 0)
            res_p   = np.zeros(len(ab2))
            res_p[valid_p] = np.log(rhoap_obs[valid_p]) - np.log(rhoap_p[valid_p])
            rms_p   = float(np.sqrt(np.mean(res_p ** 2))) / noise_frac

            # accept parabola step if it brings us closer to 1
            if abs(rms_p - 1.0) < abs(rms_hist[-1] - 1.0):
                best_delta = trial_p - ln_rho

    return best_delta


def invert(ab2, rhoap_obs, resistivities, thicknesses,
           err_min=0.0, iter_max=15,
           filter_set="gs7", fix_thicknesses=True, regularization="auto",
           noise_frac=0.05, alpha_steps=8, alpha_step=1.0/3.0):
    """
    Invert observed apparent resistivity data using the analytical Jacobian.

    Optimisation is performed in log-resistivity space so resistivities are
    guaranteed positive and large contrasts converge more reliably.

    Regularisation uses a pytem-style log-spaced alpha search with polynomial
    backtrack so the inversion targets a normalised RMS of 1.0 (fit at noise).

    Parameters
    ----------
    ab2 : array_like (M,)
    rhoap_obs : array_like (M,)
        Observed apparent resistivity [Ohm.m].
    resistivities : array_like (N,)
        Starting resistivities [Ohm.m].
    thicknesses : array_like (N-1,)
        Layer thicknesses [m].
    err_min : float
        Hard stop when normalised RMS falls below this value.
    iter_max : int
        Maximum number of iterations (default 15).
    filter_set : str
    regularization : 'auto' or float
        When 'auto', uses the pytem-style alpha search each iteration.
        When a float, uses that fixed value throughout.
    noise_frac : float
        Expected noise level as a fraction (e.g. 0.05 for 5 %).
        The normalised RMS = RMS / noise_frac; target is 1.0.
    alpha_steps : int
        Number of alpha values tested per iteration (default 8).
    alpha_step : float
        Log10 step between consecutive alpha values (default 1/3,
        i.e. ~3 steps per decade).

    Returns
    -------
    result : dict
        'resistivities' : ndarray (N,)   recovered resistivities
        'thicknesses'   : ndarray (N-1,) recovered thicknesses
        'rhoap_pred'    : ndarray (M,)   predicted sounding curve
        'rms_history'   : list[float]    normalised RMS per iteration
        'n_iter'        : int
    """
    ab2       = np.asarray(ab2,           dtype=float)
    rhoap_obs = np.asarray(rhoap_obs,     dtype=float)
    rho       = np.asarray(resistivities, dtype=float).copy()
    h         = np.asarray(thicknesses,   dtype=float).copy()

    N      = len(rho)
    ln_rho = np.log(np.clip(rho, 1e-6, 1e9))

    _stop_norm = max(float(err_min) / noise_frac if err_min > 0 else 0.0, 1.0)
    _auto_reg  = (regularization == "auto")
    _alpha_start = None   # estimated from first Jacobian

    rms_history = []
    _prev_rms   = np.inf

    for _iter in range(iter_max):
        rhoap_pred = forward(ab2, np.exp(ln_rho), h, filter_set)

        valid   = (rhoap_pred > 0) & (rhoap_obs > 0)
        res_log = np.zeros(len(ab2))
        res_log[valid] = np.log(rhoap_obs[valid]) - np.log(rhoap_pred[valid])
        rms      = float(np.sqrt(np.mean(res_log ** 2)))
        rms_norm = rms / noise_frac
        rms_history.append(rms)

        print(f"Iteration {_iter + 1:>3d}:  RMS = {rms_norm:.2f}")

        # Stop when normalised RMS reaches 1 (fit at noise level)
        if rms_norm <= _stop_norm:
            print("  RMS <= 1 - converged.")
            break
        # Stop if improvement is negligible (< 0.5 % relative)
        if _iter > 0 and (_prev_rms - rms) / max(_prev_rms, 1e-12) < 0.005:
            print("  No improvement found - stopping.")
            break
        _prev_rms = rms

        # Analytical Jacobian in log-space
        J_lin     = jacobian(ab2, np.exp(ln_rho), h, filter_set)
        rp        = np.where(rhoap_pred > 0, rhoap_pred, 1.0)[:, None]
        J_log_rho = J_lin[:, :N] * np.exp(ln_rho)[None, :] / rp

        # Initialise alpha_start from ||J^T r||_inf (same as pytem)
        if _alpha_start is None:
            _alpha_start = float(np.linalg.norm(J_log_rho.T @ res_log, np.inf) + 1e-30)
            print(f"Alpha start = {_alpha_start:.3f}")

        if _auto_reg:
            delta = _ves_alpha_search(
                ab2, rhoap_obs, J_log_rho, res_log, noise_frac,
                h, ln_rho, _alpha_start, filter_set,
                alpha_steps=alpha_steps, alpha_step=alpha_step,
                rms_norm_current=rms_norm,
            )
        else:
            # fixed regularization: single GN step
            R  = _ves_build_R(N)
            av = _ves_depth_weights(h, float(regularization))
            dm = _ves_gn_solve(J_log_rho, res_log, av, R, ln_rho)
            delta = np.clip(ln_rho + dm, np.log(1e-2), np.log(1e7)) - ln_rho

        ln_rho = np.clip(ln_rho + delta, np.log(1e-2), np.log(1e7))

    rhoap_pred = forward(ab2, np.exp(ln_rho), h, filter_set)
    return {
        "resistivities": np.exp(ln_rho),
        "thicknesses":   h,
        "rhoap_pred":    rhoap_pred,
        "rms_history":   rms_history,
        "n_iter":        len(rms_history),
    }


def fit_system(ab2, rhoap_obs, thicknesses, noise=0.03, filter_set="gs7"):
    """
    VES data as a ``pytem.invert_joint`` fit system, on the layer grid of the
    inversion (``thicknesses``), so it can be inverted alone or jointly with TEM.

    Parameters
    ----------
    ab2 : array_like (M,)       Electrode half-spacings [m].
    rhoap_obs : array_like (M,) Observed apparent resistivity [Ohm.m].
    thicknesses : array_like (N-1,) Layer thicknesses of the shared model [m].
    noise : float or array_like Fractional noise of the apparent resistivities.

    Returns
    -------
    dict with 'obs', 'noise' (absolute), 'forward' (ln rho -> rhoap) and
    'jacobian' (ln rho -> d rhoap / d ln rho, from the analytical Jacobian).
    """
    ab2 = np.asarray(ab2, dtype=float)
    rhoap_obs = np.asarray(rhoap_obs, dtype=float)
    n = len(thicknesses) + 1
    return {
        "obs": rhoap_obs,
        "noise": np.asarray(noise, dtype=float) * rhoap_obs,
        "forward": lambda log_rho: forward(ab2, np.exp(log_rho), thicknesses, filter_set),
        "jacobian": lambda log_rho: jacobian(ab2, np.exp(log_rho), thicknesses, filter_set)[:, :n] * np.exp(log_rho),
    }
