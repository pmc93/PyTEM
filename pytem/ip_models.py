"""
ip_models.py - IP (Induced Polarisation) model functions and TEM forward with IP.

Contains:
  pelton_res_rho     - Pelton et al. (1978) resistivity formulation
  cole_cole_rho      - Cole & Cole (1941) conductivity formulation
  double_pelton_rho  - Double Pelton (two relaxation terms)
  _mpa_tau_rho       - MPA helper: tau_rho from tau_phi
  _mpa_ab            - MPA helper: real/imag parts
  _mpa_m             - MPA helper: chargeability from a, b, phi_max
  get_m_taur_MPA     - Iterative MPA -> CC conversion (Fiandaca et al., 2018)
  mpa_rho            - MPA model returning complex resistivity
  tem_forward_ip     - Central-loop TEM step-off with pluggable IP per layer
"""

import numpy as np

from .transform_weights import (
    MU0,
    _HANKEL_BASE_201, _HANKEL_J1_201,
    _FOURIER_BASE_81, _FOURIER_SIN_81,
)
from .recursion import te_reflection_coeff


# ============================================================================
# IP model functions (complex resistivity at angular frequency omega)
# ============================================================================

def pelton_res_rho(rho_0, m, tau, c, omega):
    """Pelton et al. (1978) - resistivity formulation."""
    iotc = (1j * omega * tau) ** c
    return rho_0 * (1 - m * (1 - 1 / (1 + iotc)))


def cole_cole_rho(rho_0, cond_0, cond_inf, tau, c, omega):
    """Cole & Cole (1941) - conductivity formulation, returned as resistivity.
    cond_0 : DC conductivity (low-freq limit)
    cond_inf: high-freq limit conductivity
    """
    iotc = (1j * omega * tau) ** c
    cond = cond_inf + (cond_0 - cond_inf) / (1 + iotc)
    return 1.0 / cond


def double_pelton_rho(rho_0, m1, tau1, c1, m2, tau2, c2, omega):
    """Double Pelton - two relaxation terms (Pelton et al., 1978)."""
    iotc1 = (1j * omega * tau1) ** c1
    iotc2 = (1j * omega * tau2) ** c2
    r1 = m1 * (1 - 1 / (1 + iotc1))
    r2 = m2 * (1 - 1 / (1 + iotc2))
    return rho_0 * (1 - r1 - r2)


# ============================================================================
# MPA (Maximum Phase Angle) - Fiandaca et al., 2018, Appendix A
# ============================================================================

def _mpa_tau_rho(m, tau_phi, c):
    """A.05: tau_rho from tau_phi."""
    return tau_phi * (abs(1 - m)) ** (-1 / (2 * c))


def _mpa_ab(tau_rho, tau_phi, c):
    """A.06 & A.07: real (a) and imag (b) parts."""
    z = 1 / (1 + (1j * (tau_rho / tau_phi)) ** c)
    return np.real(z), np.imag(z)


def _mpa_m(a, b, phi_max):
    """A.08: chargeability from a, b, phi_max."""
    tan_phi = np.tan(-phi_max)
    return tan_phi / ((1 - a) * tan_phi + b)


def get_m_taur_MPA(phi_max, tau_phi, c, max_iters=42, threshold=1e-7):
    """Iterative MPA -> CC conversion (Fiandaca et al., 2018, A.1-A.08).
    Works for scalar inputs. Returns (m, tau_rho)."""
    if phi_max == 0:
        return 0.0, _mpa_tau_rho(0.0, tau_phi, c)

    m_n = 0.0
    for _ in range(max_iters):
        tau_r = _mpa_tau_rho(m_n, tau_phi, c)
        a, b = _mpa_ab(tau_r, tau_phi, c)
        m_new = _mpa_m(a, b, phi_max)
        if abs(m_new - m_n) / abs(m_new) <= threshold:
            return m_new, tau_r
        m_n = m_new

    raise ValueError(f'MPA iteration did not converge after {max_iters} iters.')


def mpa_rho(rho_0, phi_max, tau_phi, c, omega):
    """MPA model (Fiandaca et al., 2018) - returns complex resistivity.
    Converts (phi_max, tau_phi, c) -> (m, tau_rho), then applies Pelton."""
    m, tau_rho = get_m_taur_MPA(phi_max, tau_phi, c)
    return pelton_res_rho(rho_0, m, tau_rho, c, omega)


# ============================================================================
# Generic forward model with pluggable IP function per layer
# ============================================================================

def tem_forward_ip(thicknesses, resistivities, tx_radius, times,
                   ip_funcs=None, current=1.0, system_filter=None):
    """
    Central-loop TEM step-off dB/dt with arbitrary IP model per layer.

    Parameters
    ----------
    thicknesses  : layer thicknesses [m] (N-1 values)
    resistivities: DC resistivities rho_0 [Ohm.m] (N values)
    tx_radius    : Tx loop radius [m]
    times        : gate times [s]
    ip_funcs     : list of callables (one per layer).
                   Each callable: f(rho_0, omega) -> complex resistivity.
                   Use None for layers without IP (real rho_0 is used).
    current      : float, default 1.0
    system_filter: callable or None

    Returns
    -------
    dbdt : ndarray - dB/dt [V/m^2]
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    rho_dc = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    a_r = float(tx_radius)
    n_layers = len(rho_dc)

    if ip_funcs is None:
        ip_funcs = [None] * n_layers

    def _hz_sec(omega):
        rho_complex = np.array([
            ip_funcs[j](rho_dc[j], omega) if ip_funcs[j] is not None
            else complex(rho_dc[j])
            for j in range(n_layers)
        ])
        lam = _HANKEL_BASE_201 / a_r
        r_te = te_reflection_coeff(lam, omega, thicknesses, rho_complex)
        hankel = np.dot(r_te * lam, _HANKEL_J1_201) / a_r
        hz = 0.5 * a_r * hankel
        if system_filter is not None:
            hz *= system_filter(omega)
        return hz

    dbdt = np.zeros(len(times))
    for i, t in enumerate(times):
        omega_pts = _FOURIER_BASE_81 / t
        signal = np.zeros(len(omega_pts))
        for k, w in enumerate(omega_pts):
            signal[k] = MU0 * np.imag(_hz_sec(w))
        dbdt[i] = np.dot(signal, _FOURIER_SIN_81) / t

    dbdt *= current * 2.0 / np.pi
    return dbdt
