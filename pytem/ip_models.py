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
    tem_forward_ip     - TEM step-off with pluggable IP and loop geometry
"""

import numpy as np
from scipy.special import j0

from .transform_weights import (
    MU0,
    _HANKEL_BASE_201, _HANKEL_J0_201, _HANKEL_J1_201,
    _FOURIER_BASE_81, _FOURIER_SIN_81,
)
from .recursion import te_reflection_coeff, te_reflection_coeff_grad
from .forward import _build_square_geometry, _vmd_greenfct


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
# Geometry helpers
# ============================================================================

_IP_GEOMETRIES = {
    'circle_central', 'circle_offset', 'square_central', 'square_offset',
}


def _validate_geometry(geometry, tx_radius, tx_side):
    geometry = str(geometry).lower()
    if geometry not in _IP_GEOMETRIES:
        choices = ', '.join(sorted(_IP_GEOMETRIES))
        raise ValueError(f"geometry must be one of: {choices}")
    if geometry.startswith('circle') and float(tx_radius) <= 0.0:
        raise ValueError("tx_radius must be positive for circular geometries")
    if geometry.startswith('square') and (tx_side is None or float(tx_side) <= 0.0):
        raise ValueError("tx_side must be positive for square geometries")
    return geometry


def _te_reflection_batch(lam, omega, thicknesses, rho_complex):
    """Vectorized TE reflection coefficient for many frequencies."""
    gamma_layers = np.sqrt(
        lam[None, None, :] ** 2
        + (1j * MU0 * omega)[None, :, None] / rho_complex[:, :, None]
    )
    reflection = np.zeros((omega.size, lam.size), dtype=complex)
    for layer in range(rho_complex.shape[0] - 1, -1, -1):
        gamma_above = lam if layer == 0 else gamma_layers[layer - 1]
        gamma_layer = gamma_layers[layer]
        interface = (gamma_above - gamma_layer) / (gamma_above + gamma_layer)
        decay = (
            np.exp(-2.0 * gamma_layer * thicknesses[layer])
            if layer < rho_complex.shape[0] - 1 else 0.0
        )
        reflection = (
            interface + reflection * decay
        ) / (
            1.0 + interface * reflection * decay
        )
    return reflection


def _pelton_hz_batch(thicknesses, rho_complex, omega, tx_radius,
                     geometry, rx_offset, tx_side, rx_x, rx_y, n_quad):
    """Secondary magnetic field for circular or square loop geometry."""
    if geometry.startswith('circle'):
        lam = _HANKEL_BASE_201 / float(tx_radius)
        reflection = _te_reflection_batch(
            lam, omega, thicknesses, rho_complex)
        extra = (
            j0(lam * float(rx_offset))
            if geometry == 'circle_offset' else np.ones_like(lam)
        )
        return 0.5 * np.sum(
            reflection * lam[None, :] * extra[None, :]
            * _HANKEL_J1_201[None, :], axis=1)

    use_symmetry = geometry == 'square_central'
    dist_q, area_w = _build_square_geometry(
        float(tx_side), int(n_quad),
        rx_x=float(rx_x), rx_y=float(rx_y), use_symmetry=use_symmetry)
    hz = np.zeros(omega.size, dtype=complex)
    for distance, weight in zip(dist_q, area_w):
        lam = _HANKEL_BASE_201 / distance
        reflection = _te_reflection_batch(
            lam, omega, thicknesses, rho_complex)
        hz += weight * np.sum(
            reflection * lam[None, :] ** 2 * _HANKEL_J0_201[None, :],
            axis=1) / distance / (4.0 * np.pi)
    if use_symmetry:
        hz *= 4.0
    return hz


# ============================================================================
# Generic forward model with pluggable IP function per layer
# ============================================================================

def tem_forward_ip(thicknesses, resistivities, tx_radius, times,
                   ip_funcs=None, current=1.0, system_filter=None,
                   geometry='circle_central', rx_offset=0.0,
                   tx_side=None, rx_x=0.0, rx_y=0.0, n_quad=11):
    """
    TEM step-off dB/dt with arbitrary IP model and loop geometry.

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
    geometry     : {'circle_central', 'circle_offset',
                    'square_central', 'square_offset'}
    rx_offset    : radial receiver offset [m] for ``circle_offset``
    tx_side      : square transmitter side length [m] for square geometries
    rx_x, rx_y   : receiver coordinates [m] for ``square_offset``
    n_quad       : Gauss-Legendre points per square-loop axis

    Returns
    -------
    dbdt : ndarray - dB/dt [V/m^2]
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    rho_dc = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    a_r = float(tx_radius)
    n_layers = len(rho_dc)
    geometry = _validate_geometry(geometry, a_r, tx_side)

    if ip_funcs is None:
        ip_funcs = [None] * n_layers

    def _hz_sec(omega):
        rho_complex = np.array([
            ip_funcs[j](rho_dc[j], omega) if ip_funcs[j] is not None
            else complex(rho_dc[j])
            for j in range(n_layers)
        ])
        if geometry.startswith('circle'):
            lam = _HANKEL_BASE_201 / a_r
            r_te = te_reflection_coeff(lam, omega, thicknesses, rho_complex)
            extra = (
                j0(lam * float(rx_offset))
                if geometry == 'circle_offset' else 1.0
            )
            hz = 0.5 * np.dot(
                r_te * lam * extra, _HANKEL_J1_201)
        else:
            use_symmetry = geometry == 'square_central'
            dist_q, area_w = _build_square_geometry(
                float(tx_side), int(n_quad),
                rx_x=float(rx_x), rx_y=float(rx_y),
                use_symmetry=use_symmetry)
            hz = sum(
                weight * _vmd_greenfct(
                    distance, omega, thicknesses, rho_complex,
                    _HANKEL_BASE_201, _HANKEL_J0_201)
                for distance, weight in zip(dist_q, area_w)
            )
            if use_symmetry:
                hz *= 4.0
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


def tem_forward_pelton(thicknesses, resistivities, chargeabilities,
                       time_constants, exponents, tx_radius, times,
                       current=1.0, system_filter=None,
                       geometry='circle_central', rx_offset=0.0,
                       tx_side=None, rx_x=0.0, rx_y=0.0, n_quad=11):
    """Vectorized Pelton step-off response for central or offset loops."""
    thicknesses = np.asarray(thicknesses, dtype=float)
    rho_dc = np.asarray(resistivities, dtype=float)
    chargeabilities = np.asarray(chargeabilities, dtype=float)
    time_constants = np.asarray(time_constants, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    times = np.asarray(times, dtype=float)
    tx_radius = float(tx_radius)
    geometry = _validate_geometry(geometry, tx_radius, tx_side)

    omega = (_FOURIER_BASE_81[None, :] / times[:, None]).ravel()
    log_iot = np.log(1j * omega[None, :] * time_constants[:, None])
    z = np.exp(exponents[:, None] * log_iot)
    rho_complex = rho_dc[:, None] * (
        1.0 - chargeabilities[:, None] * z / (1.0 + z)
    )

    hz = _pelton_hz_batch(
        thicknesses, rho_complex, omega, tx_radius,
        geometry, rx_offset, tx_side, rx_x, rx_y, n_quad,
    )
    if system_filter is not None:
        hz *= system_filter(omega)

    signal = (MU0 * np.imag(hz)).reshape(times.size, _FOURIER_BASE_81.size)
    dbdt = (signal @ _FOURIER_SIN_81) / times
    return dbdt * current * 2.0 / np.pi


def tem_forward_pelton_jacobian(thicknesses, resistivities, chargeabilities,
                                time_constants, exponents, tx_radius, times,
                                current=1.0, system_filter=None):
    """Central-loop Pelton response and analytical parameter Jacobian.

    The Jacobian columns are grouped as ``ln(rho)``, ``m``, ``ln(tau)``,
    and ``c``, with one column per layer in each group.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    rho_dc = np.asarray(resistivities, dtype=float)
    chargeabilities = np.asarray(chargeabilities, dtype=float)
    time_constants = np.asarray(time_constants, dtype=float)
    exponents = np.asarray(exponents, dtype=float)
    times = np.asarray(times, dtype=float)
    n_layers = rho_dc.size
    n_parameters = 4 * n_layers
    tx_radius = float(tx_radius)

    dbdt = np.zeros(times.size)
    jacobian = np.zeros((times.size, n_parameters))

    for time_index, time in enumerate(times):
        omega_points = _FOURIER_BASE_81 / time
        signal = np.zeros(omega_points.size)
        signal_jacobian = np.zeros((omega_points.size, n_parameters))

        for frequency_index, omega in enumerate(omega_points):
            log_iot = np.log(1j * omega * time_constants)
            z = np.exp(exponents * log_iot)
            one_plus_z = 1.0 + z
            pelton_factor = 1.0 - chargeabilities * z / one_plus_z
            rho_complex = rho_dc * pelton_factor

            dlnrho = np.empty((n_parameters, n_layers), dtype=complex)
            dlnrho[:n_layers] = np.eye(n_layers)
            dlnrho[n_layers:2 * n_layers] = np.diag(
                -z / (one_plus_z * pelton_factor))
            dlnrho[2 * n_layers:3 * n_layers] = np.diag(
                -chargeabilities * exponents * z
                / (one_plus_z ** 2 * pelton_factor))
            dlnrho[3 * n_layers:] = np.diag(
                -chargeabilities * z * log_iot
                / (one_plus_z ** 2 * pelton_factor))

            lam = _HANKEL_BASE_201 / tx_radius
            r_te, dr_te_dlnrho = te_reflection_coeff_grad(
                lam, omega, thicknesses, rho_complex)
            hz = 0.5 * np.dot(r_te * lam, _HANKEL_J1_201)
            dhz_dlnrho = 0.5 * np.sum(
                dr_te_dlnrho * lam[None, :] * _HANKEL_J1_201[None, :], axis=1)
            dhz = dlnrho @ dhz_dlnrho

            if system_filter is not None:
                filter_value = system_filter(omega)
                hz *= filter_value
                dhz *= filter_value

            signal[frequency_index] = MU0 * np.imag(hz)
            signal_jacobian[frequency_index] = MU0 * np.imag(dhz)

        dbdt[time_index] = np.dot(signal, _FOURIER_SIN_81) / time
        jacobian[time_index] = (
            signal_jacobian.T @ _FOURIER_SIN_81) / time

    scale = current * 2.0 / np.pi
    return dbdt * scale, jacobian * scale
