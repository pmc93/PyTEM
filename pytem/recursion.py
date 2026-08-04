"""
recursion.py - TE reflection coefficient via upward recursion (NumPy).
"""

import numpy as np
from .transform_weights import MU0


def te_reflection_coeff(lam, omega, thicknesses, resistivities):
    """
    TE-mode surface reflection coefficient for a layered isotropic earth
    using the upward recursion.  Complex arithmetic.

    Parameters
    ----------
    lam           : (K,)   horizontal wavenumbers [1/m]
    omega         : float or complex  angular frequency [rad/s]
    thicknesses   : (N-1,) layer thicknesses [m]
    resistivities : (N,)   layer resistivities [Ohm.m] (may be complex for IP)

    Returns
    -------
    r_TE : (K,) complex128 - TE surface reflection coefficient
    """
    n_lay = len(resistivities)
    sval = 1j * omega
    resistivities = np.asarray(resistivities, dtype=complex)

    sigma = 1.0 / resistivities
    Gamma = np.sqrt(lam[None, :]**2 + (sval * MU0 * sigma)[:, None])

    # Upward recursion (paper Eq. 2): gamma_N = 0 at the base half-space, then
    #   gamma_j = (psi_j + gamma_{j+1} E_{j+1}) / (1 + psi_j gamma_{j+1} E_{j+1})
    # with psi_j = (Gamma_j - Gamma_{j+1}) / (Gamma_j + Gamma_{j+1}) and the phase
    # E_{j+1} = exp(-2 Gamma_{j+1} h_{j+1}) applied to the deeper reflection across
    # the layer below the interface.  Index 0 is the air half-space (Gamma_0 = lam).
    gamma = np.zeros(len(lam), dtype=complex)          # gamma_N = 0
    for j in range(n_lay - 1, -1, -1):
        G_above = lam if j == 0 else Gamma[j - 1]
        psi = (G_above - Gamma[j]) / (G_above + Gamma[j])
        E = np.exp(-2.0 * Gamma[j] * thicknesses[j]) if j < n_lay - 1 else 0.0
        gamma = (psi + gamma * E) / (1.0 + psi * gamma * E)

    r_TE = gamma
    return r_TE


def te_reflection_coeff_grad(lam, omega, thicknesses, resistivities):
    """
    TE reflection coefficient AND its gradient w.r.t. log(resistivity).

    Returns both r_TE and dr_TE/d(ln rho_j) for every layer j,
    computed in a single forward + backward pass through the upward recursion.

    Parameters
    ----------
    lam           : (K,)   horizontal wavenumbers [1/m]
    omega         : float or complex  angular frequency [rad/s]
    thicknesses   : (N-1,) layer thicknesses [m]
    resistivities : (N,)   layer resistivities [Ohm.m]

    Returns
    -------
    r_TE     : (K,)    complex128 - TE surface reflection coefficient
    dr_TE    : (N, K)  complex128 - d(r_TE) / d(ln rho_j)
    """
    n_lay = len(resistivities)
    K = len(lam)
    sval = 1j * omega
    resistivities = np.asarray(resistivities, dtype=complex)

    sigma = 1.0 / resistivities
    lam2 = lam ** 2
    Gamma = np.sqrt(lam2[None, :] + (sval * MU0 * sigma)[:, None])  # (N, K)

    # dGamma_j / d(ln rho_j) = -sval*MU0*sigma_j / (2*Gamma_j)  *  (-rho_j)
    #   since d(sigma)/d(ln rho) = -sigma, so d(Gamma^2)/d(ln rho) = -sval*MU0*sigma
    #   => dGamma/d(ln rho) = -sval*MU0*sigma / (2*Gamma)
    dGamma_dlnrho = -sval * MU0 * sigma[:, None] / (2.0 * Gamma)  # (N, K)

    # --- Forward pass (paper Eq. 2): store per-interface psi, phase E, and the
    #     incoming deeper reflection gamma_{j+1}.  Index 0 is the air interface. ---
    psi_store = np.empty((n_lay, K), dtype=complex)
    exp_store = np.empty((n_lay, K), dtype=complex)
    gbelow_store = np.empty((n_lay, K), dtype=complex)

    gamma = np.zeros(K, dtype=complex)          # gamma_N = 0 (base half-space)
    for j in range(n_lay - 1, -1, -1):
        G_above = lam if j == 0 else Gamma[j - 1]
        psi = (G_above - Gamma[j]) / (G_above + Gamma[j])
        E = np.exp(-2.0 * Gamma[j] * thicknesses[j]) if j < n_lay - 1 \
            else np.zeros(K, dtype=complex)
        gbelow_store[j] = gamma
        psi_store[j] = psi
        exp_store[j] = E
        gamma = (psi + gamma * E) / (1.0 + psi * gamma * E)
    r_TE = gamma

    # --- Backward pass: adjoint lam_adj_j = d r_TE / d gamma_j, seeded at the
    #     surface (gamma_0 = r_TE) and propagated toward deeper interfaces.
    #     Gamma_j enters gamma_j (below the interface, via psi_j and E_j) and
    #     gamma_{j-1} (above the interface, via psi_{j-1}). ---
    dr_TE_all = np.zeros((n_lay, K), dtype=complex)
    lam_adj = np.ones(K, dtype=complex)         # d r_TE / d gamma_0

    for j in range(n_lay):
        G_above = lam if j == 0 else Gamma[j - 1]
        Gj = Gamma[j]
        psi = psi_store[j]
        E = exp_store[j]
        g_below = gbelow_store[j]
        denom2 = (1.0 + psi * g_below * E) ** 2

        dg_dpsi = (1.0 - (g_below * E) ** 2) / denom2
        dg_dE = g_below * (1.0 - psi ** 2) / denom2
        dg_dgbelow = E * (1.0 - psi ** 2) / denom2

        # Gamma_j (below the interface): via psi_j and the phase E_j
        dpsi_dGj = -2.0 * G_above / (G_above + Gj) ** 2
        dE_dGj = (-2.0 * thicknesses[j] * E) if j < n_lay - 1 else 0.0
        dr_TE_all[j] += lam_adj * (dg_dpsi * dpsi_dGj + dg_dE * dE_dGj)

        # Gamma_{j-1} (above the interface): via psi_j only
        if j >= 1:
            dpsi_dGabove = 2.0 * Gj / (G_above + Gj) ** 2
            dr_TE_all[j - 1] += lam_adj * dg_dpsi * dpsi_dGabove

        lam_adj = lam_adj * dg_dgbelow            # propagate to gamma_{j+1}

    dr_TE_all *= dGamma_dlnrho
    return r_TE, dr_TE_all
