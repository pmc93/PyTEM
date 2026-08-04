"""
kernels_jacobian.py - Numba JIT and CuPy GPU kernels for the analytical Jacobian.

Implements the adjoint upward recursion: a single forward+backward pass per
(omega, lambda) point yields d(r_TE)/d(ln rho_j) for all N layers at once.

Backend rationale
-----------------
Numba (scalar loops, JIT-compiled to SIMD + prange over gate times)
    Tight scalar loops compile to native SIMD machine code and prange over
    gate times parallelises across CPU cores.  Allocating large intermediate
    arrays (n_f, K) inside a JIT kernel carries heap overhead and the working
    set would exceed L2 cache for typical problem sizes (30 layers, 101-pt
    Hankel filter), so a per-frequency inner loop is preferred over
    NumPy-style broadcasting.

GPU / CuPy (full (n_t, n_f, K) tensor batched in a single CuPy operation)
    CUDA throughput depends on keeping many warps in flight simultaneously.
    Batching all gate times and all filter frequencies into one (n_t, n_f, K)
    tensor maximises occupancy.  Per-frequency Python loops would serialise
    kernel launches and leave most warps idle.

System filter
-------------
A system filter H(omega) is applied by multiplying the complex kernel value
*before* taking the imaginary (DLF) or real (Euler) part.  Since H(omega)
does not depend on resistivity, the same weight applies identically to the
forward kernel and to every layer's gradient kernel:

    d/d(ln rho_j) [ H(omega) * K(omega, rho) ] = H(omega) * dK/d(ln rho_j)

Callers pre-evaluate filter_weights[n_t, n_eval] = H(omega_ki) using
_precompute_filter_dlf / _precompute_filter_euler from forward.py.
When no filter is needed, pass np.ones((n_t, n_eval), dtype=np.complex128).
"""

import numpy as np
from .transform_weights import MU0
from .backends import HAS_CUDA

try:
    import numba as nb
    HAS_NUMBA = True
    _NB_OPTS = {'nogil': True, 'cache': True}
except ImportError:
    HAS_NUMBA = False


# ============================================================================
# Numba JIT kernels
# ============================================================================
#
# Scalar loops compiled to native SIMD machine code via Numba JIT.
# prange over gate times achieves parallelism across CPU cores.
# Per-frequency inner loops beat NumPy broadcasting inside JIT because:
#   1. Heap allocation of (n_f, K) intermediates carries overhead in JIT kernels
#   2. Large tensors exceed L2 cache for typical problem sizes
#
# _te_rte_grad_jit handles both real (DLF) and complex (Euler) omega
# natively - the same JIT function is called by both DLF and Euler kernels.
# ============================================================================

if HAS_NUMBA:

    @nb.njit(**_NB_OPTS)
    def _te_rte_grad_jit(lam, omega, thicknesses, resistivities, mu0):
        """Upward recursion + adjoint gradient - single omega, scalar loops (Numba JIT).

        Returns r_te (K,) and dr_te (N, K) = d(r_TE)/d(ln rho_j) for all layers.
        Handles both real (DLF) and complex (Euler) omega natively.
        """
        n_lay = len(resistivities)
        n_lam = len(lam)
        sval  = 1j * omega

        Gamma  = np.empty((n_lay, n_lam), dtype=np.complex128)
        dGamma = np.empty((n_lay, n_lam), dtype=np.complex128)
        for j in range(n_lay):
            sigma_j = 1.0 / resistivities[j]
            prod    = sval * mu0 * sigma_j
            for m in range(n_lam):
                g            = np.sqrt(lam[m]**2 + prod)
                Gamma[j, m]  = g
                dGamma[j, m] = -prod / (2.0 * g)

        r_te  = np.empty(n_lam, dtype=np.complex128)
        dr_te = np.zeros((n_lay, n_lam), dtype=np.complex128)

        psi_s = np.empty(n_lay, dtype=np.complex128)
        e_s   = np.empty(n_lay, dtype=np.complex128)
        gb_s  = np.empty(n_lay, dtype=np.complex128)

        for m in range(n_lam):
            # Upward recursion (paper Eq. 2): gamma_N = 0 ... r_TE = gamma_0.
            gamma = 0.0 + 0.0j
            for j in range(n_lay - 1, -1, -1):
                G_above = (lam[m] + 0.0j) if j == 0 else Gamma[j - 1, m]
                Gj = Gamma[j, m]
                psi = (G_above - Gj) / (G_above + Gj)
                if j < n_lay - 1:
                    E = np.exp(-2.0 * Gj * thicknesses[j])
                else:
                    E = 0.0 + 0.0j
                gb_s[j]  = gamma
                psi_s[j] = psi
                e_s[j]   = E
                gamma = (psi + gamma * E) / (1.0 + psi * gamma * E)
            r_te[m] = gamma

            # Adjoint backward pass; contributions scaled by dGamma/d(ln rho).
            ladj = 1.0 + 0.0j
            for j in range(n_lay):
                G_above = (lam[m] + 0.0j) if j == 0 else Gamma[j - 1, m]
                Gj = Gamma[j, m]
                psi = psi_s[j]; E = e_s[j]; g = gb_s[j]
                den2 = (1.0 + psi * g * E)**2
                dg_dpsi = (1.0 - (g * E)**2) / den2
                dg_dE   = g * (1.0 - psi**2) / den2
                dg_dgb  = E * (1.0 - psi**2) / den2
                dpsi_dGj = -2.0 * G_above / (G_above + Gj)**2
                if j < n_lay - 1:
                    dE_dGj = -2.0 * thicknesses[j] * E
                else:
                    dE_dGj = 0.0 + 0.0j
                dr_te[j, m] += ladj * (dg_dpsi * dpsi_dGj + dg_dE * dE_dGj) * dGamma[j, m]
                if j >= 1:
                    dr_te[j - 1, m] += ladj * dg_dpsi * (2.0 * Gj / (G_above + Gj)**2) * dGamma[j - 1, m]
                ladj = ladj * dg_dgb

        return r_te, dr_te

    @nb.njit(**_NB_OPTS)
    def _tem_circular_grad_jit(times, thicknesses, resistivities,
                               lam, lam_hj1, mu0,
                               fourier_base, fourier_weights,
                               filter_weights):
        """Circle dB/dt + analytical Jacobian (Numba JIT, DLF).

        Works for circle_central and circle_offset - only lam_hj1 differs:
            circle_central : lam_hj1 = lam * h_j1
            circle_offset  : lam_hj1 = lam * J0(lam * rx_offset) * h_j1

        filter_weights : (n_t, n_f) complex128
            Pre-evaluated system filter H(omega_ki).
            Pass np.ones((n_t, n_f), complex) when no filter is needed.
            H(omega) multiplies the complex kernel before the imaginary part
            is taken; since H is independent of resistivity, it applies
            equally to the forward response and every layer's gradient.

        Returns dbdt (n_t,) and J_raw (n_t, N) before log-log conversion.
        """
        n_t   = len(times)
        n_f   = len(fourier_base)
        n_lam = len(lam)
        n_lay = len(resistivities)
        dbdt  = np.zeros(n_t)
        J_raw = np.zeros((n_t, n_lay))

        for i in range(n_t):
            t       = times[i]
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_f):
                omega = fourier_base[k] / t
                fw    = filter_weights[i, k]   # H(omega) for this gate/frequency
                r_te, dr_te = _te_rte_grad_jit(
                    lam, omega, thicknesses, resistivities, mu0)
                hz_c = 0.0 + 0.0j
                for m in range(n_lam):
                    hz_c += r_te[m] * lam_hj1[m]
                hz_c *= 0.5 * fw            # apply filter before taking imaginary part
                hz_acc += mu0 * hz_c.imag * fourier_weights[k]
                for j in range(n_lay):
                    dhz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        dhz_c += dr_te[j, m] * lam_hj1[m]
                    dhz_c      *= 0.5 * fw  # same H(omega): independent of resistivity
                    dhz_acc[j] += mu0 * dhz_c.imag * fourier_weights[k]
            dbdt[i]     = hz_acc  / t
            J_raw[i, :] = dhz_acc / t
        return dbdt, J_raw

    @nb.njit(**_NB_OPTS)
    def _tem_square_grad_jit(times, thicknesses, resistivities,
                             dist_q, area_w, quad_scale,
                             h_base, h_j0, mu0,
                             fourier_base, fourier_weights,
                             filter_weights, altitude=0.0):
        """Square-loop dB/dt + analytical Jacobian (Numba JIT, DLF).

        Works for square_central (one-quadrant GL with quad_scale=4.0) and
        square_offset (full-square GL with quad_scale=1.0).

        filter_weights : (n_t, n_f) complex128 - see _tem_circular_grad_jit.
        Returns dbdt (n_t,) and J_raw (n_t, N) before log-log conversion.
        """
        n_t   = len(times)
        n_f   = len(fourier_base)
        n_q   = len(dist_q)
        n_lam = len(h_base)
        n_lay = len(resistivities)
        _4pi  = 4.0 * np.pi
        dbdt  = np.zeros(n_t)
        J_raw = np.zeros((n_t, n_lay))

        lam_q  = np.empty(n_lam, dtype=np.float64)
        kern_q = np.empty(n_lam, dtype=np.float64)

        for i in range(n_t):
            t       = times[i]
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_f):
                omega = fourier_base[k] / t
                fw    = filter_weights[i, k]   # H(omega)
                hz_f  = 0.0 + 0.0j
                dhz_f = np.zeros(n_lay, dtype=np.complex128)
                for q in range(n_q):
                    rq = dist_q[q];  wq = area_w[q]
                    for m in range(n_lam):
                        lm        = h_base[m] / rq
                        lam_q[m]  = lm
                        kern_q[m] = lm * lm * h_j0[m] / (rq * _4pi)
                        if altitude != 0.0:
                            kern_q[m] *= np.exp(-lm * altitude)
                    r_te_q, dr_te_q = _te_rte_grad_jit(
                        lam_q, omega, thicknesses, resistivities, mu0)
                    hz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        hz_c += r_te_q[m] * kern_q[m]
                    hz_f += wq * hz_c
                    for j in range(n_lay):
                        dhz_c = 0.0 + 0.0j
                        for m in range(n_lam):
                            dhz_c += dr_te_q[j, m] * kern_q[m]
                        dhz_f[j] += wq * dhz_c
                # Apply quad_scale and filter, then accumulate
                hz_f  *= quad_scale * fw   # H(omega) independent of resistivity
                dhz_f *= quad_scale * fw
                hz_acc += mu0 * hz_f.imag * fourier_weights[k]
                for j in range(n_lay):
                    dhz_acc[j] += mu0 * dhz_f[j].imag * fourier_weights[k]
            dbdt[i]     = hz_acc  / t
            J_raw[i, :] = dhz_acc / t
        return dbdt, J_raw

    @nb.njit(**_NB_OPTS)
    def _tem_circular_grad_euler_jit(times, thicknesses, resistivities,
                                     lam, lam_hj1, mu0, e_eta, e_A,
                                     filter_weights):
        """Circle dB/dt + analytical Jacobian (Numba JIT, Euler-Stehfest).

        Uses complex Bromwich frequencies omega_k = k*pi/t - (A/2t)*i.
        _te_rte_grad_jit handles complex omega natively - no separate
        implementation is needed; the recursion is analytic in omega.

        filter_weights : (n_t, n_eval) complex128
            H(omega_k) for each gate and Euler term.
            Pass np.ones((n_t, n_eval), complex) when no filter is needed.
            Applied before taking the real part (Euler accumulation).

        Returns dbdt (n_t,) and J_raw (n_t, N) with step-off sign applied.
        """
        n_t    = len(times)
        n_eval = len(e_eta)
        n_lam  = len(lam)
        n_lay  = len(resistivities)
        dbdt   = np.zeros(n_t)
        J_raw  = np.zeros((n_t, n_lay))

        for i in range(n_t):
            t      = times[i]
            c      = e_A / (2.0 * t)
            h_step = np.pi / t
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_eval):
                omega  = k * h_step - c * 1j   # complex Bromwich frequency
                sign_k = (-1.0)**k * e_eta[k]
                fw     = filter_weights[i, k]  # H(omega_k)
                r_te, dr_te = _te_rte_grad_jit(
                    lam, omega, thicknesses, resistivities, mu0)
                hz_c = 0.0 + 0.0j
                for m in range(n_lam):
                    hz_c += r_te[m] * lam_hj1[m]
                hz_c *= 0.5 * fw            # apply filter before taking real part
                hz_acc += sign_k * mu0 * hz_c.real
                for j in range(n_lay):
                    dhz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        dhz_c += dr_te[j, m] * lam_hj1[m]
                    dhz_c      *= 0.5 * fw  # H(omega) independent of resistivity
                    dhz_acc[j] += sign_k * mu0 * dhz_c.real
            prefac      = np.exp(e_A / 2.0) / t
            dbdt[i]     = -prefac * hz_acc    # -1 for step-off
            J_raw[i, :] = -prefac * dhz_acc
        return dbdt, J_raw

    @nb.njit(**_NB_OPTS)
    def _tem_square_grad_euler_jit(times, thicknesses, resistivities,
                                   dist_q, area_w, quad_scale,
                                   h_base, h_j0, mu0, e_eta, e_A,
                                   filter_weights, altitude=0.0):
        """Square-loop dB/dt + analytical Jacobian (Numba JIT, Euler-Stehfest).

        filter_weights : (n_t, n_eval) complex128 - see _tem_circular_grad_euler_jit.
        Returns dbdt (n_t,) and J_raw (n_t, N) with step-off sign applied.
        """
        n_t    = len(times)
        n_eval = len(e_eta)
        n_q    = len(dist_q)
        n_lam  = len(h_base)
        n_lay  = len(resistivities)
        _4pi   = 4.0 * np.pi
        dbdt   = np.zeros(n_t)
        J_raw  = np.zeros((n_t, n_lay))

        lam_q  = np.empty(n_lam, dtype=np.float64)
        kern_q = np.empty(n_lam, dtype=np.float64)

        for i in range(n_t):
            t      = times[i]
            c      = e_A / (2.0 * t)
            h_step = np.pi / t
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_eval):
                omega  = k * h_step - c * 1j
                sign_k = (-1.0)**k * e_eta[k]
                fw     = filter_weights[i, k]   # H(omega_k)
                hz_f  = 0.0 + 0.0j
                dhz_f = np.zeros(n_lay, dtype=np.complex128)
                for q in range(n_q):
                    rq = dist_q[q];  wq = area_w[q]
                    for m in range(n_lam):
                        lm        = h_base[m] / rq
                        lam_q[m]  = lm
                        kern_q[m] = lm * lm * h_j0[m] / (rq * _4pi)
                        if altitude != 0.0:
                            kern_q[m] *= np.exp(-lm * altitude)
                    r_te_q, dr_te_q = _te_rte_grad_jit(
                        lam_q, omega, thicknesses, resistivities, mu0)
                    hz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        hz_c += r_te_q[m] * kern_q[m]
                    hz_f += wq * hz_c
                    for j in range(n_lay):
                        dhz_c = 0.0 + 0.0j
                        for m in range(n_lam):
                            dhz_c += dr_te_q[j, m] * kern_q[m]
                        dhz_f[j] += wq * dhz_c
                hz_f  *= quad_scale * fw   # H(omega) independent of resistivity
                dhz_f *= quad_scale * fw
                hz_acc  += sign_k * mu0 * hz_f.real
                for j in range(n_lay):
                    dhz_acc[j] += sign_k * mu0 * dhz_f[j].real
            prefac      = np.exp(e_A / 2.0) / t
            dbdt[i]     = -prefac * hz_acc
            J_raw[i, :] = -prefac * dhz_acc
        return dbdt, J_raw


# ============================================================================
# CuPy GPU kernels
# ============================================================================
#
# The full (n_t, n_f, K) frequency x wavenumber tensor is processed in a
# single batched CuPy operation.  CUDA throughput depends on keeping many
# warps in flight simultaneously; the large (n_t, n_f, K) batch saturates
# GPU occupancy.  Per-frequency Python loops would serialise kernel launches
# and leave most warps idle.
#
# System filter: d_filter_weights (n_t, n_f) cupy complex128, pre-evaluated
# on the GPU.  For DLF: (hz * d_filter_weights).imag  before the f_sin dot.
# For Euler: (hz * d_filter_weights).real  before summing Euler coefficients.
# ============================================================================

if HAS_CUDA:
    import cupy as cp

    def _te_reflection_coeff_grad_gpu(d_lam, omega_2d, d_thicknesses, d_resistivities):
        """Batched TE gradient on GPU - (n_t, n_f, K) tensor.

        Parameters
        ----------
        d_lam           : (K,)        cupy float64
        omega_2d        : (n_t, n_f)  cupy complex128 - real or complex omega
        d_thicknesses   : (N-1,)      cupy float64
        d_resistivities : (N,)        cupy float64

        Returns
        -------
        r_TE  : (n_t, n_f, K)    cupy complex128
        dr_TE : (N, n_t, n_f, K) cupy complex128
        """
        n_lay    = len(d_resistivities)
        n_t, n_f = omega_2d.shape
        K        = len(d_lam)
        d_sigma  = 1.0 / d_resistivities                             # (N,)
        sval     = 1j * omega_2d[:, :, None]                         # (n_t, n_f, 1)

        Gamma = cp.sqrt(
            d_lam[None, None, :]**2
            + sval * MU0 * d_sigma[:, None, None, None])             # (N, n_t, n_f, K)

        sval_4d       = 1j * omega_2d[None, :, :, None]              # (1, n_t, n_f, 1)
        dGamma_dlnrho = (-sval_4d * MU0
                         * d_sigma[:, None, None, None]
                         / (2.0 * Gamma))                             # (N, n_t, n_f, K)

        # Upward recursion (paper Eq. 2), vectorised over (n_t, n_f, K).
        psi_store = cp.empty((n_lay, n_t, n_f, K), dtype=cp.complex128)
        exp_store = cp.empty((n_lay, n_t, n_f, K), dtype=cp.complex128)
        gb_store  = cp.empty((n_lay, n_t, n_f, K), dtype=cp.complex128)

        gamma = cp.zeros((n_t, n_f, K), dtype=cp.complex128)         # gamma_N = 0
        for j in range(n_lay - 1, -1, -1):
            G_above = d_lam[None, None, :] if j == 0 else Gamma[j - 1]
            Gj = Gamma[j]
            psi = (G_above - Gj) / (G_above + Gj)
            if j < n_lay - 1:
                E = cp.exp(-2.0 * Gj * d_thicknesses[j])
            else:
                E = cp.zeros((n_t, n_f, K), dtype=cp.complex128)
            gb_store[j]  = gamma
            psi_store[j] = psi
            exp_store[j] = E
            gamma = (psi + gamma * E) / (1.0 + psi * gamma * E)
        r_TE = gamma

        # Adjoint backward pass; contributions scaled by dGamma/d(ln rho).
        dr_TE_all = cp.zeros((n_lay, n_t, n_f, K), dtype=cp.complex128)
        ladj = cp.ones((n_t, n_f, K), dtype=cp.complex128)          # d r_TE / d gamma_0
        for j in range(n_lay):
            G_above = d_lam[None, None, :] if j == 0 else Gamma[j - 1]
            Gj = Gamma[j]
            psi = psi_store[j]; E = exp_store[j]; g = gb_store[j]
            den2 = (1.0 + psi * g * E)**2
            dg_dpsi = (1.0 - (g * E)**2) / den2
            dg_dE   = g * (1.0 - psi**2) / den2
            dg_dgb  = E * (1.0 - psi**2) / den2
            dpsi_dGj = -2.0 * G_above / (G_above + Gj)**2
            dE_dGj = (-2.0 * d_thicknesses[j] * E) if j < n_lay - 1 else 0.0
            dr_TE_all[j] += ladj * (dg_dpsi * dpsi_dGj + dg_dE * dE_dGj) * dGamma_dlnrho[j]
            if j >= 1:
                dr_TE_all[j - 1] += ladj * dg_dpsi * (2.0 * Gj / (G_above + Gj)**2) * dGamma_dlnrho[j - 1]
            ladj = ladj * dg_dgb

        return r_TE, dr_TE_all

    def _tem_circular_grad_gpu(times, thicknesses, resistivities, tx_radius,
                               lam_hj1, d_h_base, d_f_base, d_f_sin,
                               d_filter_weights):
        """Circle dB/dt + analytical Jacobian on GPU (DLF).

        Works for circle_central and circle_offset via lam_hj1.

        d_filter_weights : (n_t, n_f) cupy complex128
            H(omega_ki) pre-evaluated on GPU.
            Pass cp.ones((n_t, n_f), dtype=cp.complex128) when no filter is needed.

        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times   = cp.asarray(times)
        d_thick   = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho     = cp.asarray(resistivities, dtype=cp.float64)
        d_lam     = d_h_base / float(tx_radius)
        d_lam_hj1 = cp.asarray(lam_hj1)
        n_t   = len(times)
        n_f   = len(d_f_base)
        K     = len(d_h_base)
        n_lay = len(resistivities)
        omega_full = d_f_base[None, :] / d_times[:, None]   # (n_t, n_f) real

        d_dbdt = cp.empty(n_t,          dtype=cp.float64)
        d_Jraw = cp.empty((n_t, n_lay), dtype=cp.float64)

        # Chunk the (independent) gate-time axis so the (n_lay, chunk, n_f, K)
        # recursion tensors fit GPU memory instead of thrashing on OOM retries.
        chunk = max(1, int(0.75e9 // (6 * max(1, n_lay) * n_f * K * 16)))

        for i0 in range(0, n_t, chunk):
            i1       = min(i0 + chunk, n_t)
            omega_2d = omega_full[i0:i1]
            fw_c     = d_filter_weights[i0:i1]
            dt_c     = d_times[i0:i1]

            r_te, dr_te = _te_reflection_coeff_grad_gpu(d_lam, omega_2d, d_thick, d_rho)
            # r_te: (c, n_f, K),  dr_te: (N, c, n_f, K)

            hz  = 0.5 * cp.sum(r_te  * d_lam_hj1[None, None, :],           axis=2)  # (c, n_f)
            dhz = 0.5 * cp.sum(dr_te * d_lam_hj1[None, None, None, :], axis=3)      # (N, c, n_f)

            # Apply system filter: H(omega) multiplies complex Hz before imaginary part
            sig  = MU0 * (hz  * fw_c).imag              # (c, n_f)
            dsig = MU0 * (dhz * fw_c[None, :, :]).imag  # (N, c, n_f)

            d_dbdt[i0:i1] = cp.sum(sig  * d_f_sin[None, :],           axis=1) / dt_c          # (c,)
            d_Jraw[i0:i1] = (cp.sum(dsig * d_f_sin[None, None, :], axis=2) / dt_c[None, :]).T  # (c, N)
        return cp.asnumpy(d_dbdt), cp.asnumpy(d_Jraw)

    def _tem_square_grad_gpu(times, thicknesses, resistivities,
                             dist_q, area_w, quad_scale,
                             d_h_base, d_h_j0, d_f_base, d_f_sin,
                             d_filter_weights, altitude=0.0):
        """Square-loop dB/dt + analytical Jacobian on GPU (DLF).

        Loops over n_q quadrature points; each iteration runs the full
        (n_t, n_f, K) adjoint on the GPU then accumulates with area weight.

        d_filter_weights : (n_t, n_f) cupy complex128 - see _tem_circular_grad_gpu.
        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times  = cp.asarray(times)
        d_thick  = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho    = cp.asarray(resistivities, dtype=cp.float64)
        n_t      = len(times)
        n_f      = len(d_f_base)
        n_lay    = len(resistivities)
        K        = len(d_h_base)
        omega_full = d_f_base[None, :] / d_times[:, None]   # (n_t, n_f)
        _4pi     = 4.0 * np.pi
        dist_q   = np.asarray(dist_q, dtype=float)          # host scalars
        area_w   = np.asarray(area_w, dtype=float)

        d_dbdt = cp.zeros(n_t,          dtype=cp.float64)
        d_Jraw = cp.zeros((n_t, n_lay), dtype=cp.float64)

        # Chunk the (independent) gate-time axis so the recursion's
        # (n_lay, chunk, n_f, K) tensors fit GPU memory instead of thrashing on
        # OOM retries when the full n_t batch is too large.  ~6 live tensors.
        chunk = max(1, int(0.75e9 // (6 * max(1, n_lay) * n_f * K * 16)))

        for a in range(0, n_t, chunk):
            b        = min(a + chunk, n_t)
            omega_2d = omega_full[a:b]                       # (c, n_f)
            fw_c     = d_filter_weights[a:b]                 # (c, n_f)
            dt_c     = d_times[a:b]                          # (c,)
            for q in range(len(dist_q)):
                rq       = float(dist_q[q]);  wq = float(area_w[q])
                d_lam_q  = d_h_base / rq
                d_kern_q = d_lam_q**2 * d_h_j0 / (rq * _4pi)
                if altitude != 0.0:
                    d_kern_q = d_kern_q * cp.exp(-d_lam_q * altitude)

                r_te, dr_te = _te_reflection_coeff_grad_gpu(
                    d_lam_q, omega_2d, d_thick, d_rho)
                # r_te: (c, n_f, K),  dr_te: (N, c, n_f, K)

                hz  = cp.sum(r_te  * d_kern_q[None, None, :],           axis=2)  # (c, n_f)
                dhz = cp.sum(dr_te * d_kern_q[None, None, None, :], axis=3)      # (N, c, n_f)

                sig  = MU0 * (hz  * fw_c).imag              # (c, n_f)
                dsig = MU0 * (dhz * fw_c[None, :, :]).imag  # (N, c, n_f)

                d_dbdt[a:b] += wq * cp.sum(sig  * d_f_sin[None, :],           axis=1) / dt_c
                d_Jraw[a:b] += wq * (cp.sum(dsig * d_f_sin[None, None, :], axis=2) / dt_c[None, :]).T

        d_dbdt *= quad_scale
        d_Jraw *= quad_scale
        return cp.asnumpy(d_dbdt), cp.asnumpy(d_Jraw)

    def _tem_circular_grad_euler_gpu(times, thicknesses, resistivities, tx_radius,
                                     lam_hj1, d_h_base, e_eta, e_A,
                                     d_filter_weights):
        """Circle dB/dt + analytical Jacobian on GPU (Euler-Stehfest).

        _te_reflection_coeff_grad_gpu handles complex omega_2d natively, so
        no separate Euler kernel is needed - the same GPU adjoint recursion
        works for both real (DLF) and complex (Euler) frequencies.

        d_filter_weights : (n_t, n_eval) cupy complex128
            H(omega_k) for each gate and Euler term.
            Applied before taking the real part (Euler accumulation).

        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times   = cp.asarray(times)
        d_thick   = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho     = cp.asarray(resistivities, dtype=cp.float64)
        d_lam     = d_h_base / float(tx_radius)
        d_lam_hj1 = cp.asarray(lam_hj1)
        n_t       = len(times)
        n_eval    = len(e_eta)
        K         = len(d_h_base)
        n_lay     = len(resistivities)
        k_arr     = cp.arange(n_eval, dtype=cp.float64)
        signs_k   = cp.asarray((-1.0)**np.arange(n_eval) * e_eta)   # (n_eval,)
        prefac_all = cp.exp(float(e_A) / 2.0) / d_times            # (n_t,)

        d_dbdt = cp.empty(n_t,          dtype=cp.float64)
        d_Jraw = cp.empty((n_t, n_lay), dtype=cp.float64)
        chunk  = max(1, int(0.75e9 // (6 * max(1, n_lay) * n_eval * K * 16)))

        for i0 in range(0, n_t, chunk):
            i1     = min(i0 + chunk, n_t)
            dt_c   = d_times[i0:i1]
            fw_c   = d_filter_weights[i0:i1]
            c_vals = float(e_A) / (2.0 * dt_c)
            h_vals = cp.full(i1 - i0, np.pi) / dt_c
            omega_2d = k_arr[None, :] * h_vals[:, None] - c_vals[:, None] * 1j  # (c, n_eval)

            r_te, dr_te = _te_reflection_coeff_grad_gpu(d_lam, omega_2d, d_thick, d_rho)
            hz  = 0.5 * cp.sum(r_te  * d_lam_hj1[None, None, :],           axis=2)
            dhz = 0.5 * cp.sum(dr_te * d_lam_hj1[None, None, None, :], axis=3)

            hz_filt  = hz  * fw_c
            dhz_filt = dhz * fw_c[None, :, :]
            hz_acc  = MU0 * cp.sum(signs_k[None, :]       * hz_filt.real,  axis=1)
            dhz_acc = MU0 * cp.sum(signs_k[None, None, :] * dhz_filt.real, axis=2)

            prefac = prefac_all[i0:i1]
            d_dbdt[i0:i1] = -prefac * hz_acc
            d_Jraw[i0:i1] = (-prefac[None, :] * dhz_acc).T
        return cp.asnumpy(d_dbdt), cp.asnumpy(d_Jraw)

    def _tem_square_grad_euler_gpu(times, thicknesses, resistivities,
                                   dist_q, area_w, quad_scale,
                                   d_h_base, d_h_j0, e_eta, e_A,
                                   d_filter_weights, altitude=0.0):
        """Square-loop dB/dt + analytical Jacobian on GPU (Euler-Stehfest).

        d_filter_weights : (n_t, n_eval) cupy complex128 - see _tem_circular_grad_euler_gpu.
        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times  = cp.asarray(times)
        d_thick  = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho    = cp.asarray(resistivities, dtype=cp.float64)
        n_t      = len(times)
        n_lay    = len(resistivities)
        n_eval   = len(e_eta)
        K        = len(d_h_base)
        _4pi     = 4.0 * np.pi
        dist_q   = np.asarray(dist_q, dtype=float)          # host scalars
        area_w   = np.asarray(area_w, dtype=float)
        k_arr    = cp.arange(n_eval, dtype=cp.float64)
        signs_k  = cp.asarray((-1.0)**np.arange(n_eval) * e_eta)
        prefac_all = cp.exp(float(e_A) / 2.0) / d_times

        d_hz_acc  = cp.zeros(n_t,          dtype=cp.float64)
        d_dhz_acc = cp.zeros((n_t, n_lay), dtype=cp.float64)

        chunk = max(1, int(0.75e9 // (6 * max(1, n_lay) * n_eval * K * 16)))

        for i0 in range(0, n_t, chunk):
            i1     = min(i0 + chunk, n_t)
            dt_c   = d_times[i0:i1]
            fw_c   = d_filter_weights[i0:i1]
            c_vals = float(e_A) / (2.0 * dt_c)
            h_vals = cp.full(i1 - i0, np.pi) / dt_c
            omega_2d = k_arr[None, :] * h_vals[:, None] - c_vals[:, None] * 1j  # (c, n_eval)

            for q in range(len(dist_q)):
                rq       = float(dist_q[q]);  wq = float(area_w[q])
                d_lam_q  = d_h_base / rq
                d_kern_q = d_lam_q**2 * d_h_j0 / (rq * _4pi)
                if altitude != 0.0:
                    d_kern_q = d_kern_q * cp.exp(-d_lam_q * altitude)

                r_te, dr_te = _te_reflection_coeff_grad_gpu(d_lam_q, omega_2d, d_thick, d_rho)
                # r_te: (c, n_eval, K),  dr_te: (N, c, n_eval, K)

                hz  = cp.sum(r_te  * d_kern_q[None, None, :],           axis=2)  # (c, n_eval)
                dhz = cp.sum(dr_te * d_kern_q[None, None, None, :], axis=3)      # (N, c, n_eval)

                hz_filt  = hz  * fw_c                 # (c, n_eval)
                dhz_filt = dhz * fw_c[None, :, :]     # (N, c, n_eval)

                d_hz_acc[i0:i1]  += wq * MU0 * cp.sum(signs_k[None, :]       * hz_filt.real,  axis=1)
                d_dhz_acc[i0:i1] += wq * (MU0 * cp.sum(signs_k[None, None, :] * dhz_filt.real, axis=2)).T

        d_hz_acc  *= quad_scale
        d_dhz_acc *= quad_scale
        d_dbdt     = -prefac_all * d_hz_acc
        d_Jraw     = -(prefac_all[:, None] * d_dhz_acc)
        return cp.asnumpy(d_dbdt), cp.asnumpy(d_Jraw)
