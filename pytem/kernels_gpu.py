"""
kernels_gpu.py - CuPy/CUDA GPU kernels for TEM forward modelling.

Contains:
  - _te_reflection_coeff_gpu  : upward recursion, batched complex
  - _tem_circular_gpu         : central/offset circular loop (Fourier DLF)
  - _tem_square_gpu           : square loop (Fourier DLF)
  - _tem_circular_euler_gpu   : central/offset (Euler acceleration)
  - _tem_square_euler_gpu     : square loop (Euler acceleration)

All kernels accept an optional filter_weights parameter (n_t, n_eval)
complex128. When provided, Hz(omega) is multiplied by H(omega) before
the final transform. Pass None when no system filter is needed.
"""

from .transform_weights import MU0
from .backends import HAS_CUDA

if HAS_CUDA:
    import cupy as cp
    import numpy as np

    # Gate-time chunking for the batched square kernels.  The reflection
    # recursion allocates several (n_lay, chunk, n_f, K) complex128 tensors at
    # once; on small GPUs the full n_t batch does not fit and CuPy thrashes on
    # out-of-memory retries.  Chunking the (independent) gate-time axis keeps
    # peak memory within this budget while preserving exact results.
    _GPU_TIME_CHUNK_BUDGET = 0.75e9   # ~0.75 GB peak target per chunk

    def _gpu_time_chunk(n_f, K, n_lay):
        """Largest gate-time chunk whose recursion tensors fit the budget."""
        # ~6 live (n_lay, chunk, n_f, K) complex128 tensors in the gradient
        # recursion (Gamma, dGamma, r_store, exp_store, dr_TE_all, temporaries).
        denom = 6 * max(1, n_lay) * n_f * K * 16
        return max(1, int(_GPU_TIME_CHUNK_BUDGET // denom))

    def _te_reflection_coeff_gpu(d_lam, omega_2d, d_thicknesses, d_resistivities):
        """
        GPU-batched TE reflection coefficient via upward recursion (complex).
        Works with both real omega (DLF) and complex omega (Euler/Bromwich).
        """
        n_lay = len(d_resistivities)
        sigma = 1.0 / d_resistivities
        sval = 1j * omega_2d

        lam2 = d_lam ** 2
        prod = (MU0 * sigma)[:, None, None] * sval[None, :, :]
        Gamma = cp.sqrt(lam2[None, None, None, :] + prod[:, :, :, None])

        gamma = cp.zeros((*omega_2d.shape, len(d_lam)), dtype=cp.complex128)  # gamma_N = 0
        for j in range(n_lay - 1, -1, -1):
            G_above = d_lam[None, None, :] if j == 0 else Gamma[j - 1]
            Gj = Gamma[j]
            psi = (G_above - Gj) / (G_above + Gj)
            if j < n_lay - 1:
                E = cp.exp(-2.0 * Gj * d_thicknesses[j])
            else:
                E = 0.0
            gamma = (psi + gamma * E) / (1.0 + psi * gamma * E)
        r_TE = gamma
        return r_TE

    # ------------------------------------------------------------------
    # Fourier DLF GPU - circular (central + offset unified)
    # ------------------------------------------------------------------
    def _tem_circular_gpu(times, thicknesses, resistivities, tx_radius,
                          extra_weights, d_h_base, d_h_j1, d_f_base, d_f_sin,
                          filter_weights=None):
        """Circular-loop dB/dt fully on GPU (Fourier DLF).

        The gate-time axis is processed in memory-bounded chunks so the
        (n_lay, chunk, n_f, K) reflection tensors fit GPU memory instead of
        thrashing/OOM on the full n_t batch.  Each n_t row is independent, so
        chunking is numerically exact.
        """
        d_times = cp.asarray(times)
        d_thick = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho = cp.asarray(resistivities, dtype=cp.float64)
        d_lam = d_h_base / float(tx_radius)

        n_t = len(times)
        n_f = len(d_f_base)
        K = len(d_h_base)
        n_lay = len(resistivities)
        omega_full = d_f_base[None, :] / d_times[:, None]
        d_fw = cp.asarray(filter_weights) if filter_weights is not None else None

        chunk = _gpu_time_chunk(n_f, K, n_lay)
        dbdt = cp.empty(n_t, dtype=cp.float64)

        for i0 in range(0, n_t, chunk):
            i1 = min(i0 + chunk, n_t)
            omega_2d = omega_full[i0:i1]
            r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho)
            kernel = r_te * d_lam[None, None, :] * extra_weights[None, None, :]
            hz = 0.5 * cp.sum(kernel * d_h_j1[None, None, :], axis=2)
            if d_fw is not None:
                hz = hz * d_fw[i0:i1]
            sig = MU0 * hz.imag
            dbdt[i0:i1] = cp.sum(sig * d_f_sin[None, :], axis=1) / d_times[i0:i1]

        return cp.asnumpy(dbdt)

    # ------------------------------------------------------------------
    # Fourier DLF GPU - square (VMD area integral)
    # ------------------------------------------------------------------
    def _tem_square_gpu(times, thicknesses, resistivities, dist_q, area_w,
                        d_h_base, d_h_j0, d_f_base, d_f_sin,
                        filter_weights=None, altitude=0.0):
        """Square-loop dB/dt fully on GPU (Fourier DLF + VMD area integral).

        altitude : total Tx+Rx elevation [m]; applies an exp(-lam*altitude)
        upward continuation factor per wavenumber (0.0 = on ground).

        The reflection recursion allocates (n_lay, chunk, n_f, K) complex128
        tensors.  To avoid thrashing/OOM on small GPUs, the gate-time axis is
        processed in chunks sized to a fixed peak-memory budget.  Each n_t row
        is independent, so chunking is numerically exact.
        """
        area_w = np.asarray(area_w, dtype=float)          # host scalars, no sync
        d_times = cp.asarray(times)
        d_thick = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho_lay = cp.asarray(resistivities, dtype=cp.float64)

        n_t = len(times)
        n_f = len(d_f_base)
        n_lay = len(resistivities)
        K = len(d_h_base)
        omega_full = d_f_base[None, :] / d_times[:, None]     # (n_t, n_f)
        d_fw = cp.asarray(filter_weights) if filter_weights is not None else None

        chunk = _gpu_time_chunk(n_f, K, n_lay)
        dbdt = cp.empty(n_t, dtype=cp.float64)

        for a in range(0, n_t, chunk):
            b = min(a + chunk, n_t)
            omega_2d = omega_full[a:b]
            hz_c = cp.zeros((b - a, n_f), dtype=cp.complex128)
            for q in range(len(dist_q)):
                dist = float(dist_q[q])
                d_lam = d_h_base / dist
                kern = (d_lam ** 2) * d_h_j0
                if altitude != 0.0:
                    kern = kern * cp.exp(-d_lam * altitude)
                r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho_lay)
                g = cp.sum(r_te * kern[None, None, :], axis=2) / dist / (4.0 * cp.pi)
                hz_c += float(area_w[q]) * g
            if d_fw is not None:
                hz_c = hz_c * d_fw[a:b]
            sig = MU0 * hz_c.imag
            dbdt[a:b] = cp.sum(sig * d_f_sin[None, :], axis=1) / d_times[a:b]

        return cp.asnumpy(dbdt)

    # ------------------------------------------------------------------
    # Euler GPU - circular (central + offset unified)
    # ------------------------------------------------------------------
    def _tem_circular_euler_gpu(times, thicknesses, resistivities, tx_radius,
                                extra_weights, d_h_base, d_h_j1,
                                euler_eta, euler_A, filter_weights=None):
        """Circular-loop dB/dt on GPU via Euler-accelerated Bromwich inversion.

        Gate-time axis chunked so the (n_lay, chunk, n_euler, K) tensors fit
        GPU memory; each n_t row is independent, so the result is exact.
        """
        d_times = cp.asarray(times)
        d_thick = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho = cp.asarray(resistivities, dtype=cp.float64)
        d_lam = d_h_base / float(tx_radius)

        n_t = len(times)
        n_euler = len(euler_eta)
        K = len(d_h_base)
        n_lay = len(resistivities)
        d_eta = cp.asarray(euler_eta)
        half_A = euler_A / 2.0
        ks = cp.arange(n_euler, dtype=cp.float64)
        signs = cp.asarray([(-1.0)**k for k in range(n_euler)])
        d_fw = cp.asarray(filter_weights) if filter_weights is not None else None

        chunk = _gpu_time_chunk(n_euler, K, n_lay)
        dbdt = cp.empty(n_t, dtype=cp.float64)

        for i0 in range(0, n_t, chunk):
            i1 = min(i0 + chunk, n_t)
            dt_c = d_times[i0:i1]
            c = half_A / dt_c
            h = cp.pi / dt_c
            s_2d = c[:, None] + ks[None, :] * h[:, None] * 1j
            omega_2d = s_2d / 1j
            r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho)
            kernel = r_te * d_lam[None, None, :] * extra_weights[None, None, :]
            hz = 0.5 * cp.sum(kernel * d_h_j1[None, None, :], axis=2)
            if d_fw is not None:
                hz = hz * d_fw[i0:i1]
            fvals = (MU0 * hz).real
            dbdt[i0:i1] = cp.exp(half_A) / dt_c * cp.sum(d_eta[None, :] * signs[None, :] * fvals, axis=1)

        return cp.asnumpy(dbdt)

    # ------------------------------------------------------------------
    # Euler GPU - square (VMD area integral)
    # ------------------------------------------------------------------
    def _tem_square_euler_gpu(times, thicknesses, resistivities, dist_q, area_w,
                              d_h_base, d_h_j0,
                              euler_eta, euler_A, filter_weights=None, altitude=0.0):
        """Square-loop dB/dt on GPU via Euler-accelerated Bromwich + VMD integral.

        altitude : total Tx+Rx elevation [m]; applies an exp(-lam*altitude)
        upward continuation factor per wavenumber (0.0 = on ground).
        """
        n_t = len(times)
        n_euler = len(euler_eta)
        K = len(d_h_base)
        n_lay = len(resistivities)
        area_w = np.asarray(area_w, dtype=float)
        d_eta = cp.asarray(euler_eta)
        half_A = euler_A / 2.0
        ks = cp.arange(n_euler, dtype=cp.float64)
        signs = cp.asarray([(-1.0)**k for k in range(n_euler)])
        d_fw = cp.asarray(filter_weights) if filter_weights is not None else None

        chunk = _gpu_time_chunk(n_euler, K, n_lay)
        dbdt = cp.empty(n_t, dtype=cp.float64)

        for i0 in range(0, n_t, chunk):
            i1 = min(i0 + chunk, n_t)
            dt_c = d_times[i0:i1]
            c = half_A / dt_c
            h = cp.pi / dt_c
            s_2d = c[:, None] + ks[None, :] * h[:, None] * 1j
            omega_2d = s_2d / 1j
            hz_c = cp.zeros((i1 - i0, n_euler), dtype=cp.complex128)
            for q in range(len(dist_q)):
                dist = float(dist_q[q])
                d_lam = d_h_base / dist
                kern = (d_lam ** 2) * d_h_j0
                if altitude != 0.0:
                    kern = kern * cp.exp(-d_lam * altitude)
                r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho_lay)
                g = cp.sum(r_te * kern[None, None, :], axis=2) / dist / (4.0 * cp.pi)
                hz_c += float(area_w[q]) * g
            if d_fw is not None:
                hz_c = hz_c * d_fw[i0:i1]
            fvals = (MU0 * hz_c).real
            dbdt[i0:i1] = cp.exp(half_A) / dt_c * cp.sum(d_eta[None, :] * signs[None, :] * fvals, axis=1)

        return cp.asnumpy(dbdt)
