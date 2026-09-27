"""
kernels_numba.py - Numba JIT kernels for TEM forward modelling.

Contains:
  - _te_rte_jit          : upward recursion (complex), scalar loops
  - _tem_circular_jit    : central/offset circular loop (Fourier DLF)
  - _tem_square_jit      : square loop via VMD area integral (Fourier DLF)
  - _tem_circular_euler_jit : central/offset (Euler acceleration)
  - _tem_square_euler_jit   : square loop (Euler acceleration)

All kernels accept a filter_weights array (n_t, n_eval) complex128.
When no system filter is needed, pass np.ones((n_t, n_eval), complex128).

Recursion variants (``kernel`` option of the forward and Jacobian functions):
  'exact'      : complex sqrt via hypot (default)
  'fast_sqrt'  : sqrt with |z| = sqrt(a^2 + b^2); ~1.8x faster, Euler changes
                 at the 1e-3 level (within Euler-11's own error), DLF at 1e-10
  'vectorized' : wavenumber-innermost loops with fastmath (SIMD) and the
                 fast sqrt; ~2.2x faster, same accuracy as 'fast_sqrt'
"""

import numpy as np

KERNEL_MODES = {'exact': 0, 'fast_sqrt': 1, 'vectorized': 2}

try:
    import numba as nb
    HAS_NUMBA = True
    _NB_OPTS = {'nogil': True, 'cache': True}
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:

    @nb.njit(inline='always', fastmath=True)
    def _csqrt(z):
        """Principal complex sqrt for Re(z) > 0 without hypot (fast_sqrt mode)."""
        re = np.sqrt(0.5 * (np.sqrt(z.real * z.real + z.imag * z.imag) + z.real))
        return complex(re, 0.5 * z.imag / re)

    @nb.njit(fastmath=True, **_NB_OPTS)
    def _te_rte_vec_jit(lam, omega, thicknesses, resistivities, mu0):
        """_te_rte_jit with the wavenumber loop innermost so it vectorises."""
        n, L = len(resistivities), len(lam)
        term = 1j * omega * mu0 / resistivities
        G = np.empty((n, L), dtype=np.complex128)
        E = np.zeros((n, L), dtype=np.complex128)
        att = np.zeros(L)
        depth = n
        for j in range(n):
            if j > 0 and resistivities[j] == resistivities[j - 1]:
                G[j] = G[j - 1]
            else:
                for m in range(L):
                    G[j, m] = _csqrt(lam[m] * lam[m] + term[j])
            if j == n - 1:
                break
            lowest = 1e300
            for m in range(L):
                att[m] += 2.0 * thicknesses[j] * G[j, m].real
                E[j, m] = 0.0 if att[m] > 50.0 else np.exp(-2.0 * thicknesses[j] * G[j, m])
                lowest = min(lowest, att[m])
            if lowest > 50.0:
                depth = j + 1
                break
        R = np.zeros(L, dtype=np.complex128)
        for j in range(depth - 1, -1, -1):
            if j > 0 and resistivities[j] == resistivities[j - 1]:
                for m in range(L):
                    R[m] *= E[j, m]
                continue
            for m in range(L):
                above = lam[m] + 0j if j == 0 else G[j - 1, m]
                psi = (above - G[j, m]) / (above + G[j, m])
                rd = R[m] * E[j, m]
                R[m] = (psi + rd) / (1.0 + psi * rd)
        return R

    @nb.njit(**_NB_OPTS)
    def _te_rte_jit(lam, omega, thicknesses, resistivities, mu0, mode=0):
        """JIT-compiled TE reflection coefficient (upward recursion).

        Two exact shortcuts: the recursion starts at the layer below which the
        two-way attenuation exceeds e^-50 (deeper layers cannot change r_TE in
        double precision), and a layer repeating the one above (e.g. a refined
        DOI model) reuses its gamma and decay, with psi = 0 at the interface.
        """
        if mode == 2:
            return _te_rte_vec_jit(lam, omega, thicknesses, resistivities, mu0)
        n_lay = len(resistivities)
        n_lam = len(lam)
        prod = 1j * omega * mu0 / resistivities
        G = np.empty(n_lay, dtype=np.complex128)
        E = np.empty(n_lay, dtype=np.complex128)
        r_te = np.empty(n_lam, dtype=np.complex128)
        for m in range(n_lam):
            depth, att = n_lay, 0.0
            for j in range(n_lay):
                same = j > 0 and resistivities[j] == resistivities[j - 1]
                z = lam[m]**2 + prod[j]
                G[j] = G[j - 1] if same else (_csqrt(z) if mode == 1 else np.sqrt(z))
                if j == n_lay - 1:
                    E[j] = 0.0
                    break
                E[j] = E[j - 1] if same and thicknesses[j] == thicknesses[j - 1] \
                    else np.exp(-2.0 * G[j] * thicknesses[j])
                att += 2.0 * thicknesses[j] * G[j].real
                if att > 50.0:
                    E[j] = 0.0
                    depth = j + 1
                    break
            gamma = 0.0 + 0.0j                       # gamma_N = 0 (base half-space)
            for j in range(depth - 1, -1, -1):
                if j > 0 and resistivities[j] == resistivities[j - 1]:
                    gamma *= E[j]                    # psi = 0
                    continue
                G_above = (lam[m] + 0.0j) if j == 0 else G[j - 1]
                psi = (G_above - G[j]) / (G_above + G[j])
                gamma = (psi + gamma * E[j]) / (1.0 + psi * gamma * E[j])
            r_te[m] = gamma
        return r_te

    # ------------------------------------------------------------------
    # Circular loop: central + offset (Fourier DLF)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_circular_jit(times, thicknesses, resistivities,
                          tx_radius, extra_weights, mu0,
                          hankel_base, hankel_j1,
                          fourier_base, fourier_weights,
                          filter_weights, mode=0):
        """Circular-loop dB/dt via fused Numba loops (Fourier DLF)."""
        n_t = len(times)
        n_f = len(fourier_base)
        a = tx_radius
        lam = hankel_base / a
        n_lam = len(lam)
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            accum = 0.0
            for k in range(n_f):
                omega = fourier_base[k] / t
                r_te = _te_rte_jit(lam, omega, thicknesses,
                                   resistivities, mu0, mode)
                hz_c = 0.0 + 0.0j
                for m in range(n_lam):
                    hz_c += r_te[m] * lam[m] * hankel_j1[m] * extra_weights[m]
                hz_c *= 0.5
                hz_im = (hz_c * filter_weights[i, k]).imag
                accum += mu0 * hz_im * fourier_weights[k]
            dbdt[i] = accum / t
        return dbdt

    # ------------------------------------------------------------------
    # Square loop: VMD area integral (Fourier DLF)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_square_jit(times, thicknesses, resistivities,
                        dist_q, area_w, mu0,
                        hankel_base, hankel_j0,
                        fourier_base, fourier_weights,
                        filter_weights, altitude=0.0, mode=0):
        """Square-loop dB/dt via VMD area integral (Fourier DLF, Numba).

        altitude : total Tx+Rx elevation [m]; applies an exp(-lam*altitude)
        upward continuation factor per wavenumber (0.0 = on ground).
        """
        n_t = len(times)
        n_f = len(fourier_base)
        n_q = len(dist_q)
        n_lam = len(hankel_base)
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            accum_t = 0.0
            for k in range(n_f):
                omega = fourier_base[k] / t
                hz_c = 0.0 + 0.0j

                for q in range(n_q):
                    dist = dist_q[q]
                    lam = np.empty(n_lam)
                    for m in range(n_lam):
                        lam[m] = hankel_base[m] / dist

                    r_te = _te_rte_jit(lam, omega, thicknesses,
                                       resistivities, mu0, mode)

                    g_c = 0.0 + 0.0j
                    if altitude != 0.0:
                        for m in range(n_lam):
                            lm = lam[m]
                            g_c += (r_te[m] * (lm * lm) * hankel_j0[m]
                                    * np.exp(-lm * altitude))
                    else:
                        for m in range(n_lam):
                            lm = lam[m]
                            g_c += r_te[m] * (lm * lm) * hankel_j0[m]
                    g_c = g_c / dist / (4.0 * np.pi)
                    hz_c += area_w[q] * g_c

                hz_im = (hz_c * filter_weights[i, k]).imag
                accum_t += mu0 * hz_im * fourier_weights[k]

            dbdt[i] = accum_t / t
        return dbdt

    # ------------------------------------------------------------------
    # Circular loop: central + offset (Euler acceleration)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_circular_euler_jit(times, thicknesses, resistivities,
                                tx_radius, extra_weights, mu0,
                                hankel_base, hankel_j1,
                                euler_eta, euler_A,
                                filter_weights, mode=0):
        """Circular-loop dB/dt via Euler-accelerated Bromwich inversion."""
        n_t = len(times)
        n_euler = len(euler_eta)
        a = tx_radius
        n_lam = len(hankel_base)
        lam = np.empty(n_lam)
        for m in range(n_lam):
            lam[m] = hankel_base[m] / a
        half_A = euler_A / 2.0
        pi_val = np.pi
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            c = half_A / t
            h = pi_val / t

            d = 0.0
            for k in range(n_euler):
                s = c + k * h * 1j
                omega = s / 1j
                r_te = _te_rte_jit(lam, omega, thicknesses,
                                   resistivities, mu0, mode)
                hz = 0.0 + 0.0j
                for m in range(n_lam):
                    hz += r_te[m] * lam[m] * hankel_j1[m] * extra_weights[m]
                hz *= 0.5
                hz *= filter_weights[i, k]
                fval = (mu0 * hz).real
                sign = 1.0 if k % 2 == 0 else -1.0
                d += euler_eta[k] * sign * fval

            dbdt[i] = np.exp(half_A) / t * d
        return dbdt

    # ------------------------------------------------------------------
    # Square loop: VMD area integral (Euler acceleration)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_square_euler_jit(times, thicknesses, resistivities,
                              dist_q, area_w, mu0,
                              hankel_base, hankel_j0,
                              euler_eta, euler_A,
                              filter_weights, altitude=0.0, mode=0):
        """Square-loop dB/dt via Euler-accelerated Bromwich + VMD integral.

        altitude : total Tx+Rx elevation [m]; applies an exp(-lam*altitude)
        upward continuation factor per wavenumber (0.0 = on ground).
        """
        n_t = len(times)
        n_euler = len(euler_eta)
        n_q = len(dist_q)
        n_lam = len(hankel_base)
        half_A = euler_A / 2.0
        pi_val = np.pi
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            c = half_A / t
            h = pi_val / t

            d = 0.0
            for k in range(n_euler):
                s = c + k * h * 1j
                omega = s / 1j
                hz = 0.0 + 0.0j

                for q in range(n_q):
                    dist = dist_q[q]
                    lam = np.empty(n_lam)
                    for m in range(n_lam):
                        lam[m] = hankel_base[m] / dist

                    r_te = _te_rte_jit(lam, omega, thicknesses,
                                       resistivities, mu0, mode)
                    g = 0.0 + 0.0j
                    if altitude != 0.0:
                        for m in range(n_lam):
                            lm = lam[m]
                            g += (r_te[m] * (lm * lm) * hankel_j0[m]
                                  * np.exp(-lm * altitude))
                    else:
                        for m in range(n_lam):
                            lm = lam[m]
                            g += r_te[m] * (lm * lm) * hankel_j0[m]
                    g = g / dist / (4.0 * pi_val)
                    hz += area_w[q] * g

                hz *= filter_weights[i, k]
                fval = (mu0 * hz).real
                sign = 1.0 if k % 2 == 0 else -1.0
                d += euler_eta[k] * sign * fval

            dbdt[i] = np.exp(half_A) / t * d
        return dbdt
