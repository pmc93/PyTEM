# PyTEM

1-D layered-earth Time-Domain Electromagnetic (TEM) modelling and inversion in Python.

---

## Overview

pyTEM computes the vertical dB/dt step-off response of a 1-D horizontally layered resistivity model beneath a grounded loop source. It supports four transmitter/receiver geometries, two transform methods, three compute backends, a full regularised inversion, induced polarisation models, instrument system filters, and transmitter waveform convolution.

---

## Package structure

```
pytem/
├── transform_weights.py  # DLF coefficients and Euler weights (static data)
├── recursion.py          # TE reflection coefficient — NumPy reference
├── backends.py           # CUDA/CuPy detection; GPU transform weight arrays
├── kernels_numba.py      # Numba JIT forward kernels
├── kernels_gpu.py        # CuPy GPU forward kernels
├── kernels_jacobian.py   # Numba + GPU analytical Jacobian kernels
├── euler.py              # Standalone Euler ILT reference
├── forward.py            # Public forward model API
├── system_filter.py      # Instrument frequency-domain filter
├── waveform.py           # Waveform convolution
├── inversion.py          # Jacobian + regularised Gauss-Newton inversion
├── ip_models.py          # Induced polarisation complex resistivity models
├── plotter.py            # Plotting utilities
└── __init__.py           # Public API
```

---

## Module descriptions

### `transform_weights.py`
Pure data module — no computation. Stores the pre-optimised Digital Linear Filter (DLF) coefficients from Key (2009, 2012) and the Euler–Stehfest acceleration weights:

| Registry key | Points | Use |
|---|---|---|
| `key_101` | 101 | Hankel J0/J1 (fast, default) |
| `key_201` | 201 | Hankel J0/J1 (more accurate) |
| `key_81` | 81 | Fourier sine/cosine (fast, default) |
| `key_101` | 101 | Fourier sine/cosine (more accurate) |

Euler–Stehfest weights are stored at orders 8, 11, 15, and 19 (Abate & Whitt 1995). Also stores `MU0 = 4π × 10⁻⁷`.

Exports: `MU0`, `HANKEL_FILTERS`, `FOURIER_FILTERS`, `EULER_PARAMS`

---

### `recursion.py`
Reference NumPy implementation of the Wait (1954) upward TE-mode recursion and its adjoint gradient. Used as a ground truth for testing other backends.

Exports: `te_reflection_coeff`, `te_reflection_coeff_grad`

---

### `backends.py`
Detects whether CuPy (CUDA) is available at import time. If CUDA is present, pre-transfers all transform weight arrays to device memory so forward calls pay no Host-to-Device transfer cost.

Exports: `HAS_CUDA`

---

### `kernels_numba.py`
Numba `@njit`-compiled scalar kernels for the TEM forward model. One kernel per geometry × transform combination:

- `_tem_circular_jit` — circular loop, Fourier DLF
- `_tem_circular_euler_jit` — circular loop, Euler ILT
- `_tem_square_jit` — square loop, Fourier DLF
- `_tem_square_euler_jit` — square loop, Euler ILT

All kernels accept a `filter_weights` array `(n_t, n_eval) complex128` for the system filter. The inner loop runs in prange over gate times for CPU parallelism.

Exports: `HAS_NUMBA`, and the JIT kernels when Numba is available.

---

### `kernels_gpu.py`
CuPy (CUDA) equivalents of the Numba kernels. The full `(n_t, n_f, K)` tensor is batched in a single CuPy operation to saturate GPU occupancy. Guarded by `HAS_CUDA`.

---

### `kernels_jacobian.py`
Numba JIT and CuPy implementations of the **analytical Jacobian** kernels. Each uses the adjoint Wait recursion: one forward pass stores intermediate values, one backward pass accumulates `∂r_TE / ∂(ln ρ_j)` for all layers simultaneously, at the cost of a single forward call.

Same set of geometry × transform combinations as the forward kernels. All accept `filter_weights` so the system filter is automatically included in the gradient (since `H(ω)` is independent of resistivity).

---

### `euler.py`
Standalone reference implementation of the Euler–Maclaurin inverse Laplace transform (`euler_invert`). Used for verification only; the production path uses precomputed weights from `transform_weights.py`.

Exports: `euler_invert`

---

### `forward.py`
The main public forward model. Dispatches to CUDA > Numba > NumPy automatically.

| Function | Geometry |
|---|---|
| `fwd_circle_central` | Circular loop, Rx at centre |
| `fwd_circle_offset` | Circular loop, Rx at radial offset |
| `fwd_square_central` | Square loop, Rx at centre |
| `fwd_square_offset` | Square loop, Rx at (x, y) offset |
| `fwd_analytical_central` | Magnetic dipole analytical approximation, central |
| `fwd_analytical_offset` | Magnetic dipole analytical approximation, offset |

All functions share the same keyword arguments:

```python
fwd_circle_central(
    thicknesses,        # (N-1,) layer thicknesses [m]
    resistivities,      # (N,)   resistivities [Ohm.m]
    tx_radius,          # float  equivalent circle radius [m]
    times,              # (n_t,) gate times [s]
    use_numba=True,
    use_cuda=True,
    system_filter=None, # callable H(omega) -> complex, or None
    transform='dlf',    # 'dlf' or 'euler'
    hankel_filter='key_101',
    fourier_filter='key_81',
    euler_order=11,
)
```

Also exports `_precompute_filter_dlf` and `_precompute_filter_euler` (used internally by `inversion.py`).

---

### `system_filter.py`
Instrument frequency-domain transfer functions. A filter `H(omega)` is a callable that takes an array of angular frequencies and returns complex weights. It is applied inside the forward transform before taking the imaginary/real part.

```python
H = butterworth_filter(f_low=None, f_high=3e4, order=1)
H = cascade_filter(filtfreq=3e4)   # two cascaded 1st-order Butterworth LP
```

Exports: `butterworth_filter`, `cascade_filter`

---

### `waveform.py`
Convolves a pre-computed step-off response with a piecewise-linear transmitter waveform using Gauss-Legendre quadrature per waveform segment:

$$G(t) = -\int \frac{dI}{d\tau}\, S(t - \tau)\, d\tau$$

```python
result = convolve_waveform(
    step_times,       # dense time grid [s]
    step_response,    # step-off response on that grid
    waveform_times,   # waveform break points [s]
    waveform_currents,# current at each break point [A]
    gate_times,       # output gate centre times [s]
)
```

Exports: `convolve_waveform`

---

### `inversion.py`
All inversion machinery, built around a regularised Gauss-Newton loop that minimises:

$$\phi(\mathbf{m}) = \|\mathbf{W}(\ln \mathbf{d}_\text{obs} - \ln \mathbf{d}_\text{pred}(\mathbf{m}))\|^2 + \alpha\, \mathbf{m}^T \mathbf{R}\, \mathbf{m}$$

**Public utilities:**

| Function | Purpose |
|---|---|
| `getJ_ana` | Analytical Jacobian via adjoint Wait recursion |
| `getJ_fd` | Finite-difference Jacobian (N+1 forward calls) |
| `getR` | First-order roughness matrix with damping |
| `getRMS` | Noise-normalised RMS misfit |
| `getAlpha` | Single log-spaced regularisation parameter |
| `getAlphas` | Depth-weighted regularisation vector |
| `dbdt_to_apprho` | Convert dB/dt to apparent resistivity |
| `invert` | Full regularised inversion loop |

**Private helpers:** `_gn_solve`, `_backtrack`, `_alpha_search`

**`invert()` key options:**

```python
result = pytem.invert(
    obs_data, thicknesses, log_resistivities, tx_radius, times,
    analytical_j=True,      # use getJ_ana (faster) or getJ_fd
    system_filter=H,        # frequency-domain instrument filter
    waveform_times=wf_t,    # transmitter waveform
    waveform_currents=wf_I,
    noise_std=0.02,
    maxit=20,
    use_numba=True,
)
# result keys: 'resistivities', 'thicknesses', 'rms_history',
#              'model_history', 'sensitivity', 'obs_data', 'times'
```

When both `analytical_j=True` and a waveform are provided, the waveform Jacobian is formed analytically using the chain rule: `∂G_i/∂(ln ρ_j) = conv(∂F/∂(ln ρ_j), w)_i`, avoiding N+1 full waveform convolution calls.

---

### `ip_models.py`
Complex resistivity models for induced polarisation (IP). Each returns `rho(omega)` (complex) at a given angular frequency, suitable for passing into `te_reflection_coeff`.

| Function | Model |
|---|---|
| `pelton_res_rho` | Pelton et al. (1978) |
| `cole_cole_rho` | Cole & Cole (1941) |
| `double_pelton_rho` | Double Pelton (two relaxation terms) |
| `mpa_rho` | Maximum Phase Angle (Fiandaca et al. 2018) |
| `get_m_taur_MPA` | Iterative MPA → Cole-Cole conversion |
| `tem_forward_ip` | Full TEM forward with per-layer IP |

---

### `plotter.py`
Standalone plotting functions (no class required):

```python
ax  = plot_sounding(times, obs, mod, labels=['Observed', 'Modelled'])
ax  = plot_model(thicknesses, resistivities)
fig, axs = plot_inversion(times, obs_data, mod_data, thicknesses,
                          best_rho, rms_history)
```

All functions accept an optional `ax` argument to plot into an existing axes.

---

## Data flow

```
resistivities + thicknesses
        │
        ▼
   recursion.py          ← Wait recursion (TE reflection coeff)
        │
        ▼
   forward.py            ← Hankel + Fourier/Euler DLF transforms
        │                   system_filter applied in frequency domain
        │                   dispatch: CUDA > Numba > NumPy
        ▼
   step-off dB/dt
        │
        ├──── waveform.py ─── convolve with transmitter waveform
        │
        ▼
   inversion.py
        ├── getJ_ana / getJ_fd  ← Jacobian
        ├── _alpha_search       ← regularisation ladder
        ├── _gn_solve           ← normal equations
        └── _backtrack          ← bounds enforcement
```

---

## Compute backends

| Backend | When active | Strength |
|---|---|---|
| NumPy | always | Portable, easy to debug |
| Numba JIT | `use_numba=True` and Numba installed | Fast scalar loops, CPU SIMD, prange parallelism |
| CuPy (CUDA) | `use_cuda=True` and CUDA available | Fully batched GPU; fastest for large problems |

Priority: CUDA > Numba > NumPy. Set `use_numba=False, use_cuda=False` to force NumPy.

---

## References

- Wait, J. R. (1954). Mutual coupling of loops lying on the ground. *Geophysics*, 19, 290–296.
- Key, K. (2009). 1D inversion of multicomponent, multifrequency marine CSEM data. *Geophysics*, 74(2), F9–F20.
- Key, K. (2012). Is the fast Hankel transform faster than quadrature? *Geophysics*, 77(3), F21–F30.
- Abate, J., & Whitt, W. (1995). Numerical inversion of Laplace transforms of probability distributions. *ORSA Journal on Computing*, 7(1), 36–43.
- Pelton, W. H., et al. (1978). Mineral discrimination and removal of inductive coupling with multifrequency IP. *Geophysics*, 43(3), 588–609.
- Fiandaca, G., et al. (2018). Re-parameterisations of the Cole–Cole model. *Geophysical Journal International*, 214(2), 1160–1173.
