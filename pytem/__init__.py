"""
pyTEM: 1-D layered-earth TEM modelling.

Supports circular and square loop geometries, central and offset receivers,
DLF and Euler transforms, with NumPy / Numba / CuPy backends.

================================================================================
Package map (table of contents)
================================================================================

The package is organised in layers.  Lower layers hold pure data and physics
with no project-internal dependencies; higher layers compose them into the
public forward, inversion, and plotting API.  Arrows below read "imports from".

Layer 0 - Data and constants
    transform_weights.py   MU0 and the digital-filter coefficient tables:
                           Key (2009) 201/101-pt Hankel J0/J1 filters, Key
                           81/101-pt Fourier sine/cosine filters, and the
                           Euler weights.  Exposes the HANKEL_FILTERS,
                           FOURIER_FILTERS, EULER_PARAMS registries.  No
                           internal dependencies; everything else builds on it.

Layer 1 - Core physics and backend detection
    recursion.py           te_reflection_coeff / te_reflection_coeff_grad:
                           the TE reflection coefficient of the
                           layer stack and its log-resistivity gradient (NumPy,
                           complex).  This is the physical heart of the model.
                               -> transform_weights
    backends.py            CuPy/CUDA detection (HAS_CUDA) and transfer of the
                           filter tables to device memory (GPU_HANKEL,
                           GPU_FOURIER).
                               -> transform_weights

Layer 2 - Compute kernels (one physics, three backends)
    kernels_numba.py       Numba JIT forward kernels (circular/square, DLF and
                           Euler).  Sets HAS_NUMBA.
    kernels_gpu.py         CuPy/CUDA forward kernels (mirror of kernels_numba).
                               -> transform_weights, backends
    kernels_jacobian.py    Adjoint upward recursion: forward+backward pass giving
                           d(r_TE)/d(ln rho_j) for all layers at once, in both
                           Numba and CuPy variants.  Backs the analytical
                           Jacobian.
                               -> transform_weights, backends

Layer 3 - Forward modelling
    forward.py             Public fwd_circle_* / fwd_square_* functions, the
                           analytical half-space references, and the geometry
                           builders + backend dispatchers (CUDA > Numba >
                           pure Python).  Also exports the filter-precompute
                           helpers reused by the inversion module.
                               -> transform_weights, backends, recursion,
                                  kernels_numba, kernels_gpu

Layer 4 - Inversion
    inversion.py           Analytical (getJ_ana) and finite-difference
                           (getJ_fd) Jacobians, regularisation helpers (getR,
                           getAlpha[s], getRMS), apparent-resistivity
                           conversion, and the Gauss-Newton invert() loop.
                               -> forward (+ its filter helpers),
                                  kernels_jacobian, kernels_numba, backends,
                                  transform_weights

Utilities and add-ons (composable, mostly standalone)
    waveform.py            Convolution of the step response with a piecewise-
                           linear transmitter waveform (setup_waveform,
                           convolve_waveform).      -> kernels_numba
    system_filter.py       Butterworth and WalkTEM cascade H(omega) transfer
                           functions fed in as system_filter=.   (standalone)
    ip_models.py           Complex-resistivity IP models (Pelton, Cole-Cole,
                           double-Pelton, MPA) and tem_forward_ip.
                               -> transform_weights, recursion
    euler.py               Standalone Euler inverse-Laplace transform, used to
                           verify the production Euler path.      (standalone)
    plotter.py             Matplotlib helpers: plot_sounding, plot_model,
                           plot_inversion.                        (standalone)
    data_io.py             Field-data import: read_tem_xyz / read_kenbec_xyz /
                           read_xyz parse TEM Data Manager and TEMImage-Beta
                           `.xyz` exports into TEMData / KenbecTEMData, with
                           shared dbdt/snr/to_pytem and line/UTM/distance
                           helpers.                               (standalone)
    survey.py              Survey: quick-look sounding, transect and station-
                           map plots built on data_io containers. LM/HM are
                           always plotted separately (different waveforms).
                               -> data_io

    __init__.py            Re-exports the public API listed in __all__.

--------------------------------------------------------------------------------
Dependency flow
--------------------------------------------------------------------------------

    transform_weights
        |-> backends
        |-> recursion
        |-> kernels_numba / kernels_gpu / kernels_jacobian
                |
                v
            forward  ----> inversion
                ^               ^
                |               |
            (waveform, system_filter, ip_models feed in here as options)

--------------------------------------------------------------------------------
Typical call chains
--------------------------------------------------------------------------------

Forward:
    fwd_circle_central()                              (forward.py)
        _resolve_filters()        -> transform_weights tables
        _filter_weights()         -> optional system_filter samples
        _build_circular_geometry()-> per-wavenumber weights
        _run_circular()           -> CUDA | Numba | pure-Python kernel
            te_reflection_coeff() -> upward recursion (recursion.py / kernels)
        _apply_signal_scaling()   -> current and step-off/on/impulse sign

Inversion:
    invert()                                          (inversion.py)
        loop:
            forward call (one of fwd_*)               (forward.py)
            getJ_ana()  -> adjoint kernels            (kernels_jacobian.py)
                        or getJ_fd() -> repeated forward calls
            _alpha_search() / _gn_solve() / _backtrack()  Gauss-Newton step
        optional: convolve_waveform(), system_filter=H, IP rho(omega)
"""

from .transform_weights import MU0, HANKEL_FILTERS, FOURIER_FILTERS, EULER_PARAMS
from .backends import HAS_CUDA
from .kernels_numba import HAS_NUMBA
from .recursion import te_reflection_coeff, te_reflection_coeff_grad

from .forward import (
    fwd_circle_central,
    fwd_circle_offset,
    fwd_square_central,
    fwd_square_offset,
    fwd_analytical_central,
    fwd_analytical_offset,
)

from .waveform import convolve_waveform, setup_waveform, setup_waveform_matrix
from .system_filter import butterworth_filter, cascade_filter
from .euler import euler_invert
from .inversion import (getJ_ana, getJ_fd, getR, dbdt_to_apprho, getRMS,
                        getAlpha, getAlphas, invert)
from .plotter import plot_sounding, plot_model, plot_inversion
from .data_io import TEMData, KenbecTEMData, read_tem_xyz, read_kenbec_xyz, read_xyz
from .survey import Survey

from .ip_models import (
    pelton_res_rho,
    cole_cole_rho,
    double_pelton_rho,
    get_m_taur_MPA,
    mpa_rho,
    tem_forward_ip,
)

__all__ = [
    # Constants & flags
    'MU0', 'HAS_CUDA', 'HAS_NUMBA',
    'HANKEL_FILTERS', 'FOURIER_FILTERS', 'EULER_PARAMS',
    # Core
    'te_reflection_coeff',
    'te_reflection_coeff_grad',
    # Forward models
    'fwd_circle_central',
    'fwd_circle_offset',
    'fwd_square_central',
    'fwd_square_offset',
    # Analytical
    'fwd_analytical_central',
    'fwd_analytical_offset',
    # Waveform & system filter
    'convolve_waveform',
    'setup_waveform',
    'setup_waveform_matrix',
    'butterworth_filter',
    'cascade_filter',
    # IP models
    'pelton_res_rho',
    'cole_cole_rho',
    'double_pelton_rho',
    'get_m_taur_MPA',
    'mpa_rho',
    'tem_forward_ip',
    # Euler (verification)
    'euler_invert',
    # Inversion
    'getJ_ana',
    'getJ_fd',
    'getR',
    'dbdt_to_apprho',
    'getRMS',
    'getAlpha',
    'getAlphas',
    'invert',
    # Plotting
    'plot_sounding',
    'plot_model',
    'plot_inversion',
    # Field-data import & survey plotting
    'TEMData',
    'KenbecTEMData',
    'read_tem_xyz',
    'read_kenbec_xyz',
    'read_xyz',
    'Survey',
]
