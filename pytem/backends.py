"""
backends.py - CuPy/CUDA detection and GPU filter array transfer.

Provides:
  - HAS_CUDA flag
  - GPU_HANKEL, GPU_FOURIER registries (CuPy arrays on device)
"""

import os
import glob
import sys

from .transform_weights import (
    _HANKEL_BASE_201, _HANKEL_J0_201, _HANKEL_J1_201,
    _HANKEL_BASE_101, _HANKEL_J0_101, _HANKEL_J1_101,
    _FOURIER_BASE_81, _FOURIER_SIN_81, _FOURIER_COS_81,
    _FOURIER_BASE_101, _FOURIER_SIN_101, _FOURIER_COS_101,
)


def _register_cuda_dll_dirs():
    """Make CuPy find the CUDA runtime shipped as ``nvidia-*-cu12`` pip wheels.

    On Windows the wheels install their DLLs under ``site-packages/nvidia/*/bin``.
    CuPy's bundled loader does not search there, and nvrtc loads its builtins
    DLL via the process PATH, so we add those directories to both PATH and the
    DLL search path before CuPy is imported. No-op on non-Windows or when the
    wheels are absent.
    """
    if os.name != "nt":
        return
    bins = []
    for site_dir in {os.path.dirname(os.path.dirname(__file__)), *sys.path}:
        base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(base):
            bins.extend(glob.glob(os.path.join(base, "*", "bin")))
    bins = sorted(set(p for p in bins if os.path.isdir(p)))
    if not bins:
        return
    os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ.get("PATH", "")
    for p in bins:
        try:
            os.add_dll_directory(p)
        except (OSError, AttributeError):
            pass


_register_cuda_dll_dirs()

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False

GPU_HANKEL = {}
GPU_FOURIER = {}

if HAS_CUDA:
    import cupy as cp

    _GPU_DEVICE = cp.cuda.Device(0)
    _GPU_DEVICE.use()

    # 201-pt Hankel
    _d_HANKEL_BASE_201 = cp.asarray(_HANKEL_BASE_201)
    _d_HANKEL_J0_201   = cp.asarray(_HANKEL_J0_201, dtype=cp.complex128)
    _d_HANKEL_J1_201   = cp.asarray(_HANKEL_J1_201, dtype=cp.complex128)

    # 101-pt Hankel
    _d_HANKEL_BASE_101 = cp.asarray(_HANKEL_BASE_101)
    _d_HANKEL_J0_101   = cp.asarray(_HANKEL_J0_101, dtype=cp.complex128)
    _d_HANKEL_J1_101   = cp.asarray(_HANKEL_J1_101, dtype=cp.complex128)

    # 81-pt Fourier
    _d_FOURIER_BASE_81 = cp.asarray(_FOURIER_BASE_81)
    _d_FOURIER_SIN_81  = cp.asarray(_FOURIER_SIN_81)
    _d_FOURIER_COS_81  = cp.asarray(_FOURIER_COS_81)

    # 101-pt Fourier
    _d_FOURIER_BASE_101 = cp.asarray(_FOURIER_BASE_101)
    _d_FOURIER_SIN_101  = cp.asarray(_FOURIER_SIN_101)
    _d_FOURIER_COS_101  = cp.asarray(_FOURIER_COS_101)

    GPU_HANKEL = {
        'key_201': (_d_HANKEL_BASE_201, _d_HANKEL_J0_201, _d_HANKEL_J1_201),
        'key_101': (_d_HANKEL_BASE_101, _d_HANKEL_J0_101, _d_HANKEL_J1_101),
    }
    GPU_FOURIER = {
        'key_81':  (_d_FOURIER_BASE_81,  _d_FOURIER_SIN_81,  _d_FOURIER_COS_81),
        'key_101': (_d_FOURIER_BASE_101, _d_FOURIER_SIN_101, _d_FOURIER_COS_101),
    }
