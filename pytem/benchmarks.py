"""Numerical benchmarks for pyTEM forward-model workflows."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .forward import fwd_circle_offset, fwd_square_offset
from .waveform import setup_shared_gate_matrices


def benchmark_step_grids(
    systems,
    thicknesses,
    models,
    geometry,
    tx_size,
    rx_x=0.0,
    rx_y=0.0,
    grid_points=(20, 70, 120, 170, 220, 270, 320, 370, 400),
    reference_points=1200,
    repeats=3,
    tolerance=1.0,
    system_filter=None,
    n_quad=5,
    transform='dlf',
    use_numba=True,
    use_cuda=False,
):
    """Benchmark shared step-time grids against a high-resolution reference."""
    if geometry not in ('circle_offset', 'square_offset'):
        raise ValueError("geometry must be 'circle_offset' or 'square_offset'.")
    if repeats < 1 or reference_points < 1:
        raise ValueError('repeats and reference_points must be positive.')

    thicknesses = np.asarray(thicknesses, dtype=float)
    models = {str(name): np.asarray(model, dtype=float)
              for name, model in models.items()}
    points = sorted(set(int(value) for value in grid_points))
    if not points or min(points) < 1:
        raise ValueError('grid_points must contain positive integers.')

    def predict(resistivities, step_times, gate_matrices):
        options = dict(current=1.0, signal=-1, use_numba=use_numba,
                       use_cuda=use_cuda, transform=transform,
                       system_filter=system_filter)
        if geometry == 'square_offset':
            response = -fwd_square_offset(
                thicknesses, resistivities, tx_size, rx_x, rx_y, step_times,
                n_quad=n_quad, **options)
        else:
            response = -fwd_circle_offset(
                thicknesses, resistivities, tx_size, rx_x, step_times, **options)
        return {moment: gate_matrices[moment] @ response
                for moment in systems}

    reference_times, reference_matrices = setup_shared_gate_matrices(
        systems, n_step=reference_points)
    check_times, check_matrices = setup_shared_gate_matrices(
        systems, n_step=2 * reference_points)
    references = {name: predict(model, reference_times, reference_matrices)
                  for name, model in models.items()}
    reference_errors = []
    for name in models:
        check = predict(models[name], check_times, check_matrices)
        for moment in systems:
            scale = np.maximum(np.abs(check[moment]),
                               max(np.max(np.abs(check[moment])) * 1e-10, 1e-300))
            reference_errors.append(
                np.max(100 * np.abs(references[name][moment] - check[moment]) / scale))

    rows = []
    for points_count in points:
        started = time.perf_counter()
        step_times, gate_matrices = setup_shared_gate_matrices(
            systems, n_step=points_count)
        setup_seconds = time.perf_counter() - started
        for name, model in models.items():
            predict(model, step_times, gate_matrices)
            durations = []
            prediction = None
            for _ in range(repeats):
                started = time.perf_counter()
                prediction = predict(model, step_times, gate_matrices)
                durations.append(time.perf_counter() - started)
            for moment in systems:
                reference = references[name][moment]
                scale = np.maximum(np.abs(reference),
                                   max(np.max(np.abs(reference)) * 1e-10, 1e-300))
                error = 100 * np.abs(prediction[moment] - reference) / scale
                rows.append({
                    'Points': points_count,
                    'Model': name,
                    'Moment': moment,
                    'RMS error [%]': np.sqrt(np.mean(error ** 2)),
                    'Max error [%]': error.max(),
                    'Prediction [ms]': 1000 * np.median(durations),
                    'Setup [s]': setup_seconds,
                    'Below tolerance': bool(error.max() <= tolerance),
                })

    result = pd.DataFrame(rows)
    result.attrs['reference_max_error_percent'] = max(reference_errors, default=0.0)
    result.attrs['reference_points'] = reference_points
    result.attrs['tolerance'] = tolerance
    return result
