"""
plotting.py - Plotting utilities for TEM forward modelling and inversion.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sounding(times, *curves, ax=None, figsize=(6, 4), labels=None,
                 styles=None, title='Sounding', ylabel=r'$-dB_z/dt$ [V/m^2]'):
    """Plot one or more dB/dt sounding curves on a log-log axis.

    Parameters
    ----------
    times   : (n_t,) array - gate times [s].
    *curves : one or more (n_t,) arrays of -dB/dt values.
    ax      : existing Axes, or None to create a new figure.
    figsize : figure size when ax is None.
    labels  : list of str, optional.
    styles  : list of fmt strings (e.g. ['o-', 's--']), optional.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if labels is None:
        labels = [None] * len(curves)
    if styles is None:
        styles = ['o-'] + ['s--', '^:', 'v-.', 'D-'][:len(curves) - 1]

    for curve, lbl, sty in zip(curves, labels, styles):
        ax.loglog(times, np.abs(curve), sty, markersize=4, label=lbl)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if any(l is not None for l in labels):
        ax.legend()
    return ax


def plot_model(thicknesses, resistivities, ax=None, figsize=(4, 4),
              label='Model', color=None, depth_pad=10, xlim=None,
              title='Resistivity model', linestyle='-'):
    """Step-plot of resistivity vs depth.

    Parameters
    ----------
    thicknesses   : (N-1,) layer thicknesses [m].
    resistivities : (N,) layer resistivities [Ohm.m].
    ax            : existing Axes, or None to create a new figure.
    figsize       : figure size when ax is None.
    depth_pad     : extra depth below last interface [m].
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    thicknesses = np.asarray(thicknesses)
    resistivities = np.asarray(resistivities)
    depths = np.concatenate(([0], np.cumsum(thicknesses)))
    y = np.r_[depths, depths[-1] + depth_pad]
    x = np.r_[resistivities, resistivities[-1]]

    ax.step(x, y, where='pre', label=label, color=color, linestyle=linestyle)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Resistivity [Ohm.m]')
    ax.set_ylabel('Depth [m]')
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if label is not None:
        ax.legend()
    return ax


def plot_survey_models(thicknesses, resistivity_models, rms=None, doi=None,
                       station_names=None, line_indices=None, station_ids=None,
                       n_cols=5,
                       xlim=(1, 5e2), figsize_per_panel=(3, 3),
                       model_linestyle='--'):
    """Plot 1D resistivity models with RMS and optional DOI lines."""
    thicknesses = np.asarray(thicknesses, dtype=float)
    models = np.asarray(resistivity_models, dtype=float)
    if models.ndim != 2 or models.shape[1] != thicknesses.size + 1:
        raise ValueError('resistivity_models must have shape (n_stations, n_layers).')
    if n_cols < 1:
        raise ValueError('n_cols must be positive.')

    n_stations = models.shape[0]
    rms_values = None if rms is None else np.asarray(rms, dtype=float)
    if rms_values is not None and rms_values.shape != (n_stations,):
        raise ValueError('rms must have one value per station.')
    if station_names is None:
        station_names = [str(i) for i in range(n_stations)]
    if len(station_names) != n_stations:
        raise ValueError('station_names must have one name per station.')
    if station_ids is None:
        station_ids = np.arange(n_stations)
    if len(station_ids) != n_stations:
        raise ValueError('station_ids must have one value per station.')
    if line_indices is not None and len(line_indices) != n_stations:
        raise ValueError('line_indices must have one value per station.')

    doi_values = doi_capped = None
    if doi is not None:
        if isinstance(doi, dict):
            doi_values = np.asarray(doi.get('standard'), dtype=float)
            doi_capped = np.asarray(doi.get('standard_capped',
                                             np.zeros(n_stations, dtype=bool)), dtype=bool)
        else:
            doi_values = np.asarray(doi, dtype=float)
        if doi_values.shape != (n_stations,):
            raise ValueError('doi must have one value per station.')

    n_rows = max(1, int(np.ceil(n_stations / n_cols)))
    fig = plt.figure(figsize=(figsize_per_panel[0] * n_cols,
                              figsize_per_panel[1] * n_rows))
    grid = fig.add_gridspec(
        n_rows, n_cols,
    )
    axes = np.array([
        fig.add_subplot(grid[row, col])
        for row in range(n_rows) for col in range(n_cols)
    ])

    for ax, station in zip(axes, range(n_stations)):
        title = str(station_names[station])
        if line_indices is not None:
            title = f'Line {line_indices[station]} | {title}'
        if rms_values is not None:
            title += f'  RMS={rms_values[station]:.3f}'
        if doi_values is not None and np.isfinite(doi_values[station]):
            plot_model(thicknesses, models[station], ax=ax, color='grey',
                       label=None, title=title, xlim=xlim,
                       linestyle='--')
            depths = np.concatenate(([0.0], np.cumsum(thicknesses)))
            doi_depth = doi_values[station]
            layer = np.searchsorted(depths, doi_depth, side='right') - 1
            layer = min(max(layer, 0), models.shape[1] - 1)
            upper_x = np.r_[models[station, :layer + 1], models[station, layer]]
            upper_y = np.r_[depths[:layer + 1], doi_depth]
            ax.step(upper_x, upper_y, where='pre', color='black',
                    linestyle='-', label=None)
        else:
            plot_model(thicknesses, models[station], ax=ax, color='black',
                       label=None, title=title, xlim=xlim,
                       linestyle=model_linestyle)

    for ax in axes[n_stations:]:
        ax.axis('off')

    fig.tight_layout()
    return fig, axes


def plot_survey_responses(station_results, station_ids=None, n_cols=5,
                          figsize_per_panel=(3.2, 3), moment_colors=None):
    """Plot observed and modelled LM/HM responses for survey stations."""
    if not station_results:
        raise ValueError('station_results must not be empty.')
    if station_ids is None:
        station_ids = list(station_results)
    station_ids = list(station_ids)
    n_stations = len(station_ids)
    moment_colors = moment_colors or {'LM': 'tab:blue', 'HM': 'tab:red'}
    n_rows = max(1, int(np.ceil(n_stations / n_cols)))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols,
                 figsize_per_panel[1] * n_rows),
        sharex=True, squeeze=False)
    axes = axes.ravel()

    for ax, station in zip(axes, station_ids):
        result = station_results[station]
        for moment, color in moment_colors.items():
            if moment not in result['times']:
                continue
            ax.loglog(result['times'][moment], np.abs(result['obs'][moment]),
                      'o', ms=4, color=color, label=f'{moment} obs')
            ax.loglog(result['times'][moment], np.abs(result['pred'][moment]),
                      '-', lw=1.5, color=color, label=f'{moment} model')
        ax.grid(True, which='both', ls=':', alpha=0.4)

    for ax in axes[n_stations:]:
        ax.axis('off')
    axes[0].set_ylabel('|dB/dt| [V/m$^2$]')
    axes[0].legend(fontsize=6)
    fig.supxlabel('Time [s]')
    fig.tight_layout()
    return fig, axes


def plot_inversion(times, obs_data, mod_data, thicknesses,
                  best_rho, iter_rms_list, true_rho=None,
                  true_thicknesses=None, rho_hist=None,
                  xlim_rho=None, depth_pad=10, noise=None,
                  figsize=(12, 4), model_label='Inverted', true_label='True'):
    """Three-panel summary: sounding, RMS convergence, model.

    Parameters
    ----------
    times            : gate times [s].
    obs_data         : observed -dB/dt (positive).
    mod_data         : final modelled -dB/dt (positive).
    thicknesses      : inversion layer thicknesses.
    best_rho         : best-fit resistivities.
    iter_rms_list    : RMS per iteration.
    true_rho         : true resistivities (optional overlay).
    true_thicknesses : true layer thicknesses (required if true_rho given).
    rho_hist         : list of rho arrays per iteration (optional).
    xlim_rho         : (lo, hi) for resistivity axis.
    depth_pad        : extra depth below deepest interface.
    noise            : relative noise level (float) or absolute noise array.
    figsize          : overall figure size.
    model_label      : legend label for the inverted model (default 'Inverted').
    true_label       : legend label for the true_rho overlay (default 'True').
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # --- Sounding ---
    ax = axs[0]
    ax.loglog(times, np.abs(obs_data), 'o', label='Observed', markersize=4)
    ax.loglog(times, np.abs(mod_data), '-', label='Modelled', markersize=4)
    if noise is not None:
        if np.isscalar(noise):
            obs_noise = np.abs(obs_data) * noise
        else:
            obs_noise = np.asarray(noise)
        ax.fill_between(times,
                        np.abs(obs_data) - obs_noise,
                        np.abs(obs_data) + obs_noise,
                        alpha=0.2, color='C0')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(r'$-dB_z/dt$ [V/m^2]')
    ax.set_title('Sounding')
    ax.legend()

    # --- RMS convergence ---
    ax = axs[1]
    iters = range(1, len(iter_rms_list) + 1)
    ax.plot(iters, iter_rms_list, 'o-')
    ax.axhline(1.0, ls='--', color='grey', lw=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMS misfit')
    ax.set_title('Convergence')

    # --- Model ---
    ax = axs[2]
    thicknesses = np.asarray(thicknesses)
    depths = np.concatenate(([0], np.cumsum(thicknesses)))
    y_end = depths[-1] + depth_pad

    if rho_hist is not None:
        for rho_i in rho_hist:
            ax.step(np.r_[rho_i, rho_i[-1]], np.r_[depths, y_end],
                    where='pre', color='C0', alpha=0.15, lw=0.8)

    ax.step(np.r_[best_rho, best_rho[-1]], np.r_[depths, y_end],
            where='pre', label=model_label, color='C0', lw=2)

    if true_rho is not None:
        t_thick = np.asarray(true_thicknesses if true_thicknesses is not None
                             else thicknesses)
        t_depths = np.concatenate(([0], np.cumsum(t_thick[:-1])))
        ax.step(np.r_[true_rho, true_rho[-1]],
                np.r_[t_depths, y_end],
                where='pre', label=true_label, color='C3', lw=1.5, ls='--')

    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('Resistivity [Ohm.m]')
    ax.set_ylabel('Depth [m]')
    ax.set_title('Model')
    if xlim_rho is not None:
        ax.set_xlim(xlim_rho)
    ax.legend()

    fig.tight_layout()
    return fig, axs
