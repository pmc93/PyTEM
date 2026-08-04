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
    depths = np.concatenate(([0], np.cumsum(thicknesses[:-1])))
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
    ax.legend()
    return ax


def plot_inversion(times, obs_data, mod_data, thicknesses,
                  best_rho, iter_rms_list, true_rho=None,
                  true_thicknesses=None, rho_hist=None,
                  xlim_rho=None, depth_pad=10, noise=None,
                  figsize=(12, 4)):
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
            where='pre', label='Inverted', color='C0', lw=2)

    if true_rho is not None:
        t_thick = np.asarray(true_thicknesses if true_thicknesses is not None
                             else thicknesses)
        t_depths = np.concatenate(([0], np.cumsum(t_thick[:-1])))
        ax.step(np.r_[true_rho, true_rho[-1]],
                np.r_[t_depths, y_end],
                where='pre', label='True', color='C3', lw=1.5, ls='--')

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
