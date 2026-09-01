"""
survey.py - Quick-look plots for TEM field surveys: soundings, transects
and station maps.

Wraps a parsed :class:`~pytem.data_io.TEMData` /
:class:`~pytem.data_io.KenbecTEMData` object (see :mod:`pytem.data_io`).

LM and HM are always plotted as separate curves and never summed or
concatenated into one "combined" decay curve: each moment has its own
transmitter waveform, so a naive combination would silently mix two
different system responses. A proper joint LM+HM fit instead builds one
waveform-matrix per moment (``pytem.setup_waveform_matrix``) and inverts for
a single shared model -- that is a modelling choice, not a plotting one, and
is out of scope for this module.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

MOMENT_COLOR = {"LM": "tab:blue", "HM": "tab:red"}
LINE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


class Survey:
    """Quick-look plots for a parsed TEM station/sounding dataset.

    Parameters
    ----------
    tem : pytem.data_io.TEMData or pytem.data_io.KenbecTEMData
        Typically the result of ``pytem.data_io.read_xyz(path)``.
    """

    def __init__(self, tem):
        self.tem = tem

    @property
    def moments(self) -> list:
        """Moments present in the file ('LM' and/or 'HM')."""
        return [m for m in ("LM", "HM") if m in self.tem.gate_times
               and "center" in self.tem.gate_times[m]]

    # ------------------------------------------------------------------
    # SNR
    # ------------------------------------------------------------------
    def plot_snr(self, moments=None, axes=None, figsize=None, threshold=3.0):
        """Per-gate SNR (scatter of dB/dt across stations) for one or more moments.

        Parameters
        ----------
        moments   : list of 'LM'/'HM', or None for all moments in the file.
        axes      : existing array of Axes (one per moment), or None to create them.
        threshold : SNR level marked with a horizontal reference line.
        """
        moments = moments or self.moments
        if axes is None:
            figsize = figsize or (5 * len(moments), 4)
            _, axes = plt.subplots(1, len(moments), figsize=figsize, sharey=False)
            axes = np.atleast_1d(axes)
        for ax, m in zip(axes, moments):
            t_snr, mean, sem, snr = self.tem.snr(m)
            ax.semilogx(t_snr, snr, 'o-', color=MOMENT_COLOR.get(m, 'steelblue'))
            ax.axhline(threshold, color='r', ls='--', lw=1, label=f'SNR = {threshold:g}')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('SNR [-]')
            ax.set_title(f'{m} per-gate SNR')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
        return axes

    # ------------------------------------------------------------------
    # Soundings
    # ------------------------------------------------------------------
    def plot_soundings(self, moments=None, ax=None, figsize=(6, 5), show_mean=True):
        """Raw |dB/dt| decay curves for one or more moments.

        LM and HM (if both requested) are drawn as two separately-coloured
        curve families on the same axes -- never summed or concatenated.

        Parameters
        ----------
        moments   : list of 'LM'/'HM', or None for all moments in the file.
        ax        : existing Axes, or None to create a new figure.
        show_mean : also draw the stacked mean per moment (bold line).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        for m in (moments or self.moments):
            t = self.tem.gate_times[m]["center"]
            dbdt = self.tem.dbdt(m)
            color = MOMENT_COLOR.get(m)
            for row in dbdt:
                ax.loglog(t, np.abs(row), color=color, lw=0.4, alpha=0.15)
            if show_mean:
                mean = np.nanmean(np.abs(dbdt), axis=0)
                ax.loglog(t, mean, color=color, lw=2.2, marker='o', ms=4,
                         label=f'{m} mean ({dbdt.shape[0]} soundings)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'|dB/dt|')
        ax.set_title('Soundings')
        ax.grid(True, which='both', ls=':', alpha=0.5)
        if show_mean:
            ax.legend(fontsize=8)
        return ax

    # ------------------------------------------------------------------
    # Transects
    # ------------------------------------------------------------------
    def plot_transects(self, moment, gates=None, n_gates=3, axes=None, figsize=(7, 7)):
        """Signal vs. distance-along-line, one subplot per transect line.

        Parameters
        ----------
        moment  : 'LM' or 'HM' -- exactly one moment (see module docstring).
        gates   : list of int gate indices (0-based), or None to auto-pick
                 ``n_gates`` evenly-spaced gates.
        n_gates : number of auto-picked gates when ``gates`` is None.
        axes    : existing array of Axes (one per line), or None to create them.
        """
        m = moment.upper()
        lines = self.tem.lines()
        t = self.tem.gate_times[m]["center"]
        if gates is None:
            gates = np.linspace(0, len(t) - 1, n_gates).round().astype(int)
        dbdt = self.tem.dbdt(m)

        if axes is None:
            _, axes = plt.subplots(len(lines), 1, figsize=figsize, sharey=True)
            axes = np.atleast_1d(axes)

        for ax, line, color in zip(axes, lines, LINE_COLORS):
            mask = self.tem.line_mask(line)
            dist = self.tem.distance_along_line(line)
            for g in gates:
                vals = np.abs(dbdt[mask, g])
                ax.semilogy(dist, vals, 'o-', ms=4, lw=1.4, alpha=0.85,
                           label=f't = {t[g] * 1e6:.0f} \u00b5s')
            ax.set_xlabel('Distance along line [m]')
            ax.set_title(f'Line {line}')
            ax.grid(True, which='both', ls=':', alpha=0.5)
            ax.legend(fontsize=8)
        axes[0].set_ylabel(f'|dB/dt| ({m})')
        return axes

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    def plot_map(self, ax=None, figsize=(7, 7), basemap=True, provider=None):
        """Station map coloured by line, with an optional basemap.

        Falls back to a plain scatter plot (no basemap) if ``contextily``
        is not installed, the tile server is unreachable, or the request is
        rejected (e.g. OpenStreetMap's own tile servers reject most
        scripted/headless requests under their usage policy, and CartoDB's
        free tiles now require an API key). The default provider here,
        OpenTopoMap, needs neither and has good rural/remote coverage.

        Parameters
        ----------
        provider : a contextily tile provider, or None for the default
                  (``contextily.providers.OpenTopoMap``).
        """
        e, n, epsg = self.tem.utm_coords()
        lines = self.tem.lines()
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        for i, line in enumerate(lines):
            mask = self.tem.line_mask(line)
            ax.scatter(e[mask], n[mask], c=LINE_COLORS[i % len(LINE_COLORS)], s=50,
                      label=f'Line {line}', zorder=4, edgecolors='k', linewidths=0.5)

        if basemap:
            try:
                import contextily as ctx
                source = provider if provider is not None else ctx.providers.OpenTopoMap
                ctx.add_basemap(ax, crs=f'EPSG:{epsg}', source=source, zoom='auto')
                ax.set_title('Survey station map')
            except Exception as exc:
                ax.set_title('Survey station map (basemap unavailable)')
                ax.set_facecolor('#e8e8e8')
                ax.grid(True, ls=':', alpha=0.5)
        else:
            ax.set_title('Survey station map')
            ax.grid(True, ls=':', alpha=0.5)

        ax.set_xlabel(f'Easting [m] (EPSG:{epsg})')
        ax.set_ylabel('Northing [m]')
        ax.legend(title='Line', fontsize=8, loc='upper left', framealpha=0.8)
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            ax.add_artist(ScaleBar(1, units='m', location='lower right', box_alpha=0.7))
        except ImportError:
            pass
        ax.ticklabel_format(style='plain', useOffset=False)
        return ax
