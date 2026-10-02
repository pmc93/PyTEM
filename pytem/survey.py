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
    def plot_soundings(self, moments=None, ax=None, figsize=(6, 5), show_mean=True,
                      station_mask=None):
        """Raw |dB/dt| decay curves for one or more moments.

        LM and HM (if both requested) are drawn as two separately-coloured
        curve families on the same axes -- never summed or concatenated.

        Parameters
        ----------
        moments      : list of 'LM'/'HM', or None for all moments in the file.
        ax           : existing Axes, or None to create a new figure.
        show_mean    : also draw the stacked mean per moment (bold line).
        station_mask : boolean array or integer index array selecting which
                       stations (rows) to plot, or None for all stations.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        for m in (moments or self.moments):
            t = self.tem.gate_times[m]["center"]
            dbdt = self.tem.dbdt(m)
            if station_mask is not None:
                dbdt = dbdt[station_mask]
            color = MOMENT_COLOR.get(m)
            for row in dbdt:
                ax.loglog(t, np.abs(row), color=color, lw=0.4, alpha=0.15)
            if show_mean:
                mean = np.nanmean(np.abs(dbdt), axis=0)
                ax.loglog(t, mean, color=color, lw=2.2, marker='o', ms=4,
                         label=f'{m} mean ({dbdt.shape[0]} soundings)')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(r'$|dB/dt|$ [V/m$^2$]')
        ax.grid(True, which='both', ls=':', alpha=0.5)
        if show_mean:
            ax.legend(fontsize=8)
        return ax

    # ------------------------------------------------------------------
    # Transects
    # ------------------------------------------------------------------
    def plot_transect(self, moment, gates=None, n_gates=3, axes=None, ax=None,
                      figsize=(7, 7),
                       x_axis='distance', bad_mask=None, legend=True,
                       line_numbers=None, title=None, line=None):
        """Signal vs. distance-along-line (or station index), one subplot per line.

        Parameters
        ----------
        moment  : 'LM' or 'HM' -- exactly one moment (see module docstring).
       gates   : list of int gate indices (0-based), or None to auto-pick
                 ``n_gates`` evenly-spaced gates.
       n_gates : number of auto-picked gates when ``gates`` is None.
        axes    : existing array of Axes (one per line), or None to create them.
        ax      : existing single Axes, useful when plotting one line.
        x_axis  : 'distance' (default, cumulative distance along the line
                 [m]) or 'index' (station order within the line, 0-based).
        bad_mask : (n_stations, n_gates) boolean array (True = bad/excluded
                 data), or None. Bad points are circled in black.
        legend  : whether to draw the per-gate-time legend on each subplot.
        line_numbers : sequence, optional
            Display labels for the lines. Defaults to the line identifiers
            detected from the survey data.
        title : str, optional
            Figure-level title placed above the transect subplots.
        line : optional
            Plot only this line identifier. Defaults to all lines.
        """
        m = moment.upper()
        lines = self.tem.lines()
        if line is not None:
            if str(line) not in lines:
                raise ValueError(f'Unknown line {line!r}; available lines: {lines}')
            lines = [str(line)]
        if line_numbers is None:
            line_numbers = lines
        elif len(line_numbers) != len(lines):
            raise ValueError('line_numbers must contain one label per line.')
        t = self.tem.gate_times[m]["center"]
        if gates is None:
            gates = np.arange(len(t)) if n_gates is None else np.linspace(
                0, len(t) - 1, n_gates).round().astype(int)
        dbdt = self.tem.dbdt(m)

        if ax is not None:
            if axes is not None:
                raise ValueError('Pass either ax or axes, not both.')
            if len(lines) != 1:
                raise ValueError('ax can only be used when plotting one line.')
            axes = [ax]
        elif axes is None:
            _, axes = plt.subplots(len(lines), 1, figsize=figsize, sharey=True)
        axes = np.atleast_1d(axes)

        for ax, line in zip(axes, lines):
            mask = self.tem.line_mask(line)
            x = np.arange(mask.sum()) if x_axis == 'index' else self.tem.distance_along_line(line)
            for g in gates:
                vals = np.abs(dbdt[mask, g])
                bad = np.zeros_like(vals, dtype=bool)
                if bad_mask is not None:
                    bad = bad_mask[mask, g]
                valid_line, = ax.semilogy(
                    x[~bad], vals[~bad], 'o-', ms=4, lw=1.4, alpha=0.65,
                    label=f't = {t[g] * 1e6:.0f} \u00b5s')
                if np.any(bad):
                    ax.semilogy(x[bad], vals[bad], 'o', ms=4, alpha=0.65,
                                color=valid_line.get_color(), zorder=4)
            ax.set_xlabel('Station index' if x_axis == 'index' else 'Distance along line [m]')
            ax.set_ylabel(f'{m} dB/dt [V/m$^2$]')
            ax.grid(True, which='both', ls=':', alpha=0.5)
            if legend:
                ax.legend(fontsize=8)
        if title is not None:
            axes[0].get_figure().suptitle(title)
        return axes

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    def export_xyz(self, path, resistivities, depths, doi_conservative,
                   doi_standard, residual, station_ids=None, line_num=None,
                   instrument='tTEM', epsg='epsg:25832', max_rms=None):
        """Export inversion models and survey metadata to Workbench XYZ."""
        data = self.tem.data if station_ids is None else self.tem.data.iloc[station_ids]
        x = np.asarray(data['E'], dtype=float)
        y = np.asarray(data['N'], dtype=float)
        elev = np.asarray(data['Elevation'], dtype=float)
        rhos = np.asarray(resistivities, dtype=float)
        depths = np.asarray(depths, dtype=float)
        n_soundings, n_layers = rhos.shape
        if depths.shape != rhos.shape:
            raise ValueError('depths must have the same shape as resistivities.')

        if line_num is None:
            line_num = np.asarray(data['Line'], dtype=int)
        else:
            line_num = np.asarray(line_num, dtype=int)
        columns = (['LINE_NO', 'UTMX', 'UTMY', 'ELEVATION']
                   + [f'RHO{i + 1}' for i in range(n_layers)]
                   + [f'RHO{i + 1}_STD' for i in range(n_layers)]
                   + [f'DEP_BOT{i + 1}' for i in range(n_layers)]
                   + [f'DEP_BOT{i + 1}_STD' for i in range(n_layers)]
                   + ['DOI_CONSERVATIVE', 'DOI_STANDARD', 'RESDATA'])
        output = np.full((n_soundings, len(columns)), np.nan)
        output[:, 0:4] = np.column_stack((line_num, x, y, elev))
        rho_start = 4
        depth_start = 4 + 2 * n_layers
        output[:, rho_start:rho_start + n_layers] = rhos
        output[:, depth_start:depth_start + n_layers] = depths
        output[:, -3] = doi_conservative
        output[:, -2] = doi_standard
        output[:, -1] = residual
        if max_rms is not None:
            output = output[residual <= max_rms, :]

        epsg_code = str(epsg).split(':')[-1]
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write('Aarhus Workbench XYZ export\nDATA TYPE\n')
            stream.write(f'{instrument}/DT\nCOORDINATE SYSTEM\n')
            stream.write(f'epsg:{epsg_code}\n/ ' + ' '.join(columns) + '\n')
            for row in output:
                stream.write(' '.join(f'{value:.6g}' for value in row) + '\n')
        return path

    def plot_map(self, ax=None, figsize=(7, 7), basemap=True, provider=None,
                 attribution=False):
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

        # Up to ten lines get their own colour and a legend; more are coloured in line order with a colour bar.
        few_lines, few_stations = len(lines) <= len(LINE_COLORS), len(e) <= 500
        colors = LINE_COLORS if few_lines else plt.cm.viridis(np.linspace(0, 1, len(lines)))
        for i, line in enumerate(lines):
            mask = self.tem.line_mask(line)
            ax.scatter(e[mask], n[mask], color=colors[i], s=50 if few_stations else 6, label=f'Line {line}', zorder=4,
                      edgecolors='k' if few_stations else 'none', linewidths=0.5)
        ax.set_aspect('equal')  # UTM meters: equal x/y scale so distances aren't distorted

        if basemap:
            try:
                import contextily as ctx
                source = provider if provider is not None else ctx.providers.OpenTopoMap
                ctx.add_basemap(ax, crs=f'EPSG:{epsg}', source=source, zoom='auto',
                                attribution=attribution)
            except Exception as exc:
                ax.set_facecolor('#e8e8e8')
                ax.grid(True, ls=':', alpha=0.5)
        else:
            ax.grid(True, ls=':', alpha=0.5)

        ax.set_xlabel(f'Easting [m] (EPSG:{epsg})')
        ax.set_ylabel('Northing [m]')
        if few_lines:
            ax.legend(title='Line', fontsize=8, loc='upper left', framealpha=0.8)
        else:
            bar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, len(lines) - 1), cmap='viridis'),
                                     ax=ax, label='Line', shrink=0.7)
            ticks = np.linspace(0, len(lines) - 1, 6).round().astype(int)
            bar.set_ticks(ticks, labels=[lines[k] for k in ticks])
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            ax.add_artist(ScaleBar(1, units='m', location='lower right', box_alpha=0.7))
        except ImportError:
            pass
        ax.ticklabel_format(style='plain', useOffset=False)
        return ax
