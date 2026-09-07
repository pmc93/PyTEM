"""
data_io.py - Field-data importers for TEM survey/profiler `.xyz` exports.

Two exporter formats are supported, both returning a container with the same
interface (``dbdt``, ``dbdt_std``, ``snr``, ``to_pytem``, plus the position
helpers ``lines``, ``line_mask``, ``utm_coords``, ``distance_along_line``),
so :mod:`pytem.survey` can plot either without caring which instrument
produced the file:

  * TEM Data Manager (TEM2Go / tTEM) exports -> ``read_tem_xyz()``  -> TEMData
  * TEMImage-Beta station-stacked exports     -> ``read_kenbec_xyz()`` -> KenbecTEMData

``read_xyz(path)`` auto-detects the format from the file header and
dispatches to the right reader.

LM (low moment) and HM (high moment) each use a different transmitter
waveform, so they are always kept as separate arrays/columns here. Combining
them into one "stacked" decay curve would silently mix two different system
responses; a proper joint LM+HM fit instead assembles one waveform-matrix per
moment (see ``pytem.setup_waveform_matrix``) and solves for a shared model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _floats(text: str) -> np.ndarray:
    """Parse a whitespace-separated list of floats."""
    return np.fromstring(text.strip(), sep=" ")


class _PositionMixin:
    """Line grouping, UTM projection and along-line distance for station data.

    Shared by :class:`TEMData` and :class:`KenbecTEMData`. Requires a
    ``data`` DataFrame attribute; ``Line`` is optional (falls back to a
    single line ``"1"``), ``Longitude``/``Latitude`` are required the first
    time :meth:`utm_coords` is called unless ``E``/``N`` columns already
    exist.
    """

    def lines(self) -> list:
        """Sorted list of transect/line identifiers.

        Stations without a ``Line`` column are treated as a single line.
        """
        if "Line" in self.data.columns:
            return sorted(self.data["Line"].astype(str).str.strip().unique())
        return ["1"]

    def line_mask(self, line) -> np.ndarray:
        """Boolean mask selecting the stations belonging to ``line``."""
        if "Line" in self.data.columns:
            return (self.data["Line"].astype(str).str.strip() == str(line)).to_numpy()
        return np.ones(len(self.data), dtype=bool)

    def utm_coords(self):
        """Return ``(E, N, epsg)`` UTM coordinates for every station.

        Projects from ``Longitude``/``Latitude`` (WGS84) on first use via
        ``pyproj``, caching the result in ``E``/``N`` columns and
        ``meta["utm_epsg"]``. Requires ``pyproj``.
        """
        if "E" in self.data.columns and "N" in self.data.columns:
            return (self.data["E"].to_numpy(float), self.data["N"].to_numpy(float),
                    self.meta.get("utm_epsg"))
        if "Longitude" not in self.data.columns or "Latitude" not in self.data.columns:
            raise ValueError(
                "No E/N or Longitude/Latitude columns available for UTM projection")

        from pyproj import Proj

        lon = self.data["Longitude"].to_numpy(float)
        lat = self.data["Latitude"].to_numpy(float)
        lon_c, lat_c = np.nanmean(lon), np.nanmean(lat)
        zone = int((lon_c + 180) / 6) + 1
        hem = "north" if lat_c >= 0 else "south"
        utm = Proj(proj="utm", zone=zone, ellps="WGS84", south=(hem == "south"))
        e, n = utm(lon, lat)
        epsg = (32600 if hem == "north" else 32700) + zone

        self.data["E"], self.data["N"] = e, n
        self.meta["utm_epsg"] = epsg
        return np.asarray(e, dtype=float), np.asarray(n, dtype=float), epsg

    def distance_along_line(self, line=None) -> np.ndarray:
        """Cumulative distance [m] along one line, in station order."""
        e, n, _ = self.utm_coords()
        mask = self.line_mask(line) if line is not None else np.ones(len(self.data), dtype=bool)
        x, y = e[mask], n[mask]
        return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


@dataclass
class TEMData(_PositionMixin):
    """Container for a parsed TEM Data Manager `.xyz` file.

    Usage
    -----
        tem = read_tem_xyz("2026_0701_165727_ChA.xyz")
        tem.waveforms["LM"]["time"]        # -> np.ndarray of ramp break-point times
        tem.waveforms["LM"]["amplitude"]   # -> np.ndarray of normalised current
        tem.gate_times["HM"]["center"]     # -> np.ndarray of HM gate centre times [s]
        tem.data                           # -> pandas.DataFrame with every record

        lm, hm = tem.low_moment(), tem.high_moment()   # split by moment
    """

    meta: dict = field(default_factory=dict)
    waveforms: dict = field(default_factory=dict)   # {"LM": {"time":.., "amplitude":..}, "HM": {...}}
    gate_times: dict = field(default_factory=dict)   # {"LM": {"open":.., "center":.., "close":..}, "HM": {...}}
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def low_moment(self) -> pd.DataFrame:
        """Records acquired with the low moment (Moment == 0)."""
        return self.data[self.data["Moment"] == 0].reset_index(drop=True)

    def high_moment(self) -> pd.DataFrame:
        """Records acquired with the high moment (Moment == 1)."""
        return self.data[self.data["Moment"] == 1].reset_index(drop=True)

    def dbdt(self, moment: str) -> np.ndarray:
        """Stacked dB/dt matrix (n_records, n_gates) for 'LM' or 'HM'."""
        df = self.low_moment() if moment.upper() == "LM" else self.high_moment()
        n = len(self.gate_times[moment.upper()]["center"])
        cols = [f"dbdtDat{i:03d}" for i in range(1, n + 1)]
        return df[cols].to_numpy(dtype=float)

    def dbdt_std(self, moment: str) -> np.ndarray:
        """Stacked dB/dt uncertainty matrix (n_records, n_gates) for 'LM' or 'HM'.

        Values are fractions of the corresponding dB/dt reading (dbdtStdFUnit).
        """
        df = self.low_moment() if moment.upper() == "LM" else self.high_moment()
        n = len(self.gate_times[moment.upper()]["center"])
        cols = [f"dbdtStd{i:03d}" for i in range(1, n + 1)]
        return df[cols].to_numpy(dtype=float)

    def _meta_floats(self, key: str) -> list:
        """Parse a whitespace-separated numeric metadata value into a list."""
        return [float(t) for t in self.meta.get(key, "").split()]

    def snr(self, moment: str):
        """
        Per-gate signal-to-noise ratio of the stacked sounding.

        Returns
        -------
        (times, mean, sem, snr) : tuple of np.ndarray
            Gate centre times, mean dB/dt across records, standard error of the
            mean (from the scatter across records), and |mean| / sem.
        """
        m = moment.upper()
        t = self.gate_times[m]["center"]
        dbdt = self.dbdt(m)
        n_eff = np.sum(np.isfinite(dbdt), axis=0)
        mean = np.nanmean(dbdt, axis=0)
        sem = np.nanstd(dbdt, axis=0) / np.sqrt(np.maximum(n_eff, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.abs(mean) / sem
        return t, mean, sem, snr

    def to_pytem(self, moment: str, record=None, tx_turns=None,
                signed=False, min_noise=0.03):
        """
        Convert one moment ('LM' or 'HM') into keyword arguments for pyTEM.

        The returned dict plugs directly into ``pytem.invert`` and the
        ``pytem.fwd_square_*`` forward functions.  Conventions handled here:

          * gate centre times -> ``times`` [s]
          * dB/dt data are already normalised by Rx-coil area, so their unit is
            [V/m^2] == [T/s]; usable as ``obs_data`` with ``rx_area = 1``.
          * the waveform amplitude column is normalised (0..1); it is scaled to
            absolute ampere-turns (peak current * Tx turns) so that
            ``waveform_currents`` carries the full transmitter moment.
          * ``dbdtStd`` is a *fractional* uncertainty -> ``noise_std``.
          * loop size / Rx offset are read from the header to pick the geometry.

        Parameters
        ----------
        moment    : str    'LM' (low) or 'HM' (high).
        record    : int or None
            Index of a single sounding within the moment.  ``None`` (default)
            stacks all records of that moment (mean of dB/dt).
        tx_turns  : int or None
            Number of Tx turns.  ``None`` reads ``TxLoop_NTurns`` from the header.
        signed    : bool   Keep the measured sign (default takes abs value, as
                           pyTEM expects positive dB/dt).
        min_noise : float  Floor applied to the fractional noise (default 0.03).

        Returns
        -------
        dict
            Keys: ``times``, ``obs_data``, ``noise_std``, ``waveform_times``,
            ``waveform_currents``, ``geometry``, ``tx_size``, ``rx_x``,
            ``rx_y``, plus context keys ``gate_open``, ``gate_close``,
            ``tx_turns``, ``peak_current``, ``n_records``.
        """
        m = moment.upper()
        if m not in ("LM", "HM"):
            raise ValueError("moment must be 'LM' or 'HM'")

        gt = self.gate_times[m]
        times = gt["center"]

        df = self.low_moment() if m == "LM" else self.high_moment()
        dbdt = self.dbdt(m)
        std = self.dbdt_std(m)

        if record is not None:
            obs = dbdt[record]
            frac = std[record]
            peak_current = float(df["TxCurrent"].iloc[record])
            n_records = 1
        else:
            obs = np.nanmean(dbdt, axis=0)
            # Uncertainty of the stacked mean estimated from the empirical
            # scatter across records: standard error of the mean divided by the
            # mean magnitude gives a fractional noise. This reflects the true
            # data quality far better than the stored per-record fractions.
            n_eff = np.sum(np.isfinite(dbdt), axis=0)
            sem = np.nanstd(dbdt, axis=0) / np.sqrt(np.maximum(n_eff, 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = sem / np.abs(obs)
            peak_current = float(np.nanmedian(df["TxCurrent"]))
            n_records = int(df.shape[0])

        if not signed:
            obs = np.abs(obs)
        noise_std = np.clip(np.nan_to_num(frac, nan=min_noise), min_noise, None)

        # ---- Transmitter waveform in absolute ampere-turns ----
        if tx_turns is None:
            tx_turns = int(float(self.meta.get("TxLoop_NTurns", 1) or 1))
        wf_t = self.waveforms[m]["time"]
        wf_I = self.waveforms[m]["amplitude"] * peak_current * tx_turns

        # ---- Geometry from the header ----
        tx_len = self._meta_floats("TxLoop_XYLength")
        tx_side = tx_len[0] if tx_len else None
        rx_pos = self._meta_floats("RxCoil_XYZPos")
        rx_x = rx_pos[0] if rx_pos else 0.0
        rx_y = rx_pos[1] if len(rx_pos) > 1 else 0.0
        offset = abs(rx_x) > 1e-6 or abs(rx_y) > 1e-6
        geometry = "square_offset" if offset else "square_central"

        return {
            "times": times,
            "obs_data": obs,
            "noise_std": noise_std,
            "waveform_times": wf_t,
            "waveform_currents": wf_I,
            "geometry": geometry,
            "tx_size": tx_side,
            "rx_x": rx_x,
            "rx_y": rx_y,
            # --- context (not invert kwargs) ---
            "gate_open": gt.get("open"),
            "gate_close": gt.get("close"),
            "tx_turns": tx_turns,
            "peak_current": peak_current,
            "n_records": n_records,
        }


def read_tem_xyz(path: str) -> TEMData:
    """
    Read a TEM Data Manager `.xyz` file.

    Parameters
    ----------
    path : str
        Path to the `.xyz` file.

    Returns
    -------
    TEMData
        Parsed metadata, waveforms, gate times and the sounding DataFrame.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    tem = TEMData()
    tem.waveforms = {"LM": {}, "HM": {}}
    tem.gate_times = {"LM": {}, "HM": {}}

    header_idx = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        # Section markers like "[RxTxSpecs]" carry no key/value.
        if line.startswith("[") and "]" in line:
            continue

        if "=" in line:
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()

            if key.endswith("Waveform_Time"):
                tem.waveforms[key[:2]]["time"] = _floats(val)
            elif key.endswith("Waveform_Amplitude"):
                tem.waveforms[key[:2]]["amplitude"] = _floats(val)
            elif key.endswith("OpenTime"):
                tem.gate_times[key[:2]]["open"] = _floats(val)
            elif key.endswith("CenterTime"):
                tem.gate_times[key[:2]]["center"] = _floats(val)
            elif key.endswith("CloseTime"):
                tem.gate_times[key[:2]]["close"] = _floats(val)
            else:
                tem.meta[key] = val
            continue

        # First non key/value, non-section line beginning with "Date" is the
        # column header of the data table.
        if line.startswith("Date"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"No data table header found in {path!r}")

    columns = lines[header_idx].split()
    tem.data = pd.read_csv(
        path,
        skiprows=header_idx + 1,
        sep=r"\s+",
        names=columns,
        na_values=["nan", "NaN"],
        engine="python",
    )

    return tem


@dataclass
class KenbecTEMData(_PositionMixin):
    """Container for a parsed TEMImage_Beta `.xyz` station file.

    Usage
    -----
        tem = read_kenbec_xyz(r"C:\\...\\kenbec_project_StationData.xyz")
        tem.gate_times["HM"]["center"]        # -> np.ndarray [s]
        tem.waveforms["LM"]["time"]           # -> np.ndarray [s]
        tem.waveforms["LM"]["amplitude"]      # -> 0-1 normalised
        tem.data                              # -> pandas DataFrame, one row per station

        kw = tem.to_pytem("HM")              # ready for pytem.invert / fwd_square_offset
    """

    meta: dict = field(default_factory=dict)
    waveforms: dict = field(default_factory=dict)   # {"LM": {"time":.., "amplitude":..}, "HM": {...}}
    gate_times: dict = field(default_factory=dict)   # {"LM": {"open":.., "center":.., "close":..}, "HM": {..}}
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ---------- convenience accessors ----------

    def dbdt(self, moment: str) -> np.ndarray:
        """Station x gate dB/dt matrix [V/(A*m^4)] for 'LM' or 'HM'.

        Dummy values (99999) are replaced with NaN.
        """
        m = moment.upper()
        n = len(self.gate_times[m]["center"])
        prefix = "LMgate" if m == "LM" else "HMgate"
        cols = [f"{prefix}{i:03d}" for i in range(1, n + 1)]
        arr = self.data[cols].to_numpy(dtype=float)
        arr[arr == 99999.0] = np.nan
        return arr

    def dbdt_std(self, moment: str) -> np.ndarray:
        """Station x gate fractional uncertainty for 'LM' or 'HM'.

        Dummy values (99999) are replaced with NaN.
        """
        m = moment.upper()
        n = len(self.gate_times[m]["center"])
        prefix = "LMstd" if m == "LM" else "HMstd"
        cols = [f"{prefix}{i:03d}" for i in range(1, n + 1)]
        arr = self.data[cols].to_numpy(dtype=float)
        arr[arr == 99999.0] = np.nan
        return arr

    def snr(self, moment: str):
        """Per-gate SNR from the scatter across stations.

        Returns (times, mean, sem, snr).
        """
        t = self.gate_times[moment.upper()]["center"]
        d = self.dbdt(moment)
        n_eff = np.sum(np.isfinite(d), axis=0)
        mean = np.nanmean(d, axis=0)
        sem = np.nanstd(d, axis=0) / np.sqrt(np.maximum(n_eff, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.abs(mean) / sem
        return t, mean, sem, snr

    def to_pytem(self, moment: str, station=None, min_noise=0.03,
                peak_current=None):
        """
        Convert one moment ('LM' or 'HM') into kwargs for pyTEM.

        Data unit conversion
        --------------------
        The file stores dB/dt in V/(A*m^4), i.e. the response normalised by
        Tx moment [A*m^2] and Rx area [m^2].  pyTEM's fwd_square_offset with
        ``current=1, rx_area=1`` returns dBz/dt in [T/s per A per m^2 Rx area].

        To match the data units, the observed data (in V/(A*m^4)) is scaled
        up by A_tx * N_tx to give V/m^2 per ampere, and the waveform carries
        just the current in Amperes.  Both sides then share the same 1/I_peak
        factor which cancels in the log-space misfit.

        unit derivation::

            data [V/(A*m^4)] = dBz/dt [V/m^2] / (I * A_tx * N_tx)
            obs_scaled = data * A_tx * N_tx = dBz/dt [V/m^2] / I_peak
            wf_I = amplitude * I_peak  [A]
            -> model [V/m^2] / I_peak  matches  obs_scaled

        Parameters
        ----------
        moment     : 'LM' or 'HM'
        station    : int index or None (stack mean over all stations)
        min_noise  : fractional noise floor (default 0.03)
        peak_current : override nominal current if known [A]

        Returns
        -------
        dict  with keys: times, obs_data, noise_std, waveform_times,
              waveform_currents, geometry, tx_size, rx_x, rx_y,
              plus context keys: gate_open, gate_close, tx_turns,
              tx_area, n_stations.
        """
        m = moment.upper()
        gt = self.gate_times[m]
        times = gt["center"]

        dbdt = self.dbdt(m)
        frac = self.dbdt_std(m)

        if station is not None:
            obs = dbdt[station]
            noise_frac = frac[station]
            n_stations = 1
        else:
            obs = np.nanmean(dbdt, axis=0)
            n_eff = np.sum(np.isfinite(dbdt), axis=0)
            sem = np.nanstd(dbdt, axis=0) / np.sqrt(np.maximum(n_eff, 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                noise_frac = sem / np.abs(obs)
            n_stations = int(self.data.shape[0])

        noise_std = np.clip(np.nan_to_num(noise_frac, nan=min_noise), min_noise, None)

        # ---- Geometry -------------------------------------------------------
        tx_x = float(self.meta.get("LoopX", 3.0))
        tx_y = float(self.meta.get("LoopY", 3.0))
        tx_side = float(tx_x)                           # square loop side [m]
        tx_area = float(self.meta.get("LoopArea", tx_x * tx_y))
        tx_turns = int(float(self.meta.get("LoopTurns", 1)))
        rx_x = float(self.meta.get("RXcoil_X_Position", -13.0))
        rx_y = 0.0
        geometry = "square_offset" if abs(rx_x) > 1e-6 else "square_central"

        if peak_current is None:
            peak_current = 1.0

        # ---- Unit conversion ------------------------------------------------
        # Data: V/(A*m^4) = dBz/dt [V/m^2] / (I * A_tx * N_tx)
        # obs = data * I_peak * A_tx * N_tx = dBdt_true [V/m^2]
        obs = np.abs(obs) * peak_current * tx_area * tx_turns  # [V/m^2]

        # Waveform: amplitude x (I_peak x N_tx) = total A-turns [A-turns]
        # pyTEM models a single-turn loop; scaling wf_I by N_tx accounts for
        # the actual number of transmitter turns.
        wf_t = self.waveforms[m]["time"]
        wf_I = self.waveforms[m]["amplitude"] * peak_current * tx_turns   # [A-turns]

        return {
            "times": times,
            "obs_data": obs,
            "noise_std": noise_std,
            "waveform_times": wf_t,
            "waveform_currents": wf_I,
            "geometry": geometry,
            "tx_size": tx_side,
            "rx_x": rx_x,
            "rx_y": rx_y,
            # context
            "gate_open": gt.get("open"),
            "gate_close": gt.get("close"),
            "tx_turns": tx_turns,
            "tx_area": tx_area,
            "n_stations": n_stations,
        }


def read_kenbec_xyz(path: str) -> KenbecTEMData:
    """
    Read a TEMImage_Beta station-stacked `.xyz` file.

    Parameters
    ----------
    path : str

    Returns
    -------
    KenbecTEMData
    """
    tem = KenbecTEMData()
    tem.waveforms = {"LM": {}, "HM": {}}
    tem.gate_times = {"LM": {}, "HM": {}}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    header_idx = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        if line.startswith("/"):
            content = line[1:].strip()
            if not content:
                continue

            # ---- key: value pairs ----------------------------------------
            if ":" in content:
                key, val = content.split(":", 1)
                key, val = key.strip(), val.strip()

                k_norm = key.replace(" ", "_").replace("(", "").replace(")", "")

                # Waveforms
                if "LM_WaveformTime" in k_norm:
                    tem.waveforms["LM"]["time"] = _floats(val)
                elif "LM_WaveformAmplitude" in k_norm:
                    tem.waveforms["LM"]["amplitude"] = _floats(val)
                elif "HM_WaveformTime" in k_norm:
                    tem.waveforms["HM"]["time"] = _floats(val)
                elif "HM_WaveformAmplitude" in k_norm:
                    tem.waveforms["HM"]["amplitude"] = _floats(val)

                # Gate times
                elif "LM_GateOpenTime" in k_norm:
                    tem.gate_times["LM"]["open"] = _floats(val)
                elif "LM_GateCloseTime" in k_norm:
                    tem.gate_times["LM"]["close"] = _floats(val)
                elif "LM_GateCentreTime" in k_norm:
                    tem.gate_times["LM"]["center"] = _floats(val)
                elif "HM_GateOpenTime" in k_norm:
                    tem.gate_times["HM"]["open"] = _floats(val)
                elif "HM_GateCloseTime" in k_norm:
                    tem.gate_times["HM"]["close"] = _floats(val)
                elif "HM_GateCentreTime" in k_norm:
                    tem.gate_times["HM"]["center"] = _floats(val)

                else:
                    tem.meta[k_norm] = val
            else:
                # Section header like "/[Geometry Info]" - skip
                pass

            # Data table header starts with "/Project ..."
            if content.startswith("Project"):
                header_idx = i
                break
        else:
            # Non-comment line before the header - shouldn't happen but skip
            pass

    if header_idx is None:
        raise ValueError(f"No data table header found in {path!r}")

    columns = lines[header_idx].lstrip()[1:].split()   # strip leading whitespace then '/'

    tem.data = pd.read_csv(
        path,
        skiprows=header_idx + 1,
        sep=r"\s+",
        names=columns,
        na_values=["nan", "NaN"],
        engine="python",
        comment="/",
    )

    # Replace dummy value
    tem.data.replace(99999, np.nan, inplace=True)

    return tem


def read_xyz(path: str):
    """Auto-detect the `.xyz` format and dispatch to the matching reader.

    TEMImage-Beta (Kenbec) exports use ``/`` comment/header lines; TEM Data
    Manager exports use bare ``Key=Value`` lines. Returns a
    :class:`KenbecTEMData` or :class:`TEMData` accordingly.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                return read_kenbec_xyz(path) if line.startswith("/") else read_tem_xyz(path)
    raise ValueError(f"Empty file: {path!r}")


@dataclass
class TunoeTEMData(_PositionMixin):
    """Container for a set of parsed TEMcompany `.usf` sounding files (one per station).

    Usage
    -----
        tem = read_usf(r"...\\USF_Files_Tunoe")
        tem.gate_times["HM"]["center"]        # -> np.ndarray [s]
        tem.data                              # -> pandas DataFrame, one row per sounding

        surv = Survey(tem)
        surv.plot_soundings(); surv.plot_map()
    """

    meta: dict = field(default_factory=dict)
    waveforms: dict = field(default_factory=dict)   # {"LM": {"time":.., "amplitude":..}, "HM": {...}}
    gate_times: dict = field(default_factory=dict)   # {"LM": {"open":.., "center":.., "close":..}, "HM": {..}}
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def dbdt(self, moment: str) -> np.ndarray:
        """Station x gate dB/dt matrix [V/(A*m^2)] for 'LM' or 'HM'."""
        m = moment.upper()
        n = len(self.gate_times[m]["center"])
        cols = [f"{m}gate{i:03d}" for i in range(1, n + 1)]
        return self.data[cols].to_numpy(dtype=float)

    def dbdt_std(self, moment: str) -> np.ndarray:
        """Station x gate fractional uncertainty (stack SEM / |mean|) for 'LM' or 'HM'."""
        m = moment.upper()
        n = len(self.gate_times[m]["center"])
        cols = [f"{m}std{i:03d}" for i in range(1, n + 1)]
        return self.data[cols].to_numpy(dtype=float)

    def snr(self, moment: str):
        """Per-gate SNR from the scatter across stations.

        Returns (times, mean, sem, snr).
        """
        t = self.gate_times[moment.upper()]["center"]
        d = self.dbdt(moment)
        n_eff = np.sum(np.isfinite(d), axis=0)
        mean = np.nanmean(d, axis=0)
        sem = np.nanstd(d, axis=0) / np.sqrt(np.maximum(n_eff, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.abs(mean) / sem
        return t, mean, sem, snr

    def to_pytem(self, moment: str, station=None, min_noise=0.03, peak_current=None):
        """
        Convert one moment ('LM' or 'HM') into kwargs for pyTEM.

        Data unit conversion
        --------------------
        The `.usf` header declares ``VOLTAGE_UNITS: V/AM2``: the stored
        dB/dt is the raw receiver voltage already normalised by the Tx
        current alone (the Rx effective area is an instrument calibration
        folded in beforehand), i.e. ``data [V/(A*m^2)] = dBz/dt [V/m^2] /
        I``. This is the simplest reading of the header and has *not* been
        cross-checked against an independent calibration source -- treat
        absolute recovered resistivities with that caveat.

            obs_scaled = data * I_peak = dBz/dt_true [V/m^2]
            wf_I = amplitude * I_peak * N_tx           [A-turns]

        Parameters
        ----------
        moment     : 'LM' or 'HM'
        station    : int index or None (stack mean over all stations)
        min_noise  : fractional noise floor (default 0.03)
        peak_current : override the Tx current [A]; defaults to the
                       measured mean ``/CURRENT`` for this moment.

        Returns
        -------
        dict  with keys: times, obs_data, noise_std, waveform_times,
              waveform_currents, geometry, tx_size, rx_x, rx_y,
              plus context keys: gate_open, gate_close, tx_turns,
              n_stations.
        """
        m = moment.upper()
        gt = self.gate_times[m]
        times = gt["center"]

        dbdt = self.dbdt(m)
        frac = self.dbdt_std(m)
        current_col = f"{m}current"

        if station is not None:
            obs = dbdt[station]
            noise_frac = frac[station]
            measured_current = float(self.data[current_col].iloc[station])
            n_stations = 1
        else:
            obs = np.nanmean(dbdt, axis=0)
            n_eff = np.sum(np.isfinite(dbdt), axis=0)
            sem = np.nanstd(dbdt, axis=0) / np.sqrt(np.maximum(n_eff, 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                noise_frac = sem / np.abs(obs)
            measured_current = float(self.data[current_col].mean())
            n_stations = int(self.data.shape[0])

        noise_std = np.clip(np.nan_to_num(noise_frac, nan=min_noise), min_noise, None)

        # ---- Geometry -------------------------------------------------------
        tx_side = float(self.meta.get("LoopX", 3.0))
        tx_turns = int(float(self.meta.get("LoopTurns", 1)))
        rx_x = float(self.meta.get("RXcoil_X_Position", -13.0))
        rx_y = float(self.meta.get("RXcoil_Y_Position", 0.0))
        geometry = "square_offset" if (abs(rx_x) > 1e-6 or abs(rx_y) > 1e-6) else "square_central"

        if peak_current is None:
            peak_current = measured_current

        obs = np.abs(obs) * peak_current   # [V/m^2]

        wf_t = self.waveforms[m]["time"]
        wf_I = self.waveforms[m]["amplitude"] * peak_current * tx_turns   # [A-turns]

        return {
            "times": times,
            "obs_data": obs,
            "noise_std": noise_std,
            "waveform_times": wf_t,
            "waveform_currents": wf_I,
            "geometry": geometry,
            "tx_size": tx_side,
            "rx_x": rx_x,
            "rx_y": rx_y,
            # context
            "gate_open": gt.get("open"),
            "gate_close": gt.get("close"),
            "tx_turns": tx_turns,
            "n_stations": n_stations,
        }


def _read_usf_sounding(path):
    """Parse one TEMcompany `.usf` sounding file into header metadata + raw per-sweep channel data."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()


    meta = {}
    i, n = 0, len(lines)
    while i < n and not lines[i].startswith("/SWEEP_NUMBER"):
        line = lines[i].strip()
        if line.startswith("/DATE:"):
            meta["date"] = line.split(":", 1)[1].strip()
        elif line.startswith("/SOUNDING_NAME:"):
            meta["sounding_name"] = line.split(":", 1)[1].strip()
        elif line.startswith("/SWEEPS:"):
            meta["n_sweeps"] = int(line.split(":", 1)[1])
        elif line.startswith("/LOOP_SIZE:"):
            meta["loop_size"] = tuple(float(v) for v in line.split(":", 1)[1].split(","))
        elif line.startswith("/LOCATION:"):
            meta["location"] = tuple(float(v) for v in line.split(":", 1)[1].split(","))
        elif line.startswith("/COIL_LOCATION:"):
            meta["coil_location"] = tuple(float(v) for v in line.split(":", 1)[1].split(","))
        i += 1

    # channel_id -> {"time", "gate_open", "gate_close", "volts": [sweep][gate],
    #                "currents": [], "frequencies": [], "ramp_time", "ramp_amp"}
    channels = {}
    current = frequency = channel = None
    ramp_time = ramp_amp = None
    in_table = False
    time_list = gopen_list = gclose_list = volt_row = None
    for line in lines[i:]:
        s = line.strip()
        if s.startswith("/CURRENT:"):
            current = float(s.split(":", 1)[1])
        elif s.startswith("/FREQUENCY:"):
            frequency = float(s.split(":", 1)[1])
        elif s.startswith("/TX_RAMP:"):
            vals = _floats(s.split(":", 1)[1].replace(",", " "))
            ramp_time, ramp_amp = vals[0::2], vals[1::2]
        elif s.startswith("/CHANNEL:"):
            channel = int(s.split(":", 1)[1])
        elif s.startswith("TIME,"):
            in_table = True
            time_list, gopen_list, gclose_list, volt_row = [], [], [], []
        elif in_table:
            if s.startswith("/END"):
                in_table = False
                ch = channels.setdefault(channel, {"time": time_list, "gate_open": gopen_list,
                                                     "gate_close": gclose_list, "volts": [],
                                                     "currents": [], "frequencies": [],
                                                     "ramp_time": ramp_time, "ramp_amp": ramp_amp})
                ch["volts"].append(volt_row)
                ch["currents"].append(current)
                ch["frequencies"].append(frequency)
            elif s:
                t, v, _err, _q, topen, tclose = (float(x) for x in s.split(","))
                time_list.append(t)
                gopen_list.append(topen)
                gclose_list.append(tclose)
                volt_row.append(v)
    return meta, channels


_NOTES_LINE_RE = re.compile(
    r'\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+"([^"]*)"\s+"([^"]*)"\s*$')


def _read_usf_notes(folder) -> dict:
    """Parse the optional ``Notes_*.txt`` field-log next to the .usf files.

    Maps ``(LineNo, StationNo)`` (as the zero-padded strings used in the
    filenames, e.g. ``('001', '016')``) to the free-text ``UserNote`` field
    (e.g. proximity to a tractor or a shed/water pipe noise source).
    """
    notes = {}
    for path in Path(folder).glob("Notes_*.txt"):
        with open(path, encoding="utf-8", errors="replace") as fh:
            next(fh, None)  # header line
            for line in fh:
                m = _NOTES_LINE_RE.match(line)
                if m:
                    notes[(m.group(3), m.group(4))] = m.group(7).strip()
    return notes


def read_usf(folder: str) -> TunoeTEMData:
    """
    Read every TEMcompany `.usf` sounding file in `folder` into one dataset.

    Each `.usf` file holds one station/sounding: a header (Tx loop size, Rx
    coil offset, lon/lat) followed by many repeated sweeps split into a
    fast/low-moment (LM) and a slow/high-moment (HM) channel, identified
    here by frequency (highest = LM) rather than by the raw channel number,
    since that is robust to files with a different channel ordering or a
    single moment. The repeated sweeps of each channel are the stacks and
    are averaged here into one dB/dt curve + standard error per moment,
    stored in the same gate-column layout as :class:`KenbecTEMData` so the
    shared ``dbdt``/``snr``/:class:`~pytem.survey.Survey` machinery applies
    unchanged.

    Parameters
    ----------
    folder : str
        Directory containing the `.usf` files (one per station). A sibling
    ``Notes_*.txt`` field-log, if present, is parsed too and its
    ``UserNote`` per station attached as the ``UserNote`` column.

    Returns
    -------
    TunoeTEMData
    """
    paths = sorted(Path(folder).glob("*.usf"))
    if not paths:
        raise ValueError(f"No .usf files found in {folder!r}")

    notes = _read_usf_notes(folder)

    tem = TunoeTEMData()
    rows = []
    gate_times: dict = {}
    waveforms: dict = {}

    for path in paths:
        meta, channels = _read_usf_sounding(path)
        by_freq = sorted(channels.items(), key=lambda kv: -np.mean(kv[1]["frequencies"]))
        names = ["LM", "HM"][:len(by_freq)]

        # Filenames are "L<line>_S<station>_<timestamp>.usf", matching the
        # LineNo/StationNo columns of Notes_*.txt.
        parts = path.stem.split("_")
        line_no = parts[0][1:] if parts and parts[0].startswith("L") else "1"
        station_no = parts[1][1:] if len(parts) > 1 and parts[1].startswith("S") else "0"

        lon, lat, elev = meta["location"]
        row = {
            "SoundingName": meta.get("sounding_name"),
            "Line": line_no,
            "StationNo": station_no,
            "UserNote": notes.get((line_no, station_no), ""),
            "Date": meta.get("date"),
            "Longitude": lon, "Latitude": lat, "Elevation": elev,
            "NSweeps": meta.get("n_sweeps"),
        }

        for name, (_cid, ch) in zip(names, by_freq):
            volts = np.asarray(ch["volts"], dtype=float)
            n_stacks = volts.shape[0]
            mean = volts.mean(axis=0)
            sem = volts.std(axis=0, ddof=1) / np.sqrt(n_stacks) if n_stacks > 1 else np.zeros_like(mean)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = sem / np.abs(mean)
            for g in range(mean.size):
                row[f"{name}gate{g + 1:03d}"] = mean[g]
                row[f"{name}std{g + 1:03d}"] = frac[g]
            row[f"{name}current"] = float(np.mean(ch["currents"]))
            row[f"{name}nstacks"] = n_stacks

            if name not in gate_times:
                gate_times[name] = {
                    "center": np.asarray(ch["time"], dtype=float),
                    "open": np.asarray(ch["gate_open"], dtype=float),
                    "close": np.asarray(ch["gate_close"], dtype=float),
                }
                waveforms[name] = {
                    "time": np.asarray(ch["ramp_time"], dtype=float),
                    "amplitude": np.asarray(ch["ramp_amp"], dtype=float),
                }

        if not tem.meta:
            tem.meta = {
                "LoopX": meta["loop_size"][0], "LoopY": meta["loop_size"][1],
                "RXcoil_X_Position": meta["coil_location"][0],
                "RXcoil_Y_Position": meta["coil_location"][1],
                "LoopTurns": 1,
            }

        rows.append(row)

    tem.data = pd.DataFrame(rows)
    tem.gate_times = gate_times
    tem.waveforms = waveforms
    return tem
