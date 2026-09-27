"""
gerda_io.py - Reader for GERDA-schema Firebird ``.fdb`` TEM databases.

GERDA is the schema behind Denmark's national geophysical database (GEUS),
used to export ground/airborne TEM soundings (SkyTEM, tTEM, TEM40, ...) as a
single Firebird file, e.g. an export like ``rune.fdb``. Requires the ``fdb``
package (lazily imported, so plain ``pytem`` usage never needs it).

Schema (relevant tables only)
------------------------------
    DATASET   one physical measurement; ties together everything below
    TEMHEA    header info (bounding box, system subtype, #positions)
    TEMPOS    one row per station/position within the dataset
    TEMSEG    one row per repeat "segment": its own Tx/Rx geometry + current.
              Segments sharing the same timing config (TONTIME, MAXGATE,
              TXHEIGHT, RXHEIGHT, DISTANCE) are repeat stacks of the *same*
              moment (comparable to ``RECORDNGNO`` in the TEM Data Manager
              ``.xyz`` format), not separate moments.
    TEMDAT    one row per (segment, gate): the dB/dt decay data
    TEMWAVE   normalised (0..1) Tx waveform break points; in practice only
              stored once per moment group (its first/reference segment)

Multi-moment systems (dual-moment SkyTEM, tTEM, ...) show up as more than one
distinct timing group of segments for the same DATASET/POSITION;
:func:`read_sounding` splits these into separate entries of
``GerdaSounding.moments``.

Caveats
-------
There is no official public data dictionary for several fields used here
(``QUALITY`` flag meaning, ``DBDTSHIFTC``/``DBDTSHIFTF`` correction terms).
The handling below is a best-effort reading of the raw tables; always sanity
check a freshly-read sounding (e.g. against an existing GERDA inversion model
in ``MODEL``/``ONEDVMOD``/``ODVLAYER``, if present) before trusting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_io import _PositionMixin


def _connect(fdb_path: str):
    import fdb
    return fdb.connect(dsn='localhost:' + str(fdb_path), user='sysdba',
                        password='masterkey', charset='UTF8')


def build_system_filter(cutoff_hz):
    """Build a ``system_filter`` H(omega) callable from TEMFILT cutoffs.

    Every GERDA TEM segment carries one or more cascaded first-order
    (single-pole RC) analog low-pass filters (``TEMFILT.TYPE='LowPass'``,
    ``FILTORDER=1``) in the receiver's front end, applied before gating --
    e.g. 450 kHz then 338 kHz for SkyTEM1. These attenuate/phase-shift the
    broadband high-frequency content of the earliest gates (close to the
    transmitter turn-off) and are a likely cause of the systematic offset
    often seen there if omitted. ``cutoff_hz`` is an iterable of corner
    frequencies [Hz]; returns ``None`` if empty (no filter to apply).
    """
    cutoff_hz = [float(f) for f in cutoff_hz if f]
    if not cutoff_hz:
        return None
    omega_c = np.array([2 * math.pi * f for f in cutoff_hz])
    return _CascadedLowPass(omega_c)


class _CascadedLowPass:
    """Cascade of first-order RC low-pass filters, ``H(omega)``.

    A plain callable class (not a closure) so instances survive ``pickle``,
    e.g. when sent to a worker process by :func:`invert_soundings` -- a
    nested function returned from ``build_system_filter`` would not.
    """

    def __init__(self, omega_c):
        self.omega_c = omega_c

    def __call__(self, omega):
        h = np.ones_like(omega, dtype=complex)
        for wc in self.omega_c:
            h = h / (1.0 + 1j * omega / wc)
        return h


def list_datasets(fdb_path: str, datatype: str = 'tem', project: str | None = None,
                   limit: int | None = None) -> pd.DataFrame:
    """List datasets of a given ``DATASET.DATATYPE`` (default ``'tem'``).

    Returns one row per (dataset, position) with identifying metadata and
    coordinates, handy for picking a ``dataset_id``/``position`` to feed to
    :func:`read_sounding`.
    """
    con = _connect(fdb_path)
    try:
        cur = con.cursor()
        query = (
            "SELECT D.DATASET, D.IDENT, D.PROJECT, D.NAME, H.TEMSUBTYPE, "
            "P.\"POSITION\", P.XUTM, P.YUTM, P.ELEVATION, H.NUMPOS "
            "FROM DATASET D "
            "INNER JOIN TEMHEA H ON H.DATASET = D.DATASET "
            "INNER JOIN TEMPOS P ON P.DATASET = D.DATASET "
            "WHERE D.DATATYPE = ? "
        )
        params = [datatype]
        if project is not None:
            query += "AND D.PROJECT = ? "
            params.append(project)
        query += "ORDER BY D.DATASET, P.\"POSITION\""
        cur.execute(query, params)
        columns = [d[0].strip() for d in cur.description]
        rows = cur.fetchmany(limit) if limit else cur.fetchall()
        return pd.DataFrame.from_records(rows, columns=columns)
    finally:
        con.close()


@dataclass
class GerdaSounding:
    """One GERDA sounding (dataset + position), split into moments.

    ``moments`` maps a label (``'M1'``, ``'M2'``, ...) to a dict whose keys
    line up directly with :func:`pytem.inversion.invert` keyword arguments
    (``obs_data``, ``tx_size``, ``times``, ``noise_std``, ``waveform_times``,
    ``waveform_currents``, ``geometry``, ``rx_x``, ``rx_y``, ``tx_height``,
    ``rx_height``), plus a few context-only keys (``gate_quality``,
    ``tx_turns``, ``peak_current``, ``n_segments``).
    """

    dataset_id: int
    position: int
    ident: str
    x: float
    y: float
    elevation: float
    utm_zone: int | None = None
    moments: dict = field(default_factory=dict)


def read_sounding(fdb_path: str, dataset_id: int, position: int = 1,
                   min_noise: float = 0.03, signed: bool = False,
                   con=None) -> GerdaSounding:
    """Read one GERDA sounding and package each moment for pyTEM.

    Parameters
    ----------
    fdb_path   : str    Path to the ``.fdb`` file.
    dataset_id : int    ``DATASET.DATASET`` primary key.
    position   : int    ``TEMPOS.POSITION`` (default 1; single-position
                         ground/airborne-bin datasets always use 1).
    min_noise  : float  Floor applied to the fractional noise (default 0.03).
    signed     : bool   Keep the measured dB/dt sign (default False: abs value,
                         as pyTEM expects positive dB/dt).
    con        : fdb.Connection or None
                 Reuse an existing connection instead of opening/closing one
                 (useful when reading many soundings in a loop).
    """
    owns_con = con is None
    if owns_con:
        con = _connect(fdb_path)
    try:
        cur = con.cursor()

        cur.execute(
            'SELECT IDENT FROM DATASET WHERE DATASET = ?', (dataset_id,)
        )
        row = cur.fetchone()
        ident = row[0] if row else None

        cur.execute(
            'SELECT UTMZONE FROM TEMHEA WHERE DATASET = ?', (dataset_id,)
        )
        row = cur.fetchone()
        utm_zone = int(row[0]) if row and row[0] is not None else None

        cur.execute(
            'SELECT XUTM, YUTM, ELEVATION FROM TEMPOS WHERE DATASET = ? AND "POSITION" = ?',
            (dataset_id, position)
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f'No TEMPOS row for DATASET={dataset_id}, POSITION={position}')
        x, y, elevation = row

        cur.execute(
            'SELECT SEGMENT, SEGMENTTYP, TXCURRENT, DISTANCE, ANGLE, TXHEIGHT, TXAREA, '
            'TXSIDE1, TXSIDE2, TXTURNS, RXHEIGHT, RXAREA, MAXGATE, TONTIME, '
            'DBDTSHIFTC, DBDTSHIFTF '
            'FROM TEMSEG WHERE DATASET = ? AND "POSITION" = ? ORDER BY SEGMENT',
            (dataset_id, position)
        )
        seg_cols = [d[0].strip() for d in cur.description]
        segments = pd.DataFrame.from_records(cur.fetchall(), columns=seg_cols)
        if segments.empty:
            raise ValueError(f'No TEMSEG rows for DATASET={dataset_id}, POSITION={position}')

        # SEGMENTTYP == 2 marks a single reference/template segment per dataset
        # (nominal TXCURRENT=1, only used to carry the master TEMWAVE shape),
        # not an actual repeat measurement; keep it separate from stacking.
        is_reference = segments['SEGMENTTYP'] == 2
        reference_segments = segments[is_reference]
        measurement_segments = segments[~is_reference]
        if measurement_segments.empty:
            measurement_segments = segments

        # Group repeat segments that share the same physical moment/timing.
        group_key = measurement_segments[['TONTIME', 'MAXGATE', 'TXHEIGHT', 'RXHEIGHT', 'DISTANCE']].round(6)
        measurement_segments = measurement_segments.assign(_group=group_key.apply(tuple, axis=1))
        groups = sorted(measurement_segments['_group'].unique(), key=lambda g: (g[1], -abs(g[0])))

        moments = {}
        for i, gkey in enumerate(groups, start=1):
            seg_group = measurement_segments[measurement_segments['_group'] == gkey]
            seg_ids = seg_group['SEGMENT'].tolist()

            placeholders = ','.join('?' * len(seg_ids))
            cur.execute(
                f'SELECT SEGMENT, SEQUENCE, GATECETIME, OPGATETIME, CLGATETIME, '
                f'DBDT, DBDTSTDDEV, QUALITY, FACTOR '
                f'FROM TEMDAT WHERE DATASET = ? AND "POSITION" = ? AND SEGMENT IN ({placeholders}) '
                f'ORDER BY SEGMENT, SEQUENCE',
                (dataset_id, position, *seg_ids)
            )
            dat_cols = [d[0].strip() for d in cur.description]
            dat = pd.DataFrame.from_records(cur.fetchall(), columns=dat_cols)
            if dat.empty:
                continue

            dat['DBDT_CORR'] = dat['DBDT'] * dat['FACTOR'].fillna(1.0)
            pivot = dat.pivot(index='SEQUENCE', columns='SEGMENT', values='DBDT_CORR').sort_index()

            times = (dat.groupby('SEQUENCE')['GATECETIME'].mean().reindex(pivot.index).to_numpy())
            gate_open = dat.groupby('SEQUENCE')['OPGATETIME'].mean().reindex(pivot.index).to_numpy()
            gate_close = dat.groupby('SEQUENCE')['CLGATETIME'].mean().reindex(pivot.index).to_numpy()
            quality = (dat.groupby('SEQUENCE')['QUALITY'].max().reindex(pivot.index).to_numpy())

            values = pivot.to_numpy(dtype=float)
            n_records = values.shape[1]
            obs = np.nanmean(values, axis=1)
            if n_records > 1:
                n_eff = np.sum(np.isfinite(values), axis=1)
                sem = np.nanstd(values, axis=1) / np.sqrt(np.maximum(n_eff, 1))
                with np.errstate(divide='ignore', invalid='ignore'):
                    frac = sem / np.abs(obs)
            else:
                frac = dat.set_index('SEQUENCE')['DBDTSTDDEV'].reindex(pivot.index).to_numpy(dtype=float)

            # Apply the (usually neutral) group-level shift/scale correction.
            shift_c = float(seg_group['DBDTSHIFTC'].iloc[0] or 0.0)
            shift_f = float(seg_group['DBDTSHIFTF'].iloc[0] or 1.0)
            obs = obs * shift_f + shift_c

            # pytem's forward/invert functions assume dB/dt is already
            # normalised by the Rx coil area (rx_area=1, i.e. units V/m^2),
            # matching the TEM Data Manager .xyz convention -- but GERDA's
            # DBDT is the raw per-coil reading, so divide out the real area
            # (RXAREA, m^2) here rather than silently assuming it's 1.
            rx_area = float(seg_group['RXAREA'].iloc[0] or 1.0)
            obs = obs / rx_area
            values = values / rx_area

            if not signed:
                obs = np.abs(obs)
            noise_std = np.clip(np.nan_to_num(frac, nan=min_noise), min_noise, None)

            # ---- Waveform: prefer the matching reference segment's TEMWAVE,
            # falling back to any measurement segment that carries its own.
            # TONTIME can differ slightly between the reference segment and the
            # actual stacks (nominal vs. as-measured ramp), so match on the
            # gate count/geometry only, not TONTIME. ----
            ref_key = gkey[1:]  # (MAXGATE, TXHEIGHT, RXHEIGHT, DISTANCE)
            ref_match = reference_segments[
                reference_segments[['MAXGATE', 'TXHEIGHT', 'RXHEIGHT', 'DISTANCE']]
                .round(6).apply(tuple, axis=1) == ref_key
            ]
            wf_candidates = ref_match['SEGMENT'].tolist() + seg_ids
            wf_seg = None
            wf_seg_id = None
            for seg_id in wf_candidates:
                cur.execute(
                    'SELECT TIMEDELAY, AMPLITUDE FROM TEMWAVE '
                    'WHERE DATASET = ? AND "POSITION" = ? AND SEGMENT = ? ORDER BY SEQUENCE',
                    (dataset_id, position, seg_id)
                )
                wf_rows = cur.fetchall()
                if wf_rows:
                    wf_seg = np.asarray(wf_rows, dtype=float)
                    wf_seg_id = seg_id
                    break

            tx_turns = float(seg_group['TXTURNS'].iloc[0] or 1.0)
            peak_current = float(np.nanmedian(seg_group['TXCURRENT'].astype(float)))

            # ---- Receiver front-end filter(s): cascaded analog low-pass
            # filters applied before gating (TEMFILT), present on essentially
            # every GERDA segment; use whichever segment supplied the waveform. ----
            filt_seg_id = wf_seg_id if wf_seg_id is not None else (
                seg_ids[0] if seg_ids else None)
            filter_cutoffs = []
            if filt_seg_id is not None:
                cur.execute(
                    'SELECT CUTOFFFREQ FROM TEMFILT '
                    'WHERE DATASET = ? AND "POSITION" = ? AND SEGMENT = ? AND TYPE = \'LowPass\'',
                    (dataset_id, position, filt_seg_id)
                )
                filter_cutoffs = [row[0] for row in cur.fetchall()]
            system_filter = build_system_filter(filter_cutoffs)

            if wf_seg is not None:
                # GERDA often repeats a TIMEDELAY once at the pre-transition
                # and once at the post-transition amplitude (dt=0 corner
                # points); keep the last (settled) value per unique time.
                wf_df = pd.DataFrame(wf_seg, columns=['t', 'I']).drop_duplicates(
                    subset='t', keep='last')
                wf_t = wf_df['t'].to_numpy()
                wf_I = wf_df['I'].to_numpy() * peak_current * tx_turns
            else:
                wf_t = wf_I = None

            # ---- Geometry ----
            distance = float(seg_group['DISTANCE'].iloc[0] or 0.0)
            angle = seg_group['ANGLE'].iloc[0]
            angle = float(angle) if angle is not None else 0.0
            offset = abs(distance) > 1e-6
            geometry = 'square_offset' if offset else 'square_central'
            rx_x = distance * math.cos(math.radians(angle)) if offset else 0.0
            rx_y = distance * math.sin(math.radians(angle)) if offset else 0.0

            tx_side = seg_group[['TXSIDE1', 'TXSIDE2']].iloc[0].astype(float)
            tx_size = float(np.nanmean(tx_side))
            tx_height = float(seg_group['TXHEIGHT'].iloc[0] or 0.0)
            rx_height = float(seg_group['RXHEIGHT'].iloc[0] or 0.0)

            moments[f'M{i}'] = {
                'times': times,
                'obs_data': obs,
                'noise_std': noise_std,
                'waveform_times': wf_t,
                'waveform_currents': wf_I,
                'system_filter': system_filter,
                'geometry': geometry,
                'tx_size': tx_size,
                'rx_x': rx_x,
                'rx_y': rx_y,
                'tx_height': tx_height,
                'rx_height': rx_height,
                'dbdt_matrix': values.T,  # (n_segments, n_gates), pre-abs, already rx_area-normalised
                # --- context (not invert() kwargs) ---
                'gate_open': gate_open,
                'gate_close': gate_close,
                'gate_quality': quality,
                'tx_turns': tx_turns,
                'peak_current': peak_current,
                'n_segments': n_records,
                'rx_area': rx_area,
            }

        return GerdaSounding(dataset_id=dataset_id, position=position, ident=ident,
                              x=float(x), y=float(y), elevation=float(elevation),
                              utm_zone=utm_zone, moments=moments)
    finally:
        if owns_con:
            con.close()


def read_soundings(fdb_path: str, dataset_ids, position: int = 1,
                    min_noise: float = 0.03, signed: bool = False) -> list:
    """Read several soundings (e.g. one project/flight line) over one connection.

    Datasets that fail to read (missing TEMPOS/TEMSEG rows for ``position``)
    are skipped with a printed warning rather than raising.
    """
    con = _connect(fdb_path)
    soundings = []
    try:
        for dataset_id in dataset_ids:
            try:
                soundings.append(read_sounding(
                    fdb_path, dataset_id=dataset_id, position=position,
                    min_noise=min_noise, signed=signed, con=con))
            except ValueError as exc:
                print(f'Skipping DATASET={dataset_id}: {exc}')
    finally:
        con.close()
    return soundings


class GerdaLineAdapter(_PositionMixin):
    """Adapts several :class:`GerdaSounding` (e.g. one project/flight line)
    to the multi-station ``Survey`` interface (``gate_times``/``dbdt``/
    ``lines``/``utm_coords``), so tools built for TEM Data Manager profiles
    (``plot_transects``, ``plot_map``, ``plot_soundings``, ...) work directly
    on a set of GERDA soundings.

    All soundings must share the same ``moment`` label and gate-time array
    (true for soundings from the same project/system).
    """

    def __init__(self, soundings: list, moment: str = 'M1'):
        self.soundings = soundings
        self.moment = moment
        self.gate_times = {moment: {'center': soundings[0].moments[moment]['times']}}
        self.data = pd.DataFrame({
            'E': [s.x for s in soundings],
            'N': [s.y for s in soundings],
            'DatasetID': [s.dataset_id for s in soundings],
        })
        self.meta = {}
        zone = soundings[0].utm_zone
        if zone is not None:
            # ETRS89/UTM zone N (the 'euref89' datum used throughout GERDA).
            self.meta['utm_epsg'] = 25800 + zone

    def dbdt(self, moment: str) -> np.ndarray:
        return np.vstack([s.moments[self.moment]['obs_data'] for s in self.soundings])

    def dbdt_std(self, moment: str) -> np.ndarray:
        return np.vstack([s.moments[self.moment]['noise_std'] for s in self.soundings])


class GerdaTEMAdapter:
    """Adapts a :class:`GerdaSounding` to the ``gate_times``/``dbdt``/``snr``
    interface expected by :class:`pytem.survey.Survey`, so a GERDA sounding
    can reuse the same quick-look plots as a TEM Data Manager ``.xyz`` file.

    Each moment's repeat segments (before stacking) stand in for the
    ``.xyz`` format's repeat station records.
    """

    def __init__(self, sounding: GerdaSounding):
        self.sounding = sounding
        self.gate_times = {
            label: {'center': m['times']} for label, m in sounding.moments.items()
        }

    def dbdt(self, moment: str) -> np.ndarray:
        return self.sounding.moments[moment]['dbdt_matrix']

    def dbdt_std(self, moment: str) -> np.ndarray:
        m = self.sounding.moments[moment]
        return np.tile(m['noise_std'], (m['dbdt_matrix'].shape[0], 1))

    def snr(self, moment: str):
        t = self.gate_times[moment]['center']
        dbdt = self.dbdt(moment)
        n_eff = np.sum(np.isfinite(dbdt), axis=0)
        mean = np.nanmean(dbdt, axis=0)
        sem = np.nanstd(dbdt, axis=0) / np.sqrt(np.maximum(n_eff, 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = np.abs(mean) / sem
        return t, mean, sem, snr


_INVERT_KEYS = (
    'times', 'obs_data', 'waveform_times', 'waveform_currents', 'system_filter',
    'geometry', 'tx_size', 'rx_x', 'rx_y', 'tx_height', 'rx_height',
)


def equivalent_circle(moment: dict) -> dict:
    """Replace a moment's square-loop geometry with an equal-area circle.

    ``pytem``'s square-loop forward model (2-D quadrature over the loop area)
    is roughly 10x slower per call than the circular one in practice; GERDA's
    own TXSIDE1/TXSIDE2 are themselves already an equivalent-square stand-in
    for the real (often octagonal) airborne Tx frame, so swapping to an
    equal-area circle is not meaningfully less accurate, just faster.
    Central loop (square_central) -> circle_central; offset (square_offset)
    -> circle_offset, using the radial offset distance (angle is irrelevant
    for an azimuthally-symmetric circular source).
    """
    circle = dict(moment)
    tx_side = moment['tx_size']
    circle['tx_size'] = float(tx_side / np.sqrt(np.pi))
    if moment['geometry'] == 'square_offset':
        circle['geometry'] = 'circle_offset'
        circle['rx_x'] = float(np.hypot(moment['rx_x'], moment['rx_y']))
        circle['rx_y'] = 0.0
    else:
        circle['geometry'] = 'circle_central'
    return circle


def invert_kwargs(moment: dict, as_circle: bool = False) -> dict:
    """Build kwargs for :func:`pytem.inversion.invert` from one moment dict.

    ``pytem.inversion.invert`` treats a *scalar* ``noise_std`` as fractional
    (multiplied by ``obs_data`` internally) but an *array* ``noise_std`` as
    already absolute -- our moment dicts carry fractional per-gate noise, so
    it must be converted to absolute here before being passed on.

    ``as_circle=True`` swaps in the equal-area circle geometry (see
    :func:`equivalent_circle`) for a large speed-up.
    """
    if as_circle:
        moment = equivalent_circle(moment)
    kwargs = {k: moment[k] for k in _INVERT_KEYS}
    kwargs['noise_std'] = moment['noise_std'] * np.abs(moment['obs_data'])
    return kwargs


def predict_response(thicknesses, resistivities, moment: dict, transform='dlf',
                     as_circle: bool = False, use_numba: bool = True,
                     use_cuda: bool = False):
    """Forward-model one moment's gate times for a given layered model.

    Mirrors the forward + waveform-convolution pipeline used internally by
    ``pytem.inversion.invert`` (geometry dispatch, current/waveform scaling,
    tx/rx height), so the result is directly comparable to
    ``moment['obs_data']`` for QC/fit plots. Pass ``as_circle=True`` to match
    an inversion that was run with ``invert_kwargs(moment, as_circle=True)``.

    ``use_numba``/``use_cuda`` must match whatever backend flags the
    corresponding ``invert()`` call used -- ``pytem.forward`` functions
    default to ``use_cuda=True``, so silently leaving this unset here would
    compare against a *different* backend than the one actually inverted with.
    """
    from .forward import (fwd_square_central, fwd_square_offset,
                          fwd_circle_central, fwd_circle_offset)
    from .waveform import setup_waveform

    if as_circle:
        moment = equivalent_circle(moment)

    if moment['geometry'] == 'square_offset':
        fwd, geom_args = fwd_square_offset, (moment['tx_size'], moment['rx_x'], moment['rx_y'])
    elif moment['geometry'] == 'square_central':
        fwd, geom_args = fwd_square_central, (moment['tx_size'],)
    elif moment['geometry'] == 'circle_offset':
        fwd, geom_args = fwd_circle_offset, (moment['tx_size'], moment['rx_x'])
    else:
        fwd, geom_args = fwd_circle_central, (moment['tx_size'],)

    common = dict(tx_height=moment['tx_height'], rx_height=moment['rx_height'],
                  transform=transform, use_numba=use_numba, use_cuda=use_cuda,
                  system_filter=moment.get('system_filter'))

    if moment['waveform_times'] is not None:
        comp_times, apply_wf = setup_waveform(
            moment['times'], moment['waveform_times'], moment['waveform_currents'])
        step = fwd(thicknesses, resistivities, *geom_args, comp_times, **common)

        return -apply_wf(step)
    return -fwd(thicknesses, resistivities, *geom_args, moment['times'], **common)


def _invert_one(payload):
    """Module-level worker for :func:`invert_soundings` (must be a top-level
    function, not a closure, so ``ProcessPoolExecutor``/``pickle`` can send
    it to worker processes)."""
    import contextlib
    import io

    from .inversion import invert

    thicknesses, log_resistivities, kw = payload
    dataset_id = kw.pop('dataset_id', None)
    try:
        # invert() prints its per-iteration trace unconditionally; with many
        # soundings running at once that is just noise (and interleaved
        # garbage across processes), so swallow it here.
        with contextlib.redirect_stdout(io.StringIO()):
            result = invert(thicknesses=thicknesses, log_resistivities=log_resistivities, **kw)
        result['dataset_id'] = dataset_id
        return result
    except Exception as exc:
        return {'error': str(exc), 'dataset_id': dataset_id}


def invert_soundings(soundings, moment_label, thicknesses, log_resistivities,
                     n_jobs: int = 1, backend: str = 'cpu', as_circle: bool = True,
                     n_drop_early_gates: int = 0, verbose: bool = True,
                     **invert_overrides) -> list:
    """Invert many :class:`GerdaSounding` for the same moment, in parallel.

    Meant to make it trivial to switch between a laptop and an HPC/cluster
    node without knowing in advance whether it will have more CPU cores or a
    GPU available -- just flip ``backend``:

    - ``backend='cpu'`` (default): ``n_jobs`` worker *processes*
      (``concurrent.futures.ProcessPoolExecutor``), each running the Numba
      CPU backend (``use_cuda=False``). Set ``n_jobs`` to the number of
      cores you want to use (e.g. ``os.cpu_count()``).
    - ``backend='gpu'``: single process, ``use_cuda=True``, soundings run
      one after another. ``n_jobs`` is ignored -- one physical GPU is one
      device, so spreading soundings over several *processes* would just
      make them fight over the same device rather than run any faster.
    - ``backend='serial'``: single process, no multiprocessing at all, and
      backend flags come from ``invert_overrides`` instead of being forced
      -- handy for debugging (tracebacks/print output are not mixed up
      across workers).

    Parameters
    ----------
    soundings : list of GerdaSounding
    moment_label : str, e.g. ``'M1'`` -- must exist in every sounding
    thicknesses, log_resistivities : shared starting model (as for ``invert()``)
    n_jobs : int, number of CPU worker processes (``backend='cpu'`` only)
    backend : {'cpu', 'gpu', 'serial'}
    as_circle : bool, use the faster equal-area circle geometry (default True)
    n_drop_early_gates : int, drop this many leading gates per moment
    verbose : bool, print each sounding's result (flushed) as soon as it
        finishes -- in real completion order for ``backend='cpu'`` with
        ``n_jobs>1``, since workers do not all finish in submission order
    **invert_overrides : forwarded to ``pytem.inversion.invert`` (e.g.
        transform, rho_min, rho_max, max_noise_frac, step_backtrack,
        regularization). Do not pass ``use_numba``/``use_cuda`` here for
        ``backend`` 'cpu'/'gpu' -- they are set automatically. Per-sounding
        progress printouts *from inside* ``invert()`` are always suppressed
        (see ``_invert_one``); only the one-line-per-sounding summary below
        is printed.

    Returns
    -------
    list of dict, same order as ``soundings``: each is either the normal
    ``invert()`` result dict (plus a ``'dataset_id'`` key) or, on failure,
    ``{'error': str, 'dataset_id': ...}``.
    """
    if backend not in ('cpu', 'gpu', 'serial'):
        raise ValueError(f"backend must be 'cpu', 'gpu' or 'serial', got {backend!r}")

    payloads = []
    for s in soundings:
        m = s.moments[moment_label]
        if n_drop_early_gates:
            m = dict(m)
            for key in ('times', 'obs_data', 'noise_std'):
                m[key] = m[key][n_drop_early_gates:]
        kw = invert_kwargs(m, as_circle=as_circle)
        kw.update(invert_overrides)
        kw['dataset_id'] = s.dataset_id
        if backend == 'cpu':
            kw['use_cuda'] = False
            kw.setdefault('use_numba', True)
        elif backend == 'gpu':
            kw['use_cuda'] = True
            kw.setdefault('use_numba', False)
        payloads.append((thicknesses, log_resistivities, kw))

    def _report(n_done, result):
        if not verbose:
            return
        did = result.get('dataset_id')
        if 'error' in result:
            msg = f"  ({n_done}/{len(payloads)}) DATASET={did}: FAILED ({result['error']})"
        else:
            msg = (f"  ({n_done}/{len(payloads)}) DATASET={did}: "
                  f"RMS={result['rms_history'][-1]:.3f} in {result['n_iter']} iter")
        # Plain terminal '\r' overwrite doesn't reliably replace the previous
        # line in Jupyter/VS Code notebook output (each flushed print tends to
        # render as its own new line there) -- IPython's clear_output(wait=True)
        # is the mechanism notebooks actually support for an in-place update.
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
        except ImportError:
            pass
        print(msg, flush=True)

    if backend == 'cpu' and n_jobs > 1:
        import concurrent.futures as cf
        results = [None] * len(payloads)
        n_done = 0
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_invert_one, p): i for i, p in enumerate(payloads)}
            for fut in cf.as_completed(futures):
                result = fut.result()
                results[futures[fut]] = result
                n_done += 1
                _report(n_done, result)
        return results

    # backend='gpu' (one device, run sequentially) or 'serial', or 'cpu' with n_jobs<=1
    results = []
    for p in payloads:
        result = _invert_one(p)
        results.append(result)
        _report(len(results), result)
    return results


def _gate_edges(centres):
    """Gate open/close times from centres: geometric midpoints (InverTEM's rule)."""
    b = np.sqrt(centres[:-1] * centres[1:])
    b = np.r_[centres[0] ** 2 / b[0], b, centres[-1] ** 2 / b[-1]]
    return b[:-1], b[1:]


def export_usf(fdb_path: str, out_dir: str, project: str | None = None) -> list:
    """Write GERDA TEM soundings as ``.usf`` files for InverTEM and :func:`read_usf`.

    Each moment uses GEUS' processed reference segment (``SEGMENTTYP=2``) when
    present: current- and turn-normalised dB/dt, its per-gate relative std
    (``ERROR_BAR``, percent) and its accept/reject flag (``QUALITY``, 1/0).
    Soundings without one fall back to the stacked raw segments of
    :func:`read_sounding`. Moments stored as separate datasets named
    ``..._01``/``..._02`` (e.g. SkyTEM LM/HM) are paired into one file at their
    mean position. Missing gate open/close times (-1) are derived from the
    centres. ``FREQUENCY`` is nominal, 1/(4*|TONTIME|), used only to order
    moments (highest = LM). ``TX_HEIGHT``/``RX_HEIGHT`` are extension tags
    (ignored by :func:`read_usf`, used by InverTEM). Returns the written paths.
    """
    import utm
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = _connect(fdb_path)
    try:
        cur = con.cursor()

        def query(sql):
            cur.execute(sql)
            return pd.DataFrame.from_records(cur.fetchall(), columns=[d[0].strip() for d in cur.description])

        ref_join = ('JOIN TEMSEG S ON S.DATASET = T.DATASET AND S."POSITION" = T."POSITION" '
                    'AND S.SEGMENT = T.SEGMENT WHERE S.SEGMENTTYP = 2')
        seg = query('SELECT DATASET, "POSITION", SEGMENTTYP, TXCURRENT, TXHEIGHT, RXHEIGHT, TXSIDE1, TXSIDE2, '
                    'DISTANCE, ANGLE, TONTIME FROM TEMSEG').groupby(['DATASET', 'POSITION'])
        dat = query('SELECT T.DATASET, T."POSITION", T.GATECETIME, T.DBDT, T.DBDTSTDDEV, T.QUALITY '
                    f'FROM TEMDAT T {ref_join} ORDER BY T.DATASET, T."POSITION", T.SEQUENCE').groupby(['DATASET', 'POSITION'])
        wave = query('SELECT T.DATASET, T."POSITION", T.TIMEDELAY, T.AMPLITUDE '
                     f'FROM TEMWAVE T {ref_join} ORDER BY T.DATASET, T."POSITION", T.SEQUENCE').groupby(['DATASET', 'POSITION'])
        filt = query('SELECT T.DATASET, T."POSITION", T.CUTOFFFREQ FROM TEMFILT T '
                     f"{ref_join} AND T.TYPE = 'LowPass'").groupby(['DATASET', 'POSITION'])
        zones = dict(query('SELECT DATASET, UTMZONE FROM TEMHEA').itertuples(index=False))
        ds = list_datasets(fdb_path, project=project)

        def moments_of(row):
            key = (row.DATASET, row.POSITION)
            segs = seg.get_group(key)
            meas = segs[segs.SEGMENTTYP != 2]
            if key not in dat.groups:  # no processed segment: stack the raw ones
                snd = read_sounding(fdb_path, row.DATASET, row.POSITION, con=con)
                return [dict(times=m['times'], dbdt=m['obs_data'] / (m['peak_current'] * m['tx_turns']),
                             noise=m['noise_std'], quality=np.ones(len(m['times'])),
                             wf_t=m['waveform_times'], wf_a=m['waveform_currents'] / m['waveform_currents'].max(),
                             cutoffs=list(m['system_filter'].omega_c / (2 * math.pi)) if m['system_filter'] else [],
                             current=m['peak_current'], ton=float(meas.TONTIME.iloc[0]),
                             tx_h=m['tx_height'], rx_h=m['rx_height']) for m in snd.moments.values()]
            d = dat.get_group(key)
            w = wave.get_group(key).drop_duplicates() if key in wave.groups else None
            ref = segs[segs.SEGMENTTYP == 2].iloc[0]
            return [dict(times=d.GATECETIME.to_numpy(float), dbdt=d.DBDT.to_numpy(float),
                         noise=d.DBDTSTDDEV.to_numpy(float), quality=d.QUALITY.fillna(1).to_numpy(int),
                         wf_t=None if w is None else w.TIMEDELAY.to_numpy(float),
                         wf_a=None if w is None else w.AMPLITUDE.to_numpy(float),
                         cutoffs=filt.get_group(key).CUTOFFFREQ.tolist() if key in filt.groups else [],
                         current=float(meas.TXCURRENT.mean()) if len(meas) else 1.0,
                         ton=float(ref.TONTIME), tx_h=float(ref.TXHEIGHT or 0.0), rx_h=float(ref.RXHEIGHT or 0.0))]

        # Pair '<date>_<time>_<number>_<moment>' datasets on date + number; a repeated
        # moment starts a new sounding. Other names stay one sounding per dataset.
        parts = ds['NAME'].str.extract(r'^(\d+)_\d+_(\d+)_(\d{2})$')
        ds['SOUNDING'] = (parts[0] + '_' + parts[1]).fillna(ds['NAME'])
        ds['TAG'] = parts[2].fillna('')
        soundings = []
        for name, group in ds.sort_values('NAME').groupby('SOUNDING', sort=False):
            chunks, seen = [[]], set()
            for row in group.itertuples(index=False):
                if row.TAG in seen:
                    chunks.append([])
                    seen = set()
                chunks[-1].append(row)
                seen.add(row.TAG)
            soundings += [(name if i == 0 else f'{name}_{i + 1}', pd.DataFrame(c)) for i, c in enumerate(chunks)]
        paths = []
        for name, group in soundings:
            first = seg.get_group((group.DATASET.iloc[0], group.POSITION.iloc[0])).iloc[0]
            angle = math.radians(float(first.ANGLE or 0.0))
            distance = float(first.DISTANCE or 0.0)
            lat, lon = utm.to_latlon(group.XUTM.mean(), group.YUTM.mean(),
                                     zones.get(group.DATASET.iloc[0]) or 32, northern=True)
            moments = [m for row in group.itertuples(index=False) for m in moments_of(row)]
            lines = ['//USF: Universal Sounding Format', f'//GERDA export of {Path(fdb_path).name}, datasets '
                     + ', '.join(str(d) for d in group.DATASET), f'/SOUNDING_NAME: {name}',
                     f'/LOOP_SIZE: {float(first.TXSIDE1)}, {float(first.TXSIDE2)}',
                     f'/LOCATION: {lon:.8f}, {lat:.8f}, {group.ELEVATION.mean():.3f}',
                     f'/COIL_LOCATION: {distance * math.cos(angle):.4f}, {distance * math.sin(angle):.4f}',
                     '/VOLTAGE_UNITS: V/AM2', f'/SWEEPS: {len(moments)}', '']
            for k, m in enumerate(moments, start=1):
                opens, closes = _gate_edges(m['times'])
                lines += [f'/SWEEP_NUMBER: {k}', f'/CURRENT: {m["current"]:.6g}',
                          f'/FREQUENCY: {1.0 / (4.0 * abs(m["ton"])):.6g}', '/SWEEP_IS_NOISE: 0']
                if m['wf_t'] is not None:
                    lines.append('/TX_RAMP: ' + ', '.join(f'{t:.9g}, {a:.6g}' for t, a in zip(m['wf_t'], m['wf_a'])))
                if m['cutoffs']:
                    lines.append('/LOW_PASS: ' + ', '.join(f'{c:.6g}, 1' for c in m['cutoffs']))
                lines += [f'/TX_HEIGHT: {m["tx_h"]:.3f}', f'/RX_HEIGHT: {m["rx_h"]:.3f}',
                          f'/POINTS: {len(m["times"])}', f'/CHANNEL: {k}', '/END',
                          'TIME, VOLTAGE, ERROR_BAR, QUALITY, TIMEOPEN, TIMECLOSE']
                lines += [f'{t:.6e}, {v:.6e}, {100 * e:.3f}, {q}, {o:.6e}, {c:.6e}' for t, v, e, q, o, c in
                          zip(m['times'], m['dbdt'], np.nan_to_num(m['noise'], nan=0.0), m['quality'], opens, closes)]
                lines += ['/END', '']
            path = out / f'{name}.usf'
            path.write_text('\n'.join(lines))
            paths.append(path)
        return paths
    finally:
        con.close()
