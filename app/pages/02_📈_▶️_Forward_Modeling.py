import os
import sys
import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# -- Matplotlib font sizes (mobile-friendly) --------------------------
plt.rcParams.update({
    "font.size":       16,
    "axes.labelsize":  18,
    "axes.titlesize":  18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from pytem import fwd_circle_central
from ves import forward as ves_forward
from _shared import render_footer
try:
    from _shared import is_mobile
except ImportError:  # older deployed _shared.py without the helper
    def is_mobile():
        return False


def _fig_png(fig):
    """Rasterise a Matplotlib figure to PNG bytes (cheap to cache and re-display)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    return buf.getvalue()

# ── Shared utilities ──────────────────────────────────────────────────────────
_RHO = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

def _model_ui(prefix, n, def_rho, def_h):
    """Two-column table: left = all thickness sliders, right = all resistivity sliders.
    Returns (thicknesses, resistivities)."""
    h_out, r_out = [], []
    col_h, col_r = st.columns(2)
    with col_h:
        st.markdown("**Thickness [m]**")
        for i in range(n):
            is_hs = i == n - 1
            layer_lbl = f"Layer {i+1}" if not is_hs else "Half-space"
            if not is_hs:
                h_def = int(def_h[i]) if i < len(def_h) else 20
                h_out.append(float(st.slider(layer_lbl, 1, 500, h_def,
                                             key=f"{prefix}_h{i}")))
            else:
                st.caption(f"{layer_lbl}: extends to ∞")
    with col_r:
        st.markdown("**Resistivity [Ohm.m]**")
        for i in range(n):
            layer_lbl = f"Layer {i+1}" if i < n - 1 else "Half-space"
            rho_def = min(_RHO, key=lambda x: abs(x - (def_rho[i] if i < len(def_rho) else 100)))
            r_out.append(float(st.select_slider(layer_lbl, _RHO, value=rho_def,
                                                key=f"{prefix}_r{i}")))
    return h_out, r_out


def _stair(thick, rho):
    depths = [0.0] + list(np.cumsum(thick))
    bot = depths[-1] + max(depths[-1] * 0.3, 20.0)
    r_s, d_s = [], []
    for i, r in enumerate(rho):
        d_top = depths[i]
        d_bot = depths[i + 1] if i < len(thick) else bot
        r_s += [r, r]
        d_s += [d_top, d_bot]
    return r_s, d_s


# ── Cached forward calls ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _tem_fwd(h_t, rho_t, r, times_t):
    return -fwd_circle_central(list(h_t), list(rho_t),
                               tx_radius=r, times=np.array(times_t))


@st.cache_data(show_spinner=False)
def _ves_fwd(ab2_t, rho_t, h_t, filt):
    return ves_forward(np.array(ab2_t), np.array(rho_t), np.array(h_t), filt)


# ── Cached figure builders (rebuilt only when their inputs change) ────────────
@st.cache_data(show_spinner=False)
def _build_tem_fig(times_t, dbdt_t, dbdt_ref_t, thick_t, rho_t, mobile):
    times = np.asarray(times_t)
    dbdt = np.asarray(dbdt_t)
    dbdt_ref = np.asarray(dbdt_ref_t)

    if mobile:
        fig = Figure(figsize=(8, 12))
        ax1, ax2 = fig.subplots(2, 1)
    else:
        fig = Figure(figsize=(14, 6))
        ax1, ax2 = fig.subplots(1, 2)

    ax1.loglog(times, dbdt_ref, "--", color="black", lw=1.5,
               label="100 Ohm.m Model", zorder=1)
    ax1.loglog(times, dbdt, "o-", color="steelblue", ms=4, lw=1.5,
               label="Layered model", zorder=2)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax1.grid(True, which="both", ls="--", alpha=0.8)
    ax1.legend()

    rs, ds = _stair(list(thick_t), list(rho_t))
    _span_m = max(rs) / min(r for r in rs if r > 0)
    if _span_m < 10**2.5:
        _ctr_m = (max(rs) * min(r for r in rs if r > 0)) ** 0.5
        _mlo, _mhi = _ctr_m / 10**1.25, _ctr_m * 10**1.25
    else:
        _mlo, _mhi = min(r for r in rs if r > 0) * 0.8, max(rs) * 1.25
    ax2.semilogx(rs, ds, color="steelblue", lw=2)
    ax2.set_xlim(_mlo, _mhi)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Resistivity [Ohm.m]")
    ax2.set_ylabel("Depth [m]")
    ax2.grid(True, which="both", ls="--", alpha=0.8)
    fig.tight_layout()
    return _fig_png(fig)


@st.cache_data(show_spinner=False)
def _build_ves_fig(ab2_t, rhoap_t, rhoap_ref_t, thick_t, rho_t, mobile):
    ab2 = np.asarray(ab2_t)
    rhoap = np.asarray(rhoap_t)
    rhoap_ref = np.asarray(rhoap_ref_t)

    if mobile:
        fig = Figure(figsize=(8, 12))
        ax1, ax2 = fig.subplots(2, 1)
    else:
        fig = Figure(figsize=(14, 6))
        ax1, ax2 = fig.subplots(1, 2)

    _rho_all = np.concatenate([rhoap, rhoap_ref])
    _span_v = np.log10(_rho_all.max()) - np.log10(_rho_all.min())
    if _span_v < 2.5:
        _ctr_v = (np.log10(_rho_all.max()) + np.log10(_rho_all.min())) / 2
        _vlo, _vhi = 10 ** (_ctr_v - 1.25), 10 ** (_ctr_v + 1.25)
    else:
        _vlo, _vhi = _rho_all.min() * 0.8, _rho_all.max() * 1.25
    ax1.loglog(ab2, rhoap_ref, "--", color="black", lw=1.5,
               label="100 Ohm.m Model", zorder=1)
    ax1.loglog(ab2, rhoap, "o-", color="darkorange", ms=4, lw=1.5,
               label="Layered model", zorder=2)
    ax1.set_ylim(_vlo, _vhi)
    ax1.set_xlabel(r"AB/2 [m]")
    ax1.set_ylabel("Apparent resistivity [Ohm.m]")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()

    rs, ds = _stair(list(thick_t), list(rho_t))
    _span_m = max(rs) / min(r for r in rs if r > 0)
    if _span_m < 10**2.5:
        _ctr_m = (max(rs) * min(r for r in rs if r > 0)) ** 0.5
        _mlo, _mhi = _ctr_m / 10**1.25, _ctr_m * 10**1.25
    else:
        _mlo, _mhi = min(r for r in rs if r > 0) * 0.8, max(rs) * 1.25
    ax2.semilogx(rs, ds, color="darkorange", lw=2)
    ax2.set_xlim(_mlo, _mhi)
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Resistivity [Ohm.m]")
    ax2.set_ylabel("Depth [m]")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return _fig_png(fig)


# ── Page header ───────────────────────────────────────────────────────────────
st.header(":blue[Predicted response for a layered earth model]")
st.markdown(
    "Build a layered resistivity model on and see the predicted "
    "sounding curve update in real time. The sliders change the "
    "*earth model* (layer thicknesses and resistivities); the plot is the "
    "*simulated measurement* that such an earth would produce, not something you "
    "edit directly. The layered earth model below is **shared between the TEM and "
    "VES tabs**, so switching method keeps the same structure and makes the two "
    "responses directly comparable. In both cases a homogeneous 100 Ohm.m earth is "
    "shown for reference."
)

with st.expander("Why forward modelling?", expanded=False):
    st.markdown(
        """
        **Forward modelling** addresses the question: *if the ground had this
        resistivity structure, what would the instrument measure?* It is the numerical engine
        inside every inversion. 

        Move the sliders below and watch the theoretical response change in real time.
    
        """
    )


tab_tem, tab_ves = st.tabs(["🧲 TEM", "⚡️ VES"])

# ═══════════════════════════════════════════════════════════════════════════════
# TEM TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tem:
    st.subheader(":blue-background[TEM - dB/dt sounding]", divider="blue")

    # ── 1. System parameters ──────────────────────────────────────────────────
    st.markdown("**System Parameters**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tx_side = st.number_input(
            "Tx loop side length [m]", min_value=5, max_value=500, value=40, step=5,
            key="fwd_tem_side",
            help="Side length of the square transmitter loop laid on the ground. "
                 "A bigger loop injects more energy and reaches greater depth.",
        )
        tx_r = float(np.sqrt(tx_side ** 2 / np.pi))
        n_t = int(st.number_input(
            "Time gates", 5, 50, 25, key="fwd_tem_nt",
            help="Number of time channels sampled on the decay curve, spaced "
                 "logarithmically between the early and late times below.",
        ))
    with col_s2:
        st.markdown("Early time [log10(s)]")
        t_min = st.slider(
            "Early time [s]", -6.0, -4.0, -5.0, 0.25, key="fwd_tem_tmin",
            label_visibility="collapsed",
            help="First (earliest) gate time, as a power of 10. Earlier times sense shallower ground.",
        )
        st.markdown("Late time [log10(s)]")
        t_max = st.slider(
            "Late time [s]", -3.0, -1.0, -2.0, 0.25, key="fwd_tem_tmax",
            label_visibility="collapsed",
            help="Last (latest) gate time, as a power of 10. Later times sense deeper structure.",
        )
    times = np.logspace(t_min, t_max, n_t)

    # ── 2. Layer model ────────────────────────────────────────────────────────
    st.markdown("**Layer Model**")
    n_layers_tem = int(st.number_input(
        "Number of layers", 2, 6, 3, key="fwd_tem_n",
        help="The bottom layer is a half-space (no thickness). "
             "Adjust thicknesses and resistivities to explore different earth structures.",
    ))
    thick_tem, rho_tem = _model_ui("fwd_tem", n_layers_tem, [100, 10, 300], [20, 50])

    # ── 3. Forward response ───────────────────────────────────────────────────
    st.caption("The plot updates automatically when you move a slider.")
    try:
        with st.spinner("Computing …"):
            dbdt = _tem_fwd(tuple(thick_tem), tuple(rho_tem), tx_r, tuple(times.tolist()))
            dbdt_ref = _tem_fwd((), (100.0,), tx_r, tuple(times.tolist()))
        fig = _build_tem_fig(
            tuple(times.tolist()), tuple(np.asarray(dbdt).tolist()),
            tuple(np.asarray(dbdt_ref).tolist()), tuple(thick_tem), tuple(rho_tem),
            is_mobile(),
        )
        st.image(fig, use_column_width=True)
        with st.expander("How to read the TEM curve"):
            st.markdown(
                """
                - **Time is a proxy for depth:** early gates (left) sense shallow
                  ground, late gates (right) sense deeper. A model change that only
                  moves the late-time tail is happening at depth.
                - **Conductive layers slow the decay:** eddy currents linger in a
                  conductor, holding the dB/dt curve up for longer.
                - **Resistive layers let the field diffuse away quickly,** steepening
                  the decay. Because resistors carry little induced current, TEM is
                  relatively insensitive to them, so a thin resistor barely changes
                  the curve.
                - **The late-time noise floor** (not shown here) eventually swallows
                  the signal: if your target only changes the curve below that floor,
                  it is effectively invisible in the field.
                """
            )
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Try adjusting the sliders.")

# ═══════════════════════════════════════════════════════════════════════════════
# VES TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ves:
    st.subheader(":orange-background[VES - Apparent resistivity sounding]", divider="orange")

    # ── 1. System parameters ──────────────────────────────────────────────────
    st.markdown("**Survey Parameters**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        ab2_min = st.slider(
            "AB/2 minimum [m]", 1, 30, 1, key="fwd_ves_ab2min",
            help="Smallest half-spacing between the current electrodes. Small spacings sample shallow ground.",
        )
        ab2_max = st.slider(
            "AB/2 maximum [m]", 50, 2000, 300, key="fwd_ves_ab2max",
            help="Largest half-spacing between the current electrodes. Large spacings drive current deeper.",
        )
        n_ab2 = int(st.number_input(
            "AB/2 points", 5, 60, 25, key="fwd_ves_nab2",
            help="Number of electrode spacings sampled along the sounding curve, "
                 "spaced logarithmically between the minimum and maximum.",
        ))
        filt = "gs11"
    ab2 = np.logspace(np.log10(ab2_min), np.log10(ab2_max), n_ab2)

    # ── 2. Layer model ────────────────────────────────────────────────────────
    st.markdown("**Layer Model**")
    n_layers_ves = int(st.number_input(
        "Number of layers", 2, 6, 3, key="fwd_ves_n",
        help="The bottom layer is a half-space (no thickness).",
    ))
    thick_ves, rho_ves = _model_ui("fwd_ves", n_layers_ves, [100, 10, 300], [20, 50])

    # ── 3. Forward response ───────────────────────────────────────────────────
    st.caption("The plot updates automatically when you move a slider.")
    try:
        with st.spinner("Computing …"):
            rhoap = _ves_fwd(tuple(ab2.tolist()), tuple(rho_ves), tuple(thick_ves), filt)
            rhoap_ref = _ves_fwd(tuple(ab2.tolist()), (100.0,), (), filt)
        fig = _build_ves_fig(
            tuple(ab2.tolist()), tuple(np.asarray(rhoap).tolist()),
            tuple(np.asarray(rhoap_ref).tolist()), tuple(thick_ves), tuple(rho_ves),
            is_mobile(),
        )
        st.image(fig, use_column_width=True)
        with st.expander("How to read the VES curve"):
            st.markdown(
                """
                - **Electrode spacing is a proxy for depth:** small AB/2 (left)
                  samples shallow ground, large AB/2 (right) drives current deeper.
                - **Apparent resistivity is a smoothed average** of the true layers
                  the current passes through, so sharp boundaries appear as gentle
                  rises and falls, not steps.
                - **Resistive layers push the curve up; conductive layers pull it
                  down.** VES expresses a resistive layer clearly (current is forced
                  through it), which is where it complements TEM.
                - **Equivalence:** a thin layer can often be traded for a thicker,
                  proportionally different one with almost the same curve, a key
                  ambiguity to keep in mind when interpreting.
                """
            )
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Try adjusting the sliders.")

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Sensitivity challenge + self-check
# ═══════════════════════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(":violet[Try it yourself: how sensitive is each method?]", divider="violet")
st.markdown(
    """
    Use the model sliders above (and the dashed **homogeneous 100 Ohm.m** reference
    line for comparison) to run these quick experiments, then answer below.

    1. **Buried conductor.** Start from `100 / 100 / 100` Ohm.m and drop the middle
       layer to `5` Ohm.m. Watch how the TEM decay and the VES curve move away from
       the reference line.
    2. **Buried resistor.** Now set the middle layer to `5000` Ohm.m instead. Compare
       how much the TEM curve changes versus how much the VES curve changes.
    3. **Depth of burial.** Make the conductive layer deeper by increasing the top
       layer thickness, and see which part of the curve (early vs late time) responds.
    """
)

_FWD_QUIZ = [
    {
        "q": "You bury a conductive layer (e.g. middle layer = 5 Ohm.m). Versus the "
             "homogeneous 100 Ohm.m line, the TEM decay curve:",
        "options": [
            "Stays elevated for longer (decays more slowly)",
            "Drops faster (decays more quickly)",
            "Is unchanged",
        ],
        "answer": "Stays elevated for longer (decays more slowly)",
        "why": "Eddy currents linger in the conductor, holding |dB/dt| up at later "
               "times, the hallmark TEM response to a buried conductor.",
    },
    {
        "q": "You instead bury a very resistive layer (e.g. 5000 Ohm.m). On the TEM "
             "curve, compared with the homogeneous case the change is:",
        "options": [
            "Large and obvious",
            "Small, TEM is weakly sensitive to resistive layers",
            "The curve disappears",
        ],
        "answer": "Small, TEM is weakly sensitive to resistive layers",
        "why": "Resistive layers carry little induced current, so they barely reshape "
               "the TEM decay. This is TEM's main blind spot.",
    },
    {
        "q": "That same buried resistor on the VES apparent-resistivity curve produces:",
        "options": [
            "A clear rise above the 100 Ohm.m reference model",
            "No visible change",
            "A drop below the reference",
        ],
        "answer": "A clear rise above the 100 Ohm.m reference",
        "why": "Galvanic VES forces current through resistive layers, so a buried "
               "resistor lifts the apparent resistivity, where VES complements TEM.",
    },
]

_fwd_user = [
    st.radio(_item["q"], _item["options"], index=None, key=f"fwd_quiz_{_i}")
    for _i, _item in enumerate(_FWD_QUIZ)
]

if st.button("Check my answers", key="fwd_quiz_check"):
    _score = 0
    for _i, _item in enumerate(_FWD_QUIZ):
        if _fwd_user[_i] == _item["answer"]:
            _score += 1
            st.success(f"Q{_i + 1}: Correct. {_item['why']}")
        elif _fwd_user[_i] is None:
            st.warning(
                f"Q{_i + 1}: Not answered. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
        else:
            st.error(
                f"Q{_i + 1}: Not quite. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
    st.metric("Your score", f"{_score} / {len(_FWD_QUIZ)}")
    if _score == len(_FWD_QUIZ):
        st.balloons()

render_footer()
