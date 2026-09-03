"""
Field example: hard-rock regolith groundwater exploration, West Africa.

Loads a (synthetic but realistic) central-loop TEM sounding from a CSV that
behaves like genuine field data, inverts it with pyTEM, and walks the user
through interpreting the model in a hydrogeological context: locating the
conductive weathered-saprolite aquifer above fresh crystalline basement.
"""

import io
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

plt.rcParams.update({
    "font.size":       16,
    "axes.labelsize":  16,
    "axes.titlesize":  16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

import pandas as pd
import streamlit as st

# -- Path setup ----------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from pytem import fwd_circle_central, invert as tem_invert
from ves import forward as ves_forward, invert as ves_invert
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

_DATA = os.path.join(_ROOT, "app", "data", "west_africa_regolith.csv")
_DATA_VES = os.path.join(_ROOT, "app", "data", "west_africa_regolith_ves.csv")
_TX_SIDE = 40.0                                   # 40 x 40 m transmitter loop
_TX_RADIUS = float(np.sqrt(_TX_SIDE ** 2 / np.pi))
_VES_FILTER = "gs11"

# -- Page header ---------------------------------------------------------------
st.header(":blue[Field example: finding water in regolith]")

st.markdown(
    """
    Across the crystalline basement of West Africa, 
    a significant portion of the rural water supply comes not from the fresh,
    impermeable bedrock but from the **weathered overburden, the regolith**,
    that caps it. A typical weathering profile looks like this:

    | Unit | Character | Resistivity | Role |
    |------|-----------|-------------|------|
    | Lateritic / ferricrete cap | Hard, iron-rich crust | **High** (hundreds of Ohm.m) | Dry, protective |
    | Saprolite | Clay-rich weathered rock | **Low** (tens of Ohm.m) | Stores water |
    | Saprock / fractured basement | Partly weathered, fractured | **Low-moderate** | Yields water |
    | Fresh basement | Unweathered crystalline rock | **Very high** (thousands of Ohm.m) | Aquiclude |

    The productive aquifer is the **conductive saprolite/saprock** sitting between
    a resistive cap and the resistive fresh basement.
    """
)

st.info(
    "**The task.** A field crew recorded two soundings at a candidate borehole "
    "site: a central-loop TEM sounding (40 x 40 m loop) and a VES sounding. "
    "The true ground is unknown. Invert both, compare what each method resolves."
)

# -- Load the 'field' data -----------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_field():
    df = pd.read_csv(_DATA, comment="#")
    return (
        df["time_s"].to_numpy(),
        df["dbdt_Vm2"].to_numpy(),
        df["uncertainty_Vm2"].to_numpy(),
    )


@st.cache_data(show_spinner=False)
def _load_field_ves():
    df = pd.read_csv(_DATA_VES, comment="#")
    return (
        df["ab2_m"].to_numpy(),
        df["rhoa_ohmm"].to_numpy(),
        df["uncertainty_ohmm"].to_numpy(),
    )

try:
    times, dbdt_obs, sigma = _load_field()
    ab2, rhoa_obs, rhoa_sigma = _load_field_ves()
except FileNotFoundError:
    st.error(
        "Field dataset not found. Generate it once by running "
        "`app/data/make_field_data.py` with your Python interpreter."
    )
    st.stop()

# -- Show the raw field data ---------------------------------------------------
st.subheader("1. The measured soundings")

@st.cache_data(show_spinner=False)
def _build_raw_fig(times_t, dbdt_obs_t, ab2_t, rhoa_obs_t, mobile):
    times = np.asarray(times_t)
    dbdt_obs = np.asarray(dbdt_obs_t)
    ab2 = np.asarray(ab2_t)
    rhoa_obs = np.asarray(rhoa_obs_t)

    if mobile:
        fig_raw = Figure(figsize=(8, 12), constrained_layout=True)
        ax_tem_raw, ax_ves_raw = fig_raw.subplots(2, 1)
    else:
        fig_raw = Figure(figsize=(14, 6), constrained_layout=True)
        ax_tem_raw, ax_ves_raw = fig_raw.subplots(1, 2)

    ax_tem_raw.plot(times, dbdt_obs, "o-", ms=5, color="steelblue", lw=1.5, label="Field data")
    ax_tem_raw.set_xscale("log")
    ax_tem_raw.set_yscale("log")
    ax_tem_raw.set_xlabel("Time [s]")
    ax_tem_raw.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax_tem_raw.set_title("Central-loop TEM sounding")
    ax_tem_raw.grid(True, which="both", ls="--", alpha=0.8)
    ax_tem_raw.legend()

    ax_ves_raw.plot(ab2, rhoa_obs, "o-", ms=5, color="darkorange", lw=1.5, label="Field data")
    ax_ves_raw.set_xscale("log")
    ax_ves_raw.set_yscale("log")
    ax_ves_raw.set_ylim(top=1e3)
    ax_ves_raw.set_xlabel("AB/2 [m]")
    ax_ves_raw.set_ylabel("Apparent resistivity [Ohm.m]")
    ax_ves_raw.set_title("VES sounding")
    ax_ves_raw.grid(True, which="both", ls="--", alpha=0.8)
    ax_ves_raw.legend()
    return _fig_png(fig_raw)


fig_raw = _build_raw_fig(tuple(times), tuple(dbdt_obs), tuple(ab2), tuple(rhoa_obs), is_mobile())
st.image(fig_raw, use_container_width=True)

# -- Inversion controls --------------------------------------------------------
st.subheader("2. Invert both soundings for a resistivity-depth model")

max_depth = 200  # sensible default depth of the deepest model layer node [m]

@st.cache_data(show_spinner=False)
def _invert_field(times_t, dbdt_t, sigma_t, tx_r, start_rho, max_depth):
    times = np.asarray(times_t)
    dbdt_obs = np.asarray(dbdt_t)
    noise_std = np.asarray(sigma_t)

    depths = np.logspace(np.log10(2), np.log10(max_depth), 19)
    thick = np.diff(np.concatenate([[0.0], depths])).tolist()      # 19
    log_rho0 = np.log(np.full(20, float(start_rho)))               # 20

    res = tem_invert(
        obs_data=dbdt_obs, thicknesses=thick,
        log_resistivities=log_rho0, tx_size=tx_r, times=times,
        noise_std=noise_std, alpha_steps=10, maxit=15,
        max_noise_frac=0.0,
        transform="dlf", hankel_filter="key_101", fourier_filter="key_81",
        analytical_j=True,
    )
    dbdt_pred = -fwd_circle_central(
        thick, res["resistivities"].tolist(), tx_radius=tx_r, times=times
    )
    return thick, res["resistivities"], res["rms_history"], dbdt_pred


@st.cache_data(show_spinner=False)
def _invert_field_ves(ab2_t, rhoa_t, start_rho, max_depth):
    ab2 = np.asarray(ab2_t)
    rhoa_obs = np.asarray(rhoa_t)

    depths = np.logspace(np.log10(2), np.log10(max_depth), 15)
    thick = np.diff(np.concatenate([[0.0], depths])).tolist()      # 15
    rho0 = np.full(16, float(start_rho))                           # 16

    res = ves_invert(
        ab2=ab2, rhoap_obs=rhoa_obs,
        resistivities=rho0, thicknesses=thick,
        regularization="auto", iter_max=15, filter_set=_VES_FILTER,
        fix_thicknesses=True, noise_frac=0.05,
    )
    return thick, res["resistivities"], res["rms_history"], res["rhoap_pred"]


def _stair(thick, rho, extra=50.0):
    nodes = np.concatenate([[0.0], np.cumsum(thick)])
    bottom = nodes[-1] + extra
    rs, ds = [], []
    for i, r in enumerate(rho):
        d_top = nodes[i]
        d_bot = nodes[i + 1] if i < len(thick) else bottom
        rs += [r, r]
        ds += [d_top, d_bot]
    return rs, ds


@st.cache_data(show_spinner=False)
def _build_results_fig(thick_r, rho_r, thick_v, rho_v, dbdt_pred_t, rhoa_pred_t,
                       times_t, dbdt_obs_t, ab2_t, rhoa_obs_t, show_units, mobile):
    rho_r = np.asarray(rho_r)
    rho_v = np.asarray(rho_v)
    dbdt_pred = np.asarray(dbdt_pred_t)
    rhoa_pred = np.asarray(rhoa_pred_t)
    times = np.asarray(times_t)
    dbdt_obs = np.asarray(dbdt_obs_t)
    ab2 = np.asarray(ab2_t)
    rhoa_obs = np.asarray(rhoa_obs_t)

    if mobile:
        fig = Figure(figsize=(8, 16))
        gs = fig.add_gridspec(3, 1, hspace=0.4)
        ax_model = fig.add_subplot(gs[0, 0])
        ax_tem = fig.add_subplot(gs[1, 0])
        ax_ves = fig.add_subplot(gs[2, 0])
    else:
        fig = Figure(figsize=(12, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.3)
        ax_model = fig.add_subplot(gs[:, 0])
        ax_tem = fig.add_subplot(gs[0, 1])
        ax_ves = fig.add_subplot(gs[1, 1])

    rs_t, ds_t = _stair(list(thick_r), list(rho_r))
    rs_v, ds_v = _stair(list(thick_v), list(rho_v))
    ax_model.plot(rs_t, ds_t, color="steelblue", lw=2, label="TEM recovered")
    ax_model.plot(rs_v, ds_v, color="darkorange", lw=2, label="VES recovered")
    ax_model.set_xscale("log")
    ax_model.invert_yaxis()
    ax_model.set_xlabel("Resistivity [Ohm.m]")
    ax_model.set_ylabel("Depth [m]")
    ax_model.set_title("Recovered resistivity models")
    ax_model.grid(True, which="both", ls="--", alpha=0.8)
    ax_model.legend()

    if show_units:
        _units = [
            (0.0, 6.0, "Lateritic / ferricrete cap"),
            (6.0, 36.0, "Saprolite aquifer"),
            (36.0, None, "Fresh basement"),
        ]
        _xmin, _xmax = ax_model.get_xlim()
        _ybot = ax_model.get_ylim()[0]           # deepest visible depth (y inverted)
        _x_lbl = np.sqrt(_xmin * _xmax)          # geometric centre on the log axis
        for _top, _base, _name in _units:
            ax_model.axhline(_top, color="black", ls="--", lw=1.4, alpha=0.9, zorder=1)
            _mid = (_top + (_base if _base is not None else _ybot)) / 2.0
            ax_model.text(
                _x_lbl, _mid, _name, color="black", fontsize=13, fontweight="bold",
                ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85),
            )

    ax_tem.loglog(times, dbdt_pred, "-", color="steelblue", lw=1.5, label="Predicted")
    ax_tem.loglog(times, dbdt_obs, "o", color="black", ms=4, label="TEM data")
    ax_tem.set_xlabel("Time [s]")
    ax_tem.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax_tem.set_title("TEM data fit")
    ax_tem.grid(True, which="both", ls="--", alpha=0.8)
    ax_tem.legend()

    ax_ves.loglog(ab2, rhoa_pred, "-", color="darkorange", lw=1.5, label="Predicted")
    ax_ves.loglog(ab2, rhoa_obs, "o", color="black", ms=4, label="VES data")
    ax_ves.set_xlabel("AB/2 [m]")
    ax_ves.set_ylabel("Apparent resistivity [Ohm.m]")
    ax_ves.set_title("VES data fit")
    ax_ves.grid(True, which="both", ls="--", alpha=0.8)
    ax_ves.legend()
    return _fig_png(fig)


_col_btn, _col_rho = st.columns([1, 3])
with _col_btn:
    st.write("")
    st.write("")
    _run = st.button("Run inversions", type="primary")
with _col_rho:
    start_rho = st.slider(
        "Starting resistivity [Ohm.m]",
        min_value=20, max_value=500, value=100, step=10,
        help="Both inversions start from a uniform half-space and refine it.",
    )

if _run:
    with st.spinner("Inverting the TEM and VES soundings..."):
        _res_tem = _invert_field(
            tuple(times), tuple(dbdt_obs), tuple(sigma),
            _TX_RADIUS, start_rho, max_depth,
        )
        _res_ves = _invert_field_ves(
            tuple(ab2), tuple(rhoa_obs), start_rho, max_depth,
        )
    st.session_state["wa_result"] = _res_tem
    st.session_state["wa_result_ves"] = _res_ves

if "wa_result" in st.session_state and "wa_result_ves" in st.session_state:
    thick_r, rho_r, rms_hist, dbdt_pred = st.session_state["wa_result"]
    thick_v, rho_v, rms_hist_v, rhoa_pred = st.session_state["wa_result_ves"]
    rho_r = np.asarray(rho_r)
    rho_v = np.asarray(rho_v)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("TEM final RMS", f"{rms_hist[-1]:.2f}" if rms_hist else "-",
              help="Target ~1.0 means the fit is consistent with the noise.")
    m2.metric("TEM iterations", len(rms_hist))
    _ves_rms_norm = (rms_hist_v[-1] / 0.05) if rms_hist_v else None
    m3.metric("VES final RMS", f"{_ves_rms_norm:.2f}" if _ves_rms_norm is not None else "-",
              help="Normalised by the 5% data error; target ~1.0.")
    m4.metric("VES iterations", len(rms_hist_v))

    show_units = st.toggle(
        "Show interpreted units",
        value=False,
        help="Draw the broad hydrogeological boundaries (cap / saprolite aquifer / "
             "fresh basement) as horizontal markers on the recovered-model panel.",
    )

    # -- Combined recovered models + data fits (cached) ------------------------
    fig = _build_results_fig(
        tuple(thick_r), tuple(np.asarray(rho_r).tolist()),
        tuple(thick_v), tuple(np.asarray(rho_v).tolist()),
        tuple(np.asarray(dbdt_pred).tolist()), tuple(np.asarray(rhoa_pred).tolist()),
        tuple(times), tuple(dbdt_obs), tuple(ab2), tuple(rhoa_obs),
        bool(show_units), is_mobile(),
    )
    st.image(fig, use_container_width=True)

    # -- Interpretation --------------------------------------------------------
    st.subheader("3. Hydrogeological interpretation")


    st.markdown(
        """
        **Reading the model.**

        - A **resistive surface layer** corresponds to the dry lateritic/ferricrete cap.
        - Below it, a **conductive zone** marks the clay-rich saprolite, the water-storing regolith.
        - Resistivity then rises sharply into the **fresh crystalline basement**, marking the base of the productive weathered zone.
        """
    )

    st.markdown(
        """
        **What each method contributed.** The two recovered models (left panel above)
        emphasise different parts of the section:

        - **TEM (blue)** pins down the **conductive saprolite aquifer**, its
          resistivity and its base, because inductive eddy currents concentrate in
          conductors.
        - **VES (orange)** better expresses the **resistive lateritic cap** and the
          rise into **fresh basement**, because galvanic current is forced through
          resistive layers.

        Where the two models **agree**, you can trust the interpretation: each method
        is individually non-unique, so joint agreement is the strongest evidence that
        a feature is real.
        """
    )

    st.warning(
        "Caveats: TEM resolves *conductance* (thickness x conductivity) better "
        "than the two separately, so the aquifer boundaries carry uncertainty. "
        "Low resistivity can also reflect dry, compact clay rather than usable "
        "water, so confirm with a pumping test and, ideally, several soundings."
    )

# -- Quiz: which method resolves which feature ---------------------------------
st.subheader(":violet[Check your intuition: which method resolves what?]", divider="violet")
st.markdown(
    "Decide which method "
    "is **better suited** to resolving it, then check your answers."
)

_FE_QUIZ = [
    {
        "q": "Where should the borehole target be drilled to maximize yield?",
        "options": [
            "Within the lateritic / ferricrete cap",
            "Within the saturated saprolite/saprock interval, roughly 6-36 m, and a few metres into fractured basement",
            "Only in fresh crystalline basement",
        ],
        "answer": "Within the saturated saprolite/saprock interval, roughly 6-36 m, and a few metres into fractured basement",
        "why": "That interval is the conductive regolith aquifer. Targeting it and drilling a few metres into fractured basement maximizes the chance of a productive borehole.",
    },
    {
        "q": "The conductive saprolite aquifer (its resistivity and how deep it goes):",
        "options": ["TEM", "VES", "Equally well"],
        "answer": "TEM",
        "why": "Inductive TEM drives eddy currents that concentrate in conductors, "
               "so the conductive saprolite dominates the decay and is well resolved.",
    },
    {
        "q": "The thin, resistive lateritic / ferricrete cap at the surface:",
        "options": ["TEM", "VES", "Equally well"],
        "answer": "VES",
        "why": "Galvanic VES forces current through resistive layers, so it expresses "
               "the resistive cap more sharply than inductive TEM.",
    },
    {
        "q": "The transition into the resistive fresh basement at depth:",
        "options": ["TEM", "VES", "Neither sees it"],
        "answer": "VES",
        "why": "VES resolves the rise into resistive basement better; TEM mainly "
               "constrains the base of the conductor, not the resistivity beneath it.",
    },
    {
        "q": "When both inversions agree on the aquifer base, that agreement mainly gives you:",
        "options": [
            "A faster inversion",
            "Higher confidence despite each method's non-uniqueness",
            "A greater exploration depth",
        ],
        "answer": "Higher confidence despite each method's non-uniqueness",
        "why": "Each sounding alone is non-unique; two independent methods agreeing on "
               "a feature is strong evidence that it is real.",
    },
]

_fe_user = [
    st.radio(_item["q"], _item["options"], index=None, key=f"fe_quiz_{_i}")
    for _i, _item in enumerate(_FE_QUIZ)
]

if st.button("Check my answers", key="fe_quiz_check"):
    _score = 0
    for _i, _item in enumerate(_FE_QUIZ):
        if _fe_user[_i] == _item["answer"]:
            _score += 1
            st.success(f"Q{_i + 1}: Correct. {_item['why']}")
        elif _fe_user[_i] is None:
            st.warning(
                f"Q{_i + 1}: Not answered. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
        else:
            st.error(
                f"Q{_i + 1}: Not quite. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
    st.metric("Your score", f"{_score} / {len(_FE_QUIZ)}")
    if _score == len(_FE_QUIZ):
        st.balloons()

render_footer()
