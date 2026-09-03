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
_RHO = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
_VES_FILTER = "gs11"


def _stair(thick, rho, bottom=None, extra=None):
    depths = [0.0] + list(np.cumsum(thick))
    if bottom is None:
        bottom = depths[-1] + (extra if extra is not None else max(depths[-1] * 0.3, 20.0))
    r_s, d_s = [], []
    for i, r in enumerate(rho):
        d_top = depths[i]
        d_bot = depths[i + 1] if i < len(thick) else bottom
        r_s += [r, r]
        d_s += [d_top, d_bot]
    return r_s, d_s


def _nearest_rho(value):
    return min(_RHO, key=lambda x: abs(x - value))


# ── Cached forward calls ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _tem_fwd(h_t, rho_t, r, times_t):
    return -fwd_circle_central(list(h_t), list(rho_t),
                               tx_radius=r, times=np.array(times_t))


@st.cache_data(show_spinner=False)
def _ves_fwd(ab2_t, rho_t, h_t, filt):
    return ves_forward(np.array(ab2_t), np.array(rho_t), np.array(h_t), filt)


# ── Cached figure builders (rebuilt only when the model changes) ──────────────
@st.cache_data(show_spinner=False)
def _build_match_tem_fig(times_t, obs_t, pred_t, h1, rho1, rho2, mobile):
    times = np.asarray(times_t)
    obs_t = np.asarray(obs_t)
    pred_t = np.asarray(pred_t)

    if mobile:
        fig = Figure(figsize=(8, 12))
        ax1, ax2 = fig.subplots(2, 1)
    else:
        fig = Figure(figsize=(14, 6))
        ax1, ax2 = fig.subplots(1, 2)
    ax1.loglog(times, obs_t, "o", color="black", ms=5, label="Data", zorder=3)
    ax1.loglog(times, pred_t, "-", color="steelblue", lw=2, label="Your model", zorder=4)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend()

    rs, ds = _stair([h1], [rho1, rho2])
    ax2.semilogx(rs, ds, color="steelblue", lw=2, label="Your model")
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Resistivity [Ohm.m]")
    ax2.set_ylabel("Depth [m]")
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend()
    fig.tight_layout()
    return _fig_png(fig)


@st.cache_data(show_spinner=False)
def _build_match_ves_fig(ab2_t, obs_v, pred_v, h1, rho1, rho2, mobile):
    ab2 = np.asarray(ab2_t)
    obs_v = np.asarray(obs_v)
    pred_v = np.asarray(pred_v)

    if mobile:
        fig = Figure(figsize=(8, 12))
        ax1, ax2 = fig.subplots(2, 1)
    else:
        fig = Figure(figsize=(14, 6))
        ax1, ax2 = fig.subplots(1, 2)
    ax1.loglog(ab2, obs_v, "o", color="black", ms=5, label="Data", zorder=3)
    ax1.loglog(ab2, pred_v, "-", color="darkorange", lw=2, label="Your model", zorder=4)
    ax1.set_xlabel("AB/2 [m]")
    ax1.set_ylabel("Apparent resistivity [Ohm.m]")
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend()

    rs, ds = _stair([h1], [rho1, rho2])
    ax2.semilogx(rs, ds, color="darkorange", lw=2, label="Your model")
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Resistivity [Ohm.m]")
    ax2.set_ylabel("Depth [m]")
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    ax2.legend()
    fig.tight_layout()
    return _fig_png(fig)


# ── Target-data generators (hidden truth, no added noise) ─────────────────
@st.cache_data(show_spinner=False)
def _tem_target(thick_t, rho_t, tx_r, times_t, b_coeff):
    times = np.array(times_t)
    clean = _tem_fwd(tuple(thick_t), tuple(rho_t), tx_r, times_t)
    sigma = b_coeff * times ** (-0.5)
    obs = clean
    return obs, sigma


@st.cache_data(show_spinner=False)
def _ves_target(thick_t, rho_t, ab2_t, frac):
    clean = _ves_fwd(ab2_t, tuple(rho_t), tuple(thick_t), _VES_FILTER)
    obs = clean
    sigma = frac * obs
    return obs, sigma


def _rms(pred, obs, sigma):
    """RMS misfit (0 = perfect match)."""
    return float(np.sqrt(np.mean(((pred - obs) / sigma) ** 2)))


def _fit_feedback(rms):
    if rms < 0.01:
        st.success(f"Perfect fit (RMS = {rms:.3f}). You found the true model.")
    elif rms < 0.5:
        st.success(f"Excellent fit (RMS = {rms:.2f}). Very close to the true model.")
    elif rms < 2.0:
        st.info(f"Good fit (RMS = {rms:.2f}). Fine-tune the sliders to improve further.")
    elif rms < 8.0:
        st.warning(f"Getting closer (RMS = {rms:.2f}). Which part of the curve still misses the data?")
    else:
        st.error(f"Still far off (RMS = {rms:.2f}). Try changing one parameter at a time.")


# ── Page header ───────────────────────────────────────────────────────────────
st.header(":green[Match the data by hand]")
st.markdown(
    "On the *Forward Modeling* page you built a layered earth and computed the data it "
    "would produce. In the field we have the opposite problem: we measure the data "
    "first, and we want to recover the earth model that created it. That reverse step "
    "is called **inversion**."
)
st.markdown(
    "The best way to understand inversion is to do it yourself. Here **you** are the "
    "inversion. You are handed field-like data (a set of measured points) from a "
    "**hidden two-layer earth** (a layer over a half-space). Adjust the three "
    "model parameters, the two layer resistivities and the layer thickness, until "
    "your predicted curve passes through the measurements. The **misfit** below "
    "scores how close you are."
)
st.markdown(
    "As you work, notice *how* you decide which way to turn each slider: you change a "
    "parameter, see which way the curve moves, and step towards a better fit. A "
    "computer needs that same information in a precise form so it can update the model "
    "on its own. The **next page** measures exactly this, how sensitive the data are to "
    "each parameter, which is what guides an automated **inversion**. This exercise "
    "also reveals why several different models can fit the same data (**non-uniqueness**)."
)

with st.expander("What is the RMS misfit?", expanded=False):
    st.markdown(
        r"""
        The **RMS misfit** measures how far your predicted curve is from the data
        on average, expressed as a fraction of the data values:

        $$
        \mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
        \left(\frac{d_i^{\text{pred}} - d_i^{\text{obs}}}{d_i^{\text{obs}}}\right)^2 }
        $$

        - **RMS = 0** means a perfect match.
        - **RMS < 0.05** is a very good fit.
        - **RMS ≫ 1** means the model is missing real structure in the data.
        """
    )

tab_tem, tab_ves = st.tabs(["🧲 TEM", "⚡️ VES"])

# ═══════════════════════════════════════════════════════════════════════════════
# TEM TAB  (hidden truth: conductive basement, which TEM sees well)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tem:
    st.subheader(":blue-background[TEM - match the decay curve]", divider="blue")

    tx_side = 40.0
    tx_r = float(np.sqrt(tx_side ** 2 / np.pi))
    times = np.logspace(-5.0, -2.0, 25)
    _b = 3e-11  # TEM noise-floor coefficient

    # Hidden true model (a layer over a half-space)
    _TEM_TRUE = dict(h1=30.0, rho1=10.0, rho2=100.0)
    obs_t, sig_t = _tem_target((_TEM_TRUE["h1"],),
                               (_TEM_TRUE["rho1"], _TEM_TRUE["rho2"]),
                               tx_r, tuple(times.tolist()), _b)

    st.info(
        "**Hint:** work from shallow to deep. The **early-time** signal is set by the "
        "top layer; the **late-time tail** by the half-space. Fit the top layer "
        "first, then adjust the half-space. Watch for the conductive layer, which "
        "holds the TEM curve up for longer."
    )

    st.markdown("**Your model** (a layer over a half-space)")
    col_h, col_r = st.columns(2)
    with col_h:
        h1 = float(st.slider("Thickness of layer 1 [m]", 1, 100, 20,
                             key="match_tem_h1",
                             help="How thick the top layer is."))
    with col_r:
        rho1 = float(st.select_slider("Resistivity of layer 1 [Ohm.m]", _RHO, value=50,
                                      key="match_tem_r1",
                                      help="Resistivity of the top layer."))
        rho2 = float(st.select_slider("Resistivity of half-space [Ohm.m]", _RHO, value=200,
                                      key="match_tem_r2",
                                      help="Resistivity of the basement, which extends "
                                           "to infinite depth."))

    st.caption("The plot and misfit update automatically as you move a slider.")

    try:
        pred_t = _tem_fwd((h1,), (rho1, rho2), tx_r, tuple(times.tolist()))
        rms_t = _rms(pred_t, obs_t, sig_t)

        st.metric("RMS misfit", f"{rms_t:.3f}", help="0 = perfect match; lower is better.")
        _fit_feedback(rms_t)

        fig = _build_match_tem_fig(
            tuple(times.tolist()), tuple(np.asarray(obs_t).tolist()),
            tuple(np.asarray(pred_t).tolist()), h1, rho1, rho2,
            is_mobile(),
        )
        st.image(fig, use_container_width=True)

        with st.expander("Reveal the hidden true model"):
            st.markdown(
                f"""
                The data were generated from:

                - **Layer 1:** rho = {_TEM_TRUE['rho1']:.0f} Ohm.m, thickness = {_TEM_TRUE['h1']:.0f} m
                - **Half-space:** rho = {_TEM_TRUE['rho2']:.0f} Ohm.m

                Notice how strongly the curve responds to the **conductive top layer**:
                TEM is very sensitive to conductors, so that layer is well constrained.
                The resistive half-space is harder to pin down on its own.
                """
            )
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Try adjusting the sliders.")

# ═══════════════════════════════════════════════════════════════════════════════
# VES TAB  (hidden truth: resistive basement, which VES sees well)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ves:
    st.subheader(":orange-background[VES - match the apparent-resistivity curve]",
                 divider="orange")

    ab2 = np.logspace(0, np.log10(300), 25)
    _frac = 0.05  # 5% relative error

    _VES_TRUE = dict(h1=20.0, rho1=300.0, rho2=50.0)
    obs_v, sig_v = _ves_target((_VES_TRUE["h1"],),
                               (_VES_TRUE["rho1"], _VES_TRUE["rho2"]),
                               tuple(ab2.tolist()), _frac)

    st.info(
        "**Hint:** work from shallow to deep. **Small AB/2** (left) is set by the top "
        "layer; **large AB/2** (right) by the half-space. Fit the left of the curve "
        "first, then adjust the half-space. Watch for the resistive layer, which lifts "
        "the apparent-resistivity curve."
    )

    st.markdown("**Your model** (a layer over a half-space)")
    col_h, col_r = st.columns(2)
    with col_h:
        h1 = float(st.slider("Thickness of layer 1 [m]", 1, 100, 30,
                             key="match_ves_h1",
                             help="How thick the top layer is."))
    with col_r:
        rho1 = float(st.select_slider("Resistivity of layer 1 [Ohm.m]", _RHO, value=100,
                                      key="match_ves_r1",
                                      help="Resistivity of the top layer."))
        rho2 = float(st.select_slider("Resistivity of half-space [Ohm.m]", _RHO, value=100,
                                      key="match_ves_r2",
                                      help="Resistivity of the basement, which extends "
                                           "to infinite depth."))

    st.caption("The plot and misfit update automatically as you move a slider.")

    try:
        pred_v = _ves_fwd(tuple(ab2.tolist()), (rho1, rho2), (h1,), _VES_FILTER)
        rms_v = _rms(pred_v, obs_v, sig_v)

        st.metric("RMS misfit", f"{rms_v:.3f}", help="0 = perfect match; lower is better.")
        _fit_feedback(rms_v)

        fig = _build_match_ves_fig(
            tuple(ab2.tolist()), tuple(np.asarray(obs_v).tolist()),
            tuple(np.asarray(pred_v).tolist()), h1, rho1, rho2,
            is_mobile(),
        )
        st.image(fig, use_container_width=True)

        with st.expander("Reveal the hidden true model"):
            st.markdown(
                f"""
                The data were generated from:

                - **Layer 1:** rho = {_VES_TRUE['rho1']:.0f} Ohm.m, thickness = {_VES_TRUE['h1']:.0f} m
                - **Half-space:** rho = {_VES_TRUE['rho2']:.0f} Ohm.m

                VES forces current through the **resistive top layer**, so the rising
                part of the curve constrains it well. The complementary strength to TEM.
                """
            )
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Try adjusting the sliders.")

# ═══════════════════════════════════════════════════════════════════════════════# The real earth is more complex
# ═════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader(":violet[The real earth is more complicated]", divider="violet")
st.markdown(
    """
    Matching even a **two-layer** model by hand is already fiddly, and you probably
    noticed that several settings fit almost equally well. Now consider the catch: the
    real subsurface is **not** a tidy stack of a few uniform layers.

    A handful of sliders cannot represent that, and turning every knob by hand quickly
    becomes impossible once there are many layers. That is why real interpretation uses
    an **automated inversion** (the next page): the computer represents the ground with
    *many thin layers* and adjusts them all at once to fit the data, while a
    **regularisation** (smoothing) term keeps the result geologically reasonable.

    But automation does not remove the deeper problem. Because the data constrain only
    certain **combinations** of parameters, many different models still fit the same
    measurements, this is **non-uniqueness**, discussed next. The goal is therefore not
    a single "true" model, but the range of models the data allow, ideally narrowed by
    combining complementary methods such as TEM and VES.
    """
)

render_footer()

