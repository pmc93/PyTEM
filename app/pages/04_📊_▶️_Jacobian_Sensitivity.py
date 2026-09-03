import os
import sys
import io

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
# -- Matplotlib font sizes (mobile-friendly) --------------------------
plt.rcParams.update({
    "font.size":       16,
    "axes.labelsize":  16,
    "axes.titlesize":  16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

import streamlit as st

# -- Path setup ----------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from pytem import fwd_circle_central, getJ_ana
from ves import forward as ves_forward, jacobian as ves_jacobian
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

# -- Constants -----------------------------------------------------------------
_N_DATA = 20          # fixed number of data points for both methods
_RHO    = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


# -- Shared utilities ----------------------------------------------------------
def _model_ui(n):
    h_out, r_out = [], []
    defaults_rho = [100, 20, 300, 100, 50, 200]
    defaults_h   = [10, 30, 20, 15, 40]
    col_h, col_r = st.columns(2)
    with col_h:
        st.markdown("**Thickness [m]**")
        for i in range(n):
            is_hs = i == n - 1
            layer_lbl = f"Layer {i+1}" if not is_hs else "Half-space"
            if not is_hs:
                h_def = defaults_h[i] if i < len(defaults_h) else 20
                h_out.append(float(st.slider(layer_lbl, 1, 500, h_def,
                                             key=f"jac_h{i}")))
            else:
                st.caption(f"{layer_lbl}: extends to ∞")
    with col_r:
        st.markdown("**Resistivity [Ohm.m]**")
        for i in range(n):
            layer_lbl = f"Layer {i+1}" if i < n - 1 else "Half-space"
            rho_def = defaults_rho[i] if i < len(defaults_rho) else 100
            rho_def = min(_RHO, key=lambda x: abs(x - rho_def))
            r_out.append(float(st.select_slider(layer_lbl, _RHO, value=rho_def,
                                                key=f"jac_r{i}")))
    return h_out, r_out


# -- Cached computations -------------------------------------------------------
@st.cache_data(show_spinner=False)
def _tem_jac(h_t, log_rho_t, tx_r, times_t):
    h       = list(h_t)
    log_rho = np.array(log_rho_t)
    times   = np.array(times_t)
    dbdt = fwd_circle_central(h, np.exp(log_rho).tolist(), tx_radius=tx_r, times=times)
    J    = getJ_ana(h, log_rho, tx_r, times,
                    geometry="circle_central",
                    hankel_filter="key_101", fourier_filter="key_101")
    return -dbdt, J


@st.cache_data(show_spinner=False)
def _ves_jac(ab2_t, rho_t, h_t, filt):
    ab2   = np.array(ab2_t)
    rho   = np.array(rho_t)
    h     = np.array(h_t)
    rhoap = ves_forward(ab2, rho, h, filt)
    J     = ves_jacobian(ab2, rho, h, filt)
    return rhoap, J


# -- Cached figure builders (rebuilt only when their inputs change) ------------
@st.cache_data(show_spinner=False)
def _build_jac_tem_fig(times, dbdt, J_tem, layer_lbls, n_data, mobile):
    times = np.asarray(times)
    dbdt = np.asarray(dbdt)
    J_tem = np.asarray(J_tem)
    layer_lbls = list(layer_lbls)
    N = len(layer_lbls)

    if mobile:
        fig_tem = Figure(figsize=(8, 12))
        axes_tem = fig_tem.subplots(3, 1)
        fig_tem.subplots_adjust(hspace=0.45)
    else:
        fig_tem = Figure(figsize=(18, 6))
        axes_tem = fig_tem.subplots(1, 3)
        fig_tem.subplots_adjust(wspace=0.35)

    ax = axes_tem[0]
    ax.loglog(times, dbdt, "o-", color="steelblue", ms=5, lw=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax.set_title("TEM - decay curve")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    ax = axes_tem[1]
    vmax = max(np.abs(J_tem).max(), 1e-9)
    im = ax.imshow(J_tem, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N))
    ax.set_xticklabels(layer_lbls)
    t_ticks = list(range(0, n_data, max(1, n_data // 5)))
    ax.set_yticks(t_ticks)
    ax.set_yticklabels([f"{k + 1}" for k in t_ticks])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Data index [-]")
    ax.set_title("TEM Jacobian")
    fig_tem.colorbar(im, ax=ax)

    ax = axes_tem[2]
    ax.bar(layer_lbls, np.linalg.norm(J_tem, axis=0), color="steelblue", alpha=0.8)
    ax.set_ylabel("Layer sensitivity [-]")
    ax.set_title("TEM - sensitivity per layer")
    ax.grid(axis="y", ls="--", alpha=0.4)
    return _fig_png(fig_tem)


@st.cache_data(show_spinner=False)
def _build_jac_ves_fig(ab2, rhoap, J_ves, layer_lbls, n_data, mobile):
    ab2 = np.asarray(ab2)
    rhoap = np.asarray(rhoap)
    J_ves = np.asarray(J_ves)
    layer_lbls = list(layer_lbls)
    N = len(layer_lbls)

    if mobile:
        fig_ves = Figure(figsize=(8, 12))
        axes_ves = fig_ves.subplots(3, 1)
        fig_ves.subplots_adjust(hspace=0.45)
    else:
        fig_ves = Figure(figsize=(18, 6))
        axes_ves = fig_ves.subplots(1, 3)
        fig_ves.subplots_adjust(wspace=0.35)

    ax = axes_ves[0]
    _span_ves = np.log10(rhoap.max()) - np.log10(rhoap.min())
    _ctr_ves  = (np.log10(rhoap.max()) + np.log10(rhoap.min())) / 2
    if _span_ves < 2.5:
        ax.set_ylim(10 ** (_ctr_ves - 1.25), 10 ** (_ctr_ves + 1.25))
    ax.loglog(ab2, rhoap, "o-", color="darkorange", ms=5, lw=1.5)
    ax.set_xlabel("AB/2 [m]")
    ax.set_ylabel("Apparent resistivity [Ohm.m]")
    ax.set_title("VES - sounding curve")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    ax = axes_ves[1]
    vmax = max(np.abs(J_ves).max(), 1e-9)
    im = ax.imshow(J_ves, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N))
    ax.set_xticklabels(layer_lbls)
    ab2_ticks = list(range(0, n_data, max(1, n_data // 5)))
    ax.set_yticks(ab2_ticks)
    ax.set_yticklabels([f"{k + 1}" for k in ab2_ticks])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Data index [-]")
    ax.set_title("VES Jacobian")
    fig_ves.colorbar(im, ax=ax)

    ax = axes_ves[2]
    ax.bar(layer_lbls, np.linalg.norm(J_ves, axis=0), color="darkorange", alpha=0.8)
    ax.set_ylabel("Layer sensitivity [-]")
    ax.set_title("VES - sensitivity per layer")
    ax.grid(axis="y", ls="--", alpha=0.4)
    return _fig_png(fig_ves)


# -- Page ----------------------------------------------------------------------
st.header(":violet[Which data points are sensitive to which layers?]")

st.markdown(
    r"""
    **TEM** measures a transient electromagnetic signal that decays over time.
    Early time gates see shallow structure; late gates see deeper structure.
    The data are plotted as $|\partial B_z / \partial t|$ vs time on a log-log scale.

    **VES** injects current into the ground and measures voltage.
    Increasing electrode spacing AB/2 samples progressively greater depth.
    The data are plotted as apparent resistivity $\rho_a$ vs AB/2 on a log-log scale.

    The **Jacobian** $J_{ij}$ is the sensitivity of data point $i$ to the resistivity of 
    model layer $j$:
    a large $|J_{ij}|$ means that data point carries information about that layer resistivity.
    Red cells mean increasing the layer's resistivity raises the data value; blue means the opposite.
    The **layer sensitivity**
    summarizes total sensitivity for each layer:
    a small bar means the layer is poorly constrained by the data.
    """
)

# -- Earth model ---------------------------------------------------------------
st.subheader(":blue-background[Shared earth model]")
n_layers = int(st.number_input("Number of layers", 2, 6, 3, key="jac_n"))
thick, rho = _model_ui(n_layers)

# -- Settings (collapsible for mobile) ----------------------------------------
with st.expander("TEM & VES survey settings", expanded=False):
    col_tem_s, col_ves_s = st.columns(2)
    with col_tem_s:
        st.markdown("**TEM**")
        tx_side = st.number_input("Tx loop side length [m]", min_value=5, max_value=500, value=40, step=5, key="jac_tem_side")
        tx_r    = float(np.sqrt(tx_side ** 2 / np.pi))
        t_min = st.slider("Earliest gate [log10(s)]", -6.0, -4.0, -5.0, 0.25, key="jac_tem_tmin")
        t_max = st.slider("Latest gate [log10(s)]",   -3.0, -1.0, -2.0, 0.25, key="jac_tem_tmax")
        filt_tem = "key_101"
    with col_ves_s:
        st.markdown("**VES**")
        ab2_min = st.slider("AB/2 minimum [m]",   1,    50,   1,   key="jac_ves_ab2min")
        ab2_max = st.slider("AB/2 maximum [m]",  50,  2000, 300,   key="jac_ves_ab2max")
        filt_ves = "gs11"

# -- Compute -------------------------------------------------------------------
times = np.logspace(t_min, t_max, _N_DATA)
ab2   = np.logspace(np.log10(ab2_min), np.log10(ab2_max), _N_DATA)
log_rho = np.log(np.array(rho))

with st.spinner("Computing Jacobians ..."):
    dbdt, J_tem          = _tem_jac(tuple(thick), tuple(log_rho.tolist()), tx_r, tuple(times.tolist()))
    rhoap, J_ves_raw     = _ves_jac(tuple(ab2.tolist()), tuple(rho), tuple(thick), filt_ves)

N = len(rho)
J_ves = J_ves_raw[:, :N]

# -- Plots: switch between TEM and VES (each stacked for mobile) --------------
layer_lbls = [f"L{i+1}\n{rho[i]:.0f}" for i in range(N)]

tab_tem, tab_ves = st.tabs(["🧲 TEM", "⚡️ VES"])

with tab_tem:
    fig_tem = _build_jac_tem_fig(times, dbdt, J_tem, tuple(layer_lbls), _N_DATA, is_mobile())
    st.image(fig_tem, use_container_width=True)

with tab_ves:
    fig_ves = _build_jac_ves_fig(ab2, rhoap, J_ves, tuple(layer_lbls), _N_DATA, is_mobile())
    st.image(fig_ves, use_container_width=True)

st.caption(
    "Red = increasing resistivity raises the data value; blue = opposite. "
    "Layer sensitivity = total information a layer contributes across all data points."
)

# -- Try it yourself -----------------------------------------------------------
st.divider()
st.subheader(":violet[Try it yourself: explore the sensitivities]", divider="violet")
st.markdown(
    """
    Change the **shared earth model** sliders above and watch how the Jacobian and
    the layer-sensitivity bars respond. Flip between the 🧲 TEM and ⚡️ VES tabs to
    compare the two methods on the same earth.

    1. **Bury a conductor.** Set a middle layer to a low resistivity (e.g. 5 Ohm.m).
       Its TEM layer-sensitivity bar grows: conductors imprint strongly on TEM.
    2. **Bury a resistor.** Now set that same layer to a high resistivity
       (e.g. 2000 Ohm.m). Compare the bars: VES gains sensitivity while TEM barely
       responds, the two methods are complementary.
    3. **Push a layer deep.** Increase the thickness of the upper layers so a target
       sits deeper. Its sensitivity fades in both methods, the data lose leverage
       at depth.
    4. **Make two layers alike.** Give two adjacent layers nearly the same
       resistivity and watch their Jacobian columns look almost identical, the
       fingerprint of equivalence (non-uniqueness).
    """
)

st.markdown("**Now check your understanding**")
col1, col2 = st.columns(2)
with col1:
    qa = st.radio(
        ":red[**Which TEM time gates are most sensitive to deep layers?**]",
        ["Early gates (short times)", "Late gates (long times)",
         "All gates equally", "Middle gates only"],
        index=None, key="jac_q1",
    )
    if qa == "Late gates (long times)":
        st.success("Correct! The electromagnetic signal diffuses deeper over time, so late gates sample deeper structure.")
    elif qa is not None:
        st.error("Think about how the EM field diffuses into the ground over time.")
with col2:
    qb = st.radio(
        ":red[**What does small layer sensitivity mean for a layer?**]",
        ["The layer is very resistive",
         "The layer is poorly constrained by the data",
         "The layer is too thin to detect",
         "The inversion will converge faster"],
        index=None, key="jac_q2",
    )
    if qb == "The layer is poorly constrained by the data":
        st.success("Correct! Small layer sensitivity means few data points respond to that layer, so it cannot be recovered reliably.")
    elif qb is not None:
        st.error("Layer sensitivity measures how much information the data carries about a given layer.")

col3, col4 = st.columns(2)
with col3:
    qc = st.radio(
        ":red[**A buried conductive layer in the TEM Jacobian usually shows...**]",
        ["Large layer sensitivity (strong sensitivity)",
         "Near-zero layer sensitivity",
         "Sensitivity only at the earliest gate",
         "No effect on the data"],
        index=None, key="jac_q3",
    )
    if qc == "Large layer sensitivity (strong sensitivity)":
        st.success("Correct! Conductors carry strong induced currents, so they imprint clearly on the TEM data and are well resolved.")
    elif qc is not None:
        st.error("Recall that TEM is most sensitive to conductive layers.")
with col4:
    qd = st.radio(
        ":red[**Which method gives a resistive layer larger layer sensitivity?**]",
        ["VES", "TEM", "Neither sees it"],
        index=None, key="jac_q4",
    )
    if qd == "VES":
        st.success("Correct! Galvanic VES forces current through resistive layers, so it is far more sensitive to them than inductive TEM.")
    elif qd is not None:
        st.error("Think about which method forces current through a resistor versus inducing it.")

qe = st.radio(
    ":red[**Two adjacent layers have nearly identical Jacobian columns. The inversion will most likely...**]",
    ["Resolve both layers perfectly",
     "Trade one layer off against the other (equivalence)",
     "Diverge and fail",
     "Ignore those data points"],
    index=None, key="jac_q5",
)
if qe == "Trade one layer off against the other (equivalence)":
    st.success("Correct! Collinear sensitivities mean the data cannot separate the two layers, the classic equivalence / non-uniqueness problem.")
elif qe is not None:
    st.error("If two columns carry the same information, the data cannot tell the layers apart.")

qf = st.radio(
    ":red[**A deep layer shows weak sensitivity in BOTH Jacobians. What should you expect?**]",
    ["Its resistivity is tightly constrained",
     "Its resistivity is poorly constrained; expect smoothing and equivalence",
     "The inversion will sharpen that boundary",
     "It dominates the misfit"],
    index=None, key="jac_q6",
)
if qf == "Its resistivity is poorly constrained; expect smoothing and equivalence":
    st.success("Correct! Low sensitivity at depth means little data leverage there, so the recovered model is smooth and non-unique below the resolved zone.")
elif qf is not None:
    st.error("Weak sensitivity means the data carry little information about that layer.")

render_footer()
