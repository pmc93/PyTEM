"""Compare groundwater scenarios and TEM acquisition systems."""

import io
import os
import sys

import numpy as np
from matplotlib.figure import Figure
import streamlit as st


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from pytem import fwd_circle_central, fwd_circle_offset
from _shared import is_mobile, render_footer


SYSTEMS = {
    "20 x 20 m central loop": ("central", 20.0, 0.0),
    "40 x 40 m central loop": ("central", 40.0, 0.0),
    "200 x 200 m central loop": ("central", 200.0, 0.0),
    "3 x 3 m loop, 10 m offset": ("offset", 3.0, 10.0),
    "3 x 3 m loop, 1.2 m offset": ("offset", 3.0, 1.2),
}

SCENARIOS = (
    "Confined aquifer",
    "Thin confined aquifer",
    "Unconfined aquifer",
    "Unconfined aquifer with brackish water",
    "Unconfined aquifer with saline water",
)


def _fig_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    return buffer.getvalue()


def _stair(thicknesses, resistivities, bottom):
    interfaces = np.concatenate(([0.0], np.cumsum(thicknesses), [bottom]))
    rho_step = np.repeat(resistivities, 2)
    depth_step = np.column_stack((interfaces[:-1], interfaces[1:])).ravel()
    return rho_step, depth_step


def _build_model(scenario, geometry, resistivity):
    clay = resistivity["Clay"]
    fresh = resistivity["Fresh saturated sand"]
    dry = resistivity["Unsaturated sand"]
    bedrock = resistivity["Bedrock"]

    if scenario == "Confined aquifer":
        thicknesses = [geometry["Upper clay"], geometry["Aquifer"]]
        rhos = [clay, fresh, bedrock]
        labels = ["Clay", "Fresh-water sand", "Bedrock"]
        markers = [(5.0, "Piezometric level")]
    elif scenario == "Thin confined aquifer":
        thicknesses = [
            geometry["Upper clay"], geometry["Aquifer"], geometry["Lower clay"]
        ]
        rhos = [clay, fresh, clay, bedrock]
        labels = ["Clay", "Fresh-water sand", "Clay", "Bedrock"]
        markers = [(5.0, "Piezometric level")]
    else:
        water_table = geometry["Water table"]
        interface = geometry.get("Water interface")
        sand_base = geometry["Sand base"]
        clay_thickness = geometry["Clay"]

        boundaries = [water_table]
        rhos = [dry]
        labels = ["Unsaturated sand"]
        markers = [(water_table, "Water table")]
        if interface is not None:
            boundaries.append(interface)
            rhos.append(fresh)
            labels.append("Fresh-water sand")
            water_name = "Brackish-water sand" if "brackish" in scenario.lower() else "Saline-water sand"
            water_key = "Brackish saturated sand" if "brackish" in scenario.lower() else "Saline saturated sand"
            rhos.append(resistivity[water_key])
            labels.append(water_name)
            markers.append((interface, "Water-quality interface"))
        else:
            rhos.append(fresh)
            labels.append("Fresh-water sand")

        boundaries.extend([sand_base, sand_base + clay_thickness])
        rhos.extend([clay, bedrock])
        labels.extend(["Clay", "Bedrock"])
        thicknesses = np.diff([0.0] + boundaries).tolist()

    return thicknesses, rhos, labels, markers


@st.cache_data(show_spinner=False)
def _forward(thicknesses_t, resistivities_t, systems_t, times_t):
    thicknesses = list(thicknesses_t)
    resistivities = list(resistivities_t)
    times = np.asarray(times_t)
    responses = {}
    for name in systems_t:
        geometry, side, offset = SYSTEMS[name]
        radius = side / np.sqrt(np.pi)
        if geometry == "central":
            response = fwd_circle_central(
                thicknesses, resistivities, tx_radius=radius, times=times,
                hankel_filter="key_101", fourier_filter="key_81",
            )
        else:
            response = fwd_circle_offset(
                thicknesses, resistivities, tx_radius=radius,
                rx_offset=offset, times=times,
                hankel_filter="key_101", fourier_filter="key_81",
            )
        responses[name] = np.abs(response)
    return responses


@st.cache_data(show_spinner=False)
def _build_figure(thicknesses_t, resistivities_t, labels_t, markers_t,
                  times_t, response_items_t, noise_t, noisy_items_t, mobile):
    thicknesses = np.asarray(thicknesses_t)
    resistivities = np.asarray(resistivities_t)
    times = np.asarray(times_t)
    responses = {name: np.asarray(values) for name, values in response_items_t}
    noisy = {name: np.asarray(values) for name, values in noisy_items_t}
    noise = np.asarray(noise_t)

    if mobile:
        fig = Figure(figsize=(8, 12), constrained_layout=True)
        ax_model, ax_data = fig.subplots(2, 1)
    else:
        fig = Figure(figsize=(14, 6), constrained_layout=True)
        ax_model, ax_data = fig.subplots(1, 2)

    bottom = max(60.0, float(np.sum(thicknesses)) + 30.0)
    rho_step, depth_step = _stair(thicknesses, resistivities, bottom)
    ax_model.semilogx(rho_step, depth_step, color="black", lw=2)
    interfaces = np.concatenate(([0.0], np.cumsum(thicknesses), [bottom]))
    for index, label in enumerate(labels_t):
        middle = 0.5 * (interfaces[index] + interfaces[index + 1])
        ax_model.text(resistivities[index] * 1.08, middle, label, va="center", fontsize=10)
    for depth, label in markers_t:
        ax_model.axhline(depth, color="steelblue", ls="--", lw=1.2)
        ax_model.text(ax_model.get_xlim()[0], depth, f"  {label}", color="steelblue",
                      va="bottom", fontsize=9)
    ax_model.set_ylim(bottom, 0.0)
    ax_model.set_xlabel("Bulk resistivity [Ohm.m]")
    ax_model.set_ylabel("Depth [m]")
    ax_model.set_title("Hydrogeological model")
    ax_model.grid(True, which="both", ls=":", alpha=0.5)

    for name, response in responses.items():
        line = ax_data.loglog(times, response, lw=1.8, label=name)[0]
        if name in noisy:
            ax_data.loglog(times, noisy[name], "o", ms=3, color=line.get_color(), alpha=0.55)
    ax_data.loglog(times, noise, "k--", lw=1.5, label="1-sigma noise floor")
    ax_data.loglog(times, 3.0 * noise, color="0.45", ls=":", lw=1.3, label="SNR = 3")
    ax_data.set_xlabel("Time [s]")
    ax_data.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax_data.set_title("System response and detectability")
    ax_data.grid(True, which="both", ls=":", alpha=0.5)
    ax_data.legend(fontsize=9)
    return _fig_png(fig)


st.header(":green[Groundwater scenario testing]")
st.markdown(
    "Compare how aquifer geometry, water salinity, transmitter size, receiver "
    "offset, and instrument noise affect a synthetic TEM sounding. Later "
    "detectable gates generally indicate greater depth sensitivity, but they "
    "are not a formal depth-of-investigation estimate."
)

scenario = st.selectbox("Hydrogeological scenario", SCENARIOS)

with st.expander("Scenario geometry", expanded=True):
    geometry = {}
    if scenario == "Confined aquifer":
        col1, col2 = st.columns(2)
        geometry["Upper clay"] = col1.number_input("Clay thickness [m]", 1.0, 100.0, 10.0)
        geometry["Aquifer"] = col2.number_input("Sand aquifer thickness [m]", 1.0, 100.0, 20.0)
        st.caption("The 5 m piezometric level is shown for context; it does not split the clay resistivity.")
    elif scenario == "Thin confined aquifer":
        col1, col2, col3 = st.columns(3)
        geometry["Upper clay"] = col1.number_input("Upper clay [m]", 1.0, 100.0, 15.0)
        geometry["Aquifer"] = col2.number_input("Sand aquifer [m]", 0.5, 50.0, 3.0)
        geometry["Lower clay"] = col3.number_input("Lower clay [m]", 1.0, 100.0, 17.0)
        st.caption("The 5 m piezometric level is shown for context; it does not split the clay resistivity.")
    else:
        col1, col2, col3 = st.columns(3)
        geometry["Water table"] = col1.number_input("Groundwater level [m]", 0.5, 50.0, 5.0)
        if "with" in scenario:
            geometry["Water interface"] = col2.number_input("Fresh-water interface [m]", 1.0, 100.0, 15.0)
        else:
            geometry["Water interface"] = None
            col2.metric("Water quality", "Fresh")
        geometry["Sand base"] = col3.number_input("Base of sand [m]", 2.0, 150.0, 20.0)
        geometry["Clay"] = st.number_input("Clay thickness below sand [m]", 1.0, 100.0, 10.0)

    if geometry.get("Water interface") is not None and not (
        geometry["Water table"] < geometry["Water interface"] < geometry["Sand base"]
    ):
        st.error("The water-quality interface must lie below the water table and above the base of sand.")
        st.stop()
    if "Sand base" in geometry and geometry["Water table"] >= geometry["Sand base"]:
        st.error("The groundwater level must lie above the base of sand.")
        st.stop()

with st.expander("Bulk resistivity assumptions"):
    st.caption(
        "Illustrative values only. Brackish sand uses about 16 Ohm.m, consistent "
        "with 2.5 mS/cm pore water and a formation factor near 4. Saline sand uses "
        "2 Ohm.m for pore-water EC above 20 mS/cm. Calibrate these values locally."
    )
    col1, col2, col3 = st.columns(3)
    resistivity = {
        "Unsaturated sand": col1.number_input("Unsaturated sand [Ohm.m]", 1.0, 10000.0, 500.0),
        "Fresh saturated sand": col2.number_input("Fresh saturated sand [Ohm.m]", 1.0, 10000.0, 80.0),
        "Clay": col3.number_input("Clay [Ohm.m]", 1.0, 10000.0, 20.0),
    }
    col4, col5, col6 = st.columns(3)
    resistivity["Brackish saturated sand"] = col4.number_input("Brackish sand [Ohm.m]", 0.1, 1000.0, 16.0)
    resistivity["Saline saturated sand"] = col5.number_input("Saline sand [Ohm.m]", 0.1, 1000.0, 2.0)
    resistivity["Bedrock"] = col6.number_input("Bedrock [Ohm.m]", 1.0, 100000.0, 1000.0)

st.subheader(":blue-background[Survey systems and noise]", divider="blue")
systems = st.multiselect(
    "Systems to compare", list(SYSTEMS),
    default=["40 x 40 m central loop", "3 x 3 m loop, 10 m offset", "3 x 3 m loop, 1.2 m offset"],
)
if not systems:
    st.warning("Select at least one survey system.")
    render_footer()
    st.stop()

col_noise, col_seed, col_points = st.columns(3)
log_b = col_noise.slider("Noise coefficient log10(b)", -14.0, -9.0, -11.5, 0.25,
                         help="One-sigma noise follows b * t^(-1/2).")
seed = int(col_seed.number_input("Noise seed", 0, 10000, 42))
n_times = int(col_points.slider("Number of gates", 15, 41, 25, 2))
show_noisy = st.checkbox("Show one noise realization", value=True)

thicknesses, rhos, labels, markers = _build_model(scenario, geometry, resistivity)
times = np.logspace(-5, -2, n_times)
responses = _forward(tuple(thicknesses), tuple(rhos), tuple(systems), tuple(times))
noise = 10.0 ** log_b * times ** -0.5
rng = np.random.default_rng(seed)
noisy = {
    name: np.abs(response + rng.normal(0.0, noise))
    for name, response in responses.items()
} if show_noisy else {}

figure = _build_figure(
    tuple(thicknesses), tuple(rhos), tuple(labels), tuple(markers), tuple(times),
    tuple((name, tuple(values)) for name, values in responses.items()), tuple(noise),
    tuple((name, tuple(values)) for name, values in noisy.items()), is_mobile(),
)
st.image(figure, width="stretch")

summary = []
for name, response in responses.items():
    snr = response / noise
    detectable = np.flatnonzero(snr >= 3.0)
    summary.append({
        "System": name,
        "Maximum SNR": float(np.max(snr)),
        "Latest gate with SNR >= 3 [ms]": (
            float(times[detectable[-1]] * 1e3) if detectable.size else None
        ),
        "Detectable gates": f"{detectable.size}/{len(times)}",
    })
st.dataframe(summary, hide_index=True, width="stretch")

st.caption(
    "Square-loop dimensions are represented by equal-area circular transmitters "
    "for fast scenario comparison. The offset presets use a 3 x 3 m transmitter "
    "with receiver offsets of 10 m and 1.2 m."
)

render_footer()