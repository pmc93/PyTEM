"""Compare groundwater scenarios and TEM acquisition systems."""

import io
import os
import sys

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import streamlit as st


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from pytem import fwd_circle_central, fwd_circle_offset, invert as tem_invert
from _shared import is_mobile, render_footer


SYSTEMS = {
    "20 x 20 m central loop": ("central", 20.0, 0.0),
    "40 x 40 m central loop": ("central", 40.0, 0.0),
    "200 x 200 m central loop": ("central", 200.0, 0.0),
    "3 x 3 m loop, 10 m offset": ("offset", 3.0, 10.0),
    "1.2 x 1.2 m loop, 10 m offset": ("offset", 1.2, 10.0),
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


@st.cache_data(show_spinner=False)
def _build_resistivity_ranges_figure(mobile):
    materials = (
        ("Crystalline Bedrock\n(Granite/Gneiss)", 500.0, 100000.0),
        ("Dolomite/Limestone", 100.0, 10000.0),
        ("Sandstones and\nConglomerates", 10.0, 1000.0),
        ("Shales", 3.0, 30.0),
        ("Gravels and Sands", 5.0, 500.0),
        ("Clays and Silts", 1.0, 100.0),
        ("Freshwater", 2.0, 20.0),
        ("Salt-\nwater", 0.1, 0.5),
    )
    fig = Figure(figsize=(8, 5.5) if mobile else (12, 5.5), constrained_layout=True)
    ax = fig.subplots()
    ax.set_xscale("log")
    ax.set_xlim(0.05, 200000)
    ax.set_ylim(0, len(materials))
    ax.set_yticks([])
    ax.set_xlabel("Resistivity [Ohm.m]")
    ax.set_title("Representative material resistivity ranges")

    for index, (name, rho_min, rho_max) in enumerate(materials):
        x = np.logspace(np.log10(rho_min), np.log10(rho_max), 500)
        y = np.linspace(index + 0.3, index + 0.7, 2)
        grid_x, grid_y = np.meshgrid(x, y)
        ax.pcolormesh(
            grid_x, grid_y, np.log10(grid_x), shading="auto", cmap="turbo",
            vmin=np.log10(0.01), vmax=np.log10(100000),
        )
        ax.add_patch(Rectangle(
            (rho_min, index + 0.1), rho_max - rho_min, 0.8,
            linewidth=1, edgecolor="black", facecolor="none",
        ))
        ax.text(
            np.sqrt(rho_min * rho_max), index + 0.5, name,
            ha="center", va="center", fontsize=9, fontweight="bold",
        )

    clay_index = 5
    sand_index = 4
    ax.annotate(
        "Unsaturated (Dry)", xy=(80, clay_index + 0.5), xytext=(200, clay_index + 1.2),
        ha="left", va="center", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        "Saturated with\nSalt Water", xy=(6, sand_index + 0.5),
        xytext=(2.5, sand_index), ha="right", va="center", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        "Unsaturated (Dry)", xy=(400, sand_index + 0.5),
        xytext=(1000, sand_index + 1.2), ha="left", va="center", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    def full_number_formatter(value, _position):
        return f"{value:.10f}".rstrip("0").rstrip(".")

    ax.set_xticks([0.1, 1, 10, 100, 1000, 10000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(full_number_formatter))
    ax.grid(True, which="both", axis="x", linestyle=":", linewidth=0.5)
    return _fig_png(fig)


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


@st.cache_data(show_spinner=False)
def _run_smooth_inversion(observed_t, noise_t, times_t, system_name,
                          start_rho, max_depth):
    observed = np.asarray(observed_t)
    noise = np.asarray(noise_t)
    times = np.asarray(times_t)
    geometry, side, offset = SYSTEMS[system_name]
    radius = side / np.sqrt(np.pi)
    layer_bottoms = np.logspace(np.log10(1.0), np.log10(max_depth), 19)
    inversion_thicknesses = np.diff(np.concatenate(([0.0], layer_bottoms)))

    result = tem_invert(
        obs_data=observed,
        thicknesses=inversion_thicknesses,
        log_resistivities=np.log(np.full(20, start_rho)),
        tx_size=radius,
        times=times,
        noise_std=noise,
        alpha_steps=10,
        maxit=25,
        max_noise_frac=0.0,
        transform="dlf",
        hankel_filter="key_101",
        fourier_filter="key_81",
        analytical_j=True,
        geometry="circle_central" if geometry == "central" else "circle_offset",
        rx_x=offset,
        rx_y=0.0,
    )

    recovered = np.asarray(result["resistivities"])
    if geometry == "central":
        predicted = np.abs(fwd_circle_central(
            inversion_thicknesses, recovered, tx_radius=radius, times=times,
            hankel_filter="key_101", fourier_filter="key_81",
        ))
    else:
        predicted = np.abs(fwd_circle_offset(
            inversion_thicknesses, recovered, tx_radius=radius,
            rx_offset=offset, times=times,
            hankel_filter="key_101", fourier_filter="key_81",
        ))
    normalized_rms = float(np.sqrt(np.mean(((predicted - observed) / noise) ** 2)))
    return predicted, inversion_thicknesses, recovered, normalized_rms, len(result["rms_history"])


@st.cache_data(show_spinner=False)
def _build_inversion_figure(times_t, noise_t, result_items_t,
                            true_thicknesses_t, true_rhos_t, max_depth, mobile):
    times = np.asarray(times_t)
    noise = np.asarray(noise_t)
    true_thicknesses = np.asarray(true_thicknesses_t)
    true_rhos = np.asarray(true_rhos_t)

    if mobile:
        fig = Figure(figsize=(8, 12), constrained_layout=True)
        ax_data, ax_model = fig.subplots(2, 1)
    else:
        fig = Figure(figsize=(14, 6), constrained_layout=True)
        ax_data, ax_model = fig.subplots(1, 2)

    for name, observed_t, predicted_t, inversion_thicknesses_t, recovered_t in result_items_t:
        observed = np.asarray(observed_t)
        predicted = np.asarray(predicted_t)
        inversion_thicknesses = np.asarray(inversion_thicknesses_t)
        recovered = np.asarray(recovered_t)
        line = ax_data.loglog(times, predicted, lw=2, label=f"{name}, predicted")[0]
        ax_data.errorbar(times, observed, yerr=noise, fmt="o", color=line.get_color(),
                         ms=3, ecolor=line.get_color(), alpha=0.55, capsize=1)
        recovered_step, recovered_depth = _stair(
            inversion_thicknesses, recovered, max_depth,
        )
        ax_model.semilogx(recovered_step, recovered_depth, color=line.get_color(),
                          lw=2, label=name)
    ax_data.loglog(times, noise, "k--", lw=1.2, label="1-sigma noise floor")
    ax_data.set_xlabel("Time [s]")
    ax_data.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax_data.set_title("Data fit")
    ax_data.grid(True, which="both", ls=":", alpha=0.5)
    ax_data.legend(fontsize=9)

    true_rho_step, true_depth_step = _stair(true_thicknesses, true_rhos, max_depth)
    ax_model.semilogx(true_rho_step, true_depth_step, "k--", lw=2,
                      label="True layered model")
    ax_model.set_ylim(max_depth, 0.0)
    ax_model.set_xlabel("Resistivity [Ohm.m]")
    ax_model.set_ylabel("Depth [m]")
    ax_model.set_title("Smooth inversion")
    ax_model.grid(True, which="both", ls=":", alpha=0.5)
    ax_model.legend(fontsize=9)
    return _fig_png(fig)


st.header(":green[Groundwater scenario testing]")
st.markdown(
    "Compare how aquifer geometry, water salinity, transmitter size, receiver "
    "offset, and instrument noise affect a synthetic TEM sounding. Later "
    "detectable gates generally indicate greater depth sensitivity, but they "
    "are not a formal depth-of-investigation estimate."
)
st.caption(
    "Acknowledgement: Thanks to Fritz Kalwa at TU Dresden for the motivation "
    "and discussions that helped shape the cases presented here."
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
    st.image(_build_resistivity_ranges_figure(is_mobile()), width="stretch")

st.subheader(":blue-background[Survey systems and noise]", divider="blue")
systems = st.multiselect(
    "Systems to compare", list(SYSTEMS),
    default=["40 x 40 m central loop", "3 x 3 m loop, 10 m offset", "1.2 x 1.2 m loop, 10 m offset"],
)
if not systems:
    st.warning("Select at least one survey system.")
    render_footer()
    st.stop()

col_noise, col_points = st.columns(2)
log_b = col_noise.slider("Noise coefficient log10(b)", -14.0, -9.0, -11.5, 0.25,
                         help="One-sigma noise follows b * t^(-1/2).")
n_times = int(col_points.slider("Number of gates", 15, 41, 25, 2))
show_noisy = st.checkbox("Show one noise realization", value=True)

thicknesses, rhos, labels, markers = _build_model(scenario, geometry, resistivity)
times = np.logspace(-5, -2, n_times)
responses = _forward(tuple(thicknesses), tuple(rhos), tuple(systems), tuple(times))
noise = 10.0 ** log_b * times ** -0.5
rng = np.random.default_rng(42)
noisy = {
    name: np.abs(response + rng.normal(0.0, noise))
    for name, response in responses.items()
}
displayed_noisy = noisy if show_noisy else {}

figure = _build_figure(
    tuple(thicknesses), tuple(rhos), tuple(labels), tuple(markers), tuple(times),
    tuple((name, tuple(values)) for name, values in responses.items()), tuple(noise),
    tuple((name, tuple(values)) for name, values in displayed_noisy.items()), is_mobile(),
)
st.image(figure, width="stretch")

st.caption(
    "Square-loop dimensions are represented by equal-area circular transmitters "
    "for fast scenario comparison. The offset presets are a 3 x 3 m transmitter "
    "and a 1.2 x 1.2 m transmitter, both at 10 m offset."
)

st.subheader(":violet-background[Smooth inversion]", divider="violet")
st.markdown(
    "Invert every selected system on the same fixed 20-layer depth grid. The "
    "recovered models are deliberately smooth, so thin aquifers and sharp "
    "water-quality interfaces may appear broadened or suppressed."
)
col_start, col_depth = st.columns(2)
start_rho = col_start.number_input(
    "Starting resistivity [Ohm.m]", 1.0, 5000.0, 100.0, 10.0,
)
max_depth = col_depth.number_input(
    "Maximum model depth [m]", 30.0, 500.0, 150.0, 10.0,
)

inversion_signature = (
    scenario, tuple(thicknesses), tuple(rhos), tuple(systems), tuple(times),
    tuple((name, tuple(noisy[name])) for name in systems),
    tuple(noise), start_rho, max_depth,
)
if st.button("Invert all selected systems", type="primary"):
    inversion_results = []
    with st.spinner("Running regularized smooth inversions..."):
        for system_name in systems:
            result = _run_smooth_inversion(
                tuple(noisy[system_name]), tuple(noise), tuple(times),
                system_name, start_rho, max_depth,
            )
            inversion_results.append((system_name, result))
        st.session_state["scenario_inversion"] = (
            inversion_signature, tuple(inversion_results),
        )

stored_inversion = st.session_state.get("scenario_inversion")
if stored_inversion is not None and stored_inversion[0] == inversion_signature:
    inversion_results = stored_inversion[1]
    metric_columns = st.columns(min(len(inversion_results), 3))
    result_items = []
    for index, (system_name, result) in enumerate(inversion_results):
        predicted, inversion_thicknesses, recovered, normalized_rms, iterations = result
        metric_columns[index % len(metric_columns)].metric(
            system_name, f"RMS {normalized_rms:.2f}", help=f"{iterations} iterations",
        )
        result_items.append((
            system_name, tuple(noisy[system_name]), tuple(predicted),
            tuple(inversion_thicknesses), tuple(recovered),
        ))
    st.caption(
        "Weighted RMS = sqrt(mean(((predicted - observed) / sigma)^2)); "
        "a value near 1 indicates a fit consistent with the specified noise."
    )
    inversion_figure = _build_inversion_figure(
        tuple(times), tuple(noise), tuple(result_items), tuple(thicknesses),
        tuple(rhos), max_depth, is_mobile(),
    )
    st.image(inversion_figure, width="stretch")
else:
    st.info("Press **Invert all selected systems** to compare their recovered models.")

render_footer()