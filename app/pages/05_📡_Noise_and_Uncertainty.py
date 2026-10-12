"""Explain TEM/VES noise, data uncertainty (sigma), SNR, and the error-weighted RMS."""

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

from pytem import fwd_circle_central
from _shared import is_mobile, render_footer, tem_sigma


def _fig_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def _clean_decay(times_t, tx_r):
    return -fwd_circle_central([10.0, 40.0], [100.0, 10.0, 200.0], tx_radius=tx_r,
                               times=np.array(times_t))


@st.cache_data(show_spinner=False)
def _build_noise_fig(times_t, clean_t, sigma_floor_t, noisy_t, mobile):
    times = np.asarray(times_t)
    clean = np.asarray(clean_t)
    floor = np.asarray(sigma_floor_t)
    noisy = np.asarray(noisy_t)
    fig = Figure(figsize=(8, 6) if mobile else (11, 6), constrained_layout=True)
    ax = fig.subplots()
    ax.loglog(times, clean, "-", color="steelblue", lw=2, label="Noise-free signal")
    ax.loglog(times, np.abs(noisy), "o", color="black", ms=4, label="Noisy data (|d|)")
    ax.loglog(times, floor, "--", color="black", lw=1.5, label=r"Noise floor $b\,t^{-1/2}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"|dB/dt| [V/m$^2$]")
    ax.set_title("Signal, noise and uncertainty")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    return _fig_png(fig)


st.header(":orange[Noise, uncertainty and error weighting]")
st.markdown(
    "Every measurement is the true signal plus noise, and an inversion needs to know "
    "**how much to trust each data point**. This page explains where TEM noise comes "
    "from, how the app describes data uncertainty with a standard deviation "
    r"$\sigma$, and how that uncertainty defines the misfit the inversion tries to reduce."
)

# ------------------------------------------------------------------------------
st.subheader(":blue-background[1. Sources of noise in TEM]", divider="blue")
st.markdown(
    """
    - **Instrumental (system) noise.** Thermal and electronic noise in the receiver
      coil and amplifier, plus the finite resolution of the digitiser. It is largely
      random ("white") and sets the best case in a quiet environment.
    - **Cultural electromagnetic noise.** Power lines and electrified railways
      (50/60 Hz and harmonics), pipelines, fences, buried cables, vehicles, and other
      metal infrastructure. Often the dominant problem near villages, towns and roads.
    - **Natural / atmospheric noise.** Lightning-generated sferics and slow
      geomagnetic variations. Sferics appear as spikes in individual transients and
      depend on season, time of day and region.
    - **Non-random (systematic) errors.** Loop position and shape, receiver
      calibration, the transmitted current waveform, and the 1-D assumption. They do
      not average out with more stacking, so they are modelled as a *relative* error.
    """
)
st.markdown("**What controls the noise level?**")
st.markdown(
    r"""
    - **Measurement time.** The signal decays rapidly, but the noise decays slowly. At
      late times the signal approaches the noise floor, which limits the depth of
      investigation. Late gates are also wider and average more samples, so the
      random noise in a gate falls roughly as $t^{-1/2}$.
    - **System and geometry.** A larger transmitter moment (loop area x current) gives
      more signal for the same noise. A larger or multi-turn receiver collects more
      signal. Small loops, such as a 1.2 x 1.2 m loop, have weak signals and reach the
      noise floor earlier (see the *Scenario Testing* page).
    - **Stacking and acquisition time.** Averaging $N$ repeated transients reduces
      random noise by about $\sqrt{N}$, but not systematic or cultural noise.
    - **Site.** Quiet rural sites versus industrial areas can differ by orders of
      magnitude in cultural noise.
    """
)
st.caption(
    "Absolute noise levels depend strongly on the instrument and site, so the values "
    "used in this app are illustrative. For reference, the field example uses a floor "
    "of about 1e-11 V/m\u00b2 at 1 ms for a 40 x 40 m loop (normalised to unit current)."
)

# ------------------------------------------------------------------------------
st.subheader(":green-background[2. The error model: \u03c3]", divider="green")
st.markdown(
    r"""
    The app writes the standard deviation of the measurement error of datum $i$ as
    $\sigma_i$ (the same symbol is used on every page). For TEM it combines an
    absolute noise floor with a relative error:

    $$
    \sigma_i = \sqrt{\left(b\,t_i^{-1/2}\right)^2 + \left(\varepsilon\,|d_i|\right)^2}
    $$

    - $b\,t^{-1/2}$ is the **absolute noise floor** (units of V/m²). It dominates at
      late times, where the signal is small.
    - $\varepsilon$ is the **relative error** (e.g. 3 % = 0.03). It covers
      systematic errors and dominates at early times, where the signal is large.
    - For VES the error is purely relative: $\sigma_i = \varepsilon_{\mathrm{VES}}\,|d_i|$.

    Data points with a small $\sigma_i$ are fitted tightly; data points with a large
    $\sigma_i$ have little influence on the model.
    """
)

# ------------------------------------------------------------------------------
st.subheader(":violet-background[3. Signal-to-noise ratio (SNR)]", divider="violet")
st.markdown(
    r"""
    $$\mathrm{SNR}_i = \frac{|d_i|}{\sigma_i}$$

    **SNR = 3** means the signal is three times the noise floor. On the *Scenario
    Testing* page the grey dotted line is $3\sigma$: signals above it (SNR > 3) are
    generally considered detectable, while signals below it become increasingly
    difficult to distinguish from noise. Data close to or below the noise floor
    carry almost no information and may even change sign (negative values cannot be
    plotted on a log axis).
    """
)

# ------------------------------------------------------------------------------
st.subheader(":red-background[4. Try it: the noise floor]", divider="red")
col_a, col_c = st.columns(2)
log_b = col_a.slider("Noise floor log10(b)", -13.0, -9.0, -11.5, 0.25, key="nz_logb",
                     help=r"Absolute noise floor $\sigma = b\,t^{-1/2}$, with t in seconds.")
seed = int(col_c.number_input("Noise realisation (seed)", 0, 999, 42, key="nz_seed"))

tx_r = float(np.sqrt(40.0 ** 2 / np.pi))
times = np.logspace(-5, -2, 30)
clean = _clean_decay(tuple(times.tolist()), tx_r)
b_coeff = 10.0 ** log_b
floor = tem_sigma(times, b_coeff)
noisy = clean + np.random.default_rng(seed).normal(size=times.size) * floor

st.image(
    _build_noise_fig(tuple(times.tolist()), tuple(clean.tolist()), tuple(floor.tolist()),
                     tuple(noisy.tolist()), is_mobile()),
    use_container_width=True,
)
_above = clean > 3.0 * floor
if _above.all():
    st.caption("The whole decay stays above SNR = 3 for this noise floor.")
elif not _above.any():
    st.caption("The whole decay is below SNR = 3 for this noise floor: no usable data.")
else:
    st.caption(
        f"The signal drops below SNR = 3 at about {times[~_above][0] * 1e3:.2g} ms. "
        "Later gates carry little information about the ground."
    )
st.caption(
    "Synthetic data in the inversion step are perturbed with the noise floor only. The "
    "relative error enters only the *assumed* uncertainty used to weight the inversion."
)

# ------------------------------------------------------------------------------
st.subheader(":orange-background[5. Error-weighted misfit (wRMS)]", divider="orange")
st.markdown(
    r"""
    The **inversion** pages use the **error-weighted RMS (wRMS)**, in which every
    residual is divided by its uncertainty (the *Match the Data* exercise uses a
    simpler relative RMS instead):

    $$
    \mathrm{wRMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
    \left(\frac{d_i^{\mathrm{obs}} - d_i^{\mathrm{pred}}}{\sigma_i}\right)^2 }
    $$

    The inversion evaluates this in log-data space, which is equivalent for small
    residuals.

    - **wRMS ≈ 1**: the data are fitted *within the assumed uncertainties* $\sigma_i$.
    - **wRMS ≫ 1**: the model does not explain the data (or $\sigma_i$ is too small).
    - **wRMS ≪ 1**: the model fits noise (or $\sigma_i$ is too large).

    A wRMS of 1 therefore says the fit is consistent with the **assumed error model**,
    not necessarily with the *actual* noise. If $\varepsilon$ or $b$ are set
    unrealistically, a wRMS of 1 can still be reached, but the model is then
    over- or under-interpreted.
    """
)
st.info(
    "**If an inversion fails or does not converge**, check the data and the errors "
    "first: the result depends strongly on both. Very noisy late-time data (noise floor "
    "too high), negative values, or errors that are far too small can all stop the "
    "inversion early. Try a lower noise floor, a larger relative error (e.g. 3-5 %), "
    "or a different starting resistivity."
)

render_footer()
