import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

plt.rcParams.update({
    "font.size":       16,
    "axes.labelsize":  16,
    "axes.titlesize":  16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

st.set_page_config(
    page_title="TEM vs VES",
    page_icon="🌍",
)

# ── Path setup so the intro plot can call the pyTEM forward model ─────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_APP_DIR = os.path.abspath(os.path.dirname(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from _shared import render_footer

@st.cache_data(show_spinner=False)
def _intro_tem_curve(thick, rho, tx_r, times):
    """Cached TEM forward response for the landing-page teaser plot."""
    from pytem import fwd_circle_central
    return -fwd_circle_central(list(thick), list(rho),
                               tx_radius=tx_r, times=np.array(times))


def _figure_placeholder(title):
    """Render a dashed placeholder box marking where a figure should go.

    Replace each of these with a real image via ``st.image(...)`` once the
    sketches are drawn.
    """
    st.markdown(
        f"""
<div style="border:2px dashed #8a8a8a; border-radius:8px; padding:1.4rem 1.2rem;
            text-align:center; background-color:rgba(128,128,128,0.08);
            margin:0.6rem 0;">
  <div style="font-size:1.8rem; line-height:1;">🖼️</div>
  <div style="font-weight:700; margin:0.35rem 0; color:#9aa7b4;">
    [FIGURE PLACEHOLDER] {title}
  </div>
""",
        unsafe_allow_html=True,
    )


st.header(":blue[Geophysical methods for groundwater investigation]")

st.markdown(
    r"""
    Deciding where to drill a borehole is one of the most consequential (and costly) steps
    in a groundwater investigation. Drilling is expensive, and a dry or unproductive well
    wastes resources and delays access to water. Geophysical methods allow imaging of the
    subsurface **before drilling**, mapping the depth, thickness, and physical properties of
    aquifer materials across a site.

    A key physical property in geophysical mapping of groundwater is **electrical resistivity**.
    Resistivity varies with lithology, saturation, and, to some extent, water quality. Saturated sands and
    gravels are moderately conductive, clay-rich layers are highly conductive, basement rock
    is resistive, and saline water dramatically lowers resistivity. By mapping resistivity
    with depth, we can identify potential aquifer horizons, estimate depths to the water table,
    and detect saline intrusion. 
    
    Two methods that  are sensitive to subsurface resistivity are the **TEM** (transient electromagnetic) 
    and **VES** (vertical electrical sounding) methods: both map the subsurface resistivity structure, 
    but they work in fundamentally different ways.
    """
)

# ── Aim, motivation, and target groups ────────────────────────────────────────
st.info(
    """
    **What this app is for?** 
    An interactive, hands-on introduction to how TEM and
    VES soundings "see" the subsurface and how we turn measured curves into
    resistivity-versus-depth models.

    **Who it is for?**
    - 🎓 **Students** meeting near-surface geophysics for the first time.
    - 🛠️ **Applied geophysicists & hydrogeologists** wanting quick intuition for survey design and interpretation.
    - 👩‍🏫 **Educators** looking for a live demo of forward modelling, sensitivity, and inversion.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
<div style="background-color:#1a3a5c; border-left:6px solid steelblue;
            border-radius:6px; padding:1rem 1.2rem;">
  <h4 style="color:steelblue; margin-top:0;">🧲 TEM: Transient Electromagnetic Method</h4>
  <p>A transmitter (Tx) loop carries a steady current that is abruptly switched off.
  The collapsing magnetic field induces <b>eddy currents</b> that diffuse through
  the earth and generate a secondary magnetic field. A receiver (Rx) coil records the 
                decaying secondary field dB/dt.</p>
  <ul style="margin-bottom:0;">
    <li><b>Source:</b> inductive, no ground contact needed</li>
    <li><b>Depth proxy:</b> time (early = shallow, late = deep)</li>
    <li><b>Best for:</b> conductive targets (clay, saline water)</li>
    <li><b>Data:</b> transient decay over time (V/m²)</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    with open(os.path.join(_APP_DIR, "data", "TEM.png"), "rb") as _f:
        st.image(_f.read(), use_container_width=True)
   

with col2:
    st.markdown("""
<div style="background-color:#3d2200; border-left:6px solid darkorange;
            border-radius:6px; padding:1rem 1.2rem;">
  <h4 style="color:darkorange; margin-top:0;">⚡️ VES: Vertical Electrical Sounding Method</h4>
  <p>Current is injected between two electrodes (A, B), and the resultant voltage is measured between a second pair (M, N). 
                Increasing the electrode spacing between A and B drives current deeper and samples greater depths.</p>
  <ul style="margin-bottom:0;">
    <li><b>Source:</b> galvanic, electrodes must contact the ground</li>
    <li><b>Depth proxy:</b> AB electrode spacing (small = shallow, large = deep)</li>
    <li><b>Best for:</b> resistive layers and general stratigraphy</li>
    <li><b>Data:</b> apparent resistivity curve ρ<sub>a</sub>(AB/2)</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    with open(os.path.join(_APP_DIR, "data", "VES.png"), "rb") as _f:
        st.image(_f.read(), use_container_width=True)
   

st.divider()

st.subheader("Comparison")
st.markdown(
    """
    **Use TEM when you want:**
    - Rapid coverage over a larger area without ground contact.
    - Strong sensitivity to conductive layers, saline water, and clay.
    - Deeper investigation with a loop-sized footprint.

    **Use VES when you want:**
    - Direct galvanic current injection into the ground.
    - A classic resistivity sounding that responds well to resistive layers.
    - A simple, low-cost setup for near-surface stratigraphy.

    | Property | 🧲 TEM | ⚡️ VES |
    |---|---|---|
    | How it works | Inductive loop transmits a current pulse | Electrodes inject current directly into the ground |
    | Main control on depth | Time after turn-off | Electrode spacing (AB/2) |
    | Ground contact | Not required | Required |
    | Best sensitivity | Conductive targets and groundwater salinity | Resistive layers and layered stratigraphy |
    | Main limitation | Urban cultural noise and coupling to nearby infrastructure can contaminate signal | Thin layers can suffer from equivalence |
    | Survey footprint | Loop size | Electrode spacing |
    | Typical output | Transient decay over time | Apparent resistivity curve ρ<sub>a</sub>(AB/2) |
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Self-assessment quiz: which method resolves what? ─────────────────────────
st.subheader(":violet[Check your intuition: TEM vs VES]", divider="violet")
st.markdown(
    "Resistive and conductive layers are **not** equally easy to resolve. "
    "Think about a layered earth with a **resistive basement** and a **buried "
    "conductor** (a clay-rich, water-saturated aquifer), then test yourself below."
)

_QUIZ = [
    {
        "q": "1. To map a **buried conductor** (clay-rich, saturated aquifer), which "
             "method is inherently more sensitive?",
        "options": ["TEM", "VES", "Both are equally sensitive"],
        "answer": "TEM",
        "why": "TEM induces eddy currents that diffuse and **persist in conductive** "
               "material, so a conductor produces a strong, long-lived dB/dt signal. "
               "TEM is the method of choice for conductive targets.",
    },
    {
        "q": "2. To resolve a **resistive earth** (fresh-basement high or dry gravel), "
             "which method is more reliable?",
        "options": ["TEM", "VES", "Neither can see resistors"],
        "answer": "VES",
        "why": "Galvanic VES forces current through the ground, so a resistive layer "
               "**raises the measured voltage** and stands out in the apparent-"
               "resistivity curve. TEM generates almost no eddy currents in a resistor, "
               "so it is comparatively blind to it.",
    },
    {
        "q": "3. Why does TEM struggle to pin down a thin **resistive** layer?",
        "options": [
            "Resistors carry weak eddy currents, so they add little TEM signal",
            "Resistive layers are always too deep for TEM",
            "TEM cannot measure time",
        ],
        "answer": "Resistors carry weak eddy currents, so they add little TEM signal",
        "why": "TEM responds to induced currents, which are weak in resistive material. "
               "A thin resistor barely changes the decay, so it is poorly resolved "
               "(an equivalence-type ambiguity).",
    },
]

_user = [
    st.radio(_item["q"], _item["options"], index=None, key=f"quiz_{_i}")
    for _i, _item in enumerate(_QUIZ)
]

if st.button("Check my answers", key="quiz_check"):
    _score = 0
    for _i, _item in enumerate(_QUIZ):
        if _user[_i] == _item["answer"]:
            _score += 1
            st.success(f"Q{_i + 1}: Correct. {_item['why']}")
        elif _user[_i] is None:
            st.warning(
                f"Q{_i + 1}: Not answered. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
        else:
            st.error(
                f"Q{_i + 1}: Not quite. Correct answer: "
                f"**{_item['answer']}**. {_item['why']}"
            )
    st.metric("Your score", f"{_score} / {len(_QUIZ)}")
    if _score == len(_QUIZ):
        st.balloons()

st.divider()

st.subheader(':blue[Module overview]', divider="blue")

st.markdown(
    """
    :blue[The module is organised as follows:]
    - 📈 **Forward Modeling ▶️**: Build a layered earth model and compute the predicted
      transient decay (TEM) and apparent resistivity curve (VES) in real time.
    - 📊 **Jacobian & Sensitivity ▶️**: Explore which data points are sensitive to which
      layers; compare TEM and VES sensitivity side by side.
    - 🎯 **Inversion ▶️**: Run a synthetic inversion for TEM and VES
      and inspect how well each recovers the true model.
    - 👉 **About**: References and acknowledgements.
    """
)

render_footer()


