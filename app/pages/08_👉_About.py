import os
import sys

import streamlit as st

# -- Path setup ----------------------------------------------------------------
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from _shared import render_footer

st.title("👉 About")
st.header(":blue[Background, references, and acknowledgements]")

st.subheader(":blue-background[PyTEM]", divider="blue")
st.markdown(
    """
    This app is built on top of **PyTEM**, an open-source Python package for
    1-D layered-earth Time-Domain Electromagnetic (TEM) forward modelling and inversion.
    PyTEM is available on GitHub:.

    The core of PyTEM implements:
    - 1-D forward modelling via Wait's upward TE recursion and digital linear filter (DLF) transforms
    - Four loop geometries (circular/square, central/offset) and three compute backends (NumPy, Numba, CuPy)
    - Regularised Gauss-Newton inversion with an analytical Jacobian
    - Instrument system filters, transmitter waveform convolution, and IP models (Pelton, Cole-Cole, MPA)
    """
)

st.subheader(":blue-background[VES module]", divider="blue")
st.markdown(
    """
    The VES (vertical electrical sounding) functionality is bundled within the PyTEM repository
    as a self-contained subpackage (`ves/`). It is based on the
    [PyVES library](https://github.com/asidosaputra/PyVES) by Asido Saputra, which implements
    1-D DC resistivity forward modelling using digital linear filter coefficients
    (Guptasarma & Singh 1997) and Levenberg-Marquardt inversion.
    The version included here replaces the inversion with regularised Gauss-Newton and
    has been extended to integrate with the PyTEM modelling and app framework.
    """
)

st.subheader(":blue-background[References]", divider="blue")
st.markdown(
    """
    **TEM forward modelling:**
    - Wait, J. R. (1954). Mutual electromagnetic coupling of loops over a homogeneous ground.
      *Geophysics*, 19(2), 290-296.
    - Key, K. (2009). 1D inversion of multicomponent, multifrequency marine CSEM data.
      *Geophysics*, 74(2), F9-F20.
    - Ward, S. H., & Hohmann, G. W. (1988). Electromagnetic theory for geophysical applications.
      In M. N. Nabighian (ed.), *Electromagnetic Methods in Applied Geophysics*, Vol. 1. SEG.

    **Inversion:**
    - Constable, S. C., Parker, R. L., & Constable, C. G. (1987). Occam's inversion:
      A practical algorithm for generating smooth models from electromagnetic sounding data.
      *Geophysics*, 52(3), 289-300.

    **VES forward modelling:**
    - Guptasarma, D., & Singh, B. (1997). New digital linear filters for Hankel J0 and J1 transforms.
      *Geophysical Prospecting*, 45(5), 745-762.

    """
)

st.subheader(":blue-background[Acknowledgements]", divider="blue")
st.markdown(
    """
    Thanks to **Thomas Reimann** (TU Dresden) for reviewing this app and providing
    valuable feedback that shaped its structure and teaching narrative.
    """
)

render_footer()