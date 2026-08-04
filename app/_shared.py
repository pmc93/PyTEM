"""Shared UI helpers for the pyTEM Streamlit app.

Edit the constants below in one place; they appear in the footer of every page.
"""

import datetime
import re

import streamlit as st

# --- Edit these once; they propagate to every page footer --------------------
APP_AUTHORS = " "
APP_LICENSE = "see repository LICENSE"   # TODO: set once a license is chosen
#APP_REPO_URL = "https://github.com/pmc93/pyTEM"

_MOBILE_UA = re.compile(
    r"Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile",
    re.IGNORECASE,
)


def is_mobile():
    """Return True if the request appears to come from a phone/tablet browser.

    Uses the User-Agent header (available via ``st.context`` on Streamlit
    1.37+). Falls back to False when the header cannot be read, so pages
    always have a safe desktop default.
    """
    try:
        ua = st.context.headers.get("User-Agent", "")
    except Exception:
        return False
    return bool(_MOBILE_UA.search(ua))


def render_footer():
    """Render a consistent license / author footer at the bottom of a page."""
    st.divider()
    st.caption(
        f" "
        f"© {datetime.date.today().year} {APP_AUTHORS}  ·  "
        f"License: {APP_LICENSE}  ·  "
        #f"[Source]({APP_REPO_URL})  \n"
        "For teaching and research purposes. Synthetic examples are illustrative, not field surveys."
    )
