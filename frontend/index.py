# -*- coding: utf-8 -*-
"""
app.py (project root)
======================
Single Streamlit entrypoint that combines:

  1. The cinematic landing page (landing.html — Three.js hero, GSAP scroll
     animations, dashboard preview mock) as the home view.
  2. The real live Bank Nifty dashboard (frontend/app.py) — candlestick
     chart, risk/anomaly/news panels, and the RAG chat panel for asking
     questions about anomalies & news.

Navigation between the two is done with a Streamlit query param:
  - no ?view param (or ?view=home)  -> landing page
  - ?view=dashboard                 -> live dashboard

The "Launch Dashboard" buttons in landing.html link to `?view=dashboard`
with target="_blank" (opens a new tab). This is required, not cosmetic:
`components.html` renders the landing page inside a sandboxed iframe, and
Streamlit's iframe sandbox does not include `allow-top-navigation` — so
target="_top" is silently blocked on a normal click (it only ever worked
via the browser's own "open in new tab", which bypasses the sandbox).
`allow-popups` *is* permitted, so opening a new tab works reliably.

The dashboard's "← Home" button lives on the real top-level Streamlit page
(outside the iframe), so it isn't affected by the sandbox and can freely
clear the query param + rerun.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pathlib
import sys

import streamlit as st
import streamlit.components.v1 as components

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="AI Market Surveillance — Bank Nifty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

view = st.query_params.get("view", "home")

# ---------------------------------------------------------------------------
# View: live dashboard
# ---------------------------------------------------------------------------
if view == "dashboard":
    from frontend.style import load_css

    load_css()

    top_left, top_right = st.columns([6, 1])
    with top_left:
        st.caption("AI MARKET SURVEILLANCE")
    with top_right:
        if st.button("← Home", use_container_width=True):
            st.query_params.clear()
            st.rerun()

    from frontend.app import render_dashboard

    render_dashboard()

# ---------------------------------------------------------------------------
# View: landing page (default)
# ---------------------------------------------------------------------------
else:
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            div.block-container {
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
            }
            iframe {
                display: block;
            }
            body {
                background-color: #040506;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    landing_html = (ROOT / "landing.html").read_text(encoding="utf-8")
    components.html(landing_html, height=1000, scrolling=True)
