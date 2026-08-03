# -*- coding: utf-8 -*-
"""Dashboard CSS, kept out of app.py."""

import streamlit as st

CSS = """
<style>
.stApp {
    background-color: #0d0d0d;
    color: #f5f5f5;
}
[data-testid="stMetricValue"] {
    color: #ffffff;
}
[data-testid="stMetricLabel"] {
    color: #aaaaaa;
}
.news-item {
    border-bottom: 1px solid #333;
    padding: 8px 0;
    font-size: 0.82rem;
    line-height: 1.35;
}
div[data-testid="stSidebar"] {
    background-color: #111111;
}
</style>
"""


def load_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
