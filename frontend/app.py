# -*- coding: utf-8 -*-
"""
app.py
======
Streamlit Dashboard for AI-Based Financial Market Surveillance & Risk Intelligence System.

Features:
    - User inputs stock ticker / index symbol (Default: NSE:NIFTYBANK-INDEX).
    - Communicates with FastAPI backend pipeline (with local pipeline fallback).
    - Renders 5-minute interactive Plotly candlestick chart with red anomaly markers.
    - Displays Composite Anomaly Risk Score (CARS), D_info index, and NLP sentiment metrics.
    - Bottom table listing recent detected market anomalies.
"""

import os
import sys
from datetime import datetime
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Page configuration
st.set_page_config(
    page_title="Market Surveillance & Risk Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-critical {
        color: #DC2626;
        font-weight: bold;
    }
    .status-moderate {
        color: #D97706;
        font-weight: bold;
    }
    .status-low {
        color: #16A34A;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Surveillance Control")
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker / Symbol:",
    value="NSE:NIFTYBANK-INDEX",
    help="e.g. NSE:NIFTYBANK-INDEX, AAPL, NIFTYBANK",
)

backend_url = st.sidebar.text_input(
    "Backend API Base URL:",
    value="http://127.0.0.1:8000",
    help="FastAPI backend URL",
)

run_button = st.sidebar.button("Run Live Surveillance Pipeline", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown("**System Architecture Info:**")
st.sidebar.caption("• Model Training: Done ONCE (`model/output/`)")
st.sidebar.caption("• Data Resolution: 5-minute OHLCV candles")
st.sidebar.caption("• Live Pipeline: Model Inference + NLP + CARS Engine")


def fetch_pipeline_data(ticker: str, api_url: str) -> dict:
    """
    Fetches pipeline results from FastAPI backend API.
    Falls back to direct local pipeline invocation if backend server is offline.
    """
    endpoint = f"{api_url.rstrip('/')}/run-pipeline/{ticker}"
    try:
        response = requests.get(endpoint, timeout=12)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.warning(f"Backend API offline ({e}). Running backend pipeline locally...")

    # Fallback to local pipeline runner
    from backend.pipeline import run_live_pipeline
    return run_live_pipeline(ticker=ticker)


# Main Header
st.markdown('<div class="main-title">AI Financial Market Surveillance System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time 5-minute anomaly detection, news NLP sentiment, and CARS risk intelligence</div>',
    unsafe_allow_html=True,
)

# Load data on page load or button click
with st.spinner("Executing 5-minute live surveillance pipeline..."):
    try:
        data = fetch_pipeline_data(ticker_input, backend_url)
    except Exception as err:
        st.error(f"Failed to execute surveillance pipeline: {err}")
        st.stop()

# Extract Results
risk_summary = data.get("risk_summary", {})
nlp_analysis = data.get("nlp_analysis", {})
candles = data.get("candles", [])
recent_anomalies = data.get("recent_anomalies", [])

cars_score = risk_summary.get("cars_risk_score", 0.0)
d_info = risk_summary.get("information_disconnection_index", 0.0)
risk_class = risk_summary.get("risk_classification", "N/A")
sentiment_score = nlp_analysis.get("sentiment_score", 0.0)
relevance_score = nlp_analysis.get("relevance_score", 0.0)

# Top Metrics Bar
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("CARS Risk Score", f"{cars_score} / 100", delta=None)

with col2:
    if "CRITICAL" in risk_class:
        status_color = "🔴"
    elif "MODERATE" in risk_class:
        status_color = "🟠"
    else:
        status_color = "🟢"
    st.metric("Risk Classification", f"{status_color} {risk_class.split(':')[0]}")

with col3:
    st.metric(
        "Info Disconnection (D_info)",
        f"{d_info:.3f}",
        help="Measures market move unexplained by news (1.0 = total disconnection)",
    )

with col4:
    sentiment_label = "Positive" if sentiment_score > 0.05 else ("Negative" if sentiment_score < -0.05 else "Neutral")
    st.metric("NLP News Sentiment", f"{sentiment_score} ({sentiment_label})")

st.markdown("---")

# Section 1: Interactive 5-Minute Candlestick Chart with Anomaly Overlays
st.subheader(f"5-Minute Market Chart & Model Predictions: {data.get('ticker')}")

if candles:
    df_candles = pd.DataFrame(candles)

    # Filter anomaly candles for scatter overlay
    df_anom = df_candles[df_candles["is_anomaly"]]

    # Create Subplot with Candlestick + Volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("OHLC Price & Model Anomaly Markers", "Volume Action"),
        row_width=[0.25, 0.75],
    )

    # OHLC Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_candles["datetime"],
            open=df_candles["open"],
            high=df_candles["high"],
            low=df_candles["low"],
            close=df_candles["close"],
            name="Price (5m)",
        ),
        row=1,
        col=1,
    )

    # Red Markers for Anomaly Candles
    if not df_anom.empty:
        fig.add_trace(
            go.Scatter(
                x=df_anom["datetime"],
                y=df_anom["high"] * 1.001,
                mode="markers",
                marker=dict(symbol="triangle-down", size=11, color="red"),
                name="Isolation Forest Anomaly",
                text=[f"Anomaly Score: {s}" for s in df_anom["raw_anomaly_score"]],
                hoverinfo="x+y+text",
            ),
            row=1,
            col=1,
        )

    # Volume Bars
    colors = ["red" if row["close"] < row["open"] else "green" for _, row in df_candles.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df_candles["datetime"],
            y=df_candles["volume"],
            marker_color=colors,
            name="Volume",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

# Section 2: Recent Anomalies Table
st.subheader("Recent Market Anomalies Detected")
st.caption("Filtered list of 5-minute bars flagged as anomalous by the pre-trained Isolation Forest model")

if recent_anomalies:
    df_recent = pd.DataFrame(recent_anomalies)
    df_recent.columns = [
        "Timestamp",
        "Close Price",
        "Volume",
        "Volume Z-Score",
        "RSI (14)",
        "Raw Anomaly Score",
    ]
    st.dataframe(df_recent, use_container_width=True)
else:
    st.info("No market anomalies flagged in the current lookback window.")

# Section 3: NLP & Risk Intelligence Details
st.markdown("---")
with st.expander("View Risk Engine & NLP Headline Breakdown"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**NLP News Engine Output**")
        st.json(nlp_analysis)
    with col_b:
        st.markdown("**CARS Risk Intelligence Output**")
        st.json(risk_summary)

st.caption(f"Last updated: {data.get('last_updated', 'N/A')} | Mode: 5-minute live prediction via saved model artifacts.")
