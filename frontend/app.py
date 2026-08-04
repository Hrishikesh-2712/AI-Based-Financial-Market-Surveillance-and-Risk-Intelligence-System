# -*- coding: utf-8 -*-
"""
app.py
======
Streamlit dashboard for Bank Nifty live surveillance.

Live-mode flow (per architecture):
    model inference (backend, TCN + Isolation Forest)
        + NLP output (nlp_news_engine, live or snapshot)
    -> risk engine (CARS score) -> displayed on this UI.

Each panel polls its own FastAPI endpoint on its own refresh cadence:
    /api/candles  -> candlestick chart with anomaly markers
    /api/risk     -> model inference output + CARS risk score
    /api/news     -> top-ranked news
    /api/snapshot -> aggregate (debug panel)
If the backend is offline the dashboard falls back to a local pipeline run.
"""

from __future__ import annotations

import html
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from style import load_css

# Only set page config / run as a standalone page when this file is the
# actual Streamlit entrypoint (`streamlit run frontend/app.py`). When it's
# imported by the root app.py (landing page + dashboard combined), the root
# app already owns st.set_page_config() and calls render_dashboard() itself.
if __name__ == "__main__":
    st.set_page_config(
        page_title="Bank Nifty Surveillance",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    load_css()

BACKEND_URL = os.getenv("SURVEILLANCE_API_URL", "http://127.0.0.1:5000")
REFRESH_MINUTES = 5


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def _request(method: str, endpoint: str, timeout: int = 200):
    url = f"{BACKEND_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        response = requests.request(method, url, timeout=timeout)
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _local_payload() -> dict:
    """Offline fallback: run the pipeline in-process, cached for the session."""
    if "local_payload" not in st.session_state:
        from backend.pipeline import run_live_pipeline

        st.session_state["local_payload"] = run_live_pipeline()
    return st.session_state["local_payload"]


def _panel(endpoint: str, local_keys) -> dict:
    """Prefer the backend endpoint; fall back to a local pipeline run."""
    data = _request("GET", endpoint)
    if data and "error" not in data:
        return data
    payload = _local_payload()
    return {key: payload.get(key) for key in local_keys if key in payload}


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def build_candle_chart(candles: list) -> go.Figure:
    df = pd.DataFrame(candles)
    if df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Bank Nifty",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    both = df[df["anomaly_level"] == "both"]
    either = df[df["anomaly_level"] == "either"]

    for subset, name, color, size in (
        (both, "Both models", "#ff2222", 12),
        (either, "Either model", "#ff9900", 10),
    ):
        if not subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=subset["datetime"],
                    y=subset["high"] * 1.0015,
                    mode="markers",
                    name=name,
                    marker=dict(
                        symbol="diamond",
                        size=size,
                        color=color,
                        line=dict(width=1, color="#fff"),
                    ),
                    text=subset["hover_text"],
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    up = (df["close"] >= df["open"]).to_numpy()
    fig.add_trace(
        go.Bar(
            x=df["datetime"],
            y=df["volume"],
            name="Volume",
            marker_color=["#26a69a" if u else "#ef5350" for u in up],
            opacity=0.6,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d0d0d",
        plot_bgcolor="#0d0d0d",
        height=620,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02, x=0),
        font=dict(color="#ffffff"),
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(tickformat="%d %b %H:%M", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------


def render_risk(data: dict) -> None:
    live = data.get("live_prediction") or {}
    risk = data.get("risk_summary") or {}
    components = risk.get("components", {})
    with st.container(border=True):
        st.markdown("**Live**")
        st.metric("Hybrid Risk", live.get("final_risk", "N/A"))
        st.metric("Risk Score", f"{risk.get('risk_score', 0)} / 100")
        st.caption(risk.get("risk_classification", ""))
        st.caption(
            f"Rule: {components.get('rule_score', 0):.1f} · "
            f"IF: {components.get('if_risk_score', 0):.1f} · "
            f"TCN: {components.get('tcn_risk_score', 0):.1f} · "
            f"News adj: {components.get('news_adjustment', 0):+.0f}"
        )
        if risk.get("news_reason"):
            st.caption(risk["news_reason"])
        st.caption(
            f"TCN: {'ANOMALY' if live.get('tcn_anomaly') else 'normal'} "
            f"(err {live.get('tcn_error', 0):.4f}) · "
            f"IF: {'ANOMALY' if live.get('if_anomaly') else 'normal'} "
            f"(score {live.get('if_score', 0):.4f})"
        )


def render_anomaly(data: dict) -> None:
    with st.container(border=True):
        st.markdown("**% Anomaly**")
        st.metric("Session candles flagged", f"{data.get('anomaly_pct', 0)}%")
        st.caption("Red = TCN + IF · Orange = either model")


def render_news(data: dict) -> None:
    nlp = data.get("nlp_analysis") or {}
    top_news = nlp.get("top_news", [])[:5]
    with st.container(border=True):
        st.markdown("**Top 5 News**")
        if not top_news:
            st.caption("No NLP output yet. Run nlp_news_engine/run_every_5min.py")
        else:
            for item in top_news:
                headline = html.escape(item.get("headline", ""))[:160]
                link = item.get("link", "")
                headline_html = (
                    f'<a href="{html.escape(link, quote=True)}" target="_blank" '
                    f'style="color:#4fc3f7;text-decoration:none">{headline}</a>'
                    if link
                    else headline
                )
                st.markdown(
                    f'<div class="news-item">'
                    f'<strong>[{item.get("composite_score", 0):.2f}]</strong> {headline_html}<br>'
                    f'<span style="color:#888">sentiment '
                    f'{item.get("sentiment_signed", 0):+.2f} · '
                    f'{html.escape(item.get("category", ""))} · '
                    f'{html.escape(item.get("company", ""))} · '
                    f'{html.escape(item.get("source", ""))}</span></div>',
                    unsafe_allow_html=True,
                )
        if nlp.get("generated_at"):
            st.caption(f"News as of {nlp['generated_at']} · {nlp.get('source', '')}")


def render_debug(data: dict) -> None:
    with st.expander("Recent anomalies & debug"):
        recent = data.get("recent_anomalies", [])
        if recent:
            st.dataframe(pd.DataFrame(recent), width="stretch")
        st.json(
            {
                "live_prediction": data.get("live_prediction"),
                "risk_summary": data.get("risk_summary"),
                "risk_components": data.get("risk_summary", {}).get("components"),
                "nlp_source": data.get("nlp_analysis", {}).get("source"),
            }
        )


# ---------------------------------------------------------------------------
# Auto-refreshing panels
# ---------------------------------------------------------------------------


@st.fragment(run_every=timedelta(minutes=REFRESH_MINUTES))
def chart_panel():
    data = _panel("api/candles", ["candles", "session_date", "data_source", "last_updated"])
    candles = data.get("candles", [])
    if not candles:
        st.warning("No candle data available.")
        return

    st.markdown(
        f"**Bank Nifty** · Session {data.get('session_date', '')} · "
        f"Source {data.get('data_source', 'fyers-live')} · "
        f"Updated {data.get('last_updated', '')}"
    )
    st.plotly_chart(build_candle_chart(candles), width="stretch")


@st.fragment(run_every=timedelta(minutes=REFRESH_MINUTES))
def risk_panel():
    render_risk(_panel("api/risk", ["live_prediction", "risk_summary", "last_updated"]))


@st.fragment(run_every=timedelta(minutes=REFRESH_MINUTES))
def anomaly_panel():
    render_anomaly(_panel("api/candles", ["anomaly_pct"]))


@st.fragment(run_every=timedelta(minutes=REFRESH_MINUTES))
def news_panel():
    render_news(_panel("api/news", ["nlp_analysis"]))


def debug_expander():
    data = _request("GET", "api/snapshot")
    if not data or "error" in data:
        data = _local_payload()
    render_debug(data)


# ---------------------------------------------------------------------------
# RAG chatbot (chatbot/) -- answers questions about the latest ranked news
# and about this project's own scoring fields (CARS, D_info, TCN, IF).
# Imported lazily so langchain / sentence-transformers only load once
# someone actually opens this panel, not on every dashboard render.
# ---------------------------------------------------------------------------


def chat_panel():
    st.markdown("### 💬 Ask about anomalies & news")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, text in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input(
        'Ask e.g. "What does CARS score mean?" or "Any negative HDFC Bank news today?"'
    )
    if not question:
        return

    st.session_state["chat_history"].append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from chatbot.qa_chain import ask

                result = ask(question)
                answer = result["answer"]
                if result.get("sources"):
                    answer += "\n\n**Sources:** " + "; ".join(result["sources"])
            except FileNotFoundError as exc:
                answer = str(exc)
            except Exception as exc:
                answer = (
                    "Chatbot isn't configured yet. Copy `.env.example` to `.env` "
                    "at the project root and set `LLM_PROVIDER=gemini` with a "
                    "`GOOGLE_API_KEY`, or `LLM_PROVIDER=ollama` to run fully "
                    f"locally.\n\n**Details:** {exc}"
                )
        st.markdown(answer)

    st.session_state["chat_history"].append(("assistant", answer))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


def render_dashboard() -> None:
    """Render the full live dashboard (chart + risk/anomaly/news side panels
    + chat Q&A + debug expander). Called either by this file directly
    (`streamlit run frontend/app.py`) or by the root app.py router after the
    "Launch Dashboard" button is clicked on the landing page."""
    st.markdown("## Bank Nifty Live Surveillance")

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.caption("Hybrid TCN + Isolation Forest anomaly detection · CARS risk engine · NLP news context")
    with header_right:
        if st.button("Refresh now", use_container_width=True):
            _request("POST", "api/refresh")
            st.rerun()

    chart_col, side_col = st.columns([3.2, 1])

    with chart_col:
        chart_panel()

    with side_col:
        risk_panel()
        anomaly_panel()
        news_panel()

    st.markdown("---")
    chat_panel()
    debug_expander()


if __name__ == "__main__":
    render_dashboard()
