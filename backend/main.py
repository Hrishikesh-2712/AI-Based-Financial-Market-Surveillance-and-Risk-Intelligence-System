# -*- coding: utf-8 -*-
"""
main.py
=======
FastAPI backend for Bank Nifty live surveillance.

Live-mode flow:
    fetch -> feature extract -> model inference (TCN + Isolation Forest)
        + NLP output (nlp_news_engine, live or snapshot)
    -> risk engine (CARS score) -> served to the Streamlit UI.

Endpoints (new design):
    GET  /health          artifacts + status
    GET  /api/snapshot    full pipeline output (cached)
    GET  /api/risk        model inference + risk-engine output
    GET  /api/candles     candle data with anomaly markers
    GET  /api/news        NLP ranked news
    POST /api/refresh     force a pipeline re-run

Run:
    uvicorn backend.main:app --host 127.0.0.1 --port 5000
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pipeline import run_live_pipeline

# ---------------------------------------------------------------------------
# Cached pipeline results: live data is 5-minute candles, so re-running the
# pipeline on every request would hammer the data source. A short TTL keeps
# the separate endpoints consistent with each other. A background scheduler
# refreshes the cache every 5 minutes so data is always being pulled fresh.
# ---------------------------------------------------------------------------
# Cache TTL matches the poll interval: the 5-minute scheduler owns pulling new
# data, and the endpoints serve that snapshot without triggering extra runs.
CACHE_TTL_SECONDS = 300
POLL_INTERVAL_SECONDS = 300  # pull new data every 5 minutes

_cache: Dict[str, Any] = {"timestamp": 0.0, "result": None}
_cache_lock = threading.Lock()
_scheduler_stop = threading.Event()
_scheduler_busy = False


def _snapshot(force: bool = False) -> Dict[str, Any]:
    now = time.time()
    with _cache_lock:
        cached = _cache["result"]
        if (
            not force
            and cached is not None
            and now - _cache["timestamp"] < CACHE_TTL_SECONDS
        ):
            return cached
        result = run_live_pipeline()
        _cache.update({"timestamp": now, "result": result})
        return result


def _model_artifacts() -> Dict[str, bool]:
    models_dir = PROJECT_ROOT / "saved_models"
    return {
        "tcn": (models_dir / "tcn_autoencoder.pth").exists(),
        "isolation_forest": (models_dir / "isolation_forest.pkl").exists(),
        "threshold": (models_dir / "threshold.json").exists(),
        "scaler": (models_dir / "scaler.pkl").exists(),
    }


# ---------------------------------------------------------------------------
# Background 5-minute poller
# ---------------------------------------------------------------------------


def _scheduler_loop() -> None:
    global _scheduler_busy
    while not _scheduler_stop.is_set():
        if not _scheduler_busy:
            _scheduler_busy = True
            try:
                print(f"[scheduler] pulling new data at {time.strftime('%H:%M:%S')}")
                _snapshot(force=True)
            except Exception as exc:
                print(f"[scheduler] pipeline run failed: {exc}")
            finally:
                _scheduler_busy = False
        _scheduler_stop.wait(POLL_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    print(f"[scheduler] started -- pulls new data every {POLL_INTERVAL_SECONDS // 60} min")
    thread = threading.Thread(target=_scheduler_loop, daemon=True, name="pipeline-poller")
    thread.start()
    try:
        yield
    finally:
        _scheduler_stop.set()


app = FastAPI(
    title="AI Market Surveillance & Risk Intelligence System",
    description="Hybrid TCN + Isolation Forest anomaly detection, NLP news context, "
    "and CARS (Composite Anomaly Risk Score) from the research-paper formula.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "AI Market Surveillance & Risk Intelligence System",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/api/snapshot",
            "/api/risk",
            "/api/candles",
            "/api/news",
            "/api/refresh",
        ],
    }


@app.get("/health")
def health_check():
    artifacts = _model_artifacts()
    return {
        "status": "healthy" if all(artifacts.values()) else "degraded",
        "model_artifacts": artifacts,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }


@app.get("/api/snapshot")
def api_snapshot(force: bool = False):
    """Full pipeline output (fetch -> features -> inference -> NLP -> risk)."""
    try:
        return _snapshot(force=force)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/refresh")
def api_refresh():
    """Force a fresh pipeline run and return the updated snapshot."""
    try:
        return _snapshot(force=True)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/risk")
def api_risk():
    """Model inference output + risk-engine (CARS) result."""
    try:
        snap = _snapshot()
        return {
            "live_prediction": snap.get("live_prediction"),
            "risk_summary": snap.get("risk_summary"),
            "last_updated": snap.get("last_updated"),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/candles")
def api_candles():
    """Live candle chart data with per-bar anomaly markers."""
    try:
        snap = _snapshot()
        return {
            "ticker": snap.get("ticker"),
            "session_date": snap.get("session_date"),
            "data_source": snap.get("data_source"),
            "total_candles": snap.get("total_candles"),
            "anomaly_pct": snap.get("anomaly_pct"),
            "candles": snap.get("candles"),
            "recent_anomalies": snap.get("recent_anomalies"),
            "last_updated": snap.get("last_updated"),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/news")
def api_news():
    """NLP-ranked news feeding the risk engine and the UI news panel."""
    try:
        snap = _snapshot()
        return {
            "nlp_analysis": snap.get("nlp_analysis"),
            "last_updated": snap.get("last_updated"),
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# Backward-compatible aliases so the previous single-call client still works.
# ---------------------------------------------------------------------------
@app.get("/api/surveillance")
def api_surveillance_legacy():
    return api_snapshot()


@app.get("/run-pipeline/{ticker:path}")
def run_pipeline_legacy(ticker: str = "NSE:NIFTYBANK-INDEX"):
    _ = ticker
    return api_snapshot()
