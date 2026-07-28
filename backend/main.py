# -*- coding: utf-8 -*-
"""
main.py
=======
FastAPI Backend Server for Financial Market Surveillance & Risk Intelligence System.

Serves live 5-minute pipeline analysis endpoints connecting market data,
pre-trained Isolation Forest model predictions, news NLP sentiment, and CARS risk engine.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.pipeline import run_live_pipeline

app = FastAPI(
    title="Market Surveillance & Risk Intelligence API",
    description="End-to-end 5-minute interval financial anomaly detection & risk intelligence API",
    version="1.0.0",
)


class PipelineRequest(BaseModel):
    ticker: str = "NSE:NIFTYBANK-INDEX"


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "AI Market Surveillance & Risk Intelligence System",
        "version": "1.0.0",
        "endpoints": ["/run-pipeline/{ticker}", "/health"],
    }


@app.get("/health")
def health_check():
    model_pkl = os.path.join(PROJECT_ROOT, "model", "output", "isolation_forest_model.pkl")
    scaler_pkl = os.path.join(PROJECT_ROOT, "model", "output", "scaler.pkl")
    artifacts_ready = os.path.exists(model_pkl) and os.path.exists(scaler_pkl)

    return {
        "status": "healthy" if artifacts_ready else "degraded",
        "model_artifacts_present": artifacts_ready,
        "model_path": model_pkl,
    }


@app.get("/run-pipeline/{ticker}")
def run_pipeline_endpoint(ticker: str):
    """
    Executes live 5-minute pipeline for a given ticker:
    Data Fetch -> Features -> Pre-trained IF Model -> NLP News -> CARS Risk Engine.
    """
    try:
        results = run_live_pipeline(ticker=ticker)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-pipeline")
def run_pipeline_post(req: PipelineRequest):
    """
    POST endpoint to run live pipeline.
    """
    try:
        results = run_live_pipeline(ticker=req.ticker)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
