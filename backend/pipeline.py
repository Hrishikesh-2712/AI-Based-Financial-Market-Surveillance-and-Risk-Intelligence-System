# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Central live pipeline handler in backend.
Coordinates 5-minute data fetching, feature extraction, pre-trained model inference,
news NLP sentiment analysis, and CARS risk intelligence scoring.

Policy:
    - Model is trained ONCE (saved in model/output/).
    - Live pipeline loads pre-trained model artifacts for rapid 5-minute predictions.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Add project root to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.data import get_market_data
from feature_extraction.feature_extraction import NonOverlappingIndicators, _standardize_columns
from model.model import load_artifacts, predict
from news_engine.nlp_news import analyze_news_nlp
from risk_engine.risk_engine import calculate_research_risk_score

MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model", "output")


def run_live_pipeline(ticker: str = "NSE:NIFTYBANK-INDEX") -> Dict[str, Any]:
    """
    Executes end-to-end live market surveillance pipeline for a ticker.

    1. Fetch 5-min market data via FYERS or local CSV fallback.
    2. Extract technical indicators (Log_Returns, RSI, BB, Volume_ZScore, etc.).
    3. Load pre-trained Isolation Forest model & scaler (no retraining).
    4. Predict anomalies and raw anomaly scores.
    5. Fetch NLP news sentiment and relevance index.
    6. Compute Composite Anomaly Risk Score (CARS) and D_info.
    7. Return complete analysis payload for backend API and Streamlit UI.
    """
    clean_ticker = ticker.strip() if ticker else "NSE:NIFTYBANK-INDEX"

    # Step 1: Fetch 5-minute interval OHLCV data
    df_raw = get_market_data(symbol=clean_ticker, interval="5", days=5)

    # Standardize columns to Date/Open/High/Low/Close/Volume
    df_clean = _standardize_columns(df_raw)

    # Step 2: Extract Features
    calculator = NonOverlappingIndicators(df_clean)
    df_features = calculator.calculate_all()

    if len(df_features) == 0:
        raise ValueError("Insufficient data rows to extract technical features.")

    # Step 3: Load Pre-Trained Model & Scaler Artifacts (Trained ONCE)
    model, scaler = load_artifacts(output_dir=MODEL_OUTPUT_DIR)

    # Step 4: Run Inference (Predict)
    df_scored = predict(model, scaler, df_features)

    # Step 5: Run News NLP Engine
    nlp_data = analyze_news_nlp(clean_ticker)

    # Step 6: Compute Technical Risk Score & CARS Risk Score
    # Extract latest or worst anomaly decision score for risk engine
    anomalies = df_scored[df_scored["Is_Anomaly"]]
    if not anomalies.empty:
        worst_raw_score = float(anomalies["Raw_Anomaly_Score"].min())
        is_anomaly = True
    else:
        worst_raw_score = float(df_scored["Raw_Anomaly_Score"].iloc[-1])
        is_anomaly = bool(df_scored["Is_Anomaly"].iloc[-1])

    # Technical indicator risk score derived from volume z-scores and RSI extremes
    recent_row = df_scored.iloc[-1]
    vol_z = abs(recent_row.get("Volume_ZScore", 0))
    rsi_dist = abs(recent_row.get("RSI_14", 50) - 50)
    tech_risk_score = float(np.clip((vol_z * 20) + (rsi_dist * 1.5), 0, 100))

    risk_output = calculate_research_risk_score(
        if_raw_score=worst_raw_score,
        is_anomaly=is_anomaly,
        tech_risk_score=tech_risk_score,
        nlp_results=nlp_data,
    )

    # Step 7: Format Candles & Recent Anomalies List
    df_scored_reset = df_scored.reset_index()

    # Format timestamp to ISO format string
    df_scored_reset["Date_Str"] = df_scored_reset["Date"].astype(str)

    candles_list: List[Dict[str, Any]] = []
    for _, row in df_scored_reset.iterrows():
        candles_list.append({
            "datetime": str(row["Date_Str"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "is_anomaly": bool(row["Is_Anomaly"]),
            "raw_anomaly_score": round(float(row["Raw_Anomaly_Score"]), 4),
        })

    # Recent anomalies (filtered, sorted latest first)
    df_anom_sorted = df_scored_reset[df_scored_reset["Is_Anomaly"]].sort_values(
        by="Date", ascending=False
    )
    recent_anomalies_list: List[Dict[str, Any]] = []
    for _, row in df_anom_sorted.head(15).iterrows():
        recent_anomalies_list.append({
            "datetime": str(row["Date_Str"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "volume_zscore": round(float(row.get("Volume_ZScore", 0)), 2),
            "rsi_14": round(float(row.get("RSI_14", 0)), 2),
            "raw_anomaly_score": round(float(row["Raw_Anomaly_Score"]), 4),
        })

    return {
        "ticker": clean_ticker,
        "total_candles": len(df_scored),
        "total_anomalies_found": len(df_anom_sorted),
        "risk_summary": risk_output,
        "nlp_analysis": nlp_data,
        "candles": candles_list,
        "recent_anomalies": recent_anomalies_list,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
