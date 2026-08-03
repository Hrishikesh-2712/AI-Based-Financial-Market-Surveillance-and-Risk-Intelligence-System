# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Live Bank Nifty surveillance pipeline:
FYERS fetch -> feature engineering -> hybrid inference -> NLP -> CARS risk.

SurveillanceEngine is the facade over feature extraction, hybrid model
inference, and the CARS risk score -- it loads models and the scaler once and
reuses them across runs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model"
FEATURE_DIR = PROJECT_ROOT / "feature_extract"
NLP_SNAPSHOT_PATH = PROJECT_ROOT / "nlp_news_engine" / "snapshots" / "latest.json"
CURRENT_FEATURES_CSV = FEATURE_DIR / "current_features.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from config import FEATURE_COLUMNS, SAVED_MODELS_DIR, SEQUENCE_LENGTH

from data.data import (
    INDEX_SYMBOL,
    IST_TZ,
    ROWS_PER_SESSION_DAY,
    fetch_bank_nifty_from_cache,
    fetch_incremental_bank_nifty,
    load_current_data,
    save_current_data,
)
from feature_extract.feature_extract import compute_features
from inference import HybridModel
from risk_engine.risk_engine import calculate_composite_risk_score
from risk_engine.baseline import load_baseline

RISK_BASELINE_PATH = PROJECT_ROOT / "risk_engine" / "baseline.json"

# A fallback NLP snapshot older than this is treated as stale and flagged.
SNAPSHOT_STALE_HOURS = 24


def _tech_risk_score(row: pd.Series) -> float:
    vol_z = abs(float(row.get("Volume_ZScore", 0)))
    rsi_dist = abs(float(row.get("RSI_14", 50)) - 50)
    return float(np.clip((vol_z * 20) + (rsi_dist * 1.5), 0, 100))


class SurveillanceEngine:
    """Feature extraction -> scaling -> hybrid inference -> CARS risk.

    Models and the scaler are loaded once (lazily) and reused by every run.
    """

    def __init__(self):
        self._hybrid = HybridModel(str(SAVED_MODELS_DIR))
        self._scaler = None
        self._if_baseline, self._tcn_baseline = load_baseline(str(RISK_BASELINE_PATH))
        if self._if_baseline is None:
            print(
                f"No risk baseline found at {RISK_BASELINE_PATH}; risk_engine will use "
                f"degraded linear-scaling fallback. Run risk_engine/baseline.py against "
                f"historical scored data to build one."
            )

    def load(self) -> "SurveillanceEngine":
        self._hybrid.load()
        return self

    def extract(self, df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute indicators and apply the saved scaler.

        Returns (unscaled features, scaled features), both indexed by timestamp.
        """
        df_features = compute_features(df_raw)
        if len(df_features) < SEQUENCE_LENGTH:
            raise ValueError(
                f"Insufficient rows after feature extraction ({len(df_features)}). "
                f"Need at least {SEQUENCE_LENGTH} for TCN inference."
            )

        if self._scaler is None:
            scaler_path = SAVED_MODELS_DIR / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(
                    f"Missing scaler at {scaler_path}. "
                    "Run: python feature_extract/feature_extract.py"
                )
            self._scaler = joblib.load(scaler_path)

        df_scaled = df_features.copy()
        df_scaled[FEATURE_COLUMNS] = self._scaler.transform(df_features[FEATURE_COLUMNS])
        return df_features, df_scaled

    def predict(self, df_scaled: pd.DataFrame) -> dict:
        """Single-bar risk decision for the latest row."""
        return self._hybrid.predict(df_scaled)

    def predict_batch(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """Score every bar (TCN + Isolation Forest) with per-bar markers."""
        return self._hybrid.predict_batch(df_scaled)

    def score(self, df_features: pd.DataFrame, df_scored: pd.DataFrame, nlp_data: dict) -> dict:
        """Composite risk score for the latest bar."""
        latest = df_scored.iloc[-1]
        anomaly_timestamp = df_scored.index[-1]
        if hasattr(anomaly_timestamp, "tzinfo") and anomaly_timestamp.tzinfo is not None:
            anomaly_timestamp = anomaly_timestamp.tz_localize(None)

        # Sign of the latest raw (unscaled) return -- positive = bullish move,
        # negative = bearish move. Used only for its sign in the news check.
        anomaly_direction = float(df_features["Log_Returns"].iloc[-1])

        try:
            tcn_threshold = self._hybrid._threshold  # loaded by HybridModel.load()
        except AttributeError:
            tcn_threshold = None

        return calculate_composite_risk_score(
            rule_score=_tech_risk_score(df_features.iloc[-1]),
            if_score_raw=float(latest["if_score"]),
            tcn_error_raw=float(latest["tcn_error"]),
            anomaly_timestamp=anomaly_timestamp,
            anomaly_direction=anomaly_direction,
            nlp_results=nlp_data,
            if_baseline=self._if_baseline,
            tcn_baseline=self._tcn_baseline,
            tcn_threshold=tcn_threshold,
        )


_engine = SurveillanceEngine()


# ---------------------------------------------------------------------------
# NLP snapshot handling
# ---------------------------------------------------------------------------


def _summarise_nlp(snapshot: dict, source: str) -> dict:
    """Reduce a ranked-news snapshot to the fields the CARS engine and UI need."""
    top_news = snapshot.get("top_news", [])[:5]
    if not top_news:
        return {
            "sentiment_score": 0.0,
            "relevance_score": 0.0,
            "headline_count": snapshot.get("count", 0),
            "generated_at": snapshot.get("generated_at"),
            "source": source,
            "top_news": [],
        }

    sentiments = [item.get("sentiment_signed", 0.0) for item in top_news]
    composites = [item.get("composite_score", 0.0) for item in top_news]

    return {
        "sentiment_score": round(float(np.mean(sentiments)), 3),
        "relevance_score": round(float(np.mean(composites)), 3),
        "headline_count": snapshot.get("count", len(top_news)),
        "generated_at": snapshot.get("generated_at"),
        "source": source,
        "top_news": top_news,
    }


def _load_nlp_analysis() -> dict:
    """
    NLP output for the CARS risk engine.

    Preferred: call the NLP news engine live (nlp_news_engine). Falls back to
    the latest persisted snapshot (snapshots/latest.json) when the engine's
    dependencies are missing or a fetch fails, and flags the snapshot as
    stale if it is old.
    """
    empty = {
        "sentiment_score": 0.0,
        "relevance_score": 0.0,
        "headline_count": 0,
        "generated_at": None,
        "source": "none",
        "top_news": [],
    }

    # 1) Live news from the NLP engine when its dependencies are installed.
    try:
        nlp_engine_dir = str(PROJECT_ROOT / "nlp_news_engine")
        if nlp_engine_dir not in sys.path:
            sys.path.insert(0, nlp_engine_dir)
        from nlp.fetch_and_rank import get_ranked_news
        from nlp.snapshot import save_snapshot

        ranked = get_ranked_news(datetime.now())
        if ranked:
            snapshot = save_snapshot(ranked)
            return _summarise_nlp(snapshot, source="nlp-engine-live")
    except Exception as exc:
        print(f"Live NLP unavailable ({type(exc).__name__}: {exc}); using existing snapshot.")

    # 2) Fallback: snapshot written by the news engine poller.
    if not NLP_SNAPSHOT_PATH.exists():
        print("No NLP snapshot present; running without news context.")
        return empty

    with open(NLP_SNAPSHOT_PATH, encoding="utf-8") as f:
        snapshot = json.load(f)

    source = "nlp-snapshot"
    try:
        age_hours = (
            datetime.now() - datetime.strptime(snapshot["generated_at"], "%Y-%m-%d %H:%M:%S")
        ).total_seconds() / 3600
        if age_hours > SNAPSHOT_STALE_HOURS:
            source = "nlp-snapshot-stale"
            print(f"WARNING: NLP snapshot is {age_hours:.1f}h old; news context is stale.")
    except (KeyError, ValueError, TypeError):
        pass

    return _summarise_nlp(snapshot, source=source)


# ---------------------------------------------------------------------------
# Output formatting / persistence
# ---------------------------------------------------------------------------


def _today_mask(index: pd.Index) -> pd.Series:
    session_date = index.max().date()
    return pd.Series([ts.date() == session_date for ts in index], index=index)


def _format_candles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "datetime": ts.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "tcn_anomaly": bool(row.get("tcn_anomaly", False)),
            "if_anomaly": bool(row.get("if_anomaly", False)),
            "anomaly_level": str(row.get("anomaly_level", "none")),
            "hover_text": str(row.get("hover_text", "")),
            "tcn_error": round(float(row.get("tcn_error", 0)), 4),
            "if_score": round(float(row.get("if_score", 0)), 4),
        }
        for ts, row in df.iterrows()
    ]


def _to_ist_column(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    df = df.copy()
    df["timestamp"] = ts
    return df


def _save_current_features(df_features: pd.DataFrame) -> None:
    """Persist processed feature rows (timestamp + OHLCV + indicators),
    deduplicated by timestamp so only genuinely new candles are added."""
    try:
        out = _to_ist_column(df_features.reset_index())
        out = out[~out["timestamp"].duplicated(keep="last")].sort_values("timestamp")
        out.to_csv(CURRENT_FEATURES_CSV, index=False)
    except OSError as exc:
        print(f"Could not persist feature CSV: {exc}")


def _recent_anomaly_list(df_today: pd.DataFrame) -> List[Dict[str, Any]]:
    recent = (
        df_today[df_today["anomaly_level"].isin(["both", "either"])]
        .sort_index(ascending=False)
        .head(15)
    )
    return [
        {
            "datetime": ts.isoformat(),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
            "anomaly_level": row["anomaly_level"],
            "tcn_anomaly": bool(row["tcn_anomaly"]),
            "if_anomaly": bool(row["if_anomaly"]),
            "tcn_error": round(float(row["tcn_error"]), 4),
            "if_score": round(float(row["if_score"]), 4),
        }
        for ts, row in recent.iterrows()
    ]


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------


def run_live_pipeline(ticker: str = INDEX_SYMBOL) -> Dict[str, Any]:
    """
    End-to-end live surveillance for Bank Nifty (5-minute candles).

    1. Incremental FYERS fetch (reuse today's current_data.csv, fetch the
       missing tail), CSV cache fallback when FYERS is down
    2. Persist raw candles to data/current_data.csv (deduped by timestamp)
    3. Feature engineering + scaler transform, persist processed features
    4. Hybrid TCN + Isolation Forest inference (tail-64 for the live point)
    5. NLP output -> CARS risk score
    6. Return today's candles + metrics for Streamlit
    """
    _ = ticker  # Bank Nifty only for this deployment

    try:
        df_raw = fetch_incremental_bank_nifty()
        data_source = "fyers-live"
        save_current_data(df_raw)
    except Exception as fetch_err:
        print(
            f"FYERS fetch failed ({fetch_err}); falling back to cached data. "
            "If this is an auth error, refresh the token with: python data/auth.py"
        )
        # Prefer today's accumulated cache; fall back to the historical CSV
        # window only when nothing for today exists yet.
        df_raw = load_current_data()
        if df_raw.empty:
            df_raw = fetch_bank_nifty_from_cache()
        data_source = "csv-cache"

    if df_raw.empty:
        raise ValueError("No Bank Nifty candles available (FYERS or CSV cache).")

    df_features, df_scaled = _engine.extract(df_raw)

    # Persist the processed feature rows for reuse / audit.
    _save_current_features(df_features)

    df_scored = _engine.predict_batch(df_scaled)

    # Keep unscaled RSI / volume z-score for risk engine display logic.
    for col in ["RSI_14", "Volume_ZScore", "Log_Returns"]:
        df_scored[col + "_raw"] = df_features[col].values

    today_filter = _today_mask(df_scored.index)
    df_today = df_scored[today_filter].copy()
    if df_today.empty:
        df_today = df_scored.tail(ROWS_PER_SESSION_DAY).copy()

    nlp_data = _load_nlp_analysis()
    live_prediction = _engine.predict(df_scaled)
    risk_output = _engine.score(df_features, df_scored, nlp_data)

    anomaly_mask = df_today["anomaly_level"].isin(["both", "either"])
    anomaly_pct = round(100.0 * anomaly_mask.sum() / max(len(df_today), 1), 2)

    return {
        "ticker": INDEX_SYMBOL,
        "data_source": data_source,
        "session_date": str(df_today.index.max().date()),
        "total_candles": len(df_today),
        "anomaly_pct": anomaly_pct,
        "live_prediction": live_prediction,
        "risk_summary": risk_output,
        "nlp_analysis": nlp_data,
        "candles": _format_candles(df_today),
        "recent_anomalies": _recent_anomaly_list(df_today),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
