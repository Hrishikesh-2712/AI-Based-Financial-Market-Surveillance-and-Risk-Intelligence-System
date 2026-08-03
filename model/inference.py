# -*- coding: utf-8 -*-
"""
inference.py
============
Hybrid anomaly detector: TCN autoencoder reconstruction error + Isolation
Forest. Models are loaded once per HybridModel instance and reused.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURE_COLUMNS, SAVED_MODELS_DIR, SEQUENCE_LENGTH, TCN_BATCH_SIZE

from tcn import DEVICE, TCNAutoencoder, create_sequences


def _anomaly_level(tcn_flag: bool, if_flag: bool) -> str:
    if tcn_flag and if_flag:
        return "both"
    if tcn_flag or if_flag:
        return "either"
    return "none"


def _hover_label(tcn_flag: bool, if_flag: bool, tcn_error: float, if_score: float) -> str:
    tcn_status = "Anomaly" if tcn_flag else "Normal"
    if_status = "Anomaly" if if_flag else "Normal"
    return (
        f"TCN: {tcn_status} (err={tcn_error:.4f})<br>"
        f"IF: {if_status} (score={if_score:.4f})"
    )


class HybridModel:
    """TCN + Isolation Forest hybrid detector with lazy, once-only loading."""

    def __init__(self, models_dir=SAVED_MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._tcn = None
        self._iso = None
        self._threshold = None

    def load(self) -> "HybridModel":
        if self._tcn is None:
            self._tcn = TCNAutoencoder().to(DEVICE)
            self._tcn.load_state_dict(
                torch.load(self.models_dir / "tcn_autoencoder.pth", map_location=DEVICE)
            )
            self._tcn.eval()
            self._iso = joblib.load(self.models_dir / "isolation_forest.pkl")
            with open(self.models_dir / "threshold.json") as f:
                self._threshold = json.load(f)["threshold"]
        return self

    def predict(self, features_df: pd.DataFrame) -> dict:
        """Risk decision for the latest bar. Needs >= SEQUENCE_LENGTH rows."""
        if len(features_df) < SEQUENCE_LENGTH:
            raise ValueError(f"Need at least {SEQUENCE_LENGTH} rows for TCN inference.")
        self.load()

        sequence = torch.tensor(
            features_df[FEATURE_COLUMNS].tail(SEQUENCE_LENGTH).values,
            dtype=torch.float32,
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            reconstructed = self._tcn(sequence)
            error = nn.MSELoss(reduction="none")(reconstructed, sequence)
            tcn_error = error.mean().item()

        tcn_anomaly = tcn_error > self._threshold

        latest = features_df[FEATURE_COLUMNS].iloc[[-1]]
        prediction = self._iso.predict(latest)[0]
        score = self._iso.decision_function(latest)[0]
        iso_anomaly = prediction == -1

        if tcn_anomaly and iso_anomaly:
            risk = "High Risk"
        elif tcn_anomaly or iso_anomaly:
            risk = "Medium Risk"
        else:
            risk = "Normal"

        return {
            "tcn_error": float(tcn_error),
            "tcn_anomaly": bool(tcn_anomaly),
            "if_score": float(score),
            "if_anomaly": bool(iso_anomaly),
            "final_risk": risk,
        }

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score every row with TCN (SEQUENCE_LENGTH-bar window) and Isolation
        Forest, adding columns: tcn_error, tcn_anomaly, if_score, if_anomaly,
        anomaly_level (both | either | none), hover_text."""
        if len(features_df) < SEQUENCE_LENGTH:
            raise ValueError(f"Need at least {SEQUENCE_LENGTH} rows for batch TCN inference.")
        self.load()

        result = features_df.copy()
        feature_df = result[FEATURE_COLUMNS]
        feature_matrix = feature_df.values.astype(np.float32)

        if_pred = self._iso.predict(feature_df)
        if_scores = self._iso.decision_function(feature_df)
        if_anomalies = if_pred == -1

        # Sliding windows, run in batches (fully convolutional at eval, so
        # batching only affects throughput, not results).
        windows = create_sequences(feature_matrix, SEQUENCE_LENGTH)
        window_tensor = torch.tensor(windows, dtype=torch.float32)
        window_errors = np.zeros(len(windows), dtype=np.float64)

        with torch.no_grad():
            for start in range(0, len(window_tensor), TCN_BATCH_SIZE):
                batch = window_tensor[start : start + TCN_BATCH_SIZE].to(DEVICE)
                reconstructed = self._tcn(batch)
                err = nn.MSELoss(reduction="none")(reconstructed, batch).mean(dim=(1, 2))
                window_errors[start : start + TCN_BATCH_SIZE] = err.cpu().numpy()

        # Window i reconstructs rows [i, i+63] and reports the error on row i+63.
        tcn_errors = np.zeros(len(result), dtype=np.float64)
        tcn_errors[SEQUENCE_LENGTH - 1 :] = window_errors
        tcn_anomalies = tcn_errors > self._threshold

        result["tcn_error"] = tcn_errors
        result["tcn_anomaly"] = tcn_anomalies
        result["if_score"] = if_scores
        result["if_anomaly"] = if_anomalies
        result["anomaly_level"] = [
            _anomaly_level(bool(a), bool(b))
            for a, b in zip(tcn_anomalies, if_anomalies)
        ]
        result["hover_text"] = [
            _hover_label(bool(a), bool(b), float(e), float(s))
            for a, b, e, s in zip(tcn_anomalies, if_anomalies, tcn_errors, if_scores)
        ]
        return result
