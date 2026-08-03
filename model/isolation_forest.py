# -*- coding: utf-8 -*-
"""
Train the Isolation Forest anomaly detector on scaled feature data
(feature_extract/bank_nifty_train.csv) and save it to saved_models.

Run:
    python model/isolation_forest.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FEATURE_COLUMNS

FILE_PATH = "feature_extract/bank_nifty_train.csv"


def assign_status(score, is_anom):
    if not is_anom:
        return "Normal"
    if score >= 75:
        return "High Risk Anomaly"
    if score >= 60:
        return "Medium Risk Anomaly"
    return "Low Risk Anomaly"


def generate_explanation(row):
    if not row["Is_Anomaly"]:
        return "Normal trading activity"

    reasons = []
    if row["Volume_ZScore"] > 2.5:
        reasons.append(f"Volume Surge ({row['Volume_ZScore']:.1f}x std dev)")
    if row["RSI_14"] > 70 or row["RSI_14"] < 30:
        reasons.append(f"RSI Extreme ({row['RSI_14']:.1f})")
    if abs(row["Log_Returns"]) > 0.005:
        reasons.append(f"Price Spike ({row['Log_Returns'] * 100:.2f}%)")
    if row["BB_Width"] > 0.02:
        reasons.append("High Volatility (BB Expansion)")

    if not reasons:
        reasons.append("Multi-indicator joint anomaly")

    return " | ".join(reasons)


def evaluate(df_features, model, X):
    # Rule-based ground truth: extreme volume spikes or price moves
    df_features["Ground_Truth"] = (np.abs(df_features["Volume_ZScore"]) > 3.0) | (
        np.abs(df_features["Log_Returns"]) > 0.01
    )
    y_true = df_features["Ground_Truth"]
    y_pred = df_features["Is_Anomaly"]
    raw_scores = model.decision_function(X)

    print("\nMODEL EVALUATION METRICS REPORT")
    print(classification_report(y_true, y_pred, target_names=["Normal (0)", "Anomaly (1)"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_true, -raw_scores):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives:  {cm[0][0]:<6} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]:<6} | True Positives:  {cm[1][1]}")


if __name__ == "__main__":
    print("Loading dataset...")
    df_features = pd.read_csv(FILE_PATH)
    X = df_features[FEATURE_COLUMNS]

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X)
    joblib.dump(model, "saved_models/isolation_forest.pkl")
    print("Isolation Forest model saved.")

    predictions = model.predict(X)
    raw_scores = model.decision_function(X)
    df_features["Is_Anomaly"] = predictions == -1
    df_features["Risk_Score"] = np.round(np.clip((0.5 - raw_scores) * 100, 0, 100), 2)
    df_features["Status"] = [
        assign_status(score, is_anom)
        for score, is_anom in zip(df_features["Risk_Score"], df_features["Is_Anomaly"])
    ]
    df_features["Explanation"] = df_features.apply(generate_explanation, axis=1)

    high_medium = df_features[df_features["Status"].isin(["High Risk Anomaly", "Medium Risk Anomaly"])]
    display_cols = ["Open", "High", "Low", "Close", "Volume", "Risk_Score", "Status", "Explanation"]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\nHIGH & MEDIUM RISK ANOMALIES (Total Found: {})".format(len(high_medium)))
    print("=" * 100)
    print(high_medium[display_cols].sort_values(by="Risk_Score", ascending=False).head(15))

    output_filename = "model/high_medium_risk_anomalies_only.csv"
    high_medium[display_cols].to_csv(output_filename)
    print(f"\nFiltered DataFrame saved to: '{output_filename}'")

    evaluate(df_features, model, X)
