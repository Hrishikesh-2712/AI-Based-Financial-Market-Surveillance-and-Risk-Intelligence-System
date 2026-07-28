# -*- coding: utf-8 -*-
"""
model.py
========
Reads engineered features from  ->  feature_extraction/output/
Trains an Isolation Forest anomaly-detection model and writes:
    - trained model + scaler artifacts
    - risk-scored predictions (all rows)
    - filtered high/medium risk anomalies
    - evaluation report
to  ->  model/output/

Modeling logic ported from the shared reference script
`isolation_forest_weekly_expiry.py`.

Usage:
    python model.py
    python model.py --input feature_extraction/output/features.csv --output-dir model/output
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path configuration (relative to project root, i.e. the parent of this file)
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DEFAULT_INPUT_FILE = os.path.join(
    PROJECT_ROOT, "feature_extraction", "output", "features.csv"
)
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "output")

FEATURE_COLS = [
    "Log_Returns",
    "MACD_Hist",
    "RSI_14",
    "BB_Width",
    "ATR_Normalized",
    "Supertrend_Dir",
    "Volume_ZScore",
    "OBV_Pct_Change",
]

DISPLAY_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Risk_Score",
    "Status",
    "Explanation",
]


# ---------------------------------------------------------------------------
# 1. STATUS & EXPLANATION GENERATORS  (shared logic, unchanged)
# ---------------------------------------------------------------------------
def assign_status(score, is_anom):
    if not is_anom:
        return "Normal"
    elif score >= 75:
        return "High Risk Anomaly"
    elif score >= 60:
        return "Medium Risk Anomaly"
    else:
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


# ---------------------------------------------------------------------------
# 2. Training + inference
# ---------------------------------------------------------------------------
def train_and_score(df_features: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features[FEATURE_COLS])

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    predictions = model.predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)

    df_features = df_features.copy()
    df_features["Is_Anomaly"] = predictions == -1
    df_features["Risk_Score"] = np.round(np.clip((0.5 - raw_scores) * 100, 0, 100), 2)
    df_features["Status"] = [
        assign_status(score, is_anom)
        for score, is_anom in zip(df_features["Risk_Score"], df_features["Is_Anomaly"])
    ]
    df_features["Explanation"] = df_features.apply(generate_explanation, axis=1)

    return model, scaler, df_features, X_scaled, raw_scores


def evaluate(df_features: pd.DataFrame, raw_scores: np.ndarray) -> dict:
    """Rule-based ground truth evaluation (extreme market events)."""
    ground_truth = (np.abs(df_features["Volume_ZScore"]) > 3.0) | (
        np.abs(df_features["Log_Returns"]) > 0.01
    )
    y_true = ground_truth
    y_pred = df_features["Is_Anomaly"]

    report_dict = classification_report(
        y_true, y_pred, target_names=["Normal (0)", "Anomaly (1)"],
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y_true, y_pred, target_names=["Normal (0)", "Anomaly (1)"], zero_division=0
    )

    try:
        auc = roc_auc_score(y_true, -raw_scores)
    except ValueError:
        auc = float("nan")  # only one class present in ground truth

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    print("=" * 60)
    print("MODEL EVALUATION METRICS REPORT")
    print("=" * 60)
    print(report_text)
    print("-" * 60)
    print(f"ROC-AUC Score: {auc:.4f}" if auc == auc else "ROC-AUC Score: N/A (single class in ground truth)")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"True Negatives:  {cm[0][0]:<6} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]:<6} | True Positives:  {cm[1][1]}")

    return {
        "classification_report": report_dict,
        "roc_auc": None if auc != auc else auc,
        "confusion_matrix": {
            "true_negatives": int(cm[0][0]),
            "false_positives": int(cm[0][1]),
            "false_negatives": int(cm[1][0]),
            "true_positives": int(cm[1][1]),
        },
    }


# ---------------------------------------------------------------------------
# 3. Main entry point
# ---------------------------------------------------------------------------
def run_model(input_path: str = DEFAULT_INPUT_FILE, output_dir: str = DEFAULT_OUTPUT_DIR):
    print(f"Loading features from: {input_path}")
    df_features = pd.read_csv(input_path, index_col=0, parse_dates=True)

    missing = [c for c in FEATURE_COLS if c not in df_features.columns]
    if missing:
        raise ValueError(f"Input features file is missing required columns: {missing}")

    if len(df_features) == 0:
        raise ValueError(
            "Feature file has 0 rows. Re-run feature_extraction.py with a "
            "longer history (need at least ~20 bars of warm-up data)."
        )

    os.makedirs(output_dir, exist_ok=True)

    print("Training Isolation Forest...")
    model, scaler, df_scored, X_scaled, raw_scores = train_and_score(df_features)

    # Save model + scaler artifacts
    model_path = os.path.join(output_dir, "isolation_forest_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Save full scored dataset
    all_scored_path = os.path.join(output_dir, "all_scored.csv")
    df_scored.to_csv(all_scored_path)
    print(f"Full scored dataset saved to: {all_scored_path}")

    # Filter + save high/medium risk anomalies only
    target_statuses = ["High Risk Anomaly", "Medium Risk Anomaly"]
    high_medium_df = df_scored[df_scored["Status"].isin(target_statuses)]
    anomalies_path = os.path.join(output_dir, "high_medium_risk_anomalies_only.csv")
    high_medium_df[DISPLAY_COLS].to_csv(anomalies_path)

    print("\n" + "=" * 100)
    print(f"HIGH & MEDIUM RISK ANOMALIES (Total Found: {len(high_medium_df)}):")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(high_medium_df[DISPLAY_COLS].sort_values(by="Risk_Score", ascending=False).head(15))
    print(f"\nFiltered anomalies saved to: {anomalies_path}")

    # Evaluate against rule-based ground truth
    metrics = evaluate(df_scored, raw_scores)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation metrics saved to: {metrics_path}")

    return model, scaler, df_scored


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate Isolation Forest anomaly detector")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE,
                         help="Path to input features CSV (feature_extraction/output/features.csv by default)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                         help="Directory to write model artifacts + results (model/output by default)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_model(input_path=args.input, output_dir=args.output_dir)
