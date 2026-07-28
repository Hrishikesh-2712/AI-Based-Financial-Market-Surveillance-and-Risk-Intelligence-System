# -*- coding: utf-8 -*-
"""
model.py
========
Isolation Forest anomaly detection training and inference module.

Reads engineered features from feature extraction output, trains an Isolation Forest
anomaly detection model, and exports model + scaler artifacts for external use.

Functions:
    - train_model: Fits StandardScaler and IsolationForest on input feature DataFrame.
    - predict: Runs inference using trained model and scaler.
    - save_artifacts: Saves model and scaler to disk (.pkl).
    - load_artifacts: Loads model and scaler from disk (.pkl).
    - run_training: High-level pipeline function to train model and save artifacts.

Usage:
    - As CLI:
        python model/model.py
        python model/model.py --input feature_extraction/output/features.csv --output-dir model/output
    - As Python Module:
        from model import train_model, predict, load_artifacts
"""

import argparse
import os
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path & Feature configuration
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


# ---------------------------------------------------------------------------
# 1. Core Model Functions (Training, Prediction, Artifacts)
# ---------------------------------------------------------------------------
def train_model(
    df_features: pd.DataFrame,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[IsolationForest, StandardScaler, pd.DataFrame]:
    """
    Fits StandardScaler and IsolationForest model on df_features[FEATURE_COLS].

    Args:
        df_features: Input DataFrame containing FEATURE_COLS.
        n_estimators: Number of base estimators in the Isolation Forest ensemble.
        contamination: The amount of contamination of the data set (expected anomaly ratio).
        random_state: Controls randomness for reproducible results.

    Returns:
        Tuple of (trained model, fitted scaler, DataFrame with prediction columns).
    """
    missing = [c for c in FEATURE_COLS if c not in df_features.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features[FEATURE_COLS])

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)

    df_result = df_features.copy()
    predictions = model.predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)

    df_result["Is_Anomaly"] = predictions == -1
    df_result["Raw_Anomaly_Score"] = raw_scores

    return model, scaler, df_result


def predict(
    model: IsolationForest,
    scaler: StandardScaler,
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs inference on input features using a pre-trained model and scaler.

    Args:
        model: Trained IsolationForest instance.
        scaler: Fitted StandardScaler instance.
        df_features: Input DataFrame containing FEATURE_COLS.

    Returns:
        DataFrame with 'Is_Anomaly' (boolean) and 'Raw_Anomaly_Score' (float) appended.
    """
    missing = [c for c in FEATURE_COLS if c not in df_features.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")

    X_scaled = scaler.transform(df_features[FEATURE_COLS])
    predictions = model.predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)

    df_result = df_features.copy()
    df_result["Is_Anomaly"] = predictions == -1
    df_result["Raw_Anomaly_Score"] = raw_scores

    return df_result


def save_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Tuple[str, str]:
    """
    Saves model and scaler to PKL files in output_dir.

    Returns:
        Tuple of (model_file_path, scaler_file_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "isolation_forest_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model_path, scaler_path


def load_artifacts(output_dir: str = DEFAULT_OUTPUT_DIR) -> Tuple[IsolationForest, StandardScaler]:
    """
    Loads trained IsolationForest model and StandardScaler from PKL files.

    Returns:
        Tuple of (loaded model, loaded scaler)
    """
    model_path = os.path.join(output_dir, "isolation_forest_model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


# ---------------------------------------------------------------------------
# 2. Main High-Level Entry Point
# ---------------------------------------------------------------------------
def run_training(
    input_path: str = DEFAULT_INPUT_FILE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[IsolationForest, StandardScaler, pd.DataFrame]:
    """
    Loads features CSV, trains Isolation Forest model, saves artifacts & scored CSV.
    """
    print(f"Loading features from: {input_path}")
    df_features = pd.read_csv(input_path, index_col=0, parse_dates=True)

    if len(df_features) == 0:
        raise ValueError(
            "Feature file has 0 rows. Re-run feature_extraction with sufficient history."
        )

    print("Training Isolation Forest model...")
    model, scaler, df_scored = train_model(
        df_features,
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )

    model_path, scaler_path = save_artifacts(model, scaler, output_dir=output_dir)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Save predictions
    all_scored_path = os.path.join(output_dir, "all_scored.csv")
    df_scored.to_csv(all_scored_path)
    print(f"Scored features saved to: {all_scored_path}")

    return model, scaler, df_scored


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Isolation Forest anomaly detector"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help="Path to input features CSV (feature_extraction/output/features.csv by default)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write model artifacts (model/output by default)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(input_path=args.input, output_dir=args.output_dir)
