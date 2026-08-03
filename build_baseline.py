# -*- coding: utf-8 -*-
"""
build_baseline.py
===================
One-time (and periodically re-run) script: scores the historical training
data with the already-trained TCN + Isolation Forest models, and saves the
resulting if_score / tcn_error distributions as the risk engine's baseline
for percentile normalization.

feature_extract/bank_nifty_train.csv is already scaled (same scaler.pkl
used at inference time), so it can be fed straight into HybridModel.

Run from the project root:
    python build_baseline.py

Re-run this periodically (e.g. monthly) as more live data accumulates, by
pointing INPUT_CSV at a more recent window instead of the original
training split, so the baseline doesn't go stale.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from config import SAVED_MODELS_DIR
from inference import HybridModel
from risk_engine.baseline import build_and_save_baseline

INPUT_CSV = PROJECT_ROOT / "feature_extract" / "bank_nifty_train.csv"
BASELINE_PATH = PROJECT_ROOT / "risk_engine" / "baseline.json"


def main():
    print(f"Loading historical scaled features from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    print("Loading trained models ...")
    model = HybridModel(str(SAVED_MODELS_DIR)).load()

    print(f"Scoring {len(df)} historical rows with TCN + Isolation Forest ...")
    df_scored = model.predict_batch(df)

    print(f"Saving baseline -> {BASELINE_PATH}")
    build_and_save_baseline(df_scored, str(BASELINE_PATH))
    print("Done.")


if __name__ == "__main__":
    main()
