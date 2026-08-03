# -*- coding: utf-8 -*-
"""
baseline.py
============
Builds and persists the historical if_score / tcn_error distributions that
risk_engine.py percentile-ranks new values against.

Run this once against a chunk of historical scored data (e.g. the last
20-30 trading days), and re-run it periodically (weekly/monthly) to keep
the baseline current -- same "train once, refresh periodically" pattern
used for the models themselves. Without this, risk_engine.py falls back to
a cruder linear scaling (see its module docstring).

Usage:
    from risk_engine.baseline import build_and_save_baseline
    # df_scored_historical must have 'if_score' and 'tcn_error' columns,
    # e.g. output of HybridModel.predict_batch() over a historical window.
    build_and_save_baseline(df_scored_historical, "risk_engine/baseline.json")
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_and_save_baseline(df_scored_historical: pd.DataFrame, path: str) -> dict:
    baseline = {
        "if_scores": df_scored_historical["if_score"].dropna().tolist(),
        "tcn_errors": df_scored_historical["tcn_error"].dropna().tolist(),
        "built_from_rows": len(df_scored_historical),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(baseline, f)
    return baseline


def load_baseline(path: str):
    """Returns (if_baseline, tcn_baseline) as numpy arrays, or (None, None) if missing."""
    p = Path(path)
    if not p.exists():
        return None, None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return (
        np.array(data.get("if_scores", [])),
        np.array(data.get("tcn_errors", [])),
    )
