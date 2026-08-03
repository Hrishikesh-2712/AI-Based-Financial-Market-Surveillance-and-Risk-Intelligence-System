# -*- coding: utf-8 -*-
"""
config.py
=========
Single source of truth for shared constants and artifact paths:
model window sizes, indicator periods, feature columns, saved-model locations.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 64   # TCN sliding-window length
TCN_BATCH_SIZE = 64    # TCN inference batching; affects throughput, not results

# ---------------------------------------------------------------------------
# Technical indicators (feature_extract)
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
ATR_PERIOD = 14
BB_WINDOW = 20
BB_STD = 2
SUPERTREND_MULTIPLIER = 3.0
VOLUME_WINDOW = 20
OBV_PERIODS = 5

# ---------------------------------------------------------------------------
# Feature columns consumed by the TCN and Isolation Forest
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
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
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
