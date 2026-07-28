# -*- coding: utf-8 -*-
"""
feature_extraction.py
======================
Reads raw OHLCV data from  ->  data/save/
Writes engineered features to  ->  feature_extraction/output/

Indicator logic ported from the shared reference script
`isolation_forest_weekly_expiry.py` (class `NonOverlappingIndicators`).

Usage:
    python feature_extraction.py
    python feature_extraction.py --input data/save/NIFTYBANK_5m.csv --output feature_extraction/output/features.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration (relative to project root, i.e. the parent of this file)
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

DEFAULT_INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "save")
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "output")
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "features.csv")

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
# 1. TECHNICAL INDICATORS CALCULATOR  (shared logic, unchanged)
# ---------------------------------------------------------------------------
class NonOverlappingIndicators:
    """Computes a set of non-overlapping technical indicators on OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def calculate_all(self) -> pd.DataFrame:
        df = self.df

        # 1. Log Returns
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # 2. MACD Histogram
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = macd_line - signal_line

        # 3. RSI (14)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # 4. Bollinger Band Width
        bb_middle = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df["BB_Width"] = (bb_upper - bb_lower) / bb_middle

        # 5. Normalized ATR
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        df["ATR_Normalized"] = atr / df["Close"]

        # 6. Supertrend Direction
        multiplier = 3.0
        hl2 = (df["High"] + df["Low"]) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        supertrend = np.ones(len(df))

        close_vals = df["Close"].to_numpy(copy=True)
        b_upper_vals = basic_upper.to_numpy(copy=True)
        b_lower_vals = basic_lower.to_numpy(copy=True)
        f_upper_vals = final_upper.to_numpy(copy=True)
        f_lower_vals = final_lower.to_numpy(copy=True)

        for i in range(1, len(df)):
            if (b_upper_vals[i] < f_upper_vals[i - 1]) or (
                close_vals[i - 1] > f_upper_vals[i - 1]
            ):
                f_upper_vals[i] = b_upper_vals[i]
            else:
                f_upper_vals[i] = f_upper_vals[i - 1]

            if (b_lower_vals[i] > f_lower_vals[i - 1]) or (
                close_vals[i - 1] < f_lower_vals[i - 1]
            ):
                f_lower_vals[i] = b_lower_vals[i]
            else:
                f_lower_vals[i] = f_lower_vals[i - 1]

            if supertrend[i - 1] == 1 and close_vals[i] < f_lower_vals[i]:
                supertrend[i] = -1
            elif supertrend[i - 1] == -1 and close_vals[i] > f_upper_vals[i]:
                supertrend[i] = 1
            else:
                supertrend[i] = supertrend[i - 1]

        df["Supertrend_Dir"] = supertrend

        # 7. Volume Z-Score
        vol_mean = df["Volume"].rolling(window=20).mean()
        vol_std = df["Volume"].rolling(window=20).std()
        df["Volume_ZScore"] = (df["Volume"] - vol_mean) / (vol_std + 1e-10)

        # 8. OBV Pct Change
        obv_direction = np.sign(df["Close"].diff()).fillna(0)
        obv = (obv_direction * df["Volume"]).cumsum()
        df["OBV_Pct_Change"] = obv.pct_change(periods=5)

        df.dropna(inplace=True)
        return df


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def _resolve_input_path(input_path: str) -> str:
    """If a directory is given, pick the first CSV file found inside it."""
    if os.path.isdir(input_path):
        csvs = sorted(glob.glob(os.path.join(input_path, "*.csv")))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in {input_path}")
        return csvs[0]
    return input_path


def _standardize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Maps a variety of raw column-naming conventions to the canonical
    Date / Open / High / Low / Close / Volume schema expected by
    NonOverlappingIndicators.
    """
    # normalize header names for matching (lower, strip)
    lower_map = {c.lower().strip(): c for c in df_raw.columns}

    def find(*candidates):
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    col_date = find("datetime", "date", "timestamp")
    col_open = find("open", "banknifty_open")
    col_high = find("high", "banknifty_high")
    col_low = find("low", "banknifty_low")
    col_close = find("close", "banknifty_close")
    col_volume = find("volume", "banknifty_volume")

    missing = [
        name
        for name, col in [
            ("date/datetime", col_date),
            ("open", col_open),
            ("high", col_high),
            ("low", col_low),
            ("close", col_close),
            ("volume", col_volume),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not find required column(s) {missing} in input file. "
            f"Available columns: {list(df_raw.columns)}"
        )

    df = df_raw[[col_date, col_open, col_high, col_low, col_close, col_volume]].copy()
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df.set_index("Date", inplace=True)
    return df


# ---------------------------------------------------------------------------
# 3. Main entry point
# ---------------------------------------------------------------------------
def run_feature_extraction(input_path: str = DEFAULT_INPUT_DIR,
                            output_path: str = DEFAULT_OUTPUT_FILE) -> pd.DataFrame:
    resolved_input = _resolve_input_path(input_path)
    print(f"Loading raw dataset from: {resolved_input}")
    df_raw = pd.read_csv(resolved_input)

    df = _standardize_columns(df_raw)

    print("Calculating technical indicators...")
    calculator = NonOverlappingIndicators(df)
    df_features = calculator.calculate_all()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_features.to_csv(output_path)

    print(f"Feature rows produced: {len(df_features)}")
    print(f"Features saved to: {output_path}")

    if len(df_features) == 0:
        print(
            "WARNING: 0 rows survived indicator calculation + dropna(). "
            "This happens when the input file has fewer rows than the "
            "longest rolling/EWM warm-up window (20 bars for BB/Volume "
            "Z-Score, 14 bars min_periods for RSI/ATR). Supply a longer "
            "history for non-empty output."
        )

    return df_features


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction for anomaly detection")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR,
                         help="Path to input CSV file or a directory (data/save by default)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE,
                         help="Path to output features CSV (feature_extraction/output/features.csv by default)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_feature_extraction(input_path=args.input, output_path=args.output)
