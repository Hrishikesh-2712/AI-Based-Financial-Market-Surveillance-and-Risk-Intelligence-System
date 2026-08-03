# -*- coding: utf-8 -*-
"""
feature_extract.py
==================
Technical indicator features for Bank Nifty OHLCV data.

Each indicator is a small, independent function that reads a DataFrame with
Open/High/Low/Close/Volume columns and adds its feature column(s). Compose
them with compute_features() to build the full feature set, or call them
individually for testing/debugging.

Run:
    python feature_extract/feature_extract.py   # train/test split + scaler
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ATR_PERIOD,
    BB_STD,
    BB_WINDOW,
    FEATURE_COLUMNS,
    OBV_PERIODS,
    RSI_PERIOD,
    SAVED_MODELS_DIR,
    SUPERTREND_MULTIPLIER,
    VOLUME_WINDOW,
)


# ---------------------------------------------------------------------------
# Indicator functions
# ---------------------------------------------------------------------------


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def macd(df: pd.DataFrame) -> pd.DataFrame:
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd_line - signal_line
    return df


def rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    return df


def bollinger(df: pd.DataFrame) -> pd.DataFrame:
    middle = df["Close"].rolling(BB_WINDOW).mean()
    std = df["Close"].rolling(BB_WINDOW).std()
    df["BB_Width"] = (2 * BB_STD * std) / middle
    return df


def _atr_series(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def atr(df: pd.DataFrame) -> pd.DataFrame:
    df["ATR_Normalized"] = _atr_series(df) / df["Close"]
    return df


def supertrend(df: pd.DataFrame) -> pd.DataFrame:
    """Supertrend direction (+1 uptrend / -1 downtrend), recursively smoothed."""
    hl2 = (df["High"] + df["Low"]) / 2
    atr_series = _atr_series(df)
    basic_upper = hl2 + (SUPERTREND_MULTIPLIER * atr_series)
    basic_lower = hl2 - (SUPERTREND_MULTIPLIER * atr_series)

    final_upper = basic_upper.to_numpy().copy()
    final_lower = basic_lower.to_numpy().copy()
    direction = np.ones(len(df), dtype=np.int8)
    close = df["Close"].to_numpy()
    b_upper = basic_upper.to_numpy()
    b_lower = basic_lower.to_numpy()

    for i in range(1, len(df)):
        final_upper[i] = b_upper[i] if (b_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]) else final_upper[i - 1]
        final_lower[i] = b_lower[i] if (b_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]) else final_lower[i - 1]

        if direction[i - 1] == 1 and close[i] < final_lower[i]:
            direction[i] = -1
        elif direction[i - 1] == -1 and close[i] > final_upper[i]:
            direction[i] = 1
        else:
            direction[i] = direction[i - 1]

    df["Supertrend_Dir"] = direction
    return df


def volume_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df["Volume"].rolling(VOLUME_WINDOW).mean()
    std = df["Volume"].rolling(VOLUME_WINDOW).std()
    df["Volume_ZScore"] = (df["Volume"] - mean) / (std + 1e-10)
    return df


def obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["Close"].diff()).fillna(0)
    df["OBV_Pct_Change"] = (direction * df["Volume"]).cumsum().pct_change(periods=OBV_PERIODS)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every indicator and drop rows without a full indicator window."""
    df = df.copy()
    df = log_returns(df)
    df = macd(df)
    df = rsi(df)
    df = bollinger(df)
    df = atr(df)
    df = supertrend(df)
    df = volume_zscore(df)
    df = obv(df)
    return df.dropna()


# ---------------------------------------------------------------------------
# Train/test split + scaler
# ---------------------------------------------------------------------------

BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def extract_split_and_scale(
    input_path: str,
    train_output_path: str,
    test_output_path: str,
    train_ratio: float = 0.8,
):
    print(f"[1/4] Reading data from {input_path}...")
    df = pd.read_csv(input_path)

    print("[2/4] Calculating indicators...")
    df_features = compute_features(df)[BASE_COLUMNS + FEATURE_COLUMNS].copy()

    print(f"[3/4] Splitting data chronologically (Train: {train_ratio*100}%, Test: {(1-train_ratio)*100}%)...")
    split_idx = int(len(df_features) * train_ratio)
    train_df = df_features.iloc[:split_idx].copy()
    test_df = df_features.iloc[split_idx:].copy()

    print("[4/4] Fitting StandardScaler on training data and applying to both sets...")
    scaler = StandardScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.pkl")

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Training Set: {train_df.shape[0]} rows -> {train_output_path}")
    print(f"Testing Set:  {test_df.shape[0]} rows -> {test_output_path}")


if __name__ == "__main__":
    extract_split_and_scale(
        input_path=str(PROJECT_ROOT / "data" / "bank_nifty_5min_ohlcv.csv"),
        train_output_path=str(PROJECT_ROOT / "feature_extract" / "bank_nifty_train.csv"),
        test_output_path=str(PROJECT_ROOT / "feature_extract" / "bank_nifty_test.csv"),
        train_ratio=0.8,
    )
