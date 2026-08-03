import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent

load_dotenv(DATA_DIR / ".env")

APP_ID = os.getenv("FYERS_APP_ID")
INDEX_SYMBOL = "NSE:NIFTYBANK-INDEX"
WARMUP_DAYS = 7

# FYERS returns epoch timestamps which are true instants but are parsed as
# naive UTC. NSE market hours are IST, so convert everything to Asia/Kolkata
# so the candle chart, session date, and anomaly markers show IST wall time.
IST_TZ = "Asia/Kolkata"

CACHE_CSV_PATH = DATA_DIR / "bank_nifty_5min_ohlcv.csv"
CURRENT_DATA_CSV = DATA_DIR / "current_data.csv"  # incremental live cache
ROWS_PER_SESSION_DAY = 75  # NSE 5-min candles: 09:15 -> 15:30

BANK_NIFTY_CONSTITUENTS = [
    "NSE:HDFCBANK-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:KOTAKBANK-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:SBIN-EQ",
    "NSE:INDUSINDBK-EQ",
    "NSE:BANKBARODA-EQ",
    "NSE:PNB-EQ",
    "NSE:IDFCFIRSTB-EQ",
    "NSE:AUBANK-EQ",
    "NSE:FEDERALBNK-EQ",
    "NSE:BANDHANBNK-EQ"
]

def load_access_token():
    """Reads saved token from auth.py execution."""
    candidates = [
        DATA_DIR / "access_token.txt",
        PROJECT_ROOT / "data" / "access_token.txt",
        Path("access_token.txt"),
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(
        "Run 'python data/auth.py' first to generate data/access_token.txt"
    )


def create_fyers_client():
    """Return an authenticated FYERS client."""
    access_token = load_access_token()
    return fyersModel.FyersModel(
        client_id=APP_ID,
        is_async=False,
        token=access_token,
        log_path=str(DATA_DIR),
    )


def fetch_live_bank_nifty(warmup_days: int = WARMUP_DAYS, resolution: str = "5") -> pd.DataFrame:
    """
    Fetch Bank Nifty OHLCV via FYERS for indicator warmup plus today's session.

    Returns a DataFrame indexed by timestamp with columns
    Open, High, Low, Close, Volume.
    """
    fyers = create_fyers_client()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=warmup_days)
    return fetch_bank_nifty_ohlcv(
        fyers=fyers,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
    )


def fetch_bank_nifty_from_cache(warmup_days: int = WARMUP_DAYS) -> pd.DataFrame:
    """
    Fallback data source: serve the most recent warmup window from the local
    CSV cache (data/bank_nifty_5min_ohlcv.csv) so the pipeline and UI keep
    working when the FYERS API is unreachable (offline / expired token).

    Returns a DataFrame indexed by timestamp with columns
    Open, High, Low, Close, Volume.
    """
    if not CACHE_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV cache at {CACHE_CSV_PATH}")

    df = pd.read_csv(CACHE_CSV_PATH, parse_dates=["timestamp"])

    # Normalise cached timestamps to IST: the CSV may contain either naive
    # UTC rows (saved before the IST change) or already-aware IST rows.
    ts = df["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    df["timestamp"] = ts

    df = df.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    n_rows = (warmup_days * ROWS_PER_SESSION_DAY) + 50
    return df.tail(n_rows)


def _now_ist() -> datetime:
    """Current time as a tz-aware IST datetime."""
    return datetime.now(ZoneInfo(IST_TZ))


def _to_ist(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Normalise a timestamp column to IST.

    Naive timestamps in the caches were written as UTC wall time (pre-IST
    change), so they are localised to UTC first, then converted to IST.
    Already-aware timestamps are simply converted.
    """
    ts = df[col]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(IST_TZ)
    else:
        ts = ts.dt.tz_convert(IST_TZ)
    df = df.copy()
    df[col] = ts
    return df


def load_current_data() -> pd.DataFrame:
    """Read the incremental live cache (data/current_data.csv).

    Returns a DataFrame indexed by IST-aware timestamp with OHLCV columns.
    """
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if not CURRENT_DATA_CSV.exists():
        return empty

    df = pd.read_csv(CURRENT_DATA_CSV, parse_dates=["timestamp"])
    if df.empty:
        return empty

    df = _to_ist(df)
    df = df.set_index("timestamp")[["Open", "High", "Low", "Close", "Volume"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def save_current_data(df: pd.DataFrame) -> Path:
    """Persist the incremental live cache, deduplicated by timestamp."""
    if df.empty:
        return CURRENT_DATA_CSV
    out = df.reset_index()
    out = _to_ist(out)
    out = out[~out["timestamp"].duplicated(keep="last")].sort_values("timestamp")
    CURRENT_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CURRENT_DATA_CSV, index=False)
    return CURRENT_DATA_CSV


def fetch_incremental_bank_nifty(
    warmup_days: int = WARMUP_DAYS, resolution: str = "5"
) -> pd.DataFrame:
    """
    Incremental live fetch for the 5-minute poll cycle.

    - If data/current_data.csv already holds today's candles, reuse it and
      fetch only the missing tail from FYERS (merged + deduplicated).
    - Otherwise (fresh day / empty cache) do a full warmup fetch.
    """
    cached = load_current_data()
    has_today = (
        not cached.empty and cached.index.max().date() == _now_ist().date()
    )

    if not has_today:
        print("Incremental fetch: cache empty/stale -> full warmup fetch.")
        return fetch_live_bank_nifty(warmup_days=warmup_days, resolution=resolution)

    fyers = create_fyers_client()
    last_ts = cached.index.max()
    print(
        f"Incremental fetch: reusing cached candles up to "
        f"{last_ts:%Y-%m-%d %H:%M} IST, fetching only the missing tail."
    )
    start_date = last_ts.normalize()  # from the cached session's day
    end_date = _now_ist()

    fetched = fetch_bank_nifty_ohlcv(
        fyers=fyers,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
    )

    combined = pd.concat([cached, fetched])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def fetch_history_chunked(fyers, symbol, start_date, end_date, resolution="5", chunk_days=90):
    """
    Fetches historical candle data in 90-day chunks to respect FYERS API limits.
    """
    all_candles = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": current_start.strftime("%Y-%m-%d"),
            "range_to": current_end.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }

        response = fyers.history(data=data)

        if response.get("s") == "ok" and "candles" in response:
            all_candles.extend(response["candles"])
        else:
            print(f"Warning: Chunk failed for {symbol} [{current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}] -> {response.get('message', response)}")

        current_start = current_end + timedelta(days=1)
        time.sleep(0.15)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles, 
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert(IST_TZ)
    )
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)

    return df

def fetch_bank_nifty_ohlcv(fyers, start_date, end_date, resolution="5"):
    # 1. Fetch Bank Nifty Index OHLC Data
    print(f"Fetching Bank Nifty Index ({INDEX_SYMBOL}) OHLC price data...")
    index_df = fetch_history_chunked(
        fyers=fyers,
        symbol=INDEX_SYMBOL,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution
    )

    if index_df.empty:
        raise ValueError("Failed to fetch Bank Nifty Index data from FYERS.")

    # Drop index default zero-volume column
    index_df = index_df[["open", "high", "low", "close"]]

    # 2. Fetch Constituent Volumes in Background to calculate Synthetic Volume
    print(f"\nCalculating synthetic volume from {len(BANK_NIFTY_CONSTITUENTS)} constituents...")
    all_volumes = {}

    for symbol in BANK_NIFTY_CONSTITUENTS:
        print(f"Downloading volume for {symbol}...")
        stock_df = fetch_history_chunked(
            fyers=fyers,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            resolution=resolution
        )
        if not stock_df.empty:
            all_volumes[symbol] = stock_df["volume"]

    if not all_volumes:
        raise ValueError(
            "Failed to fetch constituent volumes; cannot compute synthetic volume. "
            "Check the FYERS token has access to equity symbols."
        )

    # 3. Sum Constituent Volumes
    volume_matrix = pd.DataFrame(all_volumes).fillna(0)
    synthetic_volume = volume_matrix.sum(axis=1)

    # 4. Merge OHLC with Synthetic Volume
    index_df["volume"] = synthetic_volume
    index_df.dropna(inplace=True)

    # Capitalize column names to standard OHLCV format
    index_df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return index_df


if __name__ == "__main__":
    access_token = load_access_token()

    fyers = fyersModel.FyersModel(
        client_id=APP_ID,
        is_async=False,
        token=access_token,
        log_path=""
    )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 1)

    df = fetch_bank_nifty_ohlcv(
        fyers=fyers,
        start_date=start_date,
        end_date=end_date,
        resolution="5"
    )

    print("\n--- Bank Nifty 5-Min OHLCV Dataset ---")
    print(df.tail(10))

    df.to_csv(DATA_DIR / "bank_nifty_5min_ohlcv.csv")
    print("\nSaved Bank Nifty OHLCV data to 'bank_nifty_5min_ohlcv.csv'.")