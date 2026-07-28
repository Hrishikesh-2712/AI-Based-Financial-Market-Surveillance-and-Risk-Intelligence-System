# -*- coding: utf-8 -*-
"""
data.py
=======
Single-file, importable module that does exactly one thing end to end:

    1. Fetches OHLCV candles from the FYERS v3 API for a symbol/interval.
    2. Wherever a candle's volume is 0 (true for index symbols like
       NSE:NIFTYBANK-INDEX, which never carry real volume), recomputes it
       using the SAME formula as the original project's
       `Downloader.download_aggregate_volume`: the SUM of that candle's
       traded volume across all 12 BankNifty constituent stocks.
    3. Exports the result as a timestamped CSV into the /save folder.

No cleaning, indicators, chunked/resumable downloads, retries, etc. -
just fetch -> fix zero volume -> save.

--------------------------------------------------------------------------
Usable as a module (no CLI)
--------------------------------------------------------------------------
    from data import fetch_and_save

    path = fetch_and_save(
        symbol="NSE:NIFTYBANK-INDEX",
        interval="5",
        days=5,
    )
    print(f"Saved to {path}")

--------------------------------------------------------------------------
Login flow
--------------------------------------------------------------------------
On the first call (or once the cached token goes stale), a browser tab
opens to the FYERS login page. REDIRECT_URI must point at localhost
(e.g. http://127.0.0.1:5000/) - this module spins up a throw-away
FastAPI/uvicorn server on that exact host/port to catch the redirect,
pulls the `auth_code` query param straight off the callback request,
exchanges it for an access token, and then shuts the server down.
Nothing is ever pasted by hand. The resulting access token is cached in
`tokens/access_token.json` and reused (same-day) on every later
call/import, so login only happens once per day.

--------------------------------------------------------------------------
Requirements
--------------------------------------------------------------------------
    pip install fyers-apiv3 pandas python-dotenv fastapi uvicorn

A `.env` file (or exported env vars) with:
    FYERS_APP_ID=...
    FYERS_SECRET_KEY=...
    REDIRECT_URI=http://127.0.0.1:5000/   # host/port the local server binds to
"""

from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fyers_apiv3 import fyersModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent

SAVE_DIR = THIS_DIR / "save"
TOKENS_DIR = THIS_DIR / "tokens"
TOKEN_FILE = TOKENS_DIR / "access_token.json"

FYERS_APP_ID = os.getenv("FYERS_APP_ID", "").strip()
FYERS_SECRET_KEY = os.getenv("FYERS_SECRET_KEY", "").strip()
REDIRECT_URI = os.getenv("REDIRECT_URI", "").strip()

DEFAULT_SYMBOL = "NSE:NIFTYBANK-INDEX"
DEFAULT_INTERVAL = "5"  # minutes; also accepts "D", "W", "M", etc.
DEFAULT_DAYS = 5  # lookback window when start/end aren't given

RAW_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# The 12 BankNifty constituent stocks whose summed volume stands in for
# the index's own (always-zero) volume. Same list/formula as the
# original project's aggregate-volume feature.
BANKNIFTY_CONSTITUENTS = [
    "NSE:HDFCBANK-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:SBIN-EQ",
    "NSE:KOTAKBANK-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:INDUSINDBK-EQ",
    "NSE:BANKBARODA-EQ",
    "NSE:PNB-EQ",
    "NSE:CANBK-EQ",
    "NSE:AUBANK-EQ",
    "NSE:FEDERALBNK-EQ",
    "NSE:IDFCFIRSTB-EQ",
]

TOKEN_MAX_AGE_SECONDS = 60 * 60 * 20  # ~20h, well inside FYERS' end-of-day expiry

# Maximum number of days FYERS allows per single history request. Intraday
# resolutions are capped at 100 days; daily/weekly/monthly bars allow up to
# 366. Long date ranges are automatically split into chunks that respect
# these limits (see _chunk_date_range / _fetch_full_range below).
MAX_DAYS_PER_REQUEST = {
    "1": 100, "2": 100, "3": 100, "5": 100, "10": 100, "15": 100,
    "20": 100, "30": 100, "60": 100, "120": 100, "240": 100,
    "D": 366, "W": 366, "M": 366,
}


# ---------------------------------------------------------------------------
# Auth (minimal, self-contained - no dependency on the original src/ package)
# ---------------------------------------------------------------------------
def _validate_credentials() -> None:
    missing = [
        name
        for name, value in (
            ("FYERS_APP_ID", FYERS_APP_ID),
            ("FYERS_SECRET_KEY", FYERS_SECRET_KEY),
            ("REDIRECT_URI", REDIRECT_URI),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in a .env file (see .env.example) or the environment."
        )

    if "-" not in FYERS_APP_ID:
        print(
            "Warning: FYERS_APP_ID does not contain a '-' (e.g. 'XXXXXXX-100'). "
            "Fyers app IDs from the developer dashboard normally include this "
            "suffix - double-check you copied the full ID, not just the prefix."
        )


def _mask(value: str, keep: int = 4) -> str:
    if len(value) <= keep:
        return "*" * len(value)
    return value[:keep] + "*" * (len(value) - keep)


def _load_cached_token() -> Optional[str]:
    if not TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
        age = time.time() - data.get("generated_at", 0.0)
        if age > TOKEN_MAX_AGE_SECONDS:
            return None
        return data.get("access_token")
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _store_token(access_token: str) -> None:
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(
        json.dumps({"access_token": access_token, "generated_at": time.time()}, indent=2),
        encoding="utf-8",
    )


class _AuthCallbackServer:
    """
    A minimal, throw-away FastAPI (uvicorn) server that listens on
    REDIRECT_URI's host/port, captures the `auth_code` FYERS appends to
    the callback request, and shuts itself down as soon as it has one.
    """

    def __init__(self, host: str, port: int, path: str = "/"):
        self.auth_code: Optional[str] = None
        self.error: Optional[str] = None
        self._got_result = threading.Event()

        app = FastAPI()

        @app.get(path)
        async def callback(request: Request):
            self.auth_code = request.query_params.get("auth_code") or request.query_params.get(
                "code"
            )
            if not self.auth_code:
                self.error = f"No auth_code in callback: {dict(request.query_params)}"
            self._got_result.set()
            return HTMLResponse(
                "FYERS login successful. You can close this tab and return to the app."
            )

        config = uvicorn.Config(app, host=host, port=port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def wait_for_auth_code(self, timeout: int = 180) -> str:
        self._thread.start()
        # uvicorn.Server spins up its own event loop asynchronously in the
        # thread; give it a moment to actually start listening.
        while not self._server.started and self._thread.is_alive():
            time.sleep(0.05)
        try:
            received = self._got_result.wait(timeout=timeout)
        finally:
            self._server.should_exit = True
            self._thread.join(timeout=5)

        if not received:
            raise TimeoutError(
                f"Timed out after {timeout}s waiting for the FYERS login redirect."
            )
        if self.error or not self.auth_code:
            raise RuntimeError(self.error or "Callback received but no auth_code was present.")
        return self.auth_code


def _interactive_login() -> str:
    _validate_credentials()
    print(
        f"Using FYERS_APP_ID={_mask(FYERS_APP_ID)} "
        f"FYERS_SECRET_KEY={_mask(FYERS_SECRET_KEY)} "
        f"REDIRECT_URI={REDIRECT_URI}"
    )

    session = fyersModel.SessionModel(
        client_id=FYERS_APP_ID,
        secret_key=FYERS_SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
    )
    login_url = session.generate_authcode()

    parsed = urlparse(REDIRECT_URI)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"REDIRECT_URI must be a localhost URL with an explicit port "
            f"(e.g. http://127.0.0.1:5000/), got: {REDIRECT_URI!r}"
        )

    callback_server = _AuthCallbackServer(
        host=parsed.hostname, port=parsed.port, path=parsed.path or "/"
    )

    print(f"Opening FYERS login page in your browser...\n{login_url}")
    try:
        webbrowser.open(login_url, new=1)
    except Exception:
        print("Could not auto-open a browser; please open the URL above manually.")

    print(f"Waiting for FYERS to redirect back to {REDIRECT_URI} ...")
    auth_code = callback_server.wait_for_auth_code()
    print("Received auth_code from callback; exchanging for access token...")

    session.set_token(auth_code)
    response = session.generate_token()
    if response.get("s") != "ok" or "access_token" not in response:
        if response.get("code") == -5 or "app id hash" in str(response.get("message", "")).lower():
            raise RuntimeError(
                f"Failed to generate access token: {response}\n"
                "This is FYERS rejecting the app_id/secret_key combination, not "
                "the callback/auth_code capture (that part worked). Check:\n"
                "  1. FYERS_APP_ID in .env is the FULL id from the dashboard, "
                "including the '-100' style suffix.\n"
                "  2. FYERS_SECRET_KEY in .env is correct and has no extra "
                "quotes/spaces around it.\n"
                "  3. REDIRECT_URI in .env exactly matches (scheme, host, port, "
                "trailing slash) the redirect URI registered for this app on "
                "the FYERS developer dashboard.\n"
                f"  Currently loaded: FYERS_APP_ID={_mask(FYERS_APP_ID)}, "
                f"REDIRECT_URI={REDIRECT_URI}"
            )
        raise RuntimeError(f"Failed to generate access token: {response}")
    return response["access_token"]


def get_client(force_login: bool = False) -> fyersModel.FyersModel:
    """Return an authenticated FYERS client, reusing a cached token when possible."""
    _validate_credentials()
    token = None if force_login else _load_cached_token()
    if token is None:
        token = _interactive_login()
        _store_token(token)

    client = fyersModel.FyersModel(
        client_id=FYERS_APP_ID, token=token, is_async=False, log_path=""
    )
    profile = client.get_profile()
    if profile.get("s") != "ok":
        # Cached token rejected -> force a fresh login once.
        token = _interactive_login()
        _store_token(token)
        client = fyersModel.FyersModel(
            client_id=FYERS_APP_ID, token=token, is_async=False, log_path=""
        )
    return client


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def _fetch_candles(
    client: fyersModel.FyersModel, symbol: str, interval: str, start: date, end: date
) -> pd.DataFrame:
    """Single history call for one chunk (must respect MAX_DAYS_PER_REQUEST)."""
    payload = {
        "symbol": symbol,
        "resolution": interval,
        "date_format": "1",
        "range_from": start.isoformat(),
        "range_to": end.isoformat(),
        "cont_flag": "1",
    }
    response = client.history(data=payload)
    if response.get("s") != "ok":
        raise RuntimeError(f"FYERS history call failed for {symbol}: {response}")

    candles = response.get("candles") or []
    if not candles:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.DataFrame(candles, columns=RAW_COLUMNS)


def _chunk_date_range(start: date, end: date, interval: str) -> list[tuple[date, date]]:
    """Split [start, end] into chunks that respect FYERS' per-resolution day cap."""
    if start > end:
        raise ValueError(f"start date {start} must be <= end date {end}")

    max_days = MAX_DAYS_PER_REQUEST.get(interval, 100)
    span = timedelta(days=max_days - 1)  # inclusive span

    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + span, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _fetch_full_range(
    client: fyersModel.FyersModel, symbol: str, interval: str, start: date, end: date
) -> pd.DataFrame:
    """
    Fetch the entire [start, end] range for `symbol`, transparently
    splitting it into as many chunk requests as FYERS' per-call day limit
    for `interval` requires, and concatenating (deduped, sorted) results.
    """
    chunks = _chunk_date_range(start, end, interval)
    frames = []
    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(f"  [{symbol}] chunk {i}/{len(chunks)}: {chunk_start} -> {chunk_end}")
        frame = _fetch_candles(client, symbol, interval, chunk_start, chunk_end)
        if not frame.empty:
            frames.append(frame)
        if len(chunks) > 1:
            time.sleep(0.3)  # small, polite gap between chunk requests

    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return combined


def _fetch_aggregate_constituent_volume(
    client: fyersModel.FyersModel, interval: str, start: date, end: date
) -> pd.Series:
    """Sum of all 12 BankNifty constituent volumes, indexed by timestamp."""
    volume_series = []
    for symbol in BANKNIFTY_CONSTITUENTS:
        try:
            frame = _fetch_full_range(client, symbol, interval, start, end)
            if frame.empty:
                continue
            series = frame.set_index("timestamp")["volume"].rename(symbol)
            volume_series.append(series)
        except Exception as exc:  # one bad constituent shouldn't kill the run
            print(f"Warning: could not fetch volume for {symbol}: {exc}")

    if not volume_series:
        return pd.Series(dtype="float64", name="aggregate_volume")

    merged = pd.concat(volume_series, axis=1).fillna(0)
    return merged.sum(axis=1).rename("aggregate_volume")


def _fill_zero_volume(
    df: pd.DataFrame, client: fyersModel.FyersModel, interval: str, start: date, end: date
) -> pd.DataFrame:
    """
    For any row where volume == 0, replace it with the summed BankNifty
    constituent volume for that same candle timestamp (the "current
    formula" used elsewhere in this project for index symbols that carry
    no real volume of their own).
    """
    if df.empty or not (df["volume"] == 0).any():
        return df

    agg_volume = _fetch_aggregate_constituent_volume(client, interval, start, end)
    if agg_volume.empty:
        print("Warning: could not compute aggregate volume; zero-volume rows left as-is.")
        return df

    df = df.copy()
    zero_mask = df["volume"] == 0
    df.loc[zero_mask, "volume"] = (
        df.loc[zero_mask, "timestamp"].map(agg_volume).fillna(0).values
    )
    return df


def _add_datetime_column(df: pd.DataFrame, timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    if df.empty:
        df["datetime"] = pd.Series(dtype="datetime64[ns]")
        return df[["datetime"] + RAW_COLUMNS]

    df = df.copy()
    df["datetime"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)
    )
    return df[["datetime"] + RAW_COLUMNS]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def fetch_and_save(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    start: Optional[date] = None,
    end: Optional[date] = None,
    days: int = DEFAULT_DAYS,
    output_dir: Path | str = SAVE_DIR,
    force_login: bool = False,
) -> Path:
    """
    Fetch candles for `symbol`/`interval`, fix zero-volume rows using the
    aggregate-constituent-volume formula, and write a timestamped CSV into
    `output_dir` (default: ./save).

    Args:
        symbol: FYERS symbol, e.g. "NSE:NIFTYBANK-INDEX".
        interval: FYERS resolution, e.g. "5", "15", "D".
        start / end: explicit inclusive date range. If omitted, defaults
            to the last `days` days ending today.
        days: lookback window used when start/end aren't given.
        output_dir: folder the CSV is written into (created if missing).
        force_login: bypass the cached token and force a fresh login.

    Returns:
        Path to the CSV file that was written.
    """
    end = end or date.today()
    start = start or (end - timedelta(days=days))

    client = get_client(force_login=force_login)

    print(f"Fetching {symbol} [{interval}] {start} -> {end} ...")
    raw = _fetch_full_range(client, symbol, interval, start, end)
    print(f"Fetched {len(raw)} candle(s).")

    raw = _fill_zero_volume(raw, client, interval, start, end)
    df = _add_datetime_column(raw)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol_clean = symbol.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol_clean}_{interval}_{stamp}.csv"
    out_path = output_dir / filename

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} row(s) to: {out_path}")
    return out_path


def get_market_data(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    days: int = DEFAULT_DAYS,
) -> pd.DataFrame:
    """
    Fetches latest 5-minute OHLCV data for given symbol.
    Attempts live FYERS API fetch if credentials are configured.
    Falls back seamlessly to local input CSV data if FYERS credentials are not set/offline.
    """
    if FYERS_APP_ID and FYERS_SECRET_KEY:
        try:
            csv_path = fetch_and_save(symbol=symbol, interval=interval, days=days)
            return pd.read_csv(csv_path)
        except Exception as e:
            print(f"FYERS API fetch notice (using fallback): {e}")

    fallback_file = THIS_DIR / "input" / "NIFTYBANK_5m.csv"
    if fallback_file.exists():
        print(f"Using local market dataset from: {fallback_file}")
        return pd.read_csv(fallback_file)

    save_csvs = list(SAVE_DIR.glob("*.csv"))
    if save_csvs:
        latest = max(save_csvs, key=lambda p: p.stat().st_mtime)
        print(f"Using latest saved dataset from: {latest}")
        return pd.read_csv(latest)

    raise FileNotFoundError("No market data available from FYERS or local CSV files.")


# ---------------------------------------------------------------------------
# Manual test run (run `python data.py` with the defaults below)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    path = fetch_and_save(
        symbol=DEFAULT_SYMBOL,
        interval=DEFAULT_INTERVAL,
        days=DEFAULT_DAYS,
    )
    print(f"Test run complete. CSV written to: {path}")

