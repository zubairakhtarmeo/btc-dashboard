"""
fetch_btc_dataset.py
====================
Fetch, clean, validate, and save a production-grade BTC/USDT 1H OHLCV dataset
from Binance via ccxt.

Target: 20 000+ rows (~2.3 years of hourly candles).
Output:
  cache/price_bitcoin_20000_1h.pkl   (pandas DataFrame, UTC DatetimeIndex)
  cache/price_bitcoin_20000_1h.csv   (CSV copy for inspection)

Run:
  python fetch_btc_dataset.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────
SYMBOL          = "BTC/USDT"
TIMEFRAME       = "1h"
BATCH_SIZE      = 1000            # Binance max candles per request
TARGET_ROWS     = 20_000          # stop once we have this many clean rows
MIN_ROWS        = 10_000          # hard minimum for training use
MAX_PRICE_JUMP  = 0.20            # flag (but don't drop) single-candle moves > 20 %
CACHE_DIR       = Path("cache")
OUT_PKL         = CACHE_DIR / "price_bitcoin_20000_1h.pkl"
OUT_CSV         = CACHE_DIR / "price_bitcoin_20000_1h.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def _ts_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _init_exchange():
    """Return a ccxt Binance exchange object (no API key needed for public OHLCV)."""
    import ccxt
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return exchange


def _fetch_all_ohlcv(exchange, symbol: str, timeframe: str, target: int) -> pd.DataFrame:
    """
    Paginate backwards until we have `target` rows.
    Binance returns at most BATCH_SIZE candles per call.
    We walk back in time by setting `since` to (oldest_ts - batch_window).
    """
    ms_per_candle = 3_600_000  # 1 hour in ms
    all_candles: list[list] = []

    # Start from now, walk backwards
    until_ms: int | None = None
    print(f"[FETCH] Target: {target:,} rows of {symbol} {timeframe}")
    print(f"[FETCH] Batch size: {BATCH_SIZE} candles per request")

    attempt = 0
    max_attempts = 60

    while len(all_candles) < target and attempt < max_attempts:
        attempt += 1

        # Calculate `since` for this batch
        if until_ms is None:
            since_ms = None  # first call: latest available
        else:
            since_ms = until_ms - BATCH_SIZE * ms_per_candle

        try:
            params = {}
            if until_ms is not None:
                params["endTime"] = until_ms

            if since_ms is not None:
                batch = exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=BATCH_SIZE, params=params
                )
            else:
                batch = exchange.fetch_ohlcv(symbol, timeframe, limit=BATCH_SIZE)

        except Exception as exc:
            print(f"[FETCH] API error on attempt {attempt}: {exc}")
            time.sleep(5)
            continue

        if not batch:
            print("[FETCH] Empty batch returned — stopping.")
            break

        all_candles = batch + all_candles  # prepend older data

        oldest_ts = batch[0][0]
        newest_ts = batch[-1][0]
        until_ms  = oldest_ts  # next iteration fetches data before this

        print(
            f"[FETCH] Batch {attempt:>2}: {len(batch):>4} candles "
            f"| {_ts_to_dt(oldest_ts)} -> {_ts_to_dt(newest_ts)} "
            f"| total so far: {len(all_candles):>6}"
        )

        # Avoid hammering the API
        time.sleep(exchange.rateLimit / 1000 + 0.1)

    print(f"\n[FETCH] Raw candles collected: {len(all_candles):,}")
    if not all_candles:
        raise RuntimeError("No candles fetched. Check network / API availability.")

    cols = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(all_candles, columns=cols)
    return df


# ── cleaning ─────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3 — DATA CLEANING")
    print("=" * 60)

    raw_len = len(df)

    # 1. Convert numerics
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  [1] Numeric conversion done. Rows: {len(df):,}")

    # 2. Build UTC DatetimeIndex from timestamp_ms
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = df.set_index("timestamp")
    print(f"  [2] DatetimeIndex set (UTC).")

    # 3. Sort ascending
    df = df.sort_index()
    print(f"  [3] Sorted ascending.")

    # 4. Remove duplicate timestamps (keep last)
    n_dups = df.index.duplicated().sum()
    if n_dups > 0:
        df = df[~df.index.duplicated(keep="last")]
        print(f"  [4] Removed {n_dups} duplicate timestamps. Rows: {len(df):,}")
    else:
        print(f"  [4] No duplicate timestamps found.")

    # 5. Drop NaN rows
    n_before = len(df)
    df = df.dropna()
    dropped_nan = n_before - len(df)
    if dropped_nan:
        print(f"  [5] Dropped {dropped_nan} NaN rows. Rows: {len(df):,}")
    else:
        print(f"  [5] No NaN rows found.")

    # 6. Drop zero/negative prices
    price_mask = (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
    n_bad_price = (~price_mask).sum()
    if n_bad_price:
        df = df[price_mask]
        print(f"  [6] Dropped {n_bad_price} rows with zero/negative prices. Rows: {len(df):,}")
    else:
        print(f"  [6] All prices positive.")

    # 7. Fix OHLC consistency: ensure low <= min(open,close) and high >= max(open,close)
    #    For rows that violate, clamp rather than drop (exchange data is almost always correct).
    lo_fix = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
    hi_fix = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
    if lo_fix or hi_fix:
        df["low"]  = df[["low",  "open", "close"]].min(axis=1)
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        print(f"  [7] Clamped {lo_fix} low values and {hi_fix} high values.")
    else:
        print(f"  [7] OHLC consistency OK.")

    # 8. Drop zero-volume rows (usually exchange outages / test candles)
    n_zero_vol = (df["volume"] == 0).sum()
    if n_zero_vol > 0:
        df = df[df["volume"] > 0]
        print(f"  [8] Dropped {n_zero_vol} zero-volume candles. Rows: {len(df):,}")
    else:
        print(f"  [8] All volume > 0.")

    print(f"\n  Cleaning summary: {raw_len:,} → {len(df):,} rows "
          f"({raw_len - len(df):,} removed)")
    return df


# ── validation ────────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 60)
    print("STEP 4 — DATA VALIDATION")
    print("=" * 60)
    passed = True

    # 4.1 Row count
    n = len(df)
    ok = n >= MIN_ROWS
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Total rows: {n:,}  (minimum required: {MIN_ROWS:,})")
    if not ok:
        passed = False

    # 4.2 Date range
    date_min = df.index.min()
    date_max = df.index.max()
    span_days = (date_max - date_min).days
    print(f"  [INFO] Date range: {date_min.strftime('%Y-%m-%d %H:%M UTC')} "
          f"-> {date_max.strftime('%Y-%m-%d %H:%M UTC')} ({span_days} days)")

    # 4.3 Missing values
    pct_missing = df.isnull().mean().mean() * 100
    ok = pct_missing == 0.0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Missing values: {pct_missing:.4f}%")
    if not ok:
        passed = False

    # 4.4 Duplicate index
    n_dups = df.index.duplicated().sum()
    ok = n_dups == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Duplicate timestamps: {n_dups}")
    if not ok:
        passed = False

    # 4.5 Monotonic index
    ok = df.index.is_monotonic_increasing
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Timestamps strictly increasing: {ok}")
    if not ok:
        passed = False

    # 4.6 Price-continuity: flag large jumps but don't fail on them (can be real events)
    returns = df["close"].pct_change().abs()
    large_jumps = (returns > MAX_PRICE_JUMP).sum()
    ok_warn = large_jumps == 0
    status = "PASS" if ok_warn else "WARN"
    print(f"  [{status}] Single-candle moves > {MAX_PRICE_JUMP*100:.0f}%: {large_jumps}")
    if large_jumps > 0:
        worst = float(returns.max()) * 100
        print(f"           Largest single-candle move: {worst:.2f}%")

    # 4.7 OHLC sanity: low <= open/close <= high
    lo_ok  = (df["low"]  <= df[["open", "close"]].min(axis=1) + 1e-8).all()
    hi_ok  = (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-8).all()
    ok = lo_ok and hi_ok
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] OHLC consistency (low≤price≤high): {ok}")
    if not ok:
        passed = False

    # 4.8 Volume
    neg_vol = (df["volume"] < 0).sum()
    zero_vol = (df["volume"] == 0).sum()
    ok = neg_vol == 0 and zero_vol == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Volume (neg={neg_vol}, zero={zero_vol})")
    if not ok:
        passed = False

    return passed


# ── feature sanity ────────────────────────────────────────────────────────────

def _feature_sanity(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 5 — FEATURE SANITY CHECK")
    print("=" * 60)

    returns = df["close"].pct_change().dropna()
    vol_30  = returns.rolling(30).std().dropna()

    print(f"  Returns (pct_change):")
    print(f"    count:  {len(returns):,}")
    print(f"    mean:   {returns.mean()*100:+.4f}%")
    print(f"    std:    {returns.std()*100:.4f}%")
    print(f"    min:    {returns.min()*100:.4f}%")
    print(f"    max:    {returns.max()*100:.4f}%")
    print(f"    NaNs:   {returns.isnull().sum()}")

    all_zero = (returns.abs() < 1e-10).mean() * 100
    print(f"    near-zero fraction: {all_zero:.2f}%")

    print(f"\n  30-period rolling volatility:")
    print(f"    count: {len(vol_30):,}")
    print(f"    mean:  {vol_30.mean()*100:.4f}%")
    print(f"    std:   {vol_30.std()*100:.4f}%")
    print(f"    NaNs:  {vol_30.isnull().sum()}")

    if all_zero > 50:
        print("  [WARN] > 50% of returns are near-zero — data may be flat/corrupt.")
    else:
        print("  [PASS] Return distribution looks healthy.")


# ── save ──────────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 6 — SAVING DATA")
    print("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df.to_pickle(str(OUT_PKL))
    print(f"  [OK] Pickle saved: {OUT_PKL}  ({OUT_PKL.stat().st_size / 1e6:.2f} MB)")

    df.to_csv(str(OUT_CSV))
    print(f"  [OK] CSV saved:    {OUT_CSV}  ({OUT_CSV.stat().st_size / 1e6:.2f} MB)")


# ── report ────────────────────────────────────────────────────────────────────

def _report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("STEP 7 — FINAL REPORT")
    print("=" * 60)

    returns = df["close"].pct_change().dropna()

    print(f"  Total rows:     {len(df):,}")
    print(f"  Date range:     {df.index.min().strftime('%Y-%m-%d %H:%M UTC')} "
          f"-> {df.index.max().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Span:           {(df.index.max() - df.index.min()).days} days "
          f"({(df.index.max() - df.index.min()).days / 365:.2f} years)")
    print(f"  Columns:        {list(df.columns)}")
    print(f"\n  Price stats (close):")
    print(f"    Min:  ${df['close'].min():>12,.2f}")
    print(f"    Max:  ${df['close'].max():>12,.2f}")
    print(f"    Last: ${df['close'].iloc[-1]:>12,.2f}")
    print(f"\n  Return stats (1H pct_change):")
    print(f"    Mean:   {returns.mean()*100:+.4f}%")
    print(f"    Std:    {returns.std()*100:.4f}%")
    print(f"    Skew:   {float(returns.skew()):.3f}")
    print(f"    Kurt:   {float(returns.kurt()):.3f}")

    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())
    print(f"\n  Last 3 rows:")
    print(df.tail(3).to_string())

    print(f"\n{'=' * 60}")
    print(f"  Dataset ready: {OUT_PKL}")
    print(f"  Use this path in train_simple.py or data_collector.py.")
    print(f"{'=' * 60}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("BTC/USDT Historical Dataset Builder")
    print(f"Target: {TARGET_ROWS:,} rows of {TIMEFRAME} OHLCV from Binance")
    print("=" * 60 + "\n")

    # ── Step 1 & 2: fetch ─────────────────────────────────────────────────────
    print("STEP 1-2 — FETCHING DATA FROM BINANCE")
    print("=" * 60)
    try:
        exchange = _init_exchange()
        print(f"  Exchange: {exchange.name}  (rate limit: {exchange.rateLimit} ms)")
        raw_df = _fetch_all_ohlcv(exchange, SYMBOL, TIMEFRAME, TARGET_ROWS)
    except Exception as exc:
        print(f"\n[ERROR] Fetch failed: {exc}")
        traceback.print_exc()
        print("\nFalling back to alternative source (public Binance REST API)...")
        raw_df = _fetch_binance_rest_fallback(TARGET_ROWS)

    # ── Step 3: clean ─────────────────────────────────────────────────────────
    df = _clean(raw_df)

    # ── Step 4: validate ──────────────────────────────────────────────────────
    validation_ok = _validate(df)
    if not validation_ok:
        print("\n[ERROR] Validation failed. See messages above. Aborting save.")
        sys.exit(1)

    # ── Step 5: feature sanity ────────────────────────────────────────────────
    _feature_sanity(df)

    # ── Step 6: save ──────────────────────────────────────────────────────────
    _save(df)

    # ── Step 7: report ────────────────────────────────────────────────────────
    _report(df)


def _fetch_binance_rest_fallback(target: int) -> pd.DataFrame:
    """
    Direct Binance REST API fallback (no ccxt dependency).
    GET https://api.binance.com/api/v3/klines
    """
    import requests

    BASE_URL  = "https://api.binance.com/api/v3/klines"
    LIMIT     = 1000
    INTERVAL  = "1h"
    ms_per_h  = 3_600_000

    all_candles: list[list] = []
    end_ms: int | None = None

    print("[FALLBACK] Using direct Binance REST API")
    attempt = 0
    max_attempts = 60

    while len(all_candles) < target and attempt < max_attempts:
        attempt += 1
        params: dict = {"symbol": "BTCUSDT", "interval": INTERVAL, "limit": LIMIT}
        if end_ms is not None:
            params["endTime"] = end_ms

        try:
            resp = requests.get(BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:
            print(f"  [FALLBACK] HTTP error attempt {attempt}: {exc}")
            time.sleep(5)
            continue

        if not batch:
            print("  [FALLBACK] Empty batch — done.")
            break

        # Binance kline format: [open_time, open, high, low, close, volume, ...]
        parsed = [[c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                  for c in batch]

        all_candles = parsed + all_candles
        oldest_ts = parsed[0][0]
        newest_ts = parsed[-1][0]
        end_ms = oldest_ts  # next fetch ends here

        print(
            f"  [FALLBACK] Batch {attempt:>2}: {len(parsed):>4} candles "
            f"| {_ts_to_dt(oldest_ts)} -> {_ts_to_dt(newest_ts)} "
            f"| total: {len(all_candles):>6}"
        )
        time.sleep(0.25)

    print(f"  [FALLBACK] Collected {len(all_candles):,} raw candles")
    if not all_candles:
        raise RuntimeError("Fallback also failed. Check network connectivity.")

    cols = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    return pd.DataFrame(all_candles, columns=cols)


if __name__ == "__main__":
    main()
