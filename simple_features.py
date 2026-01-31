"""Lightweight feature generation used by training + realtime + dashboard.

Goal:
- Keep the "simple" technical features used by the current simplified model.
- Optionally enrich with news/geopolitics features (USA-weighted) without heavy deps.

Design constraints:
- Must be usable in Streamlit Cloud (avoid transformers/torch).
- Must be deterministic and non-leaky: for each candle at time t, only use news <= t.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


# USA is the key player; weight it higher.
_GEO_KEYWORDS_WEIGHTED: dict[str, float] = {
    # USA focus
    r"\busa\b": 2.5,
    r"\bu\.s\.\b": 2.5,
    r"\bunited\s+states\b": 2.5,
    r"\bwashington\b": 1.5,
    r"\bwhite\s+house\b": 1.5,
    r"\bpentagon\b": 1.5,
    r"\bfed\b": 1.0,
    r"\bfederal\s+reserve\b": 1.0,

    # Middle East / Iran (explicit)
    r"\biran\b": 2.0,
    r"\btehran\b": 1.5,
    r"\bhamas\b": 1.2,
    r"\bhezbollah\b": 1.2,
    r"\bisrael\b": 1.4,
    r"\bgaza\b": 1.4,
    r"\byemen\b": 1.1,
    r"\bhouthi\w*\b": 1.2,

    # Energy / macro spillovers
    r"\boil\b": 1.2,
    r"\bbrent\b": 1.1,
    r"\bwti\b": 1.1,
    r"\bstrait\s+of\s+hormuz\b": 1.8,

    # Conflict / escalation words
    r"\bwar\b": 1.6,
    r"\battack\w*\b": 1.4,
    r"\bmissile\w*\b": 1.4,
    r"\bstrike\w*\b": 1.3,
    r"\bdrone\w*\b": 1.2,
    r"\bescalat\w*\b": 1.4,
    r"\btension\w*\b": 1.2,

    # Policy / sanctions / regulation
    r"\bsanction\w*\b": 1.3,
    r"\bembargo\w*\b": 1.3,
    r"\btrade\s+war\b": 1.2,
    r"\bexport\s+control\w*\b": 1.2,
}

_POS_WORDS = {
    "bullish", "rally", "surge", "soar", "breakout", "approve", "approval", "adoption", "support",
    "ceasefire", "de-escalation", "deal", "agreement", "peace",
}

_NEG_WORDS = {
    "bearish", "crash", "dump", "plunge", "panic", "fear", "selloff", "sell-off",
    "war", "attack", "missile", "strike", "sanction", "embargo", "tension", "escalation",
}


def _ensure_utc_dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out


def _align_alt_timeseries(
    price_index: pd.DatetimeIndex,
    alt_df: pd.DataFrame | None,
    *,
    ts_col_candidates: Iterable[str] = ("timestamp", "fundingTime", "ts", "time"),
    prefix: str,
) -> pd.DataFrame:
    """Align an alternative-data dataframe to the candle index non-leakily.

    Semantics: for each candle time t, use the latest alt observation at-or-before t.
    """
    idx = pd.to_datetime(price_index, utc=True, errors="coerce")
    idx = idx[~idx.isna()]
    out = pd.DataFrame(index=idx)

    if alt_df is None or not isinstance(alt_df, pd.DataFrame) or alt_df.empty:
        return out

    df = alt_df.copy()
    ts_col = None
    for c in ts_col_candidates:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        return out

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        return out

    # Keep only numeric columns (avoid huge objects)
    numeric_cols = [c for c in df.columns if c != ts_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return out

    df = df[[ts_col] + numeric_cols].sort_values(ts_col)
    df["ts_hour"] = df[ts_col].dt.floor("h")
    hourly = df.groupby("ts_hour", as_index=True)[numeric_cols].last().sort_index()

    aligned = hourly.reindex(idx, method="ffill")
    aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    aligned = aligned.add_prefix(prefix)
    return aligned


def _lexicon_sentiment(text: str) -> float:
    """Very lightweight sentiment in [-1, 1]."""
    if not text:
        return 0.0
    t = str(text).lower()
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", t)
    if not words:
        return 0.0

    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)

    denom = pos + neg
    if denom == 0:
        return 0.0

    score = (pos - neg) / float(denom)
    return float(np.clip(score, -1.0, 1.0))


def _keyword_weighted_score(text: str, patterns: dict[str, float]) -> float:
    if not text:
        return 0.0
    t = str(text).lower()
    score = 0.0
    for pat, w in patterns.items():
        try:
            cnt = len(re.findall(pat, t, flags=re.IGNORECASE))
        except re.error:
            cnt = 0
        if cnt:
            score += float(cnt) * float(w)
    return float(score)


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def build_news_geopolitics_features(
    price_index: pd.DatetimeIndex,
    news_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return per-candle features aligned to price_index.

    Non-leaky alignment:
    - bucket news to hourly timestamps
    - for each candle at time t, use aggregates from latest hour <= t

    Output columns are always present (filled with 0 if no news).
    """
    idx = pd.to_datetime(price_index, utc=True, errors="coerce")
    idx = idx[~idx.isna()]

    out = pd.DataFrame(index=idx)
    out["news_count_1h"] = 0.0
    out["news_count_6h"] = 0.0
    out["news_sentiment_6h"] = 0.0
    out["geo_risk_score_1h"] = 0.0
    out["geo_risk_score_6h"] = 0.0
    out["geo_risk_sentiment_6h"] = 0.0
    out["geo_usa_mentions_6h"] = 0.0
    out["geo_iran_mentions_6h"] = 0.0

    if news_df is None or not isinstance(news_df, pd.DataFrame) or news_df.empty:
        return out

    df = news_df.copy()
    if "timestamp" not in df.columns:
        return out

    df["timestamp"] = _ensure_utc_dt(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return out

    title = df.get("title", "")
    desc = df.get("description", "")
    text = (title.fillna("").astype(str) + " " + desc.fillna("").astype(str)).str.strip()

    df["_sent"] = text.map(_lexicon_sentiment)
    df["_geo"] = text.map(lambda t: _keyword_weighted_score(t, _GEO_KEYWORDS_WEIGHTED))

    # Explicit USA / Iran mention counts (6h rolling later)
    df["_usa"] = text.str.count(r"(?i)\b(usa|u\.s\.|united\s+states|washington|white\s+house|pentagon)\b", flags=re.IGNORECASE)
    df["_iran"] = text.str.count(r"(?i)\b(iran|tehran)\b", flags=re.IGNORECASE)

    # Bucket to hour
    df["ts_hour"] = df["timestamp"].dt.floor("h")

    hourly = df.groupby("ts_hour", as_index=True).agg(
        news_count=("ts_hour", "size"),
        sentiment_mean=("_sent", "mean"),
        geo_score=("_geo", "sum"),
        usa_mentions=("_usa", "sum"),
        iran_mentions=("_iran", "sum"),
    )

    # Align to price index hourly grid then forward-fill for "last known <= t" semantics.
    hourly = hourly.sort_index()

    # Reindex to the price index hours; forward-fill non-leaky.
    hourly_on_price = hourly.reindex(idx, method="ffill")

    # Rolling windows computed on the aligned (already non-leaky) series
    out["news_count_1h"] = hourly_on_price["news_count"].fillna(0.0).astype(float)
    out["news_count_6h"] = hourly_on_price["news_count"].fillna(0.0).rolling(6, min_periods=1).sum()

    out["news_sentiment_6h"] = hourly_on_price["sentiment_mean"].fillna(0.0).rolling(6, min_periods=1).mean()

    # Geo risk: log-scaled to avoid huge spikes
    geo_1h = hourly_on_price["geo_score"].fillna(0.0).astype(float)
    out["geo_risk_score_1h"] = np.log1p(np.maximum(0.0, geo_1h))
    out["geo_risk_score_6h"] = np.log1p(np.maximum(0.0, geo_1h.rolling(6, min_periods=1).sum()))

    # Interaction: risk * (negative sentiment)
    neg_sent_6h = np.clip(-out["news_sentiment_6h"], 0.0, 1.0)
    out["geo_risk_sentiment_6h"] = out["geo_risk_score_6h"] * neg_sent_6h

    out["geo_usa_mentions_6h"] = hourly_on_price["usa_mentions"].fillna(0.0).rolling(6, min_periods=1).sum()
    out["geo_iran_mentions_6h"] = hourly_on_price["iran_mentions"].fillna(0.0).rolling(6, min_periods=1).sum()

    # Fill any remaining NaNs
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_simple_features(
    price_df: pd.DataFrame,
    *,
    news_df: pd.DataFrame | None = None,
    derivatives_df: pd.DataFrame | None = None,
    funding_df: pd.DataFrame | None = None,
    liquidations_df: pd.DataFrame | None = None,
    options_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the simplified feature set, optionally enriched with news/geopolitics."""
    if price_df is None or not isinstance(price_df, pd.DataFrame) or len(price_df) == 0:
        return pd.DataFrame()

    df = price_df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = _ensure_utc_dt(df["timestamp"])
        else:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    # Basic returns
    for h in [1, 6, 12, 24, 48, 168]:
        df[f"return_{h}h"] = df["close"].pct_change(h)
        df[f"log_return_{h}h"] = np.log1p(df[f"return_{h}h"])

    # Simple moving averages
    for period in [7, 14, 21, 50, 100, 200]:
        df[f"sma_{period}"] = df["close"].rolling(period).mean()
        df[f"distance_sma_{period}"] = (df["close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]

    # Volatility
    for period in [7, 14, 21, 50]:
        df[f"volatility_{period}"] = df["return_1h"].rolling(period).std()

    # Volume indicators
    if "volume" in df.columns:
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    else:
        df["volume_sma_20"] = 0.0
        df["volume_ratio"] = 0.0

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
    df["bb_percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

    # ATR
    if all(c in df.columns for c in ["high", "low", "close"]):
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df["atr_14"] = true_range.rolling(14).mean()
        df["atr_ratio"] = df["atr_14"] / df["close"]
    else:
        df["atr_14"] = 0.0
        df["atr_ratio"] = 0.0

    # Momentum
    df["momentum_consistency"] = (np.sign(df["return_1h"]) == np.sign(df["return_6h"])).astype(int)

    # Optional news/geopolitics features
    geo = build_news_geopolitics_features(df.index, news_df)
    for col in geo.columns:
        df[col] = geo[col].values

    # Optional derivatives / liquidations / options features (best-effort)
    deriv = _align_alt_timeseries(df.index, derivatives_df, prefix="deriv_")
    fund = _align_alt_timeseries(df.index, funding_df, prefix="funding_")
    liq = _align_alt_timeseries(df.index, liquidations_df, prefix="liq_")
    opt = _align_alt_timeseries(df.index, options_df, prefix="options_")

    for extra in (deriv, fund, liq, opt):
        if extra is None or extra.empty:
            continue
        for col in extra.columns:
            df[col] = extra[col].values

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna()
