"""
Premium Enterprise-Grade Bitcoin Forecasting Dashboard
Bloomberg Terminal Style - AI-Powered Multi-Horizon Predictions
"""

from __future__ import annotations

import json
import pickle
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from tensorflow import keras

from data_collector import CryptoDataCollector
from enhanced_predictor import (
    EnhancedCryptoPricePredictor,
    MultiHeadAttentionCustom,
    TemporalConvLayer,
)

try:
    from gsheets_logger import sync_prediction_log_records, sync_validation_24h_records
except Exception:  # Best-effort: Sheets integration must not block the dashboard
    def sync_validation_24h_records(records):
        return

    def sync_prediction_log_records(records):
        return

# Page config: keep sidebar toggle available and sidebar discoverable by default
st.set_page_config(
    page_title="BTC Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'bitcoin_real_simplified_model.h5'
METADATA_PATH = BASE_DIR / 'models' / 'bitcoin_real_simplified_metadata.pkl'
VALIDATION_24H_PATH = BASE_DIR / 'cache' / 'validation_24h.json'
PREDICTION_LOG_PATH = BASE_DIR / 'cache' / 'prediction_log.json'


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a UTC DatetimeIndex; never raises."""
    try:
        if df is None or len(df) == 0:
            return df

        out = df.copy()

        if isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, utc=True, errors='coerce')
        elif 'timestamp' in out.columns:
            ts = out['timestamp']
            unit = None
            try:
                if np.issubdtype(ts.dtype, np.number):
                    median = float(np.nanmedian(ts.astype(float)))
                    unit = 'ms' if median > 1e11 else 's'
            except Exception:
                unit = None

            if unit:
                out.index = pd.to_datetime(ts, unit=unit, utc=True, errors='coerce')
            else:
                out.index = pd.to_datetime(ts, utc=True, errors='coerce')
        else:
            out.index = pd.to_datetime(out.index, utc=True, errors='coerce')

        out = out[~out.index.isna()]
        return out
    except Exception:
        return df


def _load_validation_records() -> list[dict]:
    try:
        if not VALIDATION_24H_PATH.exists():
            return []
        with open(VALIDATION_24H_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    except Exception:
        return []


def _save_validation_records(records: list[dict]) -> None:
    try:
        VALIDATION_24H_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_24H_PATH, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def _nearest_close_at(price_data: pd.DataFrame, target_ts: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    try:
        df = _ensure_datetime_index(price_data).sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            return None, None
        if 'close' not in df.columns or len(df) == 0:
            return None, None

        target_ts = pd.to_datetime(target_ts, utc=True)

        # Use the last known close at-or-before target time (prevents accidental future lookup).
        idx = df.index.get_indexer([target_ts], method='pad')[0]
        if idx < 0:
            return None, None

        actual_ts = df.index[idx]
        # Sanity tolerance: if the closest prior candle is too far away, treat as missing.
        if (target_ts - actual_ts) > pd.Timedelta(hours=2):
            return None, None
        actual_price = float(df['close'].iloc[idx])
        return actual_price, actual_ts
    except Exception:
        return None, None


def _update_24h_validation(price_data: pd.DataFrame, predicted_24h: float) -> tuple[str, pd.DataFrame]:
    """Store current 24H prediction and, when due, fill actual price. Returns summary HTML + chart DF."""
    now_utc = pd.Timestamp.utcnow().floor('h')
    target_utc = now_utc + pd.Timedelta(hours=24)

    records = _load_validation_records()

    # Keep last ~30 records only
    records = sorted(records, key=lambda r: r.get('made_at', ''))[-30:]

    # Create a record for this hour if missing
    if not any(pd.to_datetime(r.get('made_at'), utc=True, errors='coerce') == now_utc for r in records):
        records.append({
            'made_at': now_utc.isoformat(),
            'target_at': target_utc.isoformat(),
            'predicted_24h': float(predicted_24h),
            'actual_24h': None,
            'actual_at': None
        })

    # Fill actual for any due records missing actual
    for r in records:
        t = pd.to_datetime(r.get('target_at'), utc=True, errors='coerce')
        if t is pd.NaT:
            continue
        if now_utc >= t and r.get('actual_24h') is None:
            actual_price, actual_ts = _nearest_close_at(price_data, t)
            if actual_price is not None and actual_ts is not None:
                r['actual_24h'] = float(actual_price)
                r['actual_at'] = pd.to_datetime(actual_ts, utc=True).isoformat()

    _save_validation_records(records)
    sync_validation_24h_records(records)

    # Build chart DF from completed records
    rows = []
    for r in records:
        act = r.get('actual_24h')
        pred = r.get('predicted_24h')
        target_at = pd.to_datetime(r.get('target_at'), utc=True, errors='coerce')
        if act is None or pred is None or target_at is pd.NaT:
            continue
        act_f = float(act)
        pred_f = float(pred)
        err = act_f - pred_f  # Actual - Predicted
        err_pct = (abs(err) / act_f * 100.0) if act_f else 0.0
        acc_pct = max(0.0, 100.0 - err_pct)
        rows.append({
            'target_at': target_at,
            'predicted': pred_f,
            'actual': act_f,
            'error': err,
            'accuracy_pct': acc_pct
        })

    df_chart = pd.DataFrame(rows).sort_values('target_at') if rows else pd.DataFrame(columns=['target_at', 'predicted', 'actual', 'error', 'accuracy_pct'])

    # Summary: latest completed record if available, else pending
    summary_html = ''
    if not df_chart.empty:
        last = df_chart.iloc[-1]
        summary_html = (
            "<div style='margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid var(--dsba-border);'>"
            "<div style='color: var(--dsba-text); font-size: 0.82rem; font-weight: 700;'>Last 24H Validation</div>"
            f"<div style='color: var(--dsba-text-2); font-size: 0.78rem; margin-top: 0.35rem;'>"
            f"Pred: <b>${last['predicted']:,.0f}</b> &nbsp;•&nbsp; Actual: <b>${last['actual']:,.0f}</b> &nbsp;•&nbsp; Error (A−P): <b>{last['error']:+,.0f}</b> &nbsp;•&nbsp; Accuracy: <b>{last['accuracy_pct']:.1f}%</b>"
            "</div>"
            "</div>"
        )
    else:
        summary_html = (
            "<div style='margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid var(--dsba-border);'>"
            "<div style='color: var(--dsba-text); font-size: 0.82rem; font-weight: 700;'>24H Validation</div>"
            "<div style='color: var(--dsba-text-2); font-size: 0.78rem; margin-top: 0.35rem;'>"
            "Tracking started — first comparison will appear after 24 hours."
            "</div>"
            "</div>"
        )

    return summary_html, df_chart


def _load_prediction_log() -> list[dict]:
    try:
        if not PREDICTION_LOG_PATH.exists():
            return []
        with open(PREDICTION_LOG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    except Exception:
        return []


def _save_prediction_log(records: list[dict]) -> None:
    try:
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTION_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def _append_prediction_log(prediction_cards: list[dict], current_price: float) -> None:
    """Append current predictions to a local JSON log for future predicted-vs-actual charts."""
    try:
        now_utc = pd.Timestamp.utcnow().floor('h')
        horizon_to_hours = {
            '1H': 1,
            '6H': 6,
            '12H': 12,
            '24H': 24,
            '48H': 48,
            '72H': 72,
            '7D': 168,
        }

        new_records: list[dict] = []
        for c in (prediction_cards or []):
            label = c.get('horizon')
            if label not in horizon_to_hours:
                continue
            predicted = c.get('predicted_price')
            if predicted is None or not np.isfinite(float(predicted)):
                continue
            hours_ahead = horizon_to_hours[label]
            created_at_iso = now_utc.isoformat()
            pk = f"{created_at_iso}|{int(hours_ahead)}"
            new_records.append({
                'pk': pk,
                'created_at': now_utc.isoformat(),
                'target_at': (now_utc + pd.Timedelta(hours=hours_ahead)).isoformat(),
                'horizon_label': label,
                'horizon_hours': hours_ahead,
                'current_price': float(current_price),
                'predicted_price': float(predicted),
            })

        if not new_records:
            return

        records = _load_prediction_log()

        # De-duplicate by (created_at, horizon_hours)
        existing = {(r.get('created_at'), r.get('horizon_hours')) for r in records}
        appended: list[dict] = []
        for r in new_records:
            key = (r.get('created_at'), r.get('horizon_hours'))
            if key not in existing:
                records.append(r)
                existing.add(key)
            appended.append(r)

        # Keep the file bounded
        records = records[-5000:]
        _save_prediction_log(records)

        if appended:
            sync_prediction_log_records(appended)
    except Exception:
        return


def _build_recent_predicted_vs_actual_from_log(
    price_data: pd.DataFrame,
    *,
    horizon_label: str = '1H',
    lookback_hours: int = 24,
) -> pd.DataFrame:
    """Build a recent predicted-vs-actual series from prediction_log.json.

    Uses the log entries created by `_append_prediction_log` and matches each `target_at`
    to the nearest actual close price in `price_data`.
    """
    try:
        records = _load_prediction_log()
        if not records:
            return pd.DataFrame()

        now_utc = pd.Timestamp.now(tz='UTC')
        cutoff = now_utc - pd.Timedelta(hours=int(lookback_hours))

        rows: list[dict] = []
        for r in records:
            if r.get('horizon_label') != horizon_label:
                continue

            created_at = pd.to_datetime(r.get('created_at'), utc=True, errors='coerce')
            target_at = pd.to_datetime(r.get('target_at'), utc=True, errors='coerce')
            if created_at is pd.NaT or target_at is pd.NaT:
                continue

            # Only validate targets in the requested lookback window that should already have happened.
            if target_at < cutoff or target_at > now_utc:
                continue

            predicted_price = r.get('predicted_price')
            if predicted_price is None or not np.isfinite(float(predicted_price)):
                continue

            actual_price, actual_ts = _nearest_close_at(price_data, target_at)
            if actual_price is None or actual_ts is None:
                continue

            error = float(actual_price) - float(predicted_price)
            error_pct = (abs(error) / float(actual_price) * 100.0) if float(actual_price) else 0.0
            accuracy_pct = max(0.0, 100.0 - error_pct)

            rows.append({
                'created_at': created_at,
                'target_at': target_at,
                'predicted': float(predicted_price),
                'actual': float(actual_price),
                'actual_at': pd.to_datetime(actual_ts, utc=True),
                'error': error,
                'error_pct': error_pct,
                'accuracy_pct': accuracy_pct,
            })

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        out = out.sort_values('target_at')
        return out
    except Exception:
        return pd.DataFrame()

# Custom CSS for Premium Enterprise Design
st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

     /* Theme variables (LIGHT default; DARK override when Streamlit exposes a theme attribute)
         IMPORTANT: do NOT use prefers-color-scheme here; it follows OS theme and can
         override Streamlit's in-app toggle, causing unreadable light-mode UIs. */
    :root {
        --dsba-bg-0: linear-gradient(135deg, #f7f8fc 0%, #eef2ff 100%);
        --dsba-surface-0: rgba(255, 255, 255, 0.92);
        --dsba-surface-1: rgba(255, 255, 255, 0.72);
        --dsba-border: rgba(15, 23, 42, 0.12);
        --dsba-text: rgba(15, 23, 42, 0.96);
        --dsba-text-2: rgba(15, 23, 42, 0.72);
        --dsba-text-3: rgba(15, 23, 42, 0.58);
        --dsba-pill-bg: rgba(255, 255, 255, 0.65);
        --dsba-shadow: 0 2px 10px rgba(2, 6, 23, 0.10);
        --dsba-shadow-hover: 0 8px 22px rgba(2, 6, 23, 0.12);
        --dsba-accent: #667eea;
        --dsba-accent-2: #764ba2;
        --dsba-positive: #16a34a;
        --dsba-negative: #dc2626;
        --dsba-warning: #d97706;
    }

    /* Best-effort hooks if Streamlit sets a theme attribute */
    html[data-theme="dark"], body[data-theme="dark"], [data-theme="dark"],
    [data-testid="stAppViewContainer"][data-theme="dark"] {
        --dsba-bg-0: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        --dsba-surface-0: rgba(30, 41, 59, 0.70);
        --dsba-surface-1: rgba(15, 23, 42, 0.35);
        --dsba-border: rgba(148, 163, 184, 0.16);
        --dsba-text: rgba(241, 245, 249, 0.98);
        --dsba-text-2: rgba(226, 232, 240, 0.92);
        --dsba-text-3: rgba(148, 163, 184, 0.95);
        --dsba-pill-bg: rgba(15, 23, 42, 0.35);
        --dsba-shadow: 0 2px 10px rgba(0, 0, 0, 0.20);
        --dsba-shadow-hover: 0 8px 22px rgba(99, 102, 241, 0.18);
        --dsba-positive: #22c55e;
        --dsba-negative: #ef4444;
        --dsba-warning: #fbbf24;
    }
    
    /* App background (robust across Streamlit versions) */
    .main {
        background: var(--dsba-bg-0);
        padding: 0;
    }
    [data-testid="stAppViewContainer"] {
        background: var(--dsba-bg-0);
    }
    
    .block-container {
        padding: 0.75rem 1.25rem 1.5rem 1.25rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Keep header visible so the sidebar hamburger/toggle remains accessible */
    header {visibility: visible;}
    header[data-testid="stHeader"] {
        background: transparent;
        box-shadow: none;
    }
    /* Hide Streamlit's right-side toolbar bits; keep only the sidebar toggle */
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}
    [data-testid="stSidebarCollapseButton"] {visibility: visible;}
    
    /* Thin Top Status Strip */
    .status-strip {
        background: linear-gradient(90deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%);
        border-bottom: 1px solid rgba(102, 126, 234, 0.22);
        padding: 0.45rem 1.25rem;
        margin: -0.75rem -1.25rem 0.75rem -1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
    }

    .status-left {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        min-width: 180px;
    }

    .status-brand {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        white-space: nowrap;
    }

    .status-center {
        color: var(--dsba-text-2);
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .status-right {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.75rem;
        min-width: 280px;
        color: var(--dsba-text-3);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        white-space: nowrap;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid var(--dsba-border);
        background: var(--dsba-pill-bg);
    }

    .status-pill.accuracy {
        padding: 0.3rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--dsba-text);
        background: linear-gradient(90deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%);
        border-color: rgba(102, 126, 234, 0.22);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.12);
    }

    /* Centered Price Panel (contained instrument panel, not edge-to-edge) */
    .price-panel {
        max-width: 1040px;
        width: 100%;
        margin: 0.35rem auto 0.85rem auto;
        background: linear-gradient(135deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%);
        border: 1px solid var(--dsba-border);
        border-radius: 14px;
        padding: 0.65rem 0.9rem;
        box-shadow: var(--dsba-shadow);
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 0.75rem;
    }

    .price-meta {
        display: flex;
        flex-direction: column;
        gap: 0.18rem;
        min-width: 0;
    }

    .price-meta.left { text-align: left; }
    .price-meta.right { text-align: right; }

    .meta-label {
        color: var(--dsba-text-3);
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        line-height: 1.1;
    }

    .meta-value {
        color: var(--dsba-text);
        font-size: 0.9rem;
        font-weight: 700;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .meta-value.positive { color: var(--dsba-positive); }
    .meta-value.negative { color: var(--dsba-negative); }

    .price-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-width: 0;
    }

    .price-center-label {
        color: var(--dsba-text-3);
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        line-height: 1.1;
    }

    .price-center-value {
        color: var(--dsba-text);
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: -0.8px;
        line-height: 1.05;
        margin-top: 0.15rem;
    }

    @media (max-width: 900px) {
        .price-panel { grid-template-columns: 1fr; gap: 0.55rem; }
        .price-meta.left, .price-meta.right { text-align: center; }
    }
    
    .price-label {
        color: var(--dsba-text-3);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    .price-value {
        color: var(--dsba-text);
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin: 0.5rem 0;
        text-shadow: none;
    }
    
    .price-change {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .price-change.positive {
        background: rgba(22, 163, 74, 0.12);
        color: var(--dsba-positive);
        border: 1px solid rgba(22, 163, 74, 0.20);
    }
    
    .price-change.negative {
        background: rgba(220, 38, 38, 0.10);
        color: var(--dsba-negative);
        border: 1px solid rgba(220, 38, 38, 0.20);
    }
    
    /* Premium KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%);
        border: 1px solid var(--dsba-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: var(--dsba-shadow);
        transition: all 0.2s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--dsba-shadow-hover);
        border-color: rgba(102, 126, 234, 0.28);
    }
    
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--dsba-border);
    }
    
    .kpi-time {
        color: var(--dsba-text-3);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-icon {
        font-size: 1.2rem;
        opacity: 0.6;
    }
    
    .kpi-price {
        color: var(--dsba-text);
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.75rem 0;
        letter-spacing: -0.5px;
    }
    
    .kpi-change {
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .kpi-change.up {
        color: var(--dsba-positive);
    }
    
    .kpi-change.down {
        color: var(--dsba-negative);
    }
    
    /* Confidence Gauge */
    .confidence-gauge {
        margin: 1rem 0;
    }
    
    .confidence-label {
        color: var(--dsba-text-3);
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .confidence-bar {
        height: 6px;
        background: rgba(15, 23, 42, 0.10);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
        background: linear-gradient(90deg, var(--dsba-accent) 0%, var(--dsba-accent-2) 100%);
        box-shadow: none;
    }
    
    .confidence-value {
        color: var(--dsba-text-2);
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    
    /* Signal Badge */
    .signal-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .signal-buy {
        background: rgba(34, 197, 94, 0.15);
        color: var(--dsba-positive);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .signal-sell {
        background: rgba(239, 68, 68, 0.15);
        color: var(--dsba-negative);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .signal-hold {
        background: rgba(251, 191, 36, 0.15);
        color: var(--dsba-warning);
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        color: var(--dsba-text);
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.6rem 0 1rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.25);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-icon {
        font-size: 1.4rem;
        opacity: 0.85;
    }

    .section-subtitle {
        color: var(--dsba-text-2);
        font-size: 0.85rem;
        margin-bottom: 0.9rem;
    }

    .subsection-title {
        color: var(--dsba-text);
        font-size: 1.0rem;
        font-weight: 700;
        margin: 1.1rem 0 0.9rem 0;
    }
    
    /* Chart Container */
    .chart-container {
        background: linear-gradient(135deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%);
        border: 1px solid var(--dsba-border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--dsba-shadow);
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%);
        border: 1px solid var(--dsba-border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--dsba-shadow);
    }
    
    .metric-label {
        color: var(--dsba-text-3);
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: var(--dsba-text);
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Streamlit Metric Override */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%);
        border: 1px solid var(--dsba-border);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: var(--dsba-shadow);
    }
    
    [data-testid="stMetric"] label {
        color: var(--dsba-text-3) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--dsba-text) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Loading Animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .price-value {
            font-size: 2.5rem;
        }
        .kpi-price {
            font-size: 1.4rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

def add_simple_features(df):
    """Add basic features matching train_simple.py exactly"""
    df = df.copy()
    
    # Basic returns
    for h in [1, 6, 12, 24, 48, 168]:
        df[f'return_{h}h'] = df['close'].pct_change(h)
        df[f'log_return_{h}h'] = np.log1p(df[f'return_{h}h'])
    
    # Simple moving averages
    for period in [7, 14, 21, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(period).mean()
        df[f'distance_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
    
    # Volatility
    for period in [7, 14, 21, 50]:
        df[f'volatility_{period}'] = df['return_1h'].rolling(period).std()
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    # Momentum
    df['momentum_consistency'] = (
        np.sign(df['return_1h']) == np.sign(df['return_6h'])
    ).astype(int)
    
    return df.dropna()

@st.cache_resource
def load_model_and_predictor():
    """Load model and predictor (cached for performance)"""
    try:
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        
        custom_objects = {
            'TemporalConvLayer': TemporalConvLayer,
            'MultiHeadAttentionCustom': MultiHeadAttentionCustom,
            'mse': keras.losses.MeanSquaredError()
        }
        model = keras.models.load_model(str(MODEL_PATH), custom_objects=custom_objects, compile=False)
        
        config = metadata['config']
        predictor = EnhancedCryptoPricePredictor(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            prediction_horizons=config['prediction_horizons']
        )
        predictor.model = model
        predictor.feature_scaler = metadata['scaler_X']
        predictor.price_scaler = metadata['scaler_y']
        
        return predictor, metadata
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_live_data():
    """Fetch live Bitcoin price and historical data"""
    # The dashboard already uses Streamlit in-memory caching (ttl=60).
    # Disk caching has caused stale/flat series on Streamlit Cloud, so keep it off here.
    collector = CryptoDataCollector(use_cache=False)
    
    try:
        current_price = collector.get_current_price('bitcoin')
        price_data = collector.price_collector.get_price_data('bitcoin', hours_back=2000, interval='1h')

        # Only blend live price into the last candle if it's close to the historical last close.
        # This prevents a confusing vertical "cliff" when the historical series is stale/flat.
        try:
            last_close = float(price_data['close'].iloc[-1])
            if last_close > 0 and abs(float(current_price) - last_close) / last_close <= 0.01:
                price_data.iloc[-1, price_data.columns.get_loc('close')] = current_price
        except Exception:
            pass
        
        return current_price, price_data, None
    except Exception as e:
        return None, None, str(e)

def generate_predictions(predictor, metadata, features_df):
    """Generate predictions for all horizons"""
    feature_names = metadata['feature_names']
    features_df_clean = features_df.select_dtypes(include=[np.number])
    features = features_df_clean[feature_names]
    
    feature_cols = [col for col in features.columns if col != 'close']
    features_array = features[feature_cols].values
    features_scaled = predictor.feature_scaler.transform(features_array)
    
    sequence_length = metadata['config']['sequence_length']
    X = features_scaled[-sequence_length:].reshape(1, sequence_length, -1)
    
    predictions = predictor.predict_with_uncertainty(X)
    return predictions


def _format_price_data_diagnostics(price_data: pd.DataFrame, current_price: float | None = None) -> str:
    try:
        if price_data is None or not isinstance(price_data, pd.DataFrame) or len(price_data) == 0:
            return "No price_data"

        # Read attrs from the original frame first (index conversions/copies can drop attrs)
        source = None
        cache_hit = None
        try:
            original_attrs = dict(getattr(price_data, 'attrs', {}) or {})
            source = original_attrs.get('source')
            cache_hit = original_attrs.get('cache_hit')
        except Exception:
            source = None
            cache_hit = None

        df = _ensure_datetime_index(price_data).sort_index()

        if 'close' not in df.columns:
            return f"source={source or 'unknown'} • missing close"

        close = pd.to_numeric(df['close'], errors='coerce').dropna()
        if close.empty:
            return f"source={source or 'unknown'} • empty close"

        # Focus on the visible window for flatness detection
        window = close.tail(168)
        nunique = int(window.nunique(dropna=True))
        std = float(window.std()) if len(window) > 1 else 0.0

        first_ts = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
        last_ts = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None

        last_close = float(close.iloc[-1]) if len(close) else None
        delta_str = ""
        if current_price is not None and last_close is not None and last_close != 0:
            pct = (float(current_price) - last_close) / last_close * 100.0
            delta_str = f" • last_close=${last_close:,.2f} • live=${float(current_price):,.2f} • live_vs_last={pct:+.2f}%"

        return (
            f"source={source or 'unknown'}"
            f" • cache_hit={cache_hit if cache_hit is not None else 'unknown'}"
            f" • points={len(df)}"
            f" • window_unique_close={nunique}"
            f" • window_std={std:,.6f}"
            f" • close_min=${float(window.min()):,.2f}"
            f" • close_max=${float(window.max()):,.2f}"
            f" • first={first_ts}"
            f" • last={last_ts}"
            f"{delta_str}"
        )
    except Exception as e:
        return f"diagnostics_failed={e}"


@st.cache_data(ttl=300)
def compute_historical_backtest(
    _predictor,
    _metadata,
    price_data: pd.DataFrame,
    horizon_hours: int = 1,
    days: int = 30,
) -> pd.DataFrame:
    """Compute a rolling backtest series: predicted vs actual for past data."""
    debug: dict[str, object] = {
        'horizon_hours': int(horizon_hours),
        'days': int(days),
    }

    def _empty(reason: str) -> pd.DataFrame:
        df0 = pd.DataFrame()
        try:
            debug['reason'] = reason
            df0.attrs = dict(getattr(df0, 'attrs', {}) or {})
            df0.attrs['debug'] = debug
        except Exception:
            pass
        return df0

    if _predictor is None or _metadata is None or price_data is None or len(price_data) == 0:
        return _empty('missing_inputs')

    # Ensure we have a clean datetime index
    price_df = _ensure_datetime_index(price_data)
    if not isinstance(price_df.index, pd.DatetimeIndex):
        return _empty('price_index_not_datetime')

    features_df = add_simple_features(price_df)
    if features_df.empty:
        debug['price_points'] = int(len(price_df))
        debug['price_cols'] = list(getattr(price_df, 'columns', []))
        return _empty('features_empty')

    feature_names = _metadata['feature_names']
    features_df_clean = features_df.select_dtypes(include=[np.number])
    features = features_df_clean[feature_names]

    predictor = _predictor

    feature_cols = [c for c in features.columns if c != 'close']
    features_array = features[feature_cols].values
    features_scaled = predictor.feature_scaler.transform(features_array)

    seq_len = int(_metadata['config']['sequence_length'])

    # Work on a limited window for performance
    points_needed = int(days * 24 + seq_len + horizon_hours + 5)
    debug['features_points'] = int(len(features))
    debug['seq_len'] = int(seq_len)
    debug['points_needed'] = int(points_needed)
    if len(features) > points_needed:
        start = len(features) - points_needed
        idx = features.index[start:]
        scaled = features_scaled[start:]
    else:
        idx = features.index
        scaled = features_scaled

    horizons = list(getattr(predictor, 'prediction_horizons', []) or [])
    debug['model_horizons'] = horizons
    if horizon_hours not in horizons:
        return _empty('horizon_not_supported')

    horizon_idx = list(predictor.prediction_horizons).index(horizon_hours)

    # Build sequences and target timestamps
    X_list = []
    target_ts = []
    for end in range(seq_len, len(scaled) + 1):
        current_ts = idx[end - 1]
        tts = current_ts + pd.Timedelta(hours=horizon_hours)
        # Only keep if we can evaluate against an actual close later
        X_list.append(scaled[end - seq_len:end])
        target_ts.append(tts)

    if len(X_list) == 0:
        return _empty('no_sequences')

    X = np.asarray(X_list, dtype=np.float32)

    # Single forward pass for speed (no MC dropout)
    try:
        raw_pred = predictor.model.predict(X, verbose=0)
    except Exception as e:
        debug['predict_error'] = str(e)
        return _empty('predict_failed')
    price_scaled = raw_pred[horizon_idx * 3]
    price_pred = predictor.price_scaler.inverse_transform(price_scaled).reshape(-1)

    # Ensure target_at is consistently UTC tz-aware to avoid tz-naive/aware comparison issues.
    pred_df = pd.DataFrame({'target_at': pd.to_datetime(target_ts, utc=True), 'predicted': price_pred})
    pred_df = pred_df.dropna(subset=['predicted']).sort_values('target_at')
    debug['pred_rows'] = int(len(pred_df))

    # Align actual close by last known candle at-or-before target time (prevents future leakage).
    # IMPORTANT: Use the full close series (UTC-indexed) rather than a position slice.
    # Feature engineering drops initial rows (rolling windows), so positional slicing can misalign
    # `target_at` timestamps vs the actual series and wipe out the backtest on some hosts.
    actual = price_df['close'].copy()
    actual.index = pd.to_datetime(actual.index, utc=True, errors='coerce')
    actual = actual[~actual.index.isna()].sort_index()

    # Cloud providers occasionally return missing hourly candles; keep matching non-leaky
    # (at-or-before target) but allow a wider tolerance so we don't drop the entire backtest.
    tolerance_seconds = 4 * 3600  # 4 hours

    # Robust non-leaky alignment using merge_asof (backward join):
    # - avoids per-row get_indexer edge cases on some hosts
    # - keeps the "at-or-before" semantics
    try:
        actual_df = actual.reset_index()
        actual_df.columns = ['ts', 'actual']
        actual_df['ts'] = pd.to_datetime(actual_df['ts'], utc=True, errors='coerce')
        actual_df = actual_df.dropna(subset=['ts', 'actual']).sort_values('ts')

        pred_sorted = pred_df.sort_values('target_at')
        merged = pd.merge_asof(
            pred_sorted,
            actual_df,
            left_on='target_at',
            right_on='ts',
            direction='backward',
            tolerance=pd.Timedelta(seconds=tolerance_seconds),
        )

        pred_df = merged.drop(columns=['ts'])
    except Exception as e:
        debug['actual_match_error'] = str(e)
        return _empty('actual_match_failed')

    matched = int(pred_df['actual'].notna().sum()) if 'actual' in pred_df.columns else 0
    debug['matched_actual_rows'] = matched
    debug['tolerance_hours'] = float(tolerance_seconds) / 3600.0

    pred_df = pred_df.dropna(subset=['actual'])
    debug['rows_after_actual_drop'] = int(len(pred_df))

    # Limit to last N days in the final result
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
    pred_df = pred_df[pred_df['target_at'] >= cutoff]
    debug['rows_after_cutoff'] = int(len(pred_df))

    try:
        pred_df.attrs = dict(getattr(pred_df, 'attrs', {}) or {})
        pred_df.attrs['debug'] = debug
    except Exception:
        pass
    return pred_df


def _accuracy_from_pred_actual_df(df: pd.DataFrame) -> tuple[int, float | None, float | None]:
    """Return (n, median_abs_pct_error, accuracy_100_minus_median_ape)."""
    try:
        if df is None or df.empty:
            return 0, None, None
        d = df.copy()
        d['predicted'] = pd.to_numeric(d.get('predicted'), errors='coerce')
        d['actual'] = pd.to_numeric(d.get('actual'), errors='coerce')
        d = d.dropna(subset=['predicted', 'actual'])
        if d.empty:
            return 0, None, None
        ape = (np.abs(d['actual'] - d['predicted']) / np.abs(d['actual']).replace(0, np.nan)) * 100.0
        ape = ape.replace([np.inf, -np.inf], np.nan).dropna()
        if ape.empty:
            return int(len(d)), None, None
        median_ape = float(np.nanmedian(ape.values))
        acc = max(0.0, 100.0 - median_ape)
        return int(len(ape)), median_ape, acc
    except Exception:
        return 0, None, None


def _render_status_strip(
    *,
    utc_hms: str,
    accuracy_text: str,
) -> str:
    return (
        "<div class=\"status-strip\">"
        "<div class=\"status-left\"><div class=\"status-brand\">₿ BTC INTELLIGENCE</div></div>"
        "<div class=\"status-center\">BTC Forecasting & Market Intelligence</div>"
        "<div class=\"status-right\">"
        "<span class=\"status-pill\"><span class=\"status-dot\"></span>Live</span>"
        f"<span class=\"status-pill\">UTC: {utc_hms}</span>"
        f"<span class=\"status-pill accuracy\">{accuracy_text}</span>"
        "</div></div>"
    )


def _render_price_accuracy_badge(accuracy_text: str) -> str:
    return (
        "<div style='display:flex; justify-content:center; margin: 0.15rem 0 0.65rem 0;'>"
        "<div style='padding: 0.45rem 0.9rem; border-radius: 999px; "
        "border: 1px solid rgba(102, 126, 234, 0.28); "
        "background: linear-gradient(90deg, var(--dsba-surface-0) 0%, var(--dsba-surface-1) 100%); "
        "box-shadow: var(--dsba-shadow); "
        "color: var(--dsba-text); "
        "font-weight: 800; font-size: 0.92rem; letter-spacing: 0.2px;'>"
        f"{accuracy_text}"
        "</div></div>"
    )

def create_prediction_card(horizon, label, current_price, pred_price, pred_std, direction_prob, predictor):
    """Create a prediction card for a specific horizon"""
    # Calculate metrics
    change_pct = ((pred_price - current_price) / current_price) * 100
    change_usd = pred_price - current_price
    
    prob_down, prob_neutral, prob_up = direction_prob[0], direction_prob[1], direction_prob[2]
    
    # Determine direction and signal
    if prob_up > prob_down and prob_up > prob_neutral:
        direction_str = "↑ UP"
        confidence = prob_up
        signal = "BUY" if confidence > 0.6 else "HOLD"
        signal_class = "buy-signal" if signal == "BUY" else "hold-signal"
        signal_color = "#28a745" if signal == "BUY" else "#ffc107"
    elif prob_down > prob_up and prob_down > prob_neutral:
        direction_str = "↓ DOWN"
        confidence = prob_down
        signal = "SELL" if confidence > 0.6 else "HOLD"
        signal_class = "sell-signal" if signal == "SELL" else "hold-signal"
        signal_color = "#dc3545" if signal == "SELL" else "#ffc107"
    else:
        direction_str = "↔ NEUTRAL"
        confidence = prob_neutral
        signal = "HOLD"
        signal_class = "hold-signal"
        signal_color = "#ffc107"
    
    return {
        'horizon': label,
        'predicted_price': pred_price,
        'uncertainty': pred_std,
        'change_pct': change_pct,
        'change_usd': change_usd,
        'direction': direction_str,
        'confidence': confidence,
        'prob_up': prob_up,
        'prob_down': prob_down,
        'prob_neutral': prob_neutral,
        'signal': signal,
        'signal_class': signal_class,
        'signal_color': signal_color
    }


def _render_html_block(html: str) -> None:
    """Render HTML reliably across Streamlit versions."""
    if not html:
        return
    html = str(html)
    if hasattr(st, "html"):
        # Newer Streamlit versions provide st.html which renders HTML directly.
        st.html(html)
    else:
        # Fallback for older versions.
        st.markdown(html, unsafe_allow_html=True)

def main():
    # Thin status strip (essentials first) — update once metrics are available
    status_ph = st.empty()
    status_ph.markdown(
        _render_status_strip(
            utc_hms=datetime.utcnow().strftime('%H:%M:%S'),
            accuracy_text="Model Accuracy: calculating…",
        ),
        unsafe_allow_html=True,
    )

    # Theme-aware styling for Plotly + embedded widgets
    theme_base = str(st.get_option("theme.base") or "").lower()
    is_dark = theme_base == "dark"
    plotly_template = "plotly_dark" if is_dark else "plotly_white"
    plot_title_color = "#f1f5f9" if is_dark else "#0f172a"
    plot_text_color = "#94a3b8" if is_dark else "#0f172a"
    plot_panel_bg = "rgba(30, 41, 59, 0.50)" if is_dark else "rgba(255, 255, 255, 0.80)"
    plot_legend_bg = "rgba(30, 41, 59, 0.80)" if is_dark else "rgba(255, 255, 255, 0.88)"
    plot_border = "rgba(148, 163, 184, 0.20)" if is_dark else "rgba(15, 23, 42, 0.12)"
    plot_marker_outline = "#0f172a" if not is_dark else "#f1f5f9"

    # Top-left Menu (☰): replaces sidebar controls so it's always discoverable.
    # Use Streamlit popover when available; fall back to an expander.
    if "menu_open" not in st.session_state:
        st.session_state.menu_open = False

    menu_row = st.columns([1, 10])
    with menu_row[0]:
        if hasattr(st, "popover"):
            with st.popover("☰"):
                st.markdown("### ⚙️ Controls")
                if st.button("🔄 Refresh Dashboard", use_container_width=True, key="menu_refresh"):
                    st.cache_data.clear()
                    st.rerun()

                auto_refresh = st.checkbox("Auto-refresh (60s)", key="menu_auto_refresh")
                if auto_refresh:
                    import time

                    time.sleep(60)
                    st.rerun()

                st.markdown("---")
                st.markdown("### 📊 Dashboard Info")
                st.info(
                    """
                    **Features:**
                    - Real-time BTC price
                    - 7 prediction horizons
                    - AI confidence metrics
                    - Buy/Sell/Hold signals
                    - Historical analysis
                    """
                )
        else:
            if st.button("☰", help="Open menu", key="menu_toggle"):
                st.session_state.menu_open = not st.session_state.menu_open

    if not hasattr(st, "popover"):
        with st.expander("Menu", expanded=bool(st.session_state.menu_open)):
            st.markdown("### ⚙️ Controls")
            if st.button("🔄 Refresh Dashboard", use_container_width=True, key="menu_refresh_fallback"):
                st.cache_data.clear()
                st.rerun()

            auto_refresh = st.checkbox("Auto-refresh (60s)", key="menu_auto_refresh_fallback")
            if auto_refresh:
                import time

                time.sleep(60)
                st.rerun()

            st.markdown("---")
            st.markdown("### 📊 Dashboard Info")
            st.info(
                """
                **Features:**
                - Real-time BTC price
                - 7 prediction horizons
                - AI confidence metrics
                - Buy/Sell/Hold signals
                - Historical analysis
                """
            )
    
    # Check if model exists
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("⚠️ Model files not found. Please train the model first using `python train_simple.py`")
        st.stop()
    
    # Load model
    with st.spinner("🔄 Loading AI Engine..."):
        predictor, metadata = load_model_and_predictor()
    
    if predictor is None:
        st.error("Failed to load model. Please check the model files.")
        st.stop()
    
    # Fetch live data
    with st.spinner("📡 Fetching Live Market Data..."):
        current_price, price_data, error = fetch_live_data()
    
    if error:
        st.error(f"⚠️ Failed to fetch live data: {error}")
        st.stop()

    # Credible performance: compute a 7-day historical backtest for 24H horizon (does not depend on live logs)
    bt24 = pd.DataFrame()
    backtest_err: str | None = None
    backtest_text = "24H Backtest 7D: unavailable"
    backtest_acc = None
    backtest_n = 0
    try:
        bt24 = compute_historical_backtest(predictor, metadata, price_data, horizon_hours=24, days=7)
        n_bt, med_ape_bt, acc_bt = _accuracy_from_pred_actual_df(bt24)
        backtest_n = int(n_bt)
        backtest_acc = acc_bt
        if n_bt > 0 and acc_bt is not None:
            backtest_text = f"24H Backtest 7D: {acc_bt:.1f}% (N={n_bt})"
        elif n_bt > 0:
            backtest_text = f"24H Backtest 7D: collecting (N={n_bt})"
    except Exception as e:
        backtest_text = "24H Backtest 7D: unavailable"
        backtest_acc = None
        backtest_n = 0
        backtest_err = str(e)
    
    # Compact centered price panel (contained instrument panel)
    price_24h_ago = price_data['close'].iloc[-25] if len(price_data) >= 25 else price_data['close'].iloc[0]
    change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
    change_class = "positive" if change_24h >= 0 else "negative"
    change_symbol = "▲" if change_24h >= 0 else "▼"

    st.markdown(
                f"""<div class="price-panel">
<div class="price-meta left">
    <div class="meta-label">Change (24H)</div>
    <div class="meta-value {change_class}">{change_symbol} {abs(change_24h):.2f}%</div>
</div>
<div class="price-center">
    <div class="price-center-label">BTC Price</div>
    <div class="price-center-value">${current_price:,.2f}</div>
</div>
<div class="price-meta right">
    <div class="meta-label">Interval</div>
    <div class="meta-value">1H candles</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Accuracy badge placeholder (updated after predictions/backtest are computed)
    price_accuracy_ph = st.empty()
    
    # Generate features and predictions
    with st.spinner("🧠 Generating AI Predictions..."):
        features_df = add_simple_features(price_data)
        predictions = generate_predictions(predictor, metadata, features_df)
    
    # Prediction horizons
    horizons = [1, 6, 12, 24, 48, 72, 168]
    horizon_labels = ['1H', '6H', '12H', '24H', '48H', '72H', '7D']
    horizon_icons = ['🕐', '🕕', '🕚', '📅', '📆', '📆', '📅']
    
    # Process all predictions
    prediction_cards = []
    for horizon, label in zip(horizons, horizon_labels):
        pred_scaled = predictions[f'price_{horizon}h'][0]
        pred_std_scaled = predictions[f'price_{horizon}h_std'][0]
        direction_prob = predictions[f'direction_{horizon}h'][0]
        
        if isinstance(pred_scaled, np.ndarray):
            pred_scaled = float(pred_scaled.flatten()[0])
        
        pred_price = predictor.price_scaler.inverse_transform([[pred_scaled]])[0, 0]
        
        if hasattr(predictor.price_scaler, 'scale_'):
            pred_std_price = float(pred_std_scaled.flatten()[0]) * predictor.price_scaler.scale_[0]
        else:
            pred_std_price = float(pred_std_scaled.flatten()[0]) * 1000
        
        card = create_prediction_card(horizon, label, current_price, pred_price, 
                                     pred_std_price, direction_prob, predictor)
        prediction_cards.append(card)

    # Store predictions now for future historical charts (best-effort)
    _append_prediction_log(prediction_cards, float(current_price))

    # 24H validation (Predicted vs Actual after 24H)
    pred_24h_price = next((c['predicted_price'] for c in prediction_cards if c.get('horizon') == '24H'), None)
    validation_summary_html = ''
    validation_chart_df = pd.DataFrame()
    if pred_24h_price is not None:
        validation_summary_html, validation_chart_df = _update_24h_validation(price_data, float(pred_24h_price))

    # Update status strip with reliable metrics (live 24H only once it has enough real samples)
    live_text = "24H Live: collecting (N=0)"
    live_acc = None
    live_n = 0
    try:
        n_live, med_ape_live, acc_live = _accuracy_from_pred_actual_df(
            validation_chart_df.rename(columns={'predicted': 'predicted', 'actual': 'actual'})
        )
        live_n = int(n_live)
        live_acc = acc_live
        if n_live >= 24 and acc_live is not None:
            live_text = f"24H Live: {acc_live:.1f}% (N={n_live})"
        else:
            live_text = f"24H Live: collecting (N={n_live})"
    except Exception:
        live_text = "24H Live: collecting (N=0)"
        live_acc = None
        live_n = 0

    # Prominent headline: prefer live only when it has enough samples; otherwise use backtest.
    accuracy_text = "Model Accuracy: collecting"
    if live_n >= 24 and live_acc is not None:
        accuracy_text = f"Model Accuracy (24H Live): {live_acc:.1f}% (N={live_n})"
    elif backtest_n > 0 and backtest_acc is not None:
        accuracy_text = f"Model Accuracy (24H Backtest 7D): {backtest_acc:.1f}% (N={backtest_n})"
    elif backtest_n > 0:
        accuracy_text = f"Model Accuracy (Backtest 7D): collecting (N={backtest_n})"
    else:
        accuracy_text = "Model Accuracy: collecting"

    status_ph.markdown(
        _render_status_strip(
            utc_hms=datetime.utcnow().strftime('%H:%M:%S'),
            accuracy_text=accuracy_text,
        ),
        unsafe_allow_html=True,
    )

    price_accuracy_ph.markdown(
        _render_price_accuracy_badge(accuracy_text),
        unsafe_allow_html=True,
    )
    
    # AI Predictions Section Header (keep it above the fold)
    st.markdown(
        """
        <div class="section-header" style="margin-top: 0.9rem;">
            <span class="section-icon">🔮</span>
            <span>AI Predictions</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Multi-horizon forecasts with confidence and direction</div>',
        unsafe_allow_html=True,
    )

    # Progressive disclosure: keep validation details collapsed by default
    with st.expander("24H Validation (Predicted vs Actual)", expanded=False):
        # Live 24H validation is still being saved in the background, but we keep it hidden
        # until we have enough real completed points to be meaningful.
        if live_n < 24:
            st.info(f"Live 24H validation is collecting in the background (completed points: {live_n}). It will appear here once enough data is available.")
        else:
            if validation_summary_html:
                st.markdown(validation_summary_html, unsafe_allow_html=True)

            if not validation_chart_df.empty:
                fig_val = go.Figure()
                fig_val.add_trace(go.Scatter(
                    x=validation_chart_df['target_at'],
                    y=validation_chart_df['predicted'],
                    mode='lines+markers',
                    name='Predicted (24H)',
                    line=dict(color='#22c55e', width=2, dash='dash'),
                    marker=dict(size=7, color='#22c55e', line=dict(color=plot_marker_outline, width=1)),
                    hovertemplate='<b>%{x}</b><br>Predicted: $%{y:,.0f}<extra></extra>'
                ))
                fig_val.add_trace(go.Scatter(
                    x=validation_chart_df['target_at'],
                    y=validation_chart_df['actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=7, color='#667eea', line=dict(color=plot_marker_outline, width=1)),
                    hovertemplate='<b>%{x}</b><br>Actual: $%{y:,.0f}<extra></extra>'
                ))

                fig_val.update_layout(
                    title=dict(
                        text='24H Validation: Predicted vs Actual',
                        font=dict(size=13, color=plot_title_color, weight=600)
                    ),
                    xaxis_title='Target Time (UTC)',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    height=260,
                    template=plotly_template,
                    paper_bgcolor=plot_panel_bg,
                    plot_bgcolor=plot_panel_bg,
                    font=dict(size=11, color=plot_text_color),
                    margin=dict(t=45, b=35, l=40, r=40),
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1,
                        bgcolor=plot_legend_bg,
                        bordercolor=plot_border,
                        borderwidth=1
                    )
                )

                st.plotly_chart(fig_val, use_container_width=True)

        # 7-day rolling backtest for the 24H horizon (real historical evaluation, not live tracking)
        st.markdown(
            '<div class="subsection-title" style="margin-top: 1.1rem;">Backtest (Last 7 Days): 24H Rolling Predictions</div>',
            unsafe_allow_html=True,
        )

        if isinstance(bt24, pd.DataFrame) and not bt24.empty:
            n_bt, med_ape_bt, acc_bt = _accuracy_from_pred_actual_df(bt24)
            st.caption(
                f"Backtest uses historical candles to evaluate your model. Accuracy shown is 100 − Median Abs % Error. (N={n_bt})"
            )

            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt24['target_at'],
                y=bt24['predicted'],
                mode='lines',
                name='Predicted (24H)',
                line=dict(color='#22c55e', width=2, dash='dash'),
                hovertemplate='<b>%{x}</b><br>Predicted: $%{y:,.0f}<extra></extra>'
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt24['target_at'],
                y=bt24['actual'],
                mode='lines',
                name='Actual',
                line=dict(color='#667eea', width=2),
                hovertemplate='<b>%{x}</b><br>Actual: $%{y:,.0f}<extra></extra>'
            ))

            title_suffix = f" • Accuracy: {acc_bt:.1f}%" if acc_bt is not None else ""
            fig_bt.update_layout(
                title=dict(
                    text=f"24H Backtest (7D): Predicted vs Actual{title_suffix}",
                    font=dict(size=13, color=plot_title_color, weight=600)
                ),
                xaxis_title='Target Time (UTC)',
                yaxis_title='Price (USD)',
                hovermode='x unified',
                height=320,
                template=plotly_template,
                paper_bgcolor=plot_panel_bg,
                plot_bgcolor=plot_panel_bg,
                font=dict(size=11, color=plot_text_color),
                margin=dict(t=45, b=35, l=40, r=40),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor=plot_legend_bg,
                    bordercolor=plot_border,
                    borderwidth=1
                )
            )
            st.plotly_chart(fig_bt, use_container_width=True)
        else:
            st.info(
                "Backtest is temporarily unavailable. This usually resolves after a refresh once enough 1H candles load. "
                "(It requires ~7 days of hourly candles + model sequence window.)"
            )

            try:
                dbg = getattr(bt24, 'attrs', {}).get('debug') if isinstance(bt24, pd.DataFrame) else None
            except Exception:
                dbg = None
            if dbg:
                st.caption(f"Backtest debug: {dbg}")

            # Diagnostics (best-effort) to help debug Streamlit Cloud data/provider issues.
            try:
                diag = _format_price_data_diagnostics(price_data, current_price)
            except Exception:
                diag = "diagnostics_unavailable"

            try:
                cols = list(price_data.columns) if isinstance(price_data, pd.DataFrame) else []
            except Exception:
                cols = []

            try:
                horizons = list(getattr(predictor, 'prediction_horizons', [])) if predictor is not None else []
            except Exception:
                horizons = []

            if backtest_err:
                st.caption(f"Backtest error: {backtest_err}")

            st.caption(f"Price diagnostics: {diag}")
            if cols:
                st.caption(f"Price columns: {', '.join(cols)}")
            if horizons:
                st.caption(f"Model horizons: {horizons}")

    
    
    # Premium KPI Cards - Short-term (4 cards)
    st.markdown('<div class="subsection-title">Short-term Outlook</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="large")
    
    for i, (card, icon) in enumerate(zip(prediction_cards[:4], horizon_icons[:4])):
        with cols[i]:
            change_class = "up" if card['change_pct'] >= 0 else "down"
            change_arrow = "↑" if card['change_pct'] >= 0 else "↓"
            
            signal_class_map = {
                'BUY': 'signal-buy',
                'SELL': 'signal-sell',
                'HOLD': 'signal-hold'
            }

            kpi_html = textwrap.dedent(f"""\
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-time">{card['horizon']}</span>
                    <span class="kpi-icon">{icon}</span>
                </div>
                <div class="kpi-price">${card['predicted_price']:,.0f}</div>
                <div class="kpi-change {change_class}">{change_arrow} {abs(card['change_pct']):.2f}%</div>

                <div class="confidence-gauge">
                    <div class="confidence-label">Confidence</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {card['confidence']*100:.1f}%;"></div>
                    </div>
                    <div class="confidence-value">{card['confidence']*100:.1f}%</div>
                </div>

                <div class="signal-badge {signal_class_map[card['signal']]}">{card['signal']}</div>
            </div>
            """).strip()
            _render_html_block(kpi_html)
    
    # Premium KPI Cards - Long-term (3 cards)
    st.markdown('<div class="subsection-title" style="margin-top: 1.5rem;">Long-term Outlook</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    
    for i, (card, icon) in enumerate(zip(prediction_cards[4:], horizon_icons[4:])):
        with cols[i]:
            idx = i + 4  # Continue numbering from short-term cards
            change_class = "up" if card['change_pct'] >= 0 else "down"
            change_arrow = "↑" if card['change_pct'] >= 0 else "↓"
            signal_class_map = {
                'BUY': 'signal-buy',
                'SELL': 'signal-sell',
                'HOLD': 'signal-hold'
            }

            kpi_html = textwrap.dedent(f"""\
            <div class="kpi-card">
                <div class="kpi-header">
                    <span class="kpi-time">{card['horizon']}</span>
                    <span class="kpi-icon">{icon}</span>
                </div>
                <div class="kpi-price">${card['predicted_price']:,.0f}</div>
                <div class="kpi-change {change_class}">{change_arrow} {abs(card['change_pct']):.2f}%</div>

                <div class="confidence-gauge">
                    <div class="confidence-label">Confidence</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {card['confidence']*100:.1f}%;"></div>
                    </div>
                    <div class="confidence-value">{card['confidence']*100:.1f}%</div>
                </div>

                <div class="signal-badge {signal_class_map[card['signal']]}">{card['signal']}</div>
            </div>
            """).strip()
            _render_html_block(kpi_html)
    
    # Charts Section
    st.markdown("""
        <div class="section-header" style="margin-top: 3rem;">
            <span class="section-icon">📊</span>
            <span>Price Analytics & Forecasting</span>
        </div>
    """, unsafe_allow_html=True)
    
    # TradingView Live Chart Section (progressive disclosure)
    with st.expander("Live chart (TradingView)", expanded=False):
        st.markdown('<div style="color: var(--dsba-text-2); font-size: 0.85rem; margin: 0.25rem 0 0.75rem 0;">Real-time market data with professional trading tools</div>', unsafe_allow_html=True)

        # Embedded TradingView Widget
        tv_theme = "dark" if is_dark else "light"
        tv_bg = "rgba(19, 23, 34, 1)" if is_dark else "rgba(255, 255, 255, 1)"
        tv_grid = "rgba(42, 46, 57, 0.06)" if is_dark else "rgba(15, 23, 42, 0.08)"

        tradingview_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="height: 500px; margin-bottom: 0;">
        <div class="tradingview-widget-container__widget" style="height: calc(100% - 32px); width: 100%;"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {{
        "autosize": true,
        "symbol": "BINANCE:BTCUSDT",
        "interval": "60",
        "timezone": "Etc/UTC",
        "theme": "{tv_theme}",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "backgroundColor": "{tv_bg}",
        "gridColor": "{tv_grid}",
        "hide_top_toolbar": false,
        "hide_legend": false,
        "save_image": false,
        "calendar": false,
        "support_host": "https://www.tradingview.com"
        }}
        </script>
        </div>
        <!-- TradingView Widget END -->
        """

        components.html(tradingview_html, height=520)
    
    st.markdown('<div style="margin: 1rem 0 1rem 0;"><div class="section-header" style="margin: 0;"><span class="section-icon">📈</span><span>AI Model Analytics</span></div></div>', unsafe_allow_html=True)
    
    # Create two columns for charts
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Price Prediction Chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=['Now'] + [card['horizon'] for card in prediction_cards],
            y=[current_price] + [card['predicted_price'] for card in prediction_cards],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10, color='#667eea', line=dict(color=plot_marker_outline, width=2)),
            hovertemplate='<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>'
        ))
        
        # Add uncertainty bands
        upper_bound = [current_price] + [card['predicted_price'] + card['uncertainty'] for card in prediction_cards]
        lower_bound = [current_price] + [card['predicted_price'] - card['uncertainty'] for card in prediction_cards]
        
        fig.add_trace(go.Scatter(
            x=['Now'] + [card['horizon'] for card in prediction_cards],
            y=upper_bound,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=['Now'] + [card['horizon'] for card in prediction_cards],
            y=lower_bound,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(width=0),
            name='Uncertainty Range',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(
                text="Price Prediction Trajectory",
                font=dict(size=16, color=plot_title_color, weight=600)
            ),
            xaxis_title="Time Horizon",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=400,
            template=plotly_template,
            paper_bgcolor=plot_panel_bg,
            plot_bgcolor=plot_panel_bg,
            font=dict(size=11, color=plot_text_color),
            margin=dict(t=60, b=40, l=40, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor=plot_legend_bg,
                bordercolor=plot_border,
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence Chart
        fig_conf = go.Figure()
        
        colors = ['#22c55e' if card['signal'] == 'BUY' else '#ef4444' if card['signal'] == 'SELL' else '#fbbf24' for card in prediction_cards]
        
        fig_conf.add_trace(go.Bar(
            x=[card['horizon'] for card in prediction_cards],
            y=[card['confidence'] * 100 for card in prediction_cards],
            marker=dict(
                color=colors,
                line=dict(color=plot_marker_outline, width=1.5)
            ),
            text=[f"{card['confidence']*100:.1f}%" for card in prediction_cards],
            textposition='outside',
            textfont=dict(size=11, color=plot_title_color, weight='bold'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        ))
        
        fig_conf.update_layout(
            title=dict(
                text="AI Confidence Levels",
                font=dict(size=16, color=plot_title_color, weight=600)
            ),
            xaxis_title="Time Horizon",
            yaxis_title="Confidence (%)",
            height=400,
            template=plotly_template,
            paper_bgcolor=plot_panel_bg,
            plot_bgcolor=plot_panel_bg,
            font=dict(size=11, color=plot_text_color),
            yaxis=dict(range=[0, 100]),
            margin=dict(t=60, b=40, l=40, r=40),
            showlegend=False
        )
        
        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Historical Price Chart (Actual only for now)
    recent = _ensure_datetime_index(price_data).tail(168)  # last 7 days @ 1H
    if isinstance(recent.index, pd.DatetimeIndex) and len(recent) > 0 and 'close' in recent.columns:
        fig_historical = go.Figure()
        fig_historical.add_trace(go.Scatter(
            x=recent.index,
            y=recent['close'].astype(float),
            mode='lines',
            name='BTC Actual Price (Last 7D)',
            line=dict(color='#667eea', width=2),
            hovertemplate='<b>%{x}</b><br>Actual: $%{y:,.2f}<extra></extra>'
        ))

        y_min = float(recent['close'].min()) * 0.985
        y_max = float(recent['close'].max()) * 1.015

        fig_historical.update_layout(
            title=dict(
                text='BTC Price (Last 7 Days)',
                font=dict(size=16, color=plot_title_color, weight=600)
            ),
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=350,
            template=plotly_template,
            paper_bgcolor=plot_panel_bg,
            plot_bgcolor=plot_panel_bg,
            font=dict(size=11, color=plot_text_color),
            yaxis=dict(range=[y_min, y_max]),
            margin=dict(t=60, b=40, l=40, r=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor=plot_legend_bg,
                bordercolor=plot_border,
                borderwidth=1
            )
        )

        st.plotly_chart(fig_historical, use_container_width=True)
        with st.expander("Data diagnostics", expanded=False):
            st.caption(_format_price_data_diagnostics(price_data, current_price=current_price))
        st.caption("Prediction history is being recorded from now. In ~7–8 days we can enable a second line using real stored predictions vs actual.")
    else:
        st.info("Not enough data to render the last-7-days chart yet.")
    
    # Market Metrics Section
    st.markdown("""
        <div class="section-header" style="margin-top: 3rem;">
            <span class="section-icon">💹</span>
            <span>Market Metrics</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate market metrics
    price_24h_ago = price_data['close'].iloc[-25] if len(price_data) >= 25 else price_data['close'].iloc[0]
    change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
    
    volatility_7d = price_data['close'].tail(168).pct_change().std() * 100

    # 24H volume: most exchanges return volume in base units (e.g., BTC), not USD.
    # Convert to USD notional when raw volume looks like base units.
    volume_24h = 0.0
    try:
        if 'volume' in price_data.columns and 'close' in price_data.columns:
            vol_24 = pd.to_numeric(price_data['volume'], errors='coerce').tail(24)
            close_24 = pd.to_numeric(price_data['close'], errors='coerce').tail(24)
            raw_sum = float(vol_24.sum(skipna=True))

            # Heuristic: if raw sum is too small to be USD volume, treat it as base units and convert.
            # (Typical USD 24h volume for BTC is many millions+; base-unit volume is in the thousands.)
            if raw_sum > 0 and raw_sum < 1e7:
                volume_24h = float((vol_24 * close_24).sum(skipna=True))
            else:
                volume_24h = raw_sum
    except Exception:
        volume_24h = 0.0
    avg_confidence = sum([card['confidence'] for card in prediction_cards]) / len(prediction_cards) * 100
    
    # Display metrics in cards
    met_cols = st.columns(4, gap="large")
    
    metrics = [
        {"label": "24H Change", "value": f"{change_24h:+.2f}%", "icon": "📊"},
        {"label": "7D Volatility", "value": f"{volatility_7d:.2f}%", "icon": "⚡"},
        {"label": "24H Volume", "value": f"${volume_24h/1e9:.2f}B" if volume_24h > 0 else "N/A", "icon": "💹"},
        {"label": "Avg Confidence", "value": f"{avg_confidence:.1f}%", "icon": "📈"}
    ]
    
    for col, metric in zip(met_cols, metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric['icon']} {metric['label']}</div>
                    <div class="metric-value">{metric['value']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style="margin-top: 4rem; padding: 2rem; text-align: center; border-top: 1px solid rgba(148, 163, 184, 0.1);">
            <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">
                Powered by DSBA
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # (Sidebar intentionally unused; menu is on-page.)

if __name__ == "__main__":
    main()
