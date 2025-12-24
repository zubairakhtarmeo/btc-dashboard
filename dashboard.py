"""
Premium Enterprise-Grade Bitcoin Forecasting Dashboard
Bloomberg Terminal Style - AI-Powered Multi-Horizon Predictions
"""

from __future__ import annotations

import os
import warnings

# Suppress TensorFlow CUDA/GPU errors on Streamlit Cloud (CPU-only environment)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

# Suppress scikit-learn version mismatch warnings
warnings.filterwarnings('ignore', category=UserWarning)

import json
import pickle
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Delay heavy ML imports until needed to avoid native library crashes on startup
from data_collector import CryptoDataCollector


print("[DASHBOARD] Starting imports...", flush=True)

# Supabase (PostgreSQL) logging - simple and reliable
try:
    from supabase_logger import (
        sync_prediction_log_records,
        sync_validation_24h_records,
        get_validation_history,
        get_prediction_log_history,
        cleanup_bad_validation_records
    )
    print("[DASHBOARD] supabase_logger imported OK", flush=True)
except Exception as e:
    print(f"[DASHBOARD] supabase import failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    def cleanup_bad_validation_records():
        return 0
    def sync_validation_24h_records(records):
        pass
    def sync_prediction_log_records(records):
        pass
    def get_validation_history(days=30):
        return []
    def get_prediction_log_history(days=7):
        return []

print("[DASHBOARD] Imports complete", flush=True)

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

# Optional local artifacts produced by scripts (data_collector.py / feature_engineering.py)
CACHED_PRICE_PATH = BASE_DIR / 'cache' / 'price_data.pkl'
CACHED_SIMPLE_FEATURES_PATH = BASE_DIR / 'cache' / 'simple_features.pkl'


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
    """Load validation records from Supabase first, fall back to local cache."""
    # Try loading from Supabase first (survives app restarts)
    try:
        db_records = get_validation_history(days=30)
        if db_records:
            print(f"[DASHBOARD] Loaded {len(db_records)} validation records from Supabase")
            return db_records
    except Exception as e:
        print(f"[DASHBOARD] Could not load from Supabase: {e}")
    
    # Fall back to local file cache
    try:
        if not VALIDATION_24H_PATH.exists():
            return []
        with open(VALIDATION_24H_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            local_records = [r for r in data if isinstance(r, dict)]
            print(f"[DASHBOARD] Loaded {len(local_records)} validation records from local cache")
            return local_records
        return []
    except Exception:
        return []


def _save_validation_records(records: list[dict]) -> None:
    """Save to both local cache AND Supabase for redundancy."""
    # Save to local file cache
    try:
        VALIDATION_24H_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Convert Decimal to float for JSON serialization
        records_serializable = []
        for r in records:
            r_copy = dict(r)
            for key in ['predicted_24h', 'actual_24h']:
                if r_copy.get(key) is not None:
                    r_copy[key] = float(r_copy[key])
            records_serializable.append(r_copy)
        with open(VALIDATION_24H_PATH, 'w', encoding='utf-8') as f:
            json.dump(records_serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[DASHBOARD] Could not save to local cache: {e}")
    
    # Also sync to Supabase (happens in background, non-blocking)
    try:
        sync_validation_24h_records(records)
    except Exception as e:
        print(f"[DASHBOARD] Could not sync to Supabase: {e}")


def _nearest_close_at(price_data: pd.DataFrame, target_ts: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
    try:
        df = _ensure_datetime_index(price_data).sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"[DEBUG] _nearest_close_at: index is not DatetimeIndex", flush=True)
            return None, None
        if 'close' not in df.columns or len(df) == 0:
            print(f"[DEBUG] _nearest_close_at: no close column or empty df", flush=True)
            return None, None

        # Remove duplicate timestamps (keep last occurrence)
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='last')]

        target_ts = pd.to_datetime(target_ts, utc=True)

        # Use the last known close at-or-before target time (prevents accidental future lookup).
        idx = df.index.get_indexer([target_ts], method='pad')[0]
        if idx < 0:
            print(f"[DEBUG] _nearest_close_at: no prior candle for {target_ts}", flush=True)
            return None, None

        actual_ts = df.index[idx]
        time_diff = target_ts - actual_ts
        # Sanity tolerance: if the closest prior candle is too far away, treat as missing.
        # Increased tolerance to 6 hours to handle API data gaps
        if time_diff > pd.Timedelta(hours=6):
            print(f"[DEBUG] _nearest_close_at: nearest candle at {actual_ts} is {time_diff} away from target {target_ts} (>6h)", flush=True)
            return None, None
        actual_price = float(df['close'].iloc[idx])
        return actual_price, actual_ts
    except Exception as e:
        print(f"[DEBUG] _nearest_close_at exception: {e}", flush=True)
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
                print(f"[DASHBOARD] Filled actual for {t}: ${actual_price:,.2f}")
            else:
                print(f"[DASHBOARD] Could not find actual price for {t} (data gap or missing)")

    _save_validation_records(records)

    # Build chart DF from completed records
    rows = []
    for r in records:
        act = r.get('actual_24h')
        pred = r.get('predicted_24h')
        target_at = pd.to_datetime(r.get('target_at'), utc=True, errors='coerce')
        made_at = pd.to_datetime(r.get('made_at'), utc=True, errors='coerce')
        if act is None or pred is None or target_at is pd.NaT:
            continue
        act_f = float(act)
        pred_f = float(pred)
        err = act_f - pred_f  # Actual - Predicted
        err_pct = (abs(err) / act_f * 100.0) if act_f else 0.0

        # Price variation: compute absolute percentage variance relative to the prediction.
        # Variance_i = |actual - predicted| / predicted (use NaN when predicted is zero or invalid)
        baseline = None
        if made_at is not pd.NaT:
            baseline, _ = _nearest_close_at(price_data, made_at)
        baseline_f = float(baseline) if baseline is not None else np.nan
        variance = np.nan
        if pred_f and np.isfinite(pred_f) and float(pred_f) != 0.0:
            variance = abs(act_f - pred_f) / float(pred_f)
        rows.append({
            'target_at': target_at,
            'predicted': pred_f,
            'actual': act_f,
            'error': err,
            'baseline': baseline_f,
            'abs_pct_variance': variance,
        })

    df_chart = (
        pd.DataFrame(rows).sort_values('target_at')
        if rows
        else pd.DataFrame(columns=['target_at', 'predicted', 'actual', 'error', 'baseline', 'abs_pct_variance'])
    )

    # Summary: latest completed record if available, else pending
    summary_html = ''
    if not df_chart.empty:
        last = df_chart.iloc[-1]
        # Compute tolerance-based accuracy for validation summary
        tolerance_pct = 2.0
        valid_rows = df_chart[df_chart['abs_pct_variance'].notna()].copy()
        errors_pct = valid_rows['abs_pct_variance'] * 100.0  # Convert to percentage
        tolerance_acc = None
        if not errors_pct.empty:
            correct_count = int((errors_pct <= tolerance_pct).sum())
            total_count = int(len(errors_pct))
            tolerance_acc = float((correct_count / total_count) * 100.0) if total_count > 0 else None
        
        summary_html = (
            "<div style='margin-top: 1rem; padding: 1rem; border-radius: 12px; background: var(--dsba-surface-glass); border: 1px solid var(--dsba-border);'>"
            "<div style='color: var(--dsba-accent); font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem;'>📊 Last 24H Validation</div>"
            f"<div style='color: var(--dsba-text-2); font-size: 0.82rem; display: flex; flex-wrap: wrap; gap: 1rem;'>"
            f"<span>Predicted: <b style='color: var(--dsba-text);'>${last['predicted']:,.0f}</b></span>"
            f"<span>Actual: <b style='color: var(--dsba-text);'>${last['actual']:,.0f}</b></span>"
            f"<span>Error: <b style='color: {'var(--dsba-positive)' if last['error'] >= 0 else 'var(--dsba-negative)'};'>{last['error']:+,.0f}</b></span>"
            f"<span>Accuracy within ±2%: <b style='color: var(--dsba-accent);'>{tolerance_acc:.1f}%</b></span>" if tolerance_acc is not None else "<span>Accuracy within ±2%: <b style='color: var(--dsba-accent);'>n/a</b></span>"
            "</div>"
            "</div>"
        )
    else:
        summary_html = (
            "<div style='margin-top: 1rem; padding: 1rem; border-radius: 12px; background: var(--dsba-surface-glass); border: 1px solid var(--dsba-border);'>"
            "<div style='color: var(--dsba-accent); font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem;'>📊 24H Validation</div>"
            "<div style='color: var(--dsba-text-2); font-size: 0.82rem;'>"
            "⏳ Tracking started — first comparison will appear after 24 hours."
            "</div>"
            "</div>"
        )

    return summary_html, df_chart


def _load_prediction_log() -> list[dict]:
    """Load prediction log from Supabase first, fall back to local cache."""
    # Try loading from Supabase first (survives app restarts)
    try:
        db_records = get_prediction_log_history(days=7)
        if db_records:
            print(f"[DASHBOARD] Loaded {len(db_records)} prediction log records from Supabase")
            return db_records
    except Exception as e:
        print(f"[DASHBOARD] Could not load prediction log from Supabase: {e}")
    
    # Fall back to local file cache
    try:
        if not PREDICTION_LOG_PATH.exists():
            return []
        with open(PREDICTION_LOG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            local_records = [r for r in data if isinstance(r, dict)]
            print(f"[DASHBOARD] Loaded {len(local_records)} prediction log records from local cache")
            return local_records
        return []
    except Exception:
        return []


def _save_prediction_log(records: list[dict]) -> None:
    """Save to both local cache AND Supabase for redundancy."""
    # Save to local file cache
    try:
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Convert Decimal to float for JSON serialization
        records_serializable = []
        for r in records:
            r_copy = dict(r)
            for key in ['current_price', 'predicted_price']:
                if r_copy.get(key) is not None:
                    r_copy[key] = float(r_copy[key])
            records_serializable.append(r_copy)
        with open(PREDICTION_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump(records_serializable, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[DASHBOARD] Could not save prediction log to local cache: {e}")
    
    # Supabase sync happens in _append_prediction_log (per new record)


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
            baseline_f = np.nan
            try:
                baseline_f = float(r.get('current_price'))
            except Exception:
                baseline_f = np.nan

            # Compute absolute percentage variance relative to predicted price
            variance = np.nan
            try:
                p = float(predicted_price)
                if p != 0.0 and np.isfinite(p):
                    variance = abs(float(actual_price) - p) / p
            except Exception:
                variance = np.nan

            rows.append({
                'created_at': created_at,
                'target_at': target_at,
                'predicted': float(predicted_price),
                'actual': float(actual_price),
                'actual_at': pd.to_datetime(actual_ts, utc=True),
                'error': error,
                'error_pct': error_pct,
                'baseline': baseline_f,
                'abs_pct_variance': variance,
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Theme variables - Modern Dark Theme (Default) */
    :root {
        --dsba-bg-0: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        --dsba-surface-0: rgba(30, 30, 50, 0.85);
        --dsba-surface-1: rgba(25, 25, 45, 0.75);
        --dsba-surface-glass: rgba(255, 255, 255, 0.03);
        --dsba-border: rgba(255, 255, 255, 0.08);
        --dsba-border-glow: rgba(99, 102, 241, 0.3);
        --dsba-text: #f8fafc;
        --dsba-text-2: rgba(248, 250, 252, 0.75);
        --dsba-text-3: rgba(148, 163, 184, 0.9);
        --dsba-pill-bg: rgba(30, 30, 50, 0.6);
        --dsba-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
        --dsba-shadow-hover: 0 12px 40px rgba(99, 102, 241, 0.25);
        --dsba-shadow-glow: 0 0 30px rgba(99, 102, 241, 0.15);
        --dsba-accent: #6366f1;
        --dsba-accent-2: #8b5cf6;
        --dsba-accent-3: #a855f7;
        --dsba-positive: #10b981;
        --dsba-positive-glow: rgba(16, 185, 129, 0.3);
        --dsba-negative: #ef4444;
        --dsba-negative-glow: rgba(239, 68, 68, 0.3);
        --dsba-warning: #f59e0b;
        --dsba-warning-glow: rgba(245, 158, 11, 0.3);
        --dsba-card-gradient: linear-gradient(145deg, rgba(30, 30, 50, 0.9) 0%, rgba(20, 20, 40, 0.95) 100%);
        --dsba-gold: #fbbf24;
        --dsba-gold-gradient: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    }

    /* Light theme override */
    html[data-theme="light"], body[data-theme="light"], [data-theme="light"],
    [data-testid="stAppViewContainer"][data-theme="light"] {
        --dsba-bg-0: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 50%, #f8fafc 100%);
        --dsba-surface-0: rgba(255, 255, 255, 0.95);
        --dsba-surface-1: rgba(255, 255, 255, 0.85);
        --dsba-surface-glass: rgba(255, 255, 255, 0.7);
        --dsba-border: rgba(15, 23, 42, 0.1);
        --dsba-border-glow: rgba(99, 102, 241, 0.2);
        --dsba-text: #0f172a;
        --dsba-text-2: rgba(15, 23, 42, 0.75);
        --dsba-text-3: rgba(71, 85, 105, 0.9);
        --dsba-pill-bg: rgba(255, 255, 255, 0.8);
        --dsba-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
        --dsba-shadow-hover: 0 12px 40px rgba(99, 102, 241, 0.15);
        --dsba-shadow-glow: 0 0 30px rgba(99, 102, 241, 0.08);
        --dsba-card-gradient: linear-gradient(145deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.98) 100%);
    }
    
    /* App background */
    .main, [data-testid="stAppViewContainer"] {
        background: var(--dsba-bg-0);
        background-attachment: fixed;
    }
    
    .block-container {
        padding: 0.5rem 1.5rem 2rem 1.5rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        visibility: hidden;
    }
    header {visibility: visible;}
    header[data-testid="stHeader"] {
        background: transparent;
        box-shadow: none;
    }
    [data-testid="stSidebarCollapseButton"] {visibility: visible;}
    
    /* Premium Status Strip with Glass Effect */
    .status-strip {
        background: var(--dsba-surface-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--dsba-border);
        border-radius: 16px;
        padding: 0.6rem 1.5rem;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        box-shadow: var(--dsba-shadow);
    }

    .status-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        min-width: 200px;
    }

    .status-brand {
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, var(--dsba-gold) 0%, #f59e0b 50%, var(--dsba-accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        white-space: nowrap;
        text-shadow: 0 0 30px rgba(251, 191, 36, 0.3);
    }

    .status-center {
        color: var(--dsba-text-2);
        font-size: 0.85rem;
        font-weight: 500;
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
        gap: 0.65rem;
        min-width: 300px;
        font-size: 0.75rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        border: 1px solid var(--dsba-border);
        background: var(--dsba-pill-bg);
        color: var(--dsba-text-2);
        font-size: 0.72rem;
        transition: all 0.2s ease;
    }

    .status-pill:hover {
        border-color: var(--dsba-border-glow);
        box-shadow: var(--dsba-shadow-glow);
    }

    .status-pill.accuracy {
        padding: 0.4rem 1rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--dsba-text);
        background: linear-gradient(135deg, var(--dsba-accent) 0%, var(--dsba-accent-2) 100%);
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--dsba-positive);
        box-shadow: 0 0 10px var(--dsba-positive-glow), 0 0 20px var(--dsba-positive-glow);
        animation: pulse-glow 2s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px var(--dsba-positive-glow), 0 0 20px var(--dsba-positive-glow); }
        50% { opacity: 0.7; box-shadow: 0 0 5px var(--dsba-positive-glow), 0 0 10px var(--dsba-positive-glow); }
    }

    /* Hero Price Panel */
    .price-panel {
        max-width: 1100px;
        width: 100%;
        margin: 0.5rem auto 1.25rem auto;
        background: var(--dsba-card-gradient);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--dsba-border);
        border-radius: 20px;
        padding: 1.25rem 1.5rem;
        box-shadow: var(--dsba-shadow), var(--dsba-shadow-glow);
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .price-panel::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--dsba-accent), var(--dsba-accent-2), transparent);
        opacity: 0.5;
    }

    .price-meta {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        min-width: 0;
    }

    .price-meta.left { text-align: left; }
    .price-meta.right { text-align: right; }

    .meta-label {
        color: var(--dsba-text-3);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        line-height: 1.1;
    }

    .meta-value {
        color: var(--dsba-text);
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.3;
        white-space: nowrap;
    }

    .meta-value.positive { 
        color: var(--dsba-positive); 
        text-shadow: 0 0 20px var(--dsba-positive-glow);
    }
    .meta-value.negative { 
        color: var(--dsba-negative); 
        text-shadow: 0 0 20px var(--dsba-negative-glow);
    }

    .price-center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-width: 0;
        position: relative;
    }

    .price-center-label {
        color: var(--dsba-text-3);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        line-height: 1.1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .btc-icon {
        font-size: 1.1rem;
    }

    .price-center-value {
        color: var(--dsba-text);
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-top: 0.25rem;
        background: linear-gradient(135deg, var(--dsba-text) 0%, var(--dsba-gold) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }

    @media (max-width: 900px) {
        .price-panel { grid-template-columns: 1fr; gap: 0.75rem; padding: 1rem; }
        .price-meta.left, .price-meta.right { text-align: center; }
        .price-center-value { font-size: 2.25rem; }
    }
    
    .price-label {
        color: var(--dsba-text-3);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    .price-value {
        color: var(--dsba-text);
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0.5rem 0;
    }
    
    .price-change {
        font-size: 1rem;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        border-radius: 999px;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    .price-change.positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(16, 185, 129, 0.1) 100%);
        color: var(--dsba-positive);
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 20px var(--dsba-positive-glow);
    }
    
    .price-change.negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        color: var(--dsba-negative);
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 20px var(--dsba-negative-glow);
    }
    
    /* Premium KPI Cards with Glass Morphism */
    .kpi-card {
        background: var(--dsba-card-gradient);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--dsba-border);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        box-shadow: var(--dsba-shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--dsba-accent), var(--dsba-accent-2), var(--dsba-accent-3));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--dsba-shadow-hover), var(--dsba-shadow-glow);
        border-color: var(--dsba-border-glow);
    }

    .kpi-card:hover::before {
        opacity: 1;
    }
    
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--dsba-border);
    }
    
    .kpi-time {
        color: var(--dsba-accent);
        font-size: 0.85rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .kpi-icon {
        font-size: 1.3rem;
        opacity: 0.8;
    }
    
    .kpi-price {
        color: var(--dsba-text);
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .kpi-change {
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0.35rem 0;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
    }
    
    .kpi-change.up {
        color: var(--dsba-positive);
        background: rgba(16, 185, 129, 0.15);
    }
    
    .kpi-change.down {
        color: var(--dsba-negative);
        background: rgba(239, 68, 68, 0.15);
    }
    
    /* Enhanced Confidence Gauge */
    .confidence-gauge {
        margin: 0.75rem 0;
    }
    
    .confidence-label {
        color: var(--dsba-text-3);
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.35rem;
    }
    
    .confidence-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        background: linear-gradient(90deg, var(--dsba-accent) 0%, var(--dsba-accent-2) 50%, var(--dsba-accent-3) 100%);
        box-shadow: 0 0 12px var(--dsba-border-glow);
        position: relative;
    }

    .confidence-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .confidence-value {
        color: var(--dsba-text-2);
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }
    
    /* Premium Signal Badges */
    .signal-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1.25rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.65rem;
        transition: all 0.3s ease;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(16, 185, 129, 0.15) 100%);
        color: var(--dsba-positive);
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 20px var(--dsba-positive-glow), inset 0 0 20px rgba(16, 185, 129, 0.1);
    }

    .signal-buy:hover {
        box-shadow: 0 0 30px var(--dsba-positive-glow), inset 0 0 25px rgba(16, 185, 129, 0.15);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(239, 68, 68, 0.15) 100%);
        color: var(--dsba-negative);
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 20px var(--dsba-negative-glow), inset 0 0 20px rgba(239, 68, 68, 0.1);
    }

    .signal-sell:hover {
        box-shadow: 0 0 30px var(--dsba-negative-glow), inset 0 0 25px rgba(239, 68, 68, 0.15);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(245, 158, 11, 0.15) 100%);
        color: var(--dsba-warning);
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 20px var(--dsba-warning-glow), inset 0 0 20px rgba(245, 158, 11, 0.1);
    }

    .signal-hold:hover {
        box-shadow: 0 0 30px var(--dsba-warning-glow), inset 0 0 25px rgba(245, 158, 11, 0.15);
    }
    
    /* Section Headers with Gradient */
    .section-header {
        color: var(--dsba-text);
        font-size: 1.5rem;
        font-weight: 800;
        margin: 2rem 0 1rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, var(--dsba-accent), var(--dsba-accent-2), transparent) 1;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-icon {
        font-size: 1.5rem;
        opacity: 0.9;
    }

    .section-subtitle {
        color: var(--dsba-text-2);
        font-size: 0.88rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    .subsection-title {
        color: var(--dsba-text);
        font-size: 1.05rem;
        font-weight: 700;
        margin: 1.25rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .subsection-title::before {
        content: '';
        width: 4px;
        height: 20px;
        background: linear-gradient(180deg, var(--dsba-accent), var(--dsba-accent-2));
        border-radius: 2px;
    }
    
    /* Chart Container with Glass Effect */
    .chart-container {
        background: var(--dsba-card-gradient);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--dsba-border);
        border-radius: 16px;
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
    
    /* Premium Metric Card */
    .metric-card {
        background: var(--dsba-card-gradient);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--dsba-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--dsba-shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 50%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--dsba-accent), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--dsba-shadow-hover);
        border-color: var(--dsba-border-glow);
    }

    .metric-card:hover::before {
        opacity: 1;
    }
    
    .metric-label {
        color: var(--dsba-text-3);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.65rem;
    }
    
    .metric-value {
        color: var(--dsba-text);
        font-size: 1.65rem;
        font-weight: 800;
    }
    
    /* Streamlit Metric Override */
    [data-testid="stMetric"] {
        background: var(--dsba-card-gradient);
        backdrop-filter: blur(20px);
        border: 1px solid var(--dsba-border);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: var(--dsba-shadow);
    }
    
    [data-testid="stMetric"] label {
        color: var(--dsba-text-3) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--dsba-text) !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* Streamlit Button Override */
    .stButton > button {
        background: linear-gradient(135deg, var(--dsba-accent) 0%, var(--dsba-accent-2) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.25rem;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }

    /* Expander Override - Make text visible */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 50, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
    }

    .streamlit-expanderHeader:hover {
        border-color: rgba(99, 102, 241, 0.5) !important;
        background: rgba(40, 40, 60, 0.9) !important;
    }

    .streamlit-expanderHeader p, 
    .streamlit-expanderHeader span,
    [data-testid="stExpander"] summary span {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        border-radius: 12px !important;
        background: rgba(20, 20, 35, 0.6) !important;
    }

    [data-testid="stExpander"] details {
        border: none !important;
    }

    /* Caption text - make readable */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }

    /* Info boxes */
    [data-testid="stAlert"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] p {
        color: #e2e8f0 !important;
    }

    /* Popover/Menu Override */
    [data-testid="stPopover"] {
        background: var(--dsba-card-gradient);
        backdrop-filter: blur(20px);
        border: 1px solid var(--dsba-border);
        border-radius: 16px;
        box-shadow: var(--dsba-shadow);
    }

    /* General text visibility overrides */
    .stMarkdown, .stMarkdown p, .stText {
        color: #e2e8f0 !important;
    }

    /* Checkbox labels */
    .stCheckbox label span {
        color: #e2e8f0 !important;
    }

    /* Selectbox and other inputs */
    .stSelectbox label, .stTextInput label, .stNumberInput label {
        color: #94a3b8 !important;
    }

    /* Spinner text */
    .stSpinner > div {
        color: #e2e8f0 !important;
    }

    /* Code blocks */
    .stCodeBlock {
        background: rgba(15, 15, 30, 0.9) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        border-radius: 8px !important;
    }
    
    /* Loading Animation */
    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .price-value { font-size: 2.5rem; }
        .kpi-price { font-size: 1.4rem; }
        .status-strip { flex-direction: column; gap: 0.5rem; padding: 0.75rem; }
        .status-left, .status-right { min-width: auto; justify-content: center; }
        .status-center { display: none; }
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--dsba-surface-1);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--dsba-accent), var(--dsba-accent-2));
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--dsba-accent-2), var(--dsba-accent-3));
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
def load_model_and_predictor(model_mtime: float, metadata_mtime: float):
    """Load model and predictor (cached for performance).

    The cache key includes the model/metadata mtimes so that if you retrain and
    the files change, Streamlit automatically reloads them.
    """
    try:
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        # Import heavy ML dependencies lazily to avoid startup crashes on cloud hosts
        keras = None
        EnhancedCryptoPricePredictor = None
        TemporalConvLayer = None
        MultiHeadAttentionCustom = None
        try:
            # Force CPU-only mode before TensorFlow import
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            
            from tensorflow import keras
            from enhanced_predictor import (
                EnhancedCryptoPricePredictor,
                MultiHeadAttentionCustom,
                TemporalConvLayer,
            )
        except Exception as e:
            st.error(f"Failed to import ML dependencies: {e}")
            return None, None

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
def fetch_live_data(use_cached_history: bool, cached_history_mtime: float):
    """Fetch live Bitcoin price and historical data.

    If use_cached_history is True and cache/price_data.pkl exists, use it as
    the historical series (and still fetch live current price).
    """
    # The dashboard already uses Streamlit in-memory caching (ttl=60).
    # Disk caching has caused stale/flat series on Streamlit Cloud, so keep it off here.
    collector = CryptoDataCollector(use_cache=False)
    
    try:
        current_price = None
        try:
            current_price = collector.get_current_price('bitcoin')
        except Exception:
            current_price = None

        price_data = None
        if bool(use_cached_history) and CACHED_PRICE_PATH.exists():
            try:
                price_data = pd.read_pickle(str(CACHED_PRICE_PATH))
            except Exception:
                price_data = None

        if price_data is None or not isinstance(price_data, pd.DataFrame) or len(price_data) == 0:
            price_data = collector.price_collector.get_price_data('bitcoin', hours_back=2000, interval='1h')

        # If live price failed, fall back to last close
        if current_price is None:
            try:
                current_price = float(price_data['close'].iloc[-1])
            except Exception:
                current_price = None

        # Only blend live price into the last candle if it's close to the historical last close.
        # This prevents a confusing vertical "cliff" when the historical series is stale/flat.
        try:
            last_close = float(price_data['close'].iloc[-1])
            if current_price is not None and last_close > 0 and abs(float(current_price) - last_close) / last_close <= 0.01:
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
    """Return (n, median_abs_pct_error, average_percentage_error_pct).

    Average Percentage Error (%) using predicted price as denominator.

    Per-row formula:
        error_pct_i = |actual_i - predicted_i| / predicted_i
    
    Row eligibility rule:
        Include ONLY rows where actual_i IS NOT NULL
    
    Final metric:
        average_error_pct = SUM(error_pct_i) / COUNT(rows where actual IS NOT NULL) × 100

    Notes:
    - Only rows with non-NULL actual values are included
    - Denominator is predicted price
    - Predictions with predicted == 0 are excluded to avoid division by zero
    """
    try:
        if df is None or df.empty:
            return 0, None, None
        d = df.copy()
        d['predicted'] = pd.to_numeric(d.get('predicted'), errors='coerce')
        d['actual'] = pd.to_numeric(d.get('actual'), errors='coerce')
        
        # Row eligibility: actual IS NOT NULL
        d = d.dropna(subset=['actual'])
        if d.empty:
            return 0, None, None

        # Median Absolute Percentage Error (unchanged; actual as denominator)
        ape = (np.abs(d['actual'] - d['predicted']) / np.abs(d['actual']).replace(0, np.nan)) * 100.0
        ape = ape.replace([np.inf, -np.inf], np.nan).dropna()
        median_ape = float(np.nanmedian(ape.values)) if not ape.empty else None

        # Average Percentage Error per user formula:
        # error_pct = |actual - predicted| / predicted
        # average_error_pct = SUM(error_pct) / COUNT(rows) × 100
        valid_mask = d['predicted'].notna() & np.isfinite(d['predicted']) & (d['predicted'] != 0)
        if not bool(valid_mask.any()):
            # No valid entries to compute average error
            return int(valid_mask.sum()), median_ape, None

        # Calculate error_pct for each prediction using predicted as denominator
        error_pcts = np.abs(d.loc[valid_mask, 'actual'] - d.loc[valid_mask, 'predicted']) / d.loc[valid_mask, 'predicted']
        error_pcts = error_pcts.replace([np.inf, -np.inf], np.nan).dropna()
        if error_pcts.empty:
            return int(valid_mask.sum()), median_ape, None

        # Average error (as decimal, e.g., 0.05 for 5% error)
        avg_error_decimal = float(error_pcts.mean())
        
        # Convert to accuracy: accuracy_pct = (1 - average_error_pct) × 100
        accuracy_pct = (1.0 - avg_error_decimal) * 100.0
        
        return int(len(error_pcts)), median_ape, float(accuracy_pct)
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


def _run_python_script(script_filename: str, *, timeout_s: int) -> dict:
    """Run a known local python script and capture output.

    NOTE: This executes on the machine hosting the Streamlit app.
    """
    script_path = (BASE_DIR / script_filename).resolve()
    started_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    if not script_path.exists():
        return {
            "script": script_filename,
            "started_at": started_at,
            "returncode": 127,
            "stdout": "",
            "stderr": f"Script not found: {script_path}",
            "timeout_s": timeout_s,
        }

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
        )
        return {
            "script": script_filename,
            "started_at": started_at,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "timeout_s": timeout_s,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "script": script_filename,
            "started_at": started_at,
            "returncode": 124,
            "stdout": (e.stdout or "") if hasattr(e, "stdout") else "",
            "stderr": (e.stderr or "") if hasattr(e, "stderr") else "",
            "timeout_s": timeout_s,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "script": script_filename,
            "started_at": started_at,
            "returncode": 1,
            "stdout": "",
            "stderr": f"Failed to run script: {e}",
            "timeout_s": timeout_s,
        }


def _render_run_scripts_section() -> None:
    """Render buttons that run key project scripts."""
    st.markdown("### ▶️ Run Scripts")
    st.caption("Runs on the server hosting this dashboard.")

    # Use artifacts (if present) instead of always pulling live history.
    if "use_cached_history" not in st.session_state:
        st.session_state.use_cached_history = False
    if "use_cached_features" not in st.session_state:
        st.session_state.use_cached_features = False

    st.session_state.use_cached_history = st.checkbox(
        "Use cached history from data_collector (cache/price_data.pkl)",
        value=bool(st.session_state.use_cached_history),
        key="use_cached_history_chk",
    )
    st.session_state.use_cached_features = st.checkbox(
        "Use cached engineered features (cache/simple_features.pkl)",
        value=bool(st.session_state.use_cached_features),
        key="use_cached_features_chk",
    )

    scripts = {
        "data_collector": {
            "label": "Run data_collector",
            "file": "data_collector.py",
            "timeout_s": 180,
        },
        "feature_engineering": {
            "label": "Run feature_engineering",
            "file": "feature_engineering.py",
            "timeout_s": 300,
        },
        "enhanced_predictor": {
            "label": "Run enhanced_predictor",
            "file": "enhanced_predictor.py",
            "timeout_s": 120,
        },
        "train_simple": {
            "label": "Run train_simple (training)",
            "file": "train_simple.py",
            "timeout_s": 3600,
        },
        "predict_realtime": {
            "label": "Run predict_realtime",
            "file": "predict_realtime.py",
            "timeout_s": 300,
        },
    }

    if "script_running" not in st.session_state:
        st.session_state.script_running = False

    def _run_and_store(script_key: str) -> None:
        cfg = scripts[script_key]
        st.session_state.script_running = True
        try:
            with st.spinner(f"Running {cfg['file']}…"):
                result = _run_python_script(cfg["file"], timeout_s=int(cfg["timeout_s"]))
            st.session_state.last_script_run = result

            # After scripts that update on-disk artifacts or model files, rerun so the
            # dashboard picks up the latest versions. No cache clearing needed.
            if int(result.get('returncode') or 0) == 0 and cfg.get('file') in {
                'train_simple.py',
                'data_collector.py',
                'feature_engineering.py',
            }:
                st.rerun()
        finally:
            st.session_state.script_running = False

    disabled = bool(st.session_state.script_running)

    for key, cfg in scripts.items():
        if st.button(cfg["label"], width='stretch', key=f"run_script_{key}", disabled=disabled):
            _run_and_store(key)

    last = st.session_state.get("last_script_run")
    if last:
        rc = last.get("returncode")
        timed_out = bool(last.get("timed_out"))
        status = "✅ Success" if rc == 0 else ("⏱️ Timed out" if timed_out else "❌ Failed")
        st.markdown("---")
        st.markdown("### 🧾 Last Script Output")
        st.write(f"{status} · {last.get('script')} · started {last.get('started_at')} · timeout {last.get('timeout_s')}s")
        stdout = (last.get("stdout") or "").strip()
        stderr = (last.get("stderr") or "").strip()
        combined = ""
        if stdout:
            combined += "[stdout]\n" + stdout + "\n"
        if stderr:
            combined += "\n[stderr]\n" + stderr + "\n"
        if not combined:
            combined = "(no output)"
        max_chars = 20000
        if len(combined) > max_chars:
            combined = "(truncated)\n" + combined[-max_chars:]
        st.code(combined)

def main():
    # Thin status strip (essentials first) — update once metrics are available
    status_ph = st.empty()
    status_ph.markdown(
        _render_status_strip(
            utc_hms=datetime.now(timezone.utc).strftime('%H:%M:%S'),
            accuracy_text="Accuracy: calculating…",
        ),
        unsafe_allow_html=True,
    )

    # Force dark theme for Plotly charts (matches our dark UI)
    plotly_template = "plotly_dark"
    plot_title_color = "#f8fafc"
    plot_text_color = "#94a3b8"
    plot_panel_bg = "rgba(15, 15, 30, 0.95)"
    plot_legend_bg = "rgba(25, 25, 45, 0.95)"
    plot_border = "rgba(99, 102, 241, 0.2)"
    plot_marker_outline = "#1a1a2e"
    plot_grid_color = "rgba(148, 163, 184, 0.08)"
    plot_line_color = "#6366f1"
    plot_positive_color = "#10b981"
    plot_negative_color = "#ef4444"

    # Top-left Menu (☰): replaces sidebar controls so it's always discoverable.
    # Use Streamlit popover when available; fall back to an expander.
    if "menu_open" not in st.session_state:
        st.session_state.menu_open = False

    menu_row = st.columns([1, 10])
    with menu_row[0]:
        if hasattr(st, "popover"):
            with st.popover("☰"):
                st.markdown("### ⚙️ Controls")
                if st.button("🔄 Refresh Dashboard", width='stretch', key="menu_refresh"):
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

                st.markdown("---")
                _render_run_scripts_section()
        else:
            if st.button("☰", help="Open menu", key="menu_toggle"):
                st.session_state.menu_open = not st.session_state.menu_open

    if not hasattr(st, "popover"):
        with st.expander("Menu", expanded=bool(st.session_state.menu_open)):
            st.markdown("### ⚙️ Controls")
            if st.button("🔄 Refresh Dashboard", width='stretch', key="menu_refresh_fallback"):
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

            st.markdown("---")
            _render_run_scripts_section()
    
    # Check if model exists
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        st.error("⚠️ Model files not found. Please train the model first using `python train_simple.py`")
        st.stop()
    
    # IMPORTANT: Fetch data FIRST, before loading TensorFlow
    # This prevents TensorFlow segfaults when data fetch fails on Streamlit Cloud
    with st.spinner("📡 Fetching Live Market Data..."):
        cached_history_mtime = float(CACHED_PRICE_PATH.stat().st_mtime) if CACHED_PRICE_PATH.exists() else 0.0
        current_price, price_data, error = fetch_live_data(
            bool(st.session_state.get('use_cached_history', False)),
            cached_history_mtime,
        )
    
    if error or price_data is None or len(price_data) == 0:
        st.error(f"⚠️ Failed to fetch live data: {error or 'No data returned'}")
        st.info("The data APIs may be temporarily unavailable. Please try refreshing in a few minutes.")
        st.stop()
    
    # Log price data range for debugging
    try:
        price_df_debug = _ensure_datetime_index(price_data)
        if isinstance(price_df_debug.index, pd.DatetimeIndex) and len(price_df_debug) > 0:
            print(f"[DASHBOARD] Price data range: {price_df_debug.index[0]} to {price_df_debug.index[-1]} ({len(price_df_debug)} candles)", flush=True)
            # Show timestamps around Dec 22 for debugging
            dec22_candles = price_df_debug[(price_df_debug.index >= '2025-12-22 00:00:00+00:00') & 
                                          (price_df_debug.index <= '2025-12-23 13:00:00+00:00')]
            if len(dec22_candles) > 0:
                print(f"[DASHBOARD] Dec 22-23 candles: {list(dec22_candles.index[:20])}", flush=True)
    except Exception as e:
        print(f"[DASHBOARD] Debug logging error: {e}", flush=True)
    
    # Clean up any bad validation records from previous scaling bugs (one-time cleanup)
    try:
        deleted = cleanup_bad_validation_records()
        if deleted > 0:
            print(f"[DASHBOARD] Cleaned {deleted} bad validation records on startup", flush=True)
    except Exception as e:
        print(f"[DASHBOARD] Cleanup error: {e}", flush=True)

    if current_price is None:
        try:
            current_price = float(price_data['close'].iloc[-1])
        except Exception:
            st.error("⚠️ Could not determine current price")
            st.stop()
    
    # Now load model (TensorFlow) - data is already available
    with st.spinner("🔄 Loading AI Engine..."):
        model_mtime = float(MODEL_PATH.stat().st_mtime) if MODEL_PATH.exists() else 0.0
        metadata_mtime = float(METADATA_PATH.stat().st_mtime) if METADATA_PATH.exists() else 0.0
        predictor, metadata = load_model_and_predictor(model_mtime, metadata_mtime)
    
    if predictor is None:
        st.error("Failed to load model. Please check the model files.")
        st.stop()

    # DB-based 3-day rolling validation using actual predictions with actuals
    rolling_3d_df = pd.DataFrame()
    rolling_err: str | None = None
    rolling_text = "Live Validation (Last 3 Days – DB Actuals): unavailable"
    rolling_acc = None
    rolling_n = 0
    try:
        # Load validation records from DB (last 3 days where actual_24h IS NOT NULL)
        all_records = _load_validation_records()
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=3)
        
        rows = []
        for r in all_records:
            actual = r.get('actual_24h')
            predicted = r.get('predicted_24h')
            target_at = pd.to_datetime(r.get('target_at'), utc=True, errors='coerce')
            
            # Row eligibility: actual IS NOT NULL and within last 3 days
            if actual is None or predicted is None or target_at is pd.NaT:
                continue
            if target_at < cutoff:
                continue
            
            rows.append({
                'target_at': target_at,
                'predicted': float(predicted),
                'actual': float(actual)
            })
        
        if rows:
            rolling_3d_df = pd.DataFrame(rows).sort_values('target_at')
            n_roll, med_ape_roll, acc_roll = _accuracy_from_pred_actual_df(rolling_3d_df)
            rolling_n = int(n_roll)
            rolling_acc = acc_roll
            if n_roll > 0 and acc_roll is not None:
                rolling_text = f"Live Validation (Last 3 Days – DB Actuals): {acc_roll:.1f}% (N={n_roll})"
            elif n_roll > 0:
                rolling_text = f"Live Validation (Last 3 Days – DB Actuals): collecting (N={n_roll})"
    except Exception as e:
        rolling_text = "Live Validation (Last 3 Days – DB Actuals): unavailable"
        rolling_acc = None
        rolling_n = 0
        rolling_err = str(e)
    
    # Compact centered price panel (contained instrument panel)
    price_24h_ago = price_data['close'].iloc[-25] if len(price_data) >= 25 else price_data['close'].iloc[0]
    change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
    change_class = "positive" if change_24h >= 0 else "negative"
    change_symbol = "▲" if change_24h >= 0 else "▼"

    st.markdown(
                f"""<div class="price-panel">
<div class="price-meta left">
    <div class="meta-label">24H Change</div>
    <div class="meta-value {change_class}">{change_symbol} {abs(change_24h):.2f}%</div>
    <div class="meta-label" style="margin-top: 0.5rem;">Status</div>
    <div class="meta-value" style="color: var(--dsba-positive);">● Live</div>
</div>
<div class="price-center">
    <div class="price-center-label"><span class="btc-icon">₿</span> Bitcoin Price</div>
    <div class="price-center-value">${current_price:,.2f}</div>
</div>
<div class="price-meta right">
    <div class="meta-label">Interval</div>
    <div class="meta-value">1H Candles</div>
    <div class="meta-label" style="margin-top: 0.5rem;">Data Points</div>
    <div class="meta-value">{len(price_data):,}</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Accuracy badge placeholder (updated after predictions/backtest are computed)
    price_accuracy_ph = st.empty()
    
    # Generate features and predictions
    with st.spinner("🧠 Generating AI Predictions..."):
        features_df = None
        if bool(st.session_state.get('use_cached_features', False)) and CACHED_SIMPLE_FEATURES_PATH.exists():
            try:
                cached_features = pd.read_pickle(str(CACHED_SIMPLE_FEATURES_PATH))
                cached_features = cached_features.dropna() if isinstance(cached_features, pd.DataFrame) else None
                req = set(metadata.get('feature_names') or [])
                if cached_features is not None and req and req.issubset(set(cached_features.columns)):
                    features_df = cached_features
                else:
                    features_df = None
            except Exception:
                features_df = None

        if features_df is None:
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

    # Use 3-day rolling validation accuracy for status strip
    accuracy_text = "Accuracy: collecting"
    if rolling_n > 0 and rolling_acc is not None:
        accuracy_text = f"Accuracy (3D Rolling – DB Actuals): {rolling_acc:.1f}% (N={rolling_n})"
    elif rolling_n > 0:
        accuracy_text = f"Accuracy (3D Rolling): collecting (N={rolling_n})"

    status_ph.markdown(
        _render_status_strip(
            utc_hms=datetime.now(timezone.utc).strftime('%H:%M:%S'),
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

    # Progressive disclosure: validation details collapsed by default
    with st.expander("Validation: Predicted vs Actual (Last 3 Days)", expanded=False):
        if isinstance(rolling_3d_df, pd.DataFrame) and not rolling_3d_df.empty:
            n_roll, med_ape_roll, acc_roll = _accuracy_from_pred_actual_df(rolling_3d_df)
            st.caption(
                f"DB-based validation uses real predictions and actual prices. Accuracy shows error-based performance (not ±2% tolerance). (N={n_roll})"
            )

            fig_roll = go.Figure()
            
            # Actual bars (professional, clean bars)
            fig_roll.add_trace(go.Bar(
                x=rolling_3d_df['target_at'],
                y=rolling_3d_df['actual'],
                name='Actual',
                marker=dict(
                    color='#10b981',
                    opacity=0.65,
                    line=dict(color='#059669', width=0.5)
                ),
                width=2000000,
                hovertemplate='<b>%{x}</b><br>Actual: $%{y:,.0f}<extra></extra>'
            ))
            
            # Predicted line (clean line without cluttered labels)
            fig_roll.add_trace(go.Scatter(
                x=rolling_3d_df['target_at'],
                y=rolling_3d_df['predicted'],
                mode='lines+markers',
                name='Predicted (24H)',
                line=dict(color='#818cf8', width=3),
                marker=dict(size=6, color='#818cf8', symbol='circle', line=dict(color='#4f46e5', width=1)),
                hovertemplate='<b>%{x}</b><br>Predicted: $%{y:,.0f}<extra></extra>'
            ))

            # Calculate y-axis range (expand to ~10k range to minimize visual gaps)
            y_min = min(rolling_3d_df['actual'].min(), rolling_3d_df['predicted'].min())
            y_max = max(rolling_3d_df['actual'].max(), rolling_3d_df['predicted'].max())
            y_center = (y_min + y_max) / 2
            y_range_target = 10000
            y_axis_min = max(0, y_center - y_range_target / 2)
            y_axis_max = y_center + y_range_target / 2

            title_suffix = f" • Accuracy: {acc_roll:.1f}%" if acc_roll is not None else ""
            fig_roll.update_layout(
                title=dict(
                    text=f"3-Day Rolling Validation: Predicted vs Actual{title_suffix}",
                    font=dict(size=13, color=plot_title_color, weight=600)
                ),
                xaxis_title='Target Time (UTC)',
                yaxis_title='Price (USD)',
                hovermode='x unified',
                height=380,
                template=plotly_template,
                paper_bgcolor=plot_panel_bg,
                plot_bgcolor=plot_panel_bg,
                font=dict(size=11, color=plot_text_color),
                margin=dict(t=45, b=35, l=60, r=40),
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor=plot_legend_bg,
                    bordercolor=plot_border,
                    borderwidth=1,
                    font=dict(color=plot_text_color, size=11)
                ),
                xaxis=dict(
                    gridcolor=plot_grid_color,
                    linecolor=plot_border,
                    tickfont=dict(color=plot_text_color),
                    title_font=dict(color=plot_text_color)
                ),
                yaxis=dict(
                    gridcolor=plot_grid_color,
                    linecolor=plot_border,
                    tickfont=dict(color=plot_text_color),
                    title_font=dict(color=plot_text_color),
                    tickformat='$,.0f',
                    range=[y_axis_min, y_axis_max]
                ),
                bargap=0.15
            )
            st.plotly_chart(fig_roll, width='stretch')
        else:
            st.info(
                "3-day rolling validation will appear once predictions have actual prices (24 hours after prediction time)."
            )

            try:
                dbg = getattr(rolling_3d_df, 'attrs', {}).get('debug') if isinstance(rolling_3d_df, pd.DataFrame) else None
            except Exception:
                dbg = None
            if dbg:
                st.caption(f"Rolling validation debug: {dbg}")

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

            if rolling_err:
                st.caption(f"Rolling validation error: {rolling_err}")

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

        # Embedded TradingView Widget (always dark to match our theme)
        tv_theme = "dark"
        tv_bg = "rgba(19, 23, 34, 1)"
        tv_grid = "rgba(42, 46, 57, 0.06)"

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
            line=dict(color=plot_line_color, width=3),
            marker=dict(size=12, color=plot_line_color, line=dict(color=plot_panel_bg, width=2)),
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
            fillcolor='rgba(99, 102, 241, 0.15)',
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
            margin=dict(t=60, b=40, l=60, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor=plot_legend_bg,
                bordercolor=plot_border,
                borderwidth=1,
                font=dict(color=plot_text_color)
            ),
            xaxis=dict(
                gridcolor=plot_grid_color,
                linecolor=plot_border,
                tickfont=dict(color=plot_text_color),
                title_font=dict(color=plot_text_color)
            ),
            yaxis=dict(
                gridcolor=plot_grid_color,
                linecolor=plot_border,
                tickfont=dict(color=plot_text_color),
                title_font=dict(color=plot_text_color),
                tickformat='$,.0f'
            )
        )
        
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Confidence Chart
        fig_conf = go.Figure()
        
        colors = [plot_positive_color if card['signal'] == 'BUY' else plot_negative_color if card['signal'] == 'SELL' else '#f59e0b' for card in prediction_cards]
        
        fig_conf.add_trace(go.Bar(
            x=[card['horizon'] for card in prediction_cards],
            y=[card['confidence'] * 100 for card in prediction_cards],
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.2)', width=0),
                opacity=0.9
            ),
            text=[f"{card['confidence']*100:.1f}%" for card in prediction_cards],
            textposition='outside',
            textfont=dict(size=12, color=plot_title_color, weight='bold'),
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
            yaxis=dict(
                range=[0, 100],
                gridcolor=plot_grid_color,
                linecolor=plot_border,
                tickfont=dict(color=plot_text_color),
                title_font=dict(color=plot_text_color)
            ),
            xaxis=dict(
                gridcolor=plot_grid_color,
                linecolor=plot_border,
                tickfont=dict(color=plot_text_color),
                title_font=dict(color=plot_text_color)
            ),
            margin=dict(t=60, b=40, l=60, r=40),
            showlegend=False
        )
        
        st.plotly_chart(fig_conf, width='stretch')
    
    # Historical Price Chart (Actual only for now)
    recent = _ensure_datetime_index(price_data).tail(168)  # last 7 days @ 1H
    if isinstance(recent.index, pd.DatetimeIndex) and len(recent) > 0 and 'close' in recent.columns:
        fig_historical = go.Figure()
        fig_historical.add_trace(go.Scatter(
            x=recent.index,
            y=recent['close'].astype(float),
            mode='lines',
            name='BTC Actual Price (Last 7D)',
            line=dict(color=plot_line_color, width=2),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.1)',
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
            margin=dict(t=60, b=40, l=60, r=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor=plot_legend_bg,
                bordercolor=plot_border,
                borderwidth=1,
                font=dict(color=plot_text_color)
            ),
            xaxis=dict(
                gridcolor=plot_grid_color,
                linecolor=plot_border,
                tickfont=dict(color=plot_text_color),
                title_font=dict(color=plot_text_color)
            ),
            yaxis=dict(
                gridcolor=plot_grid_color,
                linecolor=plot_border,
                tickfont=dict(color=plot_text_color),
                title_font=dict(color=plot_text_color),
                tickformat='$,.0f'
            )
        )

        st.plotly_chart(fig_historical, width='stretch')
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
        <div style="margin-top: 4rem; padding: 2.5rem; text-align: center; border-top: 1px solid var(--dsba-border); position: relative;">
            <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 150px; height: 3px; background: linear-gradient(90deg, transparent, var(--dsba-accent), var(--dsba-accent-2), transparent); border-radius: 2px;"></div>
            <div style="font-size: 2rem; font-weight: 900; letter-spacing: 2px; margin-bottom: 0.75rem; background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 30%, #6366f1 70%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: none;">
                POWERED BY DSBA
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500;">
                🧠 AI-Driven Bitcoin Intelligence • Real-time Market Analytics
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # (Sidebar intentionally unused; menu is on-page.)

if __name__ == "__main__":
    main()

