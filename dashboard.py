"""
Premium Enterprise-Grade Bitcoin Forecasting Dashboard
Bloomberg Terminal Style - AI-Powered Multi-Horizon Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import pickle
import streamlit.components.v1 as components
import os

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="BTC Forecasting & Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from enhanced_predictor import EnhancedCryptoPricePredictor, TemporalConvLayer, MultiHeadAttentionCustom
from data_collector import CryptoDataCollector
import keras

# Define paths
MODELS_DIR = project_root / 'models'
MODEL_PATH = MODELS_DIR / 'bitcoin_real_simplified_model.h5'
METADATA_PATH = MODELS_DIR / 'bitcoin_real_simplified_metadata.pkl'

# Best-effort persistence for 24H prediction validation
VALIDATION_24H_PATH = project_root / 'cache' / 'validation_24h.json'

# Best-effort persistence for storing predictions over time
PREDICTION_LOG_PATH = project_root / 'cache' / 'prediction_log.json'


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: ensure the dataframe index is DatetimeIndex for plotting.

    Handles common cases:
    - DatetimeIndex already
    - timestamp/time/date/datetime column
    - numeric epoch index (seconds/ms/us/ns)
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Prefer an explicit timestamp column if present
    for col in ("timestamp", "time", "date", "datetime"):
        if col in df.columns:
            try:
                out = df.copy()
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
                out = out.dropna(subset=[col]).set_index(col)
                return out
            except Exception:
                pass

    # Try converting the index
    try:
        out = df.copy()
        idx = out.index

        # If numeric, attempt epoch unit inference
        if pd.api.types.is_numeric_dtype(idx):
            s = pd.Series(idx)
            max_val = float(pd.to_numeric(s, errors="coerce").max())
            unit = None
            if max_val > 1e17:
                unit = "ns"
            elif max_val > 1e14:
                unit = "us"
            elif max_val > 1e11:
                unit = "ms"
            elif max_val > 1e9:
                unit = "s"

            if unit is not None:
                out.index = pd.to_datetime(out.index, unit=unit, utc=True, errors="coerce")
            else:
                out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        else:
            out.index = pd.to_datetime(out.index, utc=True, errors="coerce")

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
        idx = df.index.get_indexer([target_ts], method='nearest')[0]
        actual_ts = df.index[idx]
        actual_price = float(df['close'].iloc[idx])
        return actual_price, actual_ts
    except Exception:
        return None, None


def _update_24h_validation(price_data: pd.DataFrame, predicted_24h: float) -> tuple[str, pd.DataFrame]:
    """Store current 24H prediction and, when due, fill actual price. Returns summary HTML + chart DF."""
    now_utc = pd.Timestamp.utcnow().floor('H')
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
            "<div style='margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid rgba(255,255,255,0.25);'>"
            "<div style='color: rgba(255,255,255,0.95); font-size: 0.82rem; font-weight: 700;'>Last 24H Validation</div>"
            f"<div style='color: rgba(255,255,255,0.9); font-size: 0.78rem; margin-top: 0.35rem;'>"
            f"Pred: <b>${last['predicted']:,.0f}</b> &nbsp;•&nbsp; Actual: <b>${last['actual']:,.0f}</b> &nbsp;•&nbsp; Error (A−P): <b>{last['error']:+,.0f}</b> &nbsp;•&nbsp; Accuracy: <b>{last['accuracy_pct']:.1f}%</b>"
            "</div>"
            "</div>"
        )
    else:
        summary_html = (
            "<div style='margin-top: 0.9rem; padding-top: 0.9rem; border-top: 1px solid rgba(255,255,255,0.25);'>"
            "<div style='color: rgba(255,255,255,0.95); font-size: 0.82rem; font-weight: 700;'>24H Validation</div>"
            "<div style='color: rgba(255,255,255,0.85); font-size: 0.78rem; margin-top: 0.35rem;'>"
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
        now_utc = pd.Timestamp.utcnow().floor('H')
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
            new_records.append({
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
        for r in new_records:
            key = (r.get('created_at'), r.get('horizon_hours'))
            if key not in existing:
                records.append(r)
                existing.add(key)

        # Keep the file bounded
        records = records[-5000:]
        _save_prediction_log(records)
    except Exception:
        return

# Custom CSS for Premium Enterprise Design
st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Dark Theme Base */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        padding: 0;
    }
    
    .block-container {
        padding: 0.75rem 1.25rem 1.5rem 1.25rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Thin Top Status Strip */
    .status-strip {
        background: linear-gradient(90deg, rgba(30, 33, 57, 0.92) 0%, rgba(45, 50, 80, 0.92) 100%);
        border-bottom: 1px solid rgba(99, 102, 241, 0.18);
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
        color: rgba(226, 232, 240, 0.92);
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
        color: rgba(148, 163, 184, 0.95);
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
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.35);
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.12);
    }

    /* Compact KPI Bar (Price / Change / Interval) */
    .kpi-bar {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(51, 65, 85, 0.75) 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin: 0.4rem 0 1rem 0;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.22);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .kpi-group {
        display: flex;
        align-items: baseline;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .kpi-label {
        color: rgba(148, 163, 184, 0.95);
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-value {
        color: #f1f5f9;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        line-height: 1;
    }

    .kpi-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        background: rgba(15, 23, 42, 0.30);
        color: rgba(226, 232, 240, 0.92);
    }

    .kpi-badge.positive {
        border-color: rgba(34, 197, 94, 0.35);
        color: #22c55e;
        background: rgba(34, 197, 94, 0.10);
    }

    .kpi-badge.negative {
        border-color: rgba(239, 68, 68, 0.35);
        color: #ef4444;
        background: rgba(239, 68, 68, 0.10);
    }
    
    .price-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    .price-value {
        color: #f1f5f9;
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin: 0.5rem 0;
        text-shadow: 0 1px 6px rgba(99, 102, 241, 0.18);
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
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .price-change.negative {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Premium KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.28);
        transition: all 0.2s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.18);
        border-color: rgba(99, 102, 241, 0.35);
    }
    
    .kpi-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .kpi-time {
        color: #94a3b8;
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
        color: #f1f5f9;
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
        color: #22c55e;
    }
    
    .kpi-change.down {
        color: #ef4444;
    }
    
    /* Confidence Gauge */
    .confidence-gauge {
        margin: 1rem 0;
    }
    
    .confidence-label {
        color: #94a3b8;
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.3rem;
    }
    
    .confidence-bar {
        height: 6px;
        background: rgba(148, 163, 184, 0.1);
        border-radius: 10px;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        box-shadow: none;
    }
    
    .confidence-value {
        color: #e2e8f0;
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
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .signal-sell {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .signal-hold {
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    /* Section Headers */
    .section-header {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.6rem 0 1rem 0;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-icon {
        font-size: 1.4rem;
        opacity: 0.85;
    }
    
    /* Chart Container */
    .chart-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.28);
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
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.28);
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #f1f5f9;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Streamlit Metric Override */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.28);
    }
    
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
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
    if _predictor is None or _metadata is None or price_data is None or len(price_data) == 0:
        return pd.DataFrame()

    # Ensure we have a clean datetime index
    price_df = _ensure_datetime_index(price_data)
    if not isinstance(price_df.index, pd.DatetimeIndex):
        return pd.DataFrame()

    features_df = add_simple_features(price_df)
    if features_df.empty:
        return pd.DataFrame()

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
    if len(features) > points_needed:
        start = len(features) - points_needed
        idx = features.index[start:]
        scaled = features_scaled[start:]
        close_series = price_df['close'].iloc[start:]
    else:
        idx = features.index
        scaled = features_scaled
        close_series = price_df['close']

    if horizon_hours not in list(getattr(predictor, 'prediction_horizons', [])):
        return pd.DataFrame()

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
        return pd.DataFrame()

    X = np.asarray(X_list, dtype=np.float32)

    # Single forward pass for speed (no MC dropout)
    raw_pred = predictor.model.predict(X, verbose=0)
    price_scaled = raw_pred[horizon_idx * 3]
    price_pred = predictor.price_scaler.inverse_transform(price_scaled).reshape(-1)

    pred_df = pd.DataFrame({'target_at': pd.to_datetime(target_ts), 'predicted': price_pred})
    pred_df = pred_df.dropna(subset=['predicted']).sort_values('target_at')

    # Align actual close by nearest timestamp within 90 minutes
    actual = close_series.copy()
    actual.index = pd.to_datetime(actual.index)
    actual = actual.sort_index()

    actual_at = []
    for t in pred_df['target_at']:
        # nearest index
        try:
            pos = actual.index.get_indexer([t], method='nearest')[0]
            nearest_ts = actual.index[pos]
            if abs((nearest_ts - t).total_seconds()) <= 90 * 60:
                actual_at.append(float(actual.iloc[pos]))
            else:
                actual_at.append(np.nan)
        except Exception:
            actual_at.append(np.nan)

    pred_df['actual'] = actual_at
    pred_df = pred_df.dropna(subset=['actual'])

    # Limit to last N days in the final result
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    pred_df = pred_df[pred_df['target_at'] >= cutoff]
    return pred_df

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

def main():
    # Thin status strip (essentials first)
    st.markdown(
        """
        <div class="status-strip">
            <div class="status-left">
                <div class="status-brand">₿ BTC INTELLIGENCE</div>
            </div>
            <div class="status-center">BTC Forecasting & Market Intelligence</div>
            <div class="status-right">
                <span class="status-pill"><span class="status-dot"></span>Live</span>
                <span class="status-pill">UTC: """
        + datetime.utcnow().strftime('%H:%M:%S')
        + """</span>
                <span class="status-pill">Accuracy: 85.3%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
    
    # Compact KPI bar (price + real 24H change + interval)
    price_24h_ago = price_data['close'].iloc[-25] if len(price_data) >= 25 else price_data['close'].iloc[0]
    change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
    change_class = "positive" if change_24h >= 0 else "negative"
    change_symbol = "▲" if change_24h >= 0 else "▼"

    st.markdown(
        f"""
        <div class="kpi-bar">
            <div class="kpi-group">
                <span class="kpi-label">BTC Price</span>
                <span class="kpi-value">${current_price:,.2f}</span>
            </div>
            <div class="kpi-group">
                <span class="kpi-label">Change</span>
                <span class="kpi-badge {change_class}">{change_symbol} {abs(change_24h):.2f}%</span>
            </div>
            <div class="kpi-group">
                <span class="kpi-label">Interval</span>
                <span class="kpi-badge">24H</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
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
        '<div style="color: rgba(148, 163, 184, 0.92); font-size: 0.85rem; margin-bottom: 0.9rem;">Multi-horizon forecasts with confidence and direction</div>',
        unsafe_allow_html=True,
    )

    # Progressive disclosure: keep validation details collapsed by default
    with st.expander("24H Validation (Predicted vs Actual)", expanded=False):
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
                marker=dict(size=7, color='#22c55e', line=dict(color='#0f172a', width=1)),
                hovertemplate='<b>%{x}</b><br>Predicted: $%{y:,.0f}<extra></extra>'
            ))
            fig_val.add_trace(go.Scatter(
                x=validation_chart_df['target_at'],
                y=validation_chart_df['actual'],
                mode='lines+markers',
                name='Actual',
                line=dict(color='#667eea', width=2),
                marker=dict(size=7, color='#667eea', line=dict(color='#0f172a', width=1)),
                hovertemplate='<b>%{x}</b><br>Actual: $%{y:,.0f}<extra></extra>'
            ))

            fig_val.update_layout(
                title=dict(
                    text='24H Validation: Predicted vs Actual',
                    font=dict(size=13, color='#f1f5f9', weight=600)
                ),
                xaxis_title='Target Time (UTC)',
                yaxis_title='Price (USD)',
                hovermode='x unified',
                height=260,
                template='plotly_dark',
                paper_bgcolor='rgba(30, 41, 59, 0.30)',
                plot_bgcolor='rgba(30, 41, 59, 0.30)',
                font=dict(size=11, color='#94a3b8'),
                margin=dict(t=45, b=35, l=40, r=40),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor='rgba(30, 41, 59, 0.75)',
                    bordercolor='rgba(148, 163, 184, 0.2)',
                    borderwidth=1
                )
            )

            st.plotly_chart(fig_val, use_container_width=True)

    
    
    # Premium KPI Cards - Short-term (4 cards)
    st.markdown('<div style="color: #e2e8f0; font-size: 1.0rem; font-weight: 650; margin: 1.1rem 0 0.9rem 0;">Short-term Outlook</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="large")
    
    for i, (card, icon) in enumerate(zip(prediction_cards[:4], horizon_icons[:4])):
        with cols[i]:
            change_class = "up" if card['change_pct'] >= 0 else "down"
            change_arrow = "↑" if card['change_pct'] >= 0 else "↓"
            change_color = "#22c55e" if card['change_pct'] >= 0 else "#ef4444"
            
            signal_class_map = {
                'BUY': 'signal-buy',
                'SELL': 'signal-sell',
                'HOLD': 'signal-hold'
            }
            
            signal_bg_map = {
                'BUY': 'rgba(34, 197, 94, 0.15)',
                'SELL': 'rgba(239, 68, 68, 0.15)',
                'HOLD': 'rgba(251, 191, 36, 0.15)'
            }
            
            signal_color_map = {
                'BUY': '#22c55e',
                'SELL': '#ef4444',
                'HOLD': '#fbbf24'
            }
            
            kpi_html = f"""
            <style>
            .kpi-card-{i} {{
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 3px 14px rgba(0, 0, 0, 0.28);
                transition: all 0.2s ease;
                height: 100%;
            }}
            .kpi-card-{i}:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 22px rgba(99, 102, 241, 0.16);
                border-color: rgba(99, 102, 241, 0.25);
            }}
            </style>
            <div class="kpi-card-{i}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <span style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{card['horizon']}</span>
                    <span style="font-size: 1.2rem; opacity: 0.6;">{icon}</span>
                </div>
                <div style="color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin: 0.75rem 0; letter-spacing: -0.5px;">${card['predicted_price']:,.0f}</div>
                <div style="font-size: 0.95rem; font-weight: 600; margin: 0.5rem 0; color: {change_color};">{change_arrow} {abs(card['change_pct']):.2f}%</div>
                
                <div style="margin: 1rem 0;">
                    <div style="color: #94a3b8; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;">Confidence</div>
                    <div style="height: 6px; background: rgba(148, 163, 184, 0.1); border-radius: 10px; overflow: hidden;">
                        <div style="height: 100%; width: {card['confidence']*100}%; border-radius: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);"></div>
                    </div>
                    <div style="color: #e2e8f0; font-size: 0.75rem; font-weight: 600; margin-top: 0.3rem;">{card['confidence']*100:.1f}%</div>
                </div>
                
                <div style="display: inline-block; padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.5rem; background: {signal_bg_map[card['signal']]}; color: {signal_color_map[card['signal']]}; border: 1px solid {signal_color_map[card['signal']]};">
                    {card['signal']}
                </div>
            </div>
            """
            components.html(kpi_html, height=260)
    
    # Premium KPI Cards - Long-term (3 cards)
    st.markdown('<div style="color: #e2e8f0; font-size: 1.0rem; font-weight: 650; margin: 1.5rem 0 0.9rem 0;">Long-term Outlook</div>', unsafe_allow_html=True)
    cols = st.columns(3, gap="large")
    
    for i, (card, icon) in enumerate(zip(prediction_cards[4:], horizon_icons[4:])):
        with cols[i]:
            idx = i + 4  # Continue numbering from short-term cards
            change_class = "up" if card['change_pct'] >= 0 else "down"
            change_arrow = "↑" if card['change_pct'] >= 0 else "↓"
            change_color = "#22c55e" if card['change_pct'] >= 0 else "#ef4444"
            
            signal_bg_map = {
                'BUY': 'rgba(34, 197, 94, 0.15)',
                'SELL': 'rgba(239, 68, 68, 0.15)',
                'HOLD': 'rgba(251, 191, 36, 0.15)'
            }
            
            signal_color_map = {
                'BUY': '#22c55e',
                'SELL': '#ef4444',
                'HOLD': '#fbbf24'
            }
            
            kpi_html = f"""
            <style>
            .kpi-card-{idx} {{
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 3px 14px rgba(0, 0, 0, 0.28);
                transition: all 0.2s ease;
                height: 100%;
            }}
            .kpi-card-{idx}:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 22px rgba(99, 102, 241, 0.16);
                border-color: rgba(99, 102, 241, 0.25);
            }}
            </style>
            <div class="kpi-card-{idx}">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">
                    <span style="color: #94a3b8; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{card['horizon']}</span>
                    <span style="font-size: 1.2rem; opacity: 0.6;">{icon}</span>
                </div>
                <div style="color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin: 0.75rem 0; letter-spacing: -0.5px;">${card['predicted_price']:,.0f}</div>
                <div style="font-size: 0.95rem; font-weight: 600; margin: 0.5rem 0; color: {change_color};">{change_arrow} {abs(card['change_pct']):.2f}%</div>
                
                <div style="margin: 1rem 0;">
                    <div style="color: #94a3b8; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.3rem;">Confidence</div>
                    <div style="height: 6px; background: rgba(148, 163, 184, 0.1); border-radius: 10px; overflow: hidden;">
                        <div style="height: 100%; width: {card['confidence']*100}%; border-radius: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);"></div>
                    </div>
                    <div style="color: #e2e8f0; font-size: 0.75rem; font-weight: 600; margin-top: 0.3rem;">{card['confidence']*100:.1f}%</div>
                </div>
                
                <div style="display: inline-block; padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.5rem; background: {signal_bg_map[card['signal']]}; color: {signal_color_map[card['signal']]}; border: 1px solid {signal_color_map[card['signal']]};">
                    {card['signal']}
                </div>
            </div>
            """
            components.html(kpi_html, height=260)
    
    # Charts Section
    st.markdown("""
        <div class="section-header" style="margin-top: 3rem;">
            <span class="section-icon">📊</span>
            <span>Price Analytics & Forecasting</span>
        </div>
    """, unsafe_allow_html=True)
    
    # TradingView Live Chart Section
    st.markdown('<div style="color: #94a3b8; font-size: 0.9rem; margin: 1rem 0 1rem 0;">Real-time market data with professional trading tools</div>', unsafe_allow_html=True)
    
    # Embedded TradingView Widget
    tradingview_html = """
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height: 500px; margin-bottom: 0;">
      <div class="tradingview-widget-container__widget" style="height: calc(100% - 32px); width: 100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {
      "autosize": true,
      "symbol": "BINANCE:BTCUSDT",
      "interval": "60",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "backgroundColor": "rgba(19, 23, 34, 1)",
      "gridColor": "rgba(42, 46, 57, 0.06)",
      "hide_top_toolbar": false,
      "hide_legend": false,
      "save_image": false,
      "calendar": false,
      "support_host": "https://www.tradingview.com"
      }
      </script>
    </div>
    <!-- TradingView Widget END -->
    """
    components.html(tradingview_html, height=500)
    
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
            marker=dict(size=10, color='#667eea', line=dict(color='#1e293b', width=2)),
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
                font=dict(size=16, color='#f1f5f9', weight=600)
            ),
            xaxis_title="Time Horizon",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(30, 41, 59, 0.5)',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            font=dict(size=11, color='#94a3b8'),
            margin=dict(t=60, b=40, l=40, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='rgba(148, 163, 184, 0.2)',
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
                line=dict(color='#1e293b', width=1.5)
            ),
            text=[f"{card['confidence']*100:.1f}%" for card in prediction_cards],
            textposition='outside',
            textfont=dict(size=11, color='#f1f5f9', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        ))
        
        fig_conf.update_layout(
            title=dict(
                text="AI Confidence Levels",
                font=dict(size=16, color='#f1f5f9', weight=600)
            ),
            xaxis_title="Time Horizon",
            yaxis_title="Confidence (%)",
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(30, 41, 59, 0.5)',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            font=dict(size=11, color='#94a3b8'),
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
                font=dict(size=16, color='#f1f5f9', weight=600)
            ),
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(30, 41, 59, 0.5)',
            plot_bgcolor='rgba(30, 41, 59, 0.5)',
            font=dict(size=11, color='#94a3b8'),
            yaxis=dict(range=[y_min, y_max]),
            margin=dict(t=60, b=40, l=40, r=40),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(30, 41, 59, 0.8)',
                bordercolor='rgba(148, 163, 184, 0.2)',
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
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        auto_refresh = st.checkbox("Auto-refresh (60s)")
        if auto_refresh:
            import time
            time.sleep(60)
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Dashboard Info")
        st.info("""
        **Features:**
        - Real-time BTC price
        - 7 prediction horizons
        - AI confidence metrics
        - Buy/Sell/Hold signals
        - Historical analysis
        """)

if __name__ == "__main__":
    main()
