"""Supabase (PostgreSQL) logging for BTC predictions.

Simple, reliable cloud persistence. No OAuth complexity.

Setup:
1. Create free account at https://supabase.com
2. Create a project
3. Go to Settings → Database → Connection string (URI)
4. Add to Streamlit secrets:
   SUPABASE_DB_URL = "postgresql://postgres:PASSWORD@db.xxx.supabase.co:5432/postgres"
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

# Will be imported lazily
_engine = None
_tables_created = False


def _get_db_url() -> str | None:
    """Get database URL from env or Streamlit secrets."""
    url = os.getenv("SUPABASE_DB_URL")
    if url:
        return url.strip()
    
    try:
        import streamlit as st
        if "SUPABASE_DB_URL" in st.secrets:
            return str(st.secrets["SUPABASE_DB_URL"]).strip()
    except Exception:
        pass
    
    return None


def _get_engine():
    """Get or create SQLAlchemy engine."""
    global _engine
    if _engine is not None:
        return _engine
    
    url = _get_db_url()
    if not url:
        return None
    
    try:
        from sqlalchemy import create_engine
        _engine = create_engine(url, pool_pre_ping=True, pool_size=2)
        print("[SUPABASE] Database connection established")
        return _engine
    except Exception as e:
        print(f"[SUPABASE] Connection failed: {e}")
        return None


def _ensure_tables():
    """Create tables if they don't exist."""
    global _tables_created
    if _tables_created:
        return True
    
    engine = _get_engine()
    if not engine:
        return False
    
    try:
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Validation table (24H predictions vs actuals)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS validation_24h (
                    id SERIAL PRIMARY KEY,
                    made_at TIMESTAMP WITH TIME ZONE UNIQUE,
                    target_at TIMESTAMP WITH TIME ZONE,
                    predicted_24h NUMERIC,
                    actual_24h NUMERIC,
                    actual_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """))
            
            # Prediction log table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prediction_log (
                    id SERIAL PRIMARY KEY,
                    pk VARCHAR(100) UNIQUE,
                    created_at TIMESTAMP WITH TIME ZONE,
                    target_at TIMESTAMP WITH TIME ZONE,
                    horizon_label VARCHAR(20),
                    horizon_hours INTEGER,
                    current_price NUMERIC,
                    predicted_price NUMERIC
                )
            """))
            
            conn.commit()
        
        _tables_created = True
        print("[SUPABASE] Tables ready")
        return True
    except Exception as e:
        print(f"[SUPABASE] Table creation failed: {e}")
        return False


def sync_validation_24h_records(records: list[dict[str, Any]]) -> None:
    """Sync validation records to Supabase (best-effort)."""
    try:
        if not _get_db_url():
            return
        
        if not _ensure_tables():
            return
        
        engine = _get_engine()
        if not engine:
            return
        
        from sqlalchemy import text
        
        inserted = 0
        updated = 0
        
        with engine.connect() as conn:
            for rec in records:
                made_at = rec.get("made_at")
                if not made_at:
                    continue
                
                # Try insert, on conflict update
                result = conn.execute(text("""
                    INSERT INTO validation_24h (made_at, target_at, predicted_24h, actual_24h, actual_at)
                    VALUES (:made_at, :target_at, :predicted_24h, :actual_24h, :actual_at)
                    ON CONFLICT (made_at) DO UPDATE SET
                        actual_24h = COALESCE(EXCLUDED.actual_24h, validation_24h.actual_24h),
                        actual_at = COALESCE(EXCLUDED.actual_at, validation_24h.actual_at)
                    RETURNING (xmax = 0) AS inserted
                """), {
                    "made_at": made_at,
                    "target_at": rec.get("target_at"),
                    "predicted_24h": rec.get("predicted_24h"),
                    "actual_24h": rec.get("actual_24h"),
                    "actual_at": rec.get("actual_at"),
                })
                
                row = result.fetchone()
                if row and row[0]:
                    inserted += 1
                else:
                    updated += 1
            
            conn.commit()
        
        if inserted or updated:
            print(f"[SUPABASE] validation: inserted={inserted}, updated={updated}")
    
    except Exception as e:
        print(f"[SUPABASE] validation sync error: {e}")


def sync_prediction_log_records(records: list[dict[str, Any]]) -> None:
    """Sync prediction log records to Supabase (best-effort)."""
    try:
        if not _get_db_url():
            return
        
        if not _ensure_tables():
            return
        
        engine = _get_engine()
        if not engine:
            return
        
        from sqlalchemy import text
        
        inserted = 0
        
        with engine.connect() as conn:
            for rec in records:
                pk = rec.get("pk")
                if not pk:
                    continue
                
                result = conn.execute(text("""
                    INSERT INTO prediction_log (pk, created_at, target_at, horizon_label, horizon_hours, current_price, predicted_price)
                    VALUES (:pk, :created_at, :target_at, :horizon_label, :horizon_hours, :current_price, :predicted_price)
                    ON CONFLICT (pk) DO NOTHING
                    RETURNING id
                """), {
                    "pk": pk,
                    "created_at": rec.get("created_at"),
                    "target_at": rec.get("target_at"),
                    "horizon_label": rec.get("horizon_label"),
                    "horizon_hours": rec.get("horizon_hours"),
                    "current_price": rec.get("current_price"),
                    "predicted_price": rec.get("predicted_price"),
                })
                
                if result.fetchone():
                    inserted += 1
            
            conn.commit()
        
        if inserted:
            print(f"[SUPABASE] predictions: inserted={inserted}")
    
    except Exception as e:
        print(f"[SUPABASE] prediction sync error: {e}")


def get_validation_history(days: int = 30) -> list[dict]:
    """Fetch recent validation records."""
    try:
        engine = _get_engine()
        if not engine or not _ensure_tables():
            return []
        
        from sqlalchemy import text
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT made_at, target_at, predicted_24h, actual_24h, actual_at
                FROM validation_24h
                WHERE made_at > NOW() - INTERVAL ':days days'
                ORDER BY made_at DESC
            """), {"days": days})
            
            return [dict(row._mapping) for row in result]
    
    except Exception as e:
        print(f"[SUPABASE] fetch error: {e}")
        return []
