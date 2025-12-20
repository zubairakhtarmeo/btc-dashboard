"""Google Sheets logging helpers (service-account based).

Design goals:
- Best-effort: never crash the app if Sheets is misconfigured/unavailable.
- Minimal writes: append new rows; update existing rows by primary key.
- No UI changes: dashboard continues to log locally; Sheets is optional.

Configuration (Streamlit secrets):
- GSHEETS_ENABLED: '1'/'true' to enable
- GSHEETS_SPREADSHEET_ID: Spreadsheet key (from the URL)
- [gcp_service_account]: Service account credentials section
"""

from __future__ import annotations

import logging
import os
import json
import re
from collections.abc import Mapping
from typing import Any, Iterable

_logger = logging.getLogger(__name__)
_client_email_for_logs: str | None = None


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from env first, then Streamlit secrets."""
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    try:
        import streamlit as st
        if name in st.secrets:
            v = st.secrets.get(name)
            if v is None:
                return default
            if isinstance(v, (dict, list)):
                return json.dumps(v)
            v_str = str(v).strip()
            return v_str if v_str else default
    except Exception:
        pass

    return default


def _normalize_spreadsheet_id(raw: str | None) -> str | None:
    """Accept either a spreadsheet ID or a full Google Sheets URL."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)

    return s.strip().strip('"').strip("'").strip() or None


def _a1_col(n: int) -> str:
    """1-indexed column number to Excel-style column letters."""
    letters: list[str] = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(65 + rem))
    return "".join(reversed(letters))


def _as_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _get_client():
    global _client_email_for_logs
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    # Try Streamlit secrets [gcp_service_account]
    try:
        import streamlit as st
        gcp_info = st.secrets.get("gcp_service_account")
        if isinstance(gcp_info, Mapping):
            gcp_dict = dict(gcp_info)
            creds = Credentials.from_service_account_info(gcp_dict, scopes=scopes)
            _client_email_for_logs = gcp_dict.get('client_email')
            return gspread.authorize(creds)
    except Exception:
        pass

    raise RuntimeError("Google Sheets credentials missing. Provide [gcp_service_account] in Streamlit secrets.")


def _open_spreadsheet(spreadsheet_id: str):
    client = _get_client()
    return client.open_by_key(spreadsheet_id)


def _ensure_worksheet(spreadsheet, title: str, headers: list[str]):
    try:
        ws = spreadsheet.worksheet(title)
    except Exception:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=max(12, len(headers) + 2))

    try:
        existing = ws.row_values(1)
        if [h.strip() for h in existing] != headers:
            ws.update(f"A1:{_a1_col(len(headers))}1", [headers])
    except Exception:
        ws.update(f"A1:{_a1_col(len(headers))}1", [headers])

    return ws


def _upsert_by_key(ws, *, headers: list[str], key_field: str, records: Iterable[dict[str, Any]]) -> tuple[int, int]:
    """Upsert records into worksheet using key_field as primary key."""
    if key_field not in headers:
        raise ValueError(f"key_field {key_field!r} must be in headers")

    key_col_idx = headers.index(key_field) + 1
    existing_keys = ws.col_values(key_col_idx)
    key_to_row: dict[str, int] = {}

    for row_num, key in enumerate(existing_keys, start=1):
        if row_num == 1 or key is None:
            continue
        key_str = str(key).strip()
        if key_str:
            key_to_row[key_str] = row_num

    appends: list[list[str]] = []
    updates: list[dict[str, Any]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        key_val = rec.get(key_field)
        key_str = str(key_val).strip() if key_val is not None else ""
        if not key_str:
            continue

        row = [_as_cell(rec.get(h)) for h in headers]

        if key_str in key_to_row:
            row_num = key_to_row[key_str]
            updates.append({"range": f"A{row_num}:{_a1_col(len(headers))}{row_num}", "values": [row]})
        else:
            appends.append(row)

    if updates:
        ws.batch_update(updates)
    if appends:
        ws.append_rows(appends, value_input_option="RAW")

    return len(appends), len(updates)


def sync_validation_24h_records(records: list[dict[str, Any]]) -> None:
    """Sync validation_24h.json-style records into Google Sheets (best-effort)."""
    try:
        print(f"[GSHEETS] sync called with {len(records)} records")

        enabled_val = _get_setting("GSHEETS_ENABLED", "")
        print(f"[GSHEETS] GSHEETS_ENABLED = {enabled_val!r}")
        if not _is_truthy(enabled_val):
            print("[GSHEETS] DISABLED - set GSHEETS_ENABLED = 'true' in secrets")
            return

        raw_id = _get_setting("GSHEETS_SPREADSHEET_ID")
        print(f"[GSHEETS] GSHEETS_SPREADSHEET_ID = {raw_id!r}")
        spreadsheet_id = _normalize_spreadsheet_id(raw_id)
        if not spreadsheet_id:
            print("[GSHEETS] ERROR: spreadsheet_id is empty")
            return

        headers = ["made_at", "target_at", "predicted_24h", "actual_24h", "actual_at"]

        print(f"[GSHEETS] Opening spreadsheet...")
        spreadsheet = _open_spreadsheet(spreadsheet_id)
        print(f"[GSHEETS] Opened! Getting worksheet...")
        ws = _ensure_worksheet(spreadsheet, "validation_24h", headers)
        print(f"[GSHEETS] Upserting records...")
        appended, updated = _upsert_by_key(ws, headers=headers, key_field="made_at", records=records)

        email = _client_email_for_logs or "(unknown)"
        print(f"[GSHEETS] SUCCESS! appended={appended}, updated={updated}, account={email}")
    except Exception as e:
        print(f"[GSHEETS] ERROR: {type(e).__name__}: {e}")


def sync_prediction_log_records(records: list[dict[str, Any]]) -> None:
    """Sync prediction_log records (best-effort)."""
    try:
        if not _is_truthy(_get_setting("GSHEETS_ENABLED", "")):
            return
        if not _is_truthy(_get_setting("GSHEETS_SYNC_PREDICTIONS", "")):
            return

        spreadsheet_id = _normalize_spreadsheet_id(_get_setting("GSHEETS_SPREADSHEET_ID"))
        if not spreadsheet_id:
            return

        headers = ["pk", "created_at", "target_at", "horizon_label", "horizon_hours", "current_price", "predicted_price"]

        spreadsheet = _open_spreadsheet(spreadsheet_id)
        ws = _ensure_worksheet(spreadsheet, "prediction_log", headers)
        appended, updated = _upsert_by_key(ws, headers=headers, key_field="pk", records=records)
        if appended or updated:
            print(f"[GSHEETS] predictions: appended={appended}, updated={updated}")
    except Exception as e:
        print(f"[GSHEETS] prediction sync error: {type(e).__name__}: {e}")
