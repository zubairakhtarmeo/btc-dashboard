"""Google Sheets logging helpers (service-account based).

Design goals:
- Best-effort: never crash the app if Sheets is misconfigured/unavailable.
- Minimal writes: append new rows; update existing rows by primary key.
- No UI changes: dashboard continues to log locally; Sheets is optional.

Configuration (environment variables):
- GSHEETS_ENABLED: '1'/'true' to enable
- GSHEETS_SPREADSHEET_ID: Spreadsheet key (from the URL)
- GSHEETS_SERVICE_ACCOUNT_JSON: Absolute path to service-account JSON

Optional:
- GSHEETS_VALIDATION_TAB: worksheet name for 24H validation (default: 'validation_24h')
- GSHEETS_PREDICTION_TAB: worksheet name for prediction log (default: 'prediction_log')
- GSHEETS_SYNC_PREDICTIONS: '1'/'true' to also sync prediction log
"""

from __future__ import annotations

import logging
import os
import json
import re
from collections.abc import Mapping
from typing import Any, Iterable

_logger = logging.getLogger(__name__)
_warned: set[str] = set()
_connected_printed = False


def _warn_once(key: str, message: str) -> None:
    """Best-effort logging to Streamlit Cloud logs without spamming."""
    global _warned
    if key in _warned:
        return
    _warned.add(key)
    try:
        print(message)
    except Exception:
        pass
    try:
        _logger.warning(message)
    except Exception:
        pass


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from env first, then Streamlit secrets (if available)."""
    value = _get_env(name)
    if value is not None:
        return value

    # Streamlit Community Cloud prefers secrets instead of env vars/files.
    try:
        import streamlit as st  # type: ignore

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

    # If user pasted the full URL, extract the key between /d/ and the next slash.
    # Example: https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", s)
    if m:
        return m.group(1)

    # Sometimes users paste with quotes; strip again.
    s = s.strip().strip('"').strip("'").strip()
    return s or None


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
    # Lazy import so the dashboard can run without these deps.
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
    ]

    # Option A (local/dev): path to service-account JSON file
    creds_path = _get_setting("GSHEETS_SERVICE_ACCOUNT_JSON")
    if creds_path and os.path.exists(creds_path):
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        return gspread.authorize(creds)

    # Option B (Streamlit Cloud): service-account JSON content or dict via secrets
    info_raw = _get_setting("GSHEETS_SERVICE_ACCOUNT_INFO")
    if info_raw:
        try:
            info = json.loads(info_raw)
        except Exception:
            info = None
        if isinstance(info, dict):
            creds = Credentials.from_service_account_info(info, scopes=scopes)
            return gspread.authorize(creds)

    # Option C (Streamlit Cloud conventional name): [gcp_service_account] in secrets
    try:
        import streamlit as st  # type: ignore

        gcp_info = st.secrets.get("gcp_service_account")
        # Streamlit returns an AttrDict-like mapping, not always a plain dict.
        if isinstance(gcp_info, Mapping):
            creds = Credentials.from_service_account_info(dict(gcp_info), scopes=scopes)
            return gspread.authorize(creds)
    except Exception:
        pass

    raise RuntimeError(
        "Google Sheets credentials missing. Set GSHEETS_SERVICE_ACCOUNT_JSON (path) or GSHEETS_SERVICE_ACCOUNT_INFO (JSON) or provide [gcp_service_account] in Streamlit secrets."
    )


def _open_spreadsheet(spreadsheet_id: str):
    client = _get_client()
    return client.open_by_key(spreadsheet_id)


def _ensure_worksheet(spreadsheet, title: str, headers: list[str]):
    try:
        ws = spreadsheet.worksheet(title)
    except Exception:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=max(12, len(headers) + 2))

    # Ensure header row
    try:
        existing = ws.row_values(1)
        if [h.strip() for h in existing] != headers:
            ws.update(f"A1:{_a1_col(len(headers))}1", [headers])
    except Exception:
        ws.update(f"A1:{_a1_col(len(headers))}1", [headers])

    return ws


def _upsert_by_key(
    ws,
    *,
    headers: list[str],
    key_field: str,
    records: Iterable[dict[str, Any]],
) -> tuple[int, int]:
    """Upsert records into worksheet using key_field as primary key.

    Returns (appended_count, updated_count).
    """

    if key_field not in headers:
        raise ValueError(f"key_field {key_field!r} must be included in headers")

    key_col_idx = headers.index(key_field) + 1

    # Read existing keys (small sheets are expected; safe and simple)
    existing_keys = ws.col_values(key_col_idx)
    key_to_row: dict[str, int] = {}

    # col_values includes header at row 1
    for row_num, key in enumerate(existing_keys, start=1):
        if row_num == 1:
            continue
        if key is None:
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
            start = "A" + str(row_num)
            end = _a1_col(len(headers)) + str(row_num)
            updates.append({"range": f"{start}:{end}", "values": [row]})
        else:
            appends.append(row)

    appended_count = 0
    updated_count = 0

    if updates:
        # gspread expects the "data" key for batch_update
        ws.batch_update(updates)
        updated_count = len(updates)

    if appends:
        ws.append_rows(appends, value_input_option="RAW")
        appended_count = len(appends)

    return appended_count, updated_count


def sync_validation_24h_records(records: list[dict[str, Any]]) -> None:
    """Sync validation_24h.json-style records into Google Sheets (best-effort)."""
    try:
        if not _is_truthy(_get_setting("GSHEETS_ENABLED", "")):
            return

        spreadsheet_id = _normalize_spreadsheet_id(_get_setting("GSHEETS_SPREADSHEET_ID"))
        if not spreadsheet_id:
            _warn_once("missing_spreadsheet_id", "[GSHEETS] Enabled but GSHEETS_SPREADSHEET_ID is missing")
            return

        tab = _get_setting("GSHEETS_VALIDATION_TAB", "validation_24h") or "validation_24h"

        headers = [
            "made_at",
            "target_at",
            "predicted_24h",
            "actual_24h",
            "actual_at",
        ]

        spreadsheet = _open_spreadsheet(spreadsheet_id)
        ws = _ensure_worksheet(spreadsheet, tab, headers)
        appended, updated = _upsert_by_key(ws, headers=headers, key_field="made_at", records=records)

        global _connected_printed
        if not _connected_printed:
            _warn_once("connected", f"[GSHEETS] Connected OK. Writing to tab '{tab}'.")
            _connected_printed = True

        # Only log writes when something actually changed.
        if appended or updated:
            _warn_once(f"validation_write_{tab}", f"[GSHEETS] validation sync wrote rows (appended={appended}, updated={updated})")
    except Exception as e:
        # Best-effort; never block the dashboard, but DO log once so misconfig is visible.
        if type(e).__name__ == "SpreadsheetNotFound":
            _warn_once(
                "spreadsheet_not_found_hint",
                "[GSHEETS] SpreadsheetNotFound (404). This usually means the Spreadsheet ID is wrong OR the sheet is not shared with the service account client_email (Google returns 404 when unauthorized).",
            )
        _warn_once("validation_error", f"[GSHEETS] validation sync failed: {type(e).__name__}: {e}")
        return


def sync_prediction_log_records(records: list[dict[str, Any]]) -> None:
    """Sync prediction_log.json-style records into Google Sheets (best-effort)."""
    try:
        if not _is_truthy(_get_setting("GSHEETS_ENABLED", "")):
            return
        if not _is_truthy(_get_setting("GSHEETS_SYNC_PREDICTIONS", "")):
            return

        spreadsheet_id = _normalize_spreadsheet_id(_get_setting("GSHEETS_SPREADSHEET_ID"))
        if not spreadsheet_id:
            _warn_once("missing_spreadsheet_id", "[GSHEETS] Enabled but GSHEETS_SPREADSHEET_ID is missing")
            return

        tab = _get_setting("GSHEETS_PREDICTION_TAB", "prediction_log") or "prediction_log"

        headers = [
            "pk",
            "created_at",
            "target_at",
            "horizon_label",
            "horizon_hours",
            "current_price",
            "predicted_price",
        ]

        spreadsheet = _open_spreadsheet(spreadsheet_id)
        ws = _ensure_worksheet(spreadsheet, tab, headers)

        appended, updated = _upsert_by_key(ws, headers=headers, key_field="pk", records=records)
        if appended or updated:
            _warn_once(f"pred_write_{tab}", f"[GSHEETS] prediction sync wrote rows (appended={appended}, updated={updated})")
    except Exception as e:
        if type(e).__name__ == "SpreadsheetNotFound":
            _warn_once(
                "spreadsheet_not_found_hint",
                "[GSHEETS] SpreadsheetNotFound (404). This usually means the Spreadsheet ID is wrong OR the sheet is not shared with the service account client_email (Google returns 404 when unauthorized).",
            )
        _warn_once("prediction_error", f"[GSHEETS] prediction sync failed: {type(e).__name__}: {e}")
        return
