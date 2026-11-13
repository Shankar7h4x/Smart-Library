"""Fetch the newest public occupancy rows from the Hugging Face dataset viewer."""

from datetime import datetime
from threading import Lock, Thread
from time import monotonic
from zoneinfo import ZoneInfo

import requests
import streamlit as st
from src.data_quality import is_fresh_timestamp, valid_occupancy

from config import ALL_ZONES, ZONE_MAP

HF_DATASET = "davnas/real-time-library-occupancy"
HF_API = "https://datasets-server.huggingface.co"
KTH_TZ = ZoneInfo("Europe/Stockholm")
_status_lock = Lock()
_status_snapshot = None
_status_error = None
_status_refreshing = False
_status_checked_at = 0.0


def _fetch_latest_occupancy():
    """Return the most recent row and source timestamp, or a safe error result."""
    try:
        size_response = requests.get(
            f"{HF_API}/size", params={"dataset": HF_DATASET}, timeout=4
        )
        size_response.raise_for_status()
        size_data = size_response.json()["size"]
        total_rows = int(size_data["dataset"]["num_rows"])
        if total_rows < 1:
            return None, "The Hugging Face dataset is empty."

        row_response = requests.get(
            f"{HF_API}/rows",
            params={
                "dataset": HF_DATASET,
                "config": "default",
                "split": "train",
                "offset": max(0, total_rows - 50),
                "length": min(total_rows, 50),
            },
            timeout=5,
        )
        row_response.raise_for_status()
        rows = row_response.json().get("rows", [])
        if not rows:
            return None, "Hugging Face returned no occupancy rows."

        # Parse timestamps before comparison: ISO strings with different offsets do not
        # sort reliably as plain text.
        parsed_rows = []
        for item in rows:
            row = item.get("row", {})
            raw_time = row.get("time")
            if not raw_time:
                continue
            try:
                parsed = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=KTH_TZ)
                parsed_rows.append((parsed.astimezone(KTH_TZ), row))
            except (TypeError, ValueError):
                continue
        if not parsed_rows:
            return None, "The dataset contains no valid observation timestamps."
        observed_at, source_row = max(parsed_rows, key=lambda item: item[0])
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=KTH_TZ)
        else:
            observed_at = observed_at.astimezone(KTH_TZ)

        values = {}
        for display_name, zone in ZONE_MAP.items():
            if zone not in ALL_ZONES or display_name not in source_row:
                continue
            value = source_row[display_name]
            if not valid_occupancy(value):
                continue
            values[zone] = float(value)
        if "library" not in values:
            return None, "The latest Hugging Face row is missing the library occupancy field."

        return {"observed_at": observed_at, "values": values}, None
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        return None, f"Could not refresh archived occupancy data ({type(exc).__name__})."


@st.cache_data(ttl=300, show_spinner=False)
def fetch_latest_occupancy():
    """Return a cached snapshot for pages that explicitly need the archive."""
    return _fetch_latest_occupancy()


def _refresh_status_in_background():
    global _status_snapshot, _status_error, _status_refreshing, _status_checked_at
    try:
        snapshot, error = _fetch_latest_occupancy()
    except Exception as exc:
        snapshot, error = None, f"Could not refresh archived occupancy data ({type(exc).__name__})."
    with _status_lock:
        _status_snapshot = snapshot
        _status_error = error
        _status_checked_at = monotonic()
        _status_refreshing = False


def get_data_status_info() -> dict:
    """Return cached freshness immediately while refreshing the remote archive in the background."""
    global _status_refreshing
    now = datetime.now(KTH_TZ)
    with _status_lock:
        if not _status_refreshing and (
            _status_checked_at == 0.0 or monotonic() - _status_checked_at >= 300
        ):
            _status_refreshing = True
            Thread(target=_refresh_status_in_background, daemon=True).start()
        source_row = _status_snapshot
        source_error = _status_error
        is_refreshing = _status_refreshing

    if source_row is None:
        last_updated = "Checking archive" if is_refreshing else "Archive unavailable"
        status_text = (
            "Historical model forecast (archive refresh in background)"
            if is_refreshing else "Historical model forecast (no fresh observation)"
        )
        return {
            "last_updated": last_updated,
            "status_text": status_text,
            "kind": "offline",
            "is_live": False,
            "is_recent": False,
            "observed_at": None,
            "values": {},
            "error": source_error,
        }

    if source_row:
        observed_at = source_row["observed_at"]
        is_recent = is_fresh_timestamp(observed_at, now)
        time_str = observed_at.strftime("%d %b %Y, %H:%M")
        if is_recent:
            status_text = f"Recent archive observation ({time_str} Stockholm · not a live sensor)"
            kind = "recent"
        else:
            status_text = f"Historical archive ({time_str} Stockholm · not live)"
            kind = "stale"
        return {
            "last_updated": time_str,
            "status_text": status_text,
            "kind": kind,
            "is_live": False,  # archive provenance is not a verified live sensor
            "is_recent": is_recent,
            "observed_at": observed_at,
            "values": source_row["values"],
            "error": None,
        }
    return {
        "last_updated": "Archive unavailable",
        "status_text": "Historical model forecast (no fresh observation)",
        "kind": "offline",
        "is_live": False,
        "is_recent": False,
        "observed_at": None,
        "values": {},
        "error": source_error,
    }
