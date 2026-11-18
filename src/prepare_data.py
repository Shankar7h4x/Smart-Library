"""Prepare raw library occupancy data into hourly resampled features."""

import json
import shutil
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ALL_ZONES,
    EXAM_PERIODS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    ZONE_MAP,
)


def is_in_exam_period(dt: pd.Timestamp) -> int:
    """Check if a timestamp falls within predefined Swedish university exam periods."""
    m, d = dt.month, dt.day
    for (sm, sd), (em, ed) in EXAM_PERIODS:
        if sm == em:
            if m == sm and sd <= d <= ed:
                return 1
        elif sm < em:
            if (m == sm and d >= sd) or (m == em and d <= ed) or (sm < m < em):
                return 1
    return 0


def prepare_data():
    """Load, clean, resample, feature engineer, and save library occupancy dataset."""

    # ── A. Load & validate raw data ──────────────────────────────────────────
    print(f"1. Loading raw dataset from {RAW_DATA_PATH} ...")
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: Raw data file not found at {RAW_DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    df_raw = pd.read_parquet(RAW_DATA_PATH)
    print(f"   Raw records: {len(df_raw)}")

    # Parse 'time' as datetime
    df = df_raw.copy()
    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    invalid_ts = int(parsed_time.isna().sum())
    if invalid_ts:
        print(f"   Warning: dropping {invalid_ts} rows with invalid timestamps.")
    df = df.loc[parsed_time.notna()].copy()
    parsed_time = parsed_time.loc[parsed_time.notna()]

    # Localize to Europe/Stockholm
    if getattr(parsed_time.dt, "tz", None) is None:
        parsed_time = parsed_time.dt.tz_localize(
            "Europe/Stockholm", ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        parsed_time = parsed_time.dt.tz_convert("Europe/Stockholm")

    df["time"] = parsed_time
    df = df.dropna(subset=["time"])

    # Rename zone columns
    df = df.rename(columns=ZONE_MAP)
    valid_zones = [z for z in ALL_ZONES if z in df.columns]
    missing_zones = sorted(set(ALL_ZONES) - set(valid_zones))
    if missing_zones:
        raise ValueError(f"Missing required occupancy columns: {', '.join(missing_zones)}")

    # Ensure numeric
    for z in valid_zones:
        df[z] = pd.to_numeric(df[z], errors="coerce")

    df = df.set_index("time")

    # Remove duplicates (keep last), sort chronologically
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df_sorted = df  # reference for quality report
    removed = len(df_raw) - len(df)
    print(f"   After dedup/sort: {len(df)} records ({removed} removed)")

    # ── B. Resample to 1H ────────────────────────────────────────────────────
    print("2. Resampling to 1H ...")
    hourly = df[valid_zones].resample("1h").mean()
    if hourly.empty:
        raise ValueError("No valid timestamps after resampling.")

    # is_open: library mean > 1 (threshold to exclude near-zero noise)
    hourly["is_open"] = (hourly["library"] > 1).astype(int)
    # Do NOT set closed hours to NaN, do NOT filter rows
    print(f"   Hourly rows: {len(hourly)} | Open: {int(hourly['is_open'].sum())} | Closed: {int((hourly['is_open']==0).sum())}")

    # ── C. Feature engineering ───────────────────────────────────────────────
    print("3. Adding features ...")

    # Calendar features
    idx = hourly.index  # DatetimeIndex in Stockholm time
    hourly["hour"] = idx.hour
    hourly["dayofweek"] = idx.dayofweek
    hourly["month"] = idx.month
    hourly["day_of_month"] = idx.day
    hourly["week_of_year"] = idx.isocalendar().week.astype(int)
    hourly["quarter"] = idx.quarter
    hourly["is_weekend"] = idx.dayofweek.isin([5, 6]).astype(int)

    # Holidays
    se_holidays = holidays.Sweden(years=range(idx.min().year, idx.max().year + 2))
    hourly["is_holiday"] = [1 if d.date() in se_holidays else 0 for d in idx]

    # Exam periods
    hourly["is_exam_period"] = [is_in_exam_period(d) for d in idx]

    # Cyclical encodings
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)
    hourly["dow_sin"] = np.sin(2 * np.pi * hourly["dayofweek"] / 7)
    hourly["dow_cos"] = np.cos(2 * np.pi * hourly["dayofweek"] / 7)
    hourly["month_sin"] = np.sin(2 * np.pi * hourly["month"] / 12)
    hourly["month_cos"] = np.cos(2 * np.pi * hourly["month"] / 12)

    # Lag features (shift to avoid leakage)
    hourly["lag_1h"] = hourly["library"].shift(1)
    hourly["lag_24h"] = hourly["library"].shift(24)
    hourly["lag_168h"] = hourly["library"].shift(168)
    hourly[["lag_1h", "lag_24h", "lag_168h"]] = (
        hourly[["lag_1h", "lag_24h", "lag_168h"]].fillna(0)
    )

    # Rolling features (shift before rolling to avoid leakage)
    lib_shifted = hourly["library"].shift(1)
    hourly["roll_24h_mean"] = lib_shifted.rolling(24, min_periods=1).mean().fillna(0)
    hourly["roll_24h_std"] = lib_shifted.rolling(24, min_periods=1).std().fillna(0)
    hourly["roll_7d_mean"] = lib_shifted.rolling(168, min_periods=1).mean().fillna(0)

    # Time-since-open (hours elapsed since library opened on this day)
    hourly["date_col"] = hourly.index.date

    def hours_since_open_fn(df_day):
        open_mask = df_day["is_open"] == 1
        if not open_mask.any():
            return pd.Series(0.0, index=df_day.index)
        first_open = df_day.index[open_mask][0]
        elapsed = pd.Series(
            (df_day.index - first_open).total_seconds() / 3600,
            index=df_day.index,
        )
        result = elapsed.where(open_mask, 0.0).clip(lower=0)
        return result

    hourly["hours_since_open"] = hourly.groupby(
        hourly.index.date, group_keys=False
    ).apply(hours_since_open_fn)
    hourly = hourly.drop(columns=["date_col"])

    print(f"   Final shape: {hourly.shape}")
    print(f"   Columns: {list(hourly.columns)}")

    # ── D. Save ───────────────────────────────────────────────────────────────
    print("4. Saving ...")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Backup existing
    if PROCESSED_DATA_PATH.exists():
        backup_path = PROCESSED_DATA_PATH.parent / "hourly_backup.parquet"
        shutil.copy(PROCESSED_DATA_PATH, backup_path)
        print(f"   Backed up existing file to {backup_path}")

    hourly.to_parquet(PROCESSED_DATA_PATH)
    print(f"   Saved processed hourly dataset to {PROCESSED_DATA_PATH}")

    # Data quality JSON
    quality = {
        "raw_records": int(len(df_raw)),
        "after_dedup": int(len(df_sorted)),
        "hourly_total": int(len(hourly)),
        "open_hours": int(hourly["is_open"].sum()),
        "closed_hours": int((hourly["is_open"] == 0).sum()),
        "date_range_start": str(hourly.index.min()),
        "date_range_end": str(hourly.index.max()),
        "features": list(hourly.columns),
        "feature_count": int(len(hourly.columns)),
    }
    quality_path = PROCESSED_DATA_PATH.parent / "data_quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality, f, indent=2)
    print(f"   Saved data quality report to {quality_path}")

    # Summary
    total_hours = len(hourly)
    open_hours = int(hourly["is_open"].sum())
    closed_hours = total_hours - open_hours
    open_pct = (open_hours / total_hours) * 100 if total_hours > 0 else 0

    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Raw records:              {len(df_raw)}")
    print(f"After dedup/sort:         {len(df_sorted)}")
    print(f"Dataset shape:            {hourly.shape}")
    print(f"Date range:               {hourly.index.min()} to {hourly.index.max()}")
    print(f"Total hours:              {total_hours}")
    print(f"Open hours:               {open_hours} ({open_pct:.2f}%)")
    print(f"Closed hours:             {closed_hours} ({100 - open_pct:.2f}%)")
    print(f"Feature count:            {len(hourly.columns)}")
    print("=" * 60)

    return hourly


if __name__ == "__main__":
    prepare_data()
