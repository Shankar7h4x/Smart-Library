"""Prepare raw library occupancy data into hourly resampled features."""

import sys
from pathlib import Path
import holidays
import numpy as np
import pandas as pd
from datasets import load_dataset

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
    """Download, clean, resample, feature engineer, and save library occupancy dataset."""
    print("1. Downloading dataset 'davnas/real-time-library-occupancy' from Hugging Face...")
    try:
        ds = load_dataset("davnas/real-time-library-occupancy")
        df_raw = ds["train"].to_pandas()
    except Exception as exc:
        print(f"ERROR: Failed to download dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Save raw Parquet copy
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(RAW_DATA_PATH, index=False)
    print(f"2. Saved raw dataset to {RAW_DATA_PATH} ({len(df_raw)} records)")

    # 3. Parse datetime
    df = df_raw.copy()
    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    if parsed_time.isna().any():
        dropped = int(parsed_time.isna().sum())
        print(f"Warning: dropping {dropped} rows with invalid timestamps.")
    df = df.loc[parsed_time.notna()].copy()
    parsed_time = parsed_time.loc[parsed_time.notna()]
    # Source timestamps without offsets are interpreted as Stockholm local time,
    # matching the dataset's documented campus-local reporting convention.
    if getattr(parsed_time.dt, "tz", None) is None:
        parsed_time = parsed_time.dt.tz_localize(
            "Europe/Stockholm", ambiguous="NaT", nonexistent="shift_forward"
        )
    else:
        parsed_time = parsed_time.dt.tz_convert("Europe/Stockholm")
    df["time"] = parsed_time
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # 6. Rename zone columns to IDs
    df = df.rename(columns=ZONE_MAP)
    valid_zones = [z for z in ALL_ZONES if z in df.columns]
    missing_zones = sorted(set(ALL_ZONES) - set(valid_zones))
    if missing_zones:
        raise ValueError(f"The source dataset is missing required occupancy columns: {', '.join(missing_zones)}")

    # Ensure zone columns are numeric
    for z in valid_zones:
        df[z] = pd.to_numeric(df[z], errors="coerce")

    # 7. Aggregate duplicate observations within each hour rather than
    # arbitrarily discarding all but the first timestamp.
    df = df.set_index("time")
    hourly = df[valid_zones].resample("1h").mean()
    if hourly.empty:
        raise ValueError("The source dataset contains no valid timestamps to prepare.")

    # 8. Determine is_open using overall hourly mean > 0
    # An hourly library reading > 0 indicates open hours
    is_open = (hourly["library"] > 0).astype(int)
    hourly["is_open"] = is_open

    # 9. Set closed-hour occupancy values to NaN
    for z in valid_zones:
        hourly.loc[hourly["is_open"] == 0, z] = np.nan

    # 10. Add features: hour, dayofweek, is_weekend, is_holiday, is_exam_period
    idx = hourly.index
    hourly["hour"] = idx.hour
    hourly["dayofweek"] = idx.dayofweek
    hourly["is_weekend"] = hourly["dayofweek"].isin([5, 6]).astype(int)

    # 11. Swedish holidays
    se_holidays = holidays.Sweden(years=range(idx.min().year, idx.max().year + 2))
    hourly["is_holiday"] = [1 if d.date() in se_holidays else 0 for d in idx]
    hourly["is_exam_period"] = [is_in_exam_period(d) for d in idx]

    # 12. Save to data/processed/hourly.parquet
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(PROCESSED_DATA_PATH)
    print(f"12. Saved processed hourly dataset to {PROCESSED_DATA_PATH}")

    # Summary statistics
    total_hours = len(hourly)
    open_hours = int(hourly["is_open"].sum())
    closed_hours = total_hours - open_hours
    open_pct = (open_hours / total_hours) * 100 if total_hours > 0 else 0

    print("=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Dataset shape:            {hourly.shape}")
    print(f"Date range:               {idx.min()} to {idx.max()}")
    print(f"Total hours:              {total_hours}")
    print(f"Open hours:               {open_hours} ({open_pct:.2f}%)")
    print(f"Closed hours:             {closed_hours} ({100 - open_pct:.2f}%)")
    print("=" * 60)

    return hourly


if __name__ == "__main__":
    prepare_data()

