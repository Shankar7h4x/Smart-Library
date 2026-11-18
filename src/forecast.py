"""Forecasting engine: pure functions for library occupancy predictions and recommendations."""

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Union
import holidays
import numpy as np
import pandas as pd
from config import (
    ALL_ZONES,
    EXAM_PERIODS,
    FEATURES,
    FEATURES_NO_LAG,
    STATUS_BUSY,
    STATUS_CLOSED,
    STATUS_FREE,
    STATUS_FULL,
    STATUS_THRESHOLDS,
    SUB_ZONES,
    ZONE_CAPACITIES,
    ZONE_NAMES,
)


def is_in_exam_period(dt: pd.Timestamp) -> int:
    """Check if date falls within predefined recurring exam periods."""
    m, d = dt.month, dt.day
    for (sm, sd), (em, ed) in EXAM_PERIODS:
        if sm == em:
            if m == sm and sd <= d <= ed:
                return 1
        elif sm < em:
            if (m == sm and d >= sd) or (m == em and d <= ed) or (sm < m < em):
                return 1
    return 0


def build_features(datetimes: Union[datetime, List[datetime], pd.DatetimeIndex, pd.Series], exam: bool = False) -> pd.DataFrame:
    """Build all model features. Lag features are set to 0 (not available at future prediction time)."""
    if isinstance(datetimes, (datetime, pd.Timestamp)):
        values = [datetimes]
    else:
        values = list(datetimes)

    if not values:
        return pd.DataFrame(columns=FEATURES)

    stockholm = ZoneInfo("Europe/Stockholm")
    normalized = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("Forecast timestamps must not contain missing values.")
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(stockholm)
        else:
            stamp = stamp.tz_convert(stockholm)
        normalized.append(stamp.tz_localize(None))
    dt_index = pd.DatetimeIndex(normalized)

    se_holidays = holidays.Sweden(years=range(dt_index.min().year, dt_index.max().year + 2))
    hours = dt_index.hour.values
    dows = dt_index.dayofweek.values
    months = dt_index.month.values
    day_of_months = dt_index.day.values
    week_of_years = dt_index.isocalendar().week.values.astype(int)
    quarters = dt_index.quarter.values
    is_weekends = np.isin(dows, [5, 6]).astype(int)
    is_hols = np.array([1 if d.date() in se_holidays else 0 for d in dt_index], dtype=int)
    is_exams = (
        np.ones(len(dt_index), dtype=int)
        if exam else np.array([is_in_exam_period(d) for d in dt_index], dtype=int)
    )

    df = pd.DataFrame({
        "hour": hours,
        "dayofweek": dows,
        "month": months,
        "day_of_month": day_of_months,
        "week_of_year": week_of_years,
        "quarter": quarters,
        "is_weekend": is_weekends,
        "is_holiday": is_hols,
        "is_exam_period": is_exams,
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "dow_sin": np.sin(2 * np.pi * dows / 7),
        "dow_cos": np.cos(2 * np.pi * dows / 7),
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        # Lag features: 0 at prediction time (not available for future forecasts)
        "lag_1h": 0.0,
        "lag_24h": 0.0,
        "lag_168h": 0.0,
        "roll_24h_mean": 0.0,
        "roll_24h_std": 0.0,
        "roll_7d_mean": 0.0,
        "hours_since_open": 0.0,
    })

    # Return only the columns in FEATURES (in correct order)
    feat_cols = [c for c in FEATURES if c in df.columns]
    return df[feat_cols]


def is_open(bundle: dict, dt: datetime) -> bool:
    """Check the learned open profile using Stockholm local calendar time."""
    stamp = pd.Timestamp(dt)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(ZoneInfo("Europe/Stockholm"))
    else:
        stamp = stamp.tz_convert(ZoneInfo("Europe/Stockholm"))
    dow = int(stamp.day_of_week if hasattr(stamp, "day_of_week") else stamp.dayofweek)
    h = int(stamp.hour)
    prof = bundle.get("open_profile", {})
    prob = prof.get(f"{dow}_{h}", prof.get((dow, h), 0.0))
    # Threshold 0.4 clearly separates open hours from overnight closed hours
    return bool(prob >= 0.4)


def next_open(bundle: dict, dt: datetime) -> Optional[datetime]:
    """Find the next opening hour starting from dt."""
    curr = dt.replace(minute=0, second=0, microsecond=0)
    for step in range(1, 14 * 24 + 1):
        candidate = curr + timedelta(hours=step)
        if is_open(bundle, candidate):
            return candidate
    return None


def status(p: Optional[float]) -> str:
    """Classify occupancy percentage into Free, Busy, Full, or Closed."""
    if p is None or not math.isfinite(float(p)):
        return STATUS_CLOSED
    if p < STATUS_THRESHOLDS["free_max"]:
        return STATUS_FREE
    if p <= STATUS_THRESHOLDS["busy_max"]:
        return STATUS_BUSY
    return STATUS_FULL


def free_seats(p: Optional[float], zone: str, bundle: Optional[dict] = None) -> int:
    """Estimate available seats for a given occupancy percentage."""
    if p is None or not math.isfinite(float(p)):
        return 0
    capacities = bundle["capacities"] if bundle and "capacities" in bundle else ZONE_CAPACITIES
    cap = capacities.get(zone, ZONE_CAPACITIES.get(zone))
    if cap is None or int(cap) <= 0:
        return 0
    seats = round(cap * (1.0 - (max(0.0, min(100.0, p)) / 100.0)))
    return max(0, seats)


def predict(bundle: dict, dt: Union[datetime, List[datetime], pd.DatetimeIndex], zone: str, exam: bool = False):
    """Predict occupancy percentage for single or multiple datetimes."""
    is_single = isinstance(dt, (datetime, pd.Timestamp))
    dt_list = [dt] if is_single else list(dt)
    if not dt_list:
        return np.array([], dtype=float)

    features_df = build_features(dt_list, exam=exam)
    model = bundle["models"].get(zone)
    if model is None:
        return None if is_single else np.array([np.nan] * len(dt_list))

    # Get feature columns expected by this bundle
    bundle_features = bundle.get("features", features_df.columns.tolist())
    # Fill any missing columns with 0
    for col in bundle_features:
        if col not in features_df.columns:
            features_df[col] = 0.0
    # Use only columns that exist in features_df
    use_cols = [c for c in bundle_features if c in features_df.columns]

    try:
        preds = np.clip(model.predict(features_df[bundle_features]), 0, 100)
    except Exception:
        preds = np.clip(model.predict(features_df[use_cols]), 0, 100)

    results = []
    for item_dt, pred_val in zip(dt_list, preds):
        if not is_open(bundle, item_dt):
            results.append(np.nan)
        else:
            results.append(float(pred_val))

    if is_single:
        return results[0]
    return np.array(results)


def predict_with_ci(bundle: dict, dt, zone: str, exam: bool = False, z_score: float = 1.645):
    """Predict with 90% confidence interval using historical residual std."""
    pred = predict(bundle, dt, zone, exam=exam)
    if pred is None or not math.isfinite(float(pred)):
        return None, None, None
    std = bundle.get("resid_std", {}).get(zone, 10.0)
    margin = z_score * std
    return pred, max(0.0, pred - margin), min(100.0, pred + margin)


def day_curve(bundle: dict, target_date: date, zone: str, exam: bool = False) -> pd.DataFrame:
    """Generate 24-hour occupancy forecast curve for a target date."""
    dts = [datetime.combine(target_date, datetime.min.time()) + timedelta(hours=h) for h in range(24)]
    preds = predict(bundle, dts, zone, exam=exam)
    resid_std = bundle.get("resid_std", {}).get(zone, 12.0)

    rows = []
    for h, dt_item, p in zip(range(24), dts, preds):
        op = is_open(bundle, dt_item)
        st = status(p)
        seats = free_seats(p, zone, bundle)
        rows.append({
            "hour": h,
            "datetime": dt_item,
            "occupancy": p,
            "status": st,
            "is_open": op,
            "free_seats": seats,
            "lower_bound": max(0.0, p - resid_std) if not np.isnan(p) else np.nan,
            "upper_bound": min(100.0, p + resid_std) if not np.isnan(p) else np.nan,
        })
    return pd.DataFrame(rows)


def best_hour(curve: pd.DataFrame) -> Optional[dict]:
    """Find the quietest open hour from a day curve."""
    open_curve = curve[curve["is_open"] & curve["occupancy"].notna()]
    if open_curve.empty:
        return None
    best_row = open_curve.loc[open_curve["occupancy"].idxmin()]
    return {
        "hour": int(best_row["hour"]),
        "occupancy": float(best_row["occupancy"]),
        "status": str(best_row["status"]),
        "free_seats": int(best_row["free_seats"]),
    }


def peak_hours(bundle: dict, zone: str, dayofweek: int, n: int = 3) -> List[dict]:
    """Identify top n busiest hours for a given day of the week."""
    # Build typical week datetimes for that day of week
    base = datetime(2025, 4, 7) + timedelta(days=dayofweek)  # 2025-04-07 is a Monday
    dts = [datetime.combine(base.date(), datetime.min.time()) + timedelta(hours=h) for h in range(24)]
    preds = predict(bundle, dts, zone, exam=False)

    open_hours = []
    for h, p in zip(range(24), preds):
        if not np.isnan(p):
            open_hours.append({"hour": h, "occupancy": float(p), "status": status(p)})

    open_hours.sort(key=lambda x: x["occupancy"], reverse=True)
    return open_hours[:n]


def alternative(bundle: dict, dt: datetime, current_zone: str, exam: bool = False) -> Optional[dict]:
    """Recommend a less crowded alternative zone or better time."""
    candidate_zones = [z for z in SUB_ZONES if z != current_zone]
    if not candidate_zones:
        candidate_zones = SUB_ZONES

    curr_p = predict(bundle, dt, current_zone, exam=exam)
    best_alt = None
    min_occ = float("inf")

    for z in candidate_zones:
        p = predict(bundle, dt, z, exam=exam)
        if not np.isnan(p) and p < min_occ:
            min_occ = p
            best_alt = z

    if best_alt is not None:
        return {
            "zone": best_alt,
            "display_name": ZONE_NAMES[best_alt],
            "occupancy": min_occ,
            "status": status(min_occ),
            "free_seats": free_seats(min_occ, best_alt, bundle),
            "current_occupancy": curr_p,
            "savings": (curr_p - min_occ) if (curr_p is not None and not np.isnan(curr_p)) else 0.0,
        }
    return None


def best_window(bundle: dict, target_date: date, zone: str, hours: int, exam: bool = False,
                earliest_start_hour: int = 0) -> Optional[dict]:
    """Find the best consecutive study window of specified duration on target_date."""
    if not isinstance(hours, int) or not 1 <= hours <= 24:
        raise ValueError("hours must be an integer between 1 and 24")
    if not 0 <= int(earliest_start_hour) <= 23:
        raise ValueError("earliest_start_hour must be between 0 and 23")
    curve = day_curve(bundle, target_date, zone, exam=exam)
    valid_windows = []

    for start_h in range(int(earliest_start_hour), 24 - hours + 1):
        win = curve.iloc[start_h : start_h + hours]
        if win["is_open"].all() and win["occupancy"].notna().all():
            avg_occ = float(win["occupancy"].mean())
            seats = free_seats(avg_occ, zone, bundle)
            valid_windows.append({
                "start_hour": start_h,
                "end_hour": start_h + hours,
                "start_time": f"{start_h:02d}:00",
                "end_time": f"{start_h + hours:02d}:00",
                "avg_occupancy": avg_occ,
                "free_seats": seats,
                "zone": zone,
            })

    if not valid_windows:
        return None
    valid_windows.sort(key=lambda x: x["avg_occupancy"])
    return valid_windows[0]


def week_grid(bundle: dict, start_date: date, zone: str, days: int = 7, exam: bool = False) -> pd.DataFrame:
    """Generate occupancy matrix for days x 24 hours."""
    grid = []
    day_labels = []

    for d in range(days):
        cur_date = start_date + timedelta(days=d)
        curve = day_curve(bundle, cur_date, zone, exam=exam)
        grid.append(curve["occupancy"].values)
        day_labels.append(cur_date.strftime("%a %d"))

    return pd.DataFrame(grid, index=day_labels, columns=[f"{h:02d}:00" for h in range(24)])


def top_slots(bundle: dict, start_date: date, zone: str, hours: int = 2, days: int = 7, n: int = 3, exam: bool = False) -> List[dict]:
    """Return top n recommended visit slots over the upcoming days."""
    if not isinstance(hours, int) or not 1 <= hours <= 24:
        raise ValueError("hours must be an integer between 1 and 24")
    if not isinstance(days, int) or days < 0 or not isinstance(n, int) or n < 0:
        raise ValueError("days and n must be non-negative integers")
    candidates = []
    local_now = datetime.now(ZoneInfo("Europe/Stockholm"))
    for d in range(days):
        cur_date = start_date + timedelta(days=d)
        curve = day_curve(bundle, cur_date, zone, exam=exam)
        first_hour = local_now.hour + (1 if local_now.minute else 0) if cur_date == local_now.date() else 0

        for start_h in range(first_hour, 24 - hours + 1):
            win = curve.iloc[start_h : start_h + hours]
            if win["is_open"].all() and win["occupancy"].notna().all():
                avg_occ = float(win["occupancy"].mean())
                seats = free_seats(avg_occ, zone, bundle)
                candidates.append({
                    "date": cur_date,
                    "date_str": cur_date.strftime("%a, %b %d"),
                    "start_time": f"{start_h:02d}:00 - {start_h + hours:02d}:00",
                    "duration_hours": hours,
                    "avg_occupancy": avg_occ,
                    "free_seats": seats,
                })

    candidates.sort(key=lambda x: x["avg_occupancy"])
    return candidates[:n]


def smart_pick(bundle: dict, dt: datetime, goal: str, exam: bool = False) -> dict:
    """Recommend zone and time based on student visit goal."""
    # Check if open right now
    currently_open = is_open(bundle, dt)
    upcoming_open = None if currently_open else next_open(bundle, dt)
    if not currently_open and upcoming_open is None:
        return {
            "zone": None,
            "zone_name": None,
            "time": "No upcoming opening found",
            "explanation": "No valid opening hour is available in the loaded schedule. No study zone can be recommended.",
        }
    active_dt = dt if currently_open else upcoming_open
    time_label = f"Right now ({active_dt.strftime('%H:%M')})" if currently_open else f"Next opening ({active_dt.strftime('%a %H:%M')})"

    zone_scores = []
    for z in SUB_ZONES:
        occ = predict(bundle, active_dt, z, exam=exam)
        if occ is None or not np.isfinite(float(occ)):
            continue
        seats = free_seats(occ, z, bundle)
        zone_scores.append({"zone": z, "name": ZONE_NAMES[z], "occupancy": float(occ), "seats": seats})

    if not zone_scores:
        return {"zone": None, "zone_name": None, "time": time_label,
                "explanation": "No valid zone forecasts are available for this time."}

    if goal == "quietest_now":
        zone_scores.sort(key=lambda x: x["occupancy"] if not np.isnan(x["occupancy"]) else 999)
        picked = zone_scores[0]
        return {
            "zone": picked["zone"],
            "zone_name": picked["name"],
            "time": time_label,
            "explanation": f"{picked['name']} is currently quietest with only {picked['occupancy']:.0f}% occupancy.",
        }

    if goal == "most_seats":
        zone_scores.sort(key=lambda x: x["seats"], reverse=True)
        picked = zone_scores[0]
        return {
            "zone": picked["zone"],
            "zone_name": picked["name"],
            "time": time_label,
            "explanation": f"{picked['name']} offers the highest seat availability with ~{picked['seats']} free seats.",
        }

    if goal == "longest_stay":
        # Look ahead 3 hours
        stay_scores = []
        for z in SUB_ZONES:
            dts = [active_dt + timedelta(hours=i) for i in range(3)]
            preds = predict(bundle, dts, z, exam=exam)
            valid_preds = preds[~np.isnan(preds)]
            avg_stay = float(np.mean(valid_preds)) if len(valid_preds) == len(dts) else np.nan
            stay_scores.append({"zone": z, "name": ZONE_NAMES[z], "avg_occ": avg_stay})

        stay_scores.sort(key=lambda x: x["avg_occ"] if not np.isnan(x["avg_occ"]) else 999)
        picked = stay_scores[0]
        if np.isnan(picked["avg_occ"]):
            return {
                "zone": picked["zone"],
                "zone_name": picked["name"],
                "time": f"{active_dt.strftime('%H:%M')} for next 3 hours",
                "explanation": "No zone is open for the full next 3 hours; try a shorter stay or a later time.",
            }
        return {
            "zone": picked["zone"],
            "zone_name": picked["name"],
            "time": f"{active_dt.strftime('%H:%M')} for next 3 hours",
            "explanation": f"{picked['name']} maintains lowest sustained crowding averaging {picked['avg_occ']:.0f}% over 3 hours.",
        }

    picked = min(zone_scores, key=lambda item: item["occupancy"])
    return {
        "zone": picked["zone"],
        "zone_name": picked["name"],
        "time": time_label,
        "explanation": f"{picked['name']} has the lowest available model forecast ({picked['occupancy']:.0f}%).",
    }

