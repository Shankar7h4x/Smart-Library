"""Unit and integration checks for the occupancy forecast engine."""

import math
import sys
from datetime import date, datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ALL_ZONES, MODEL_PATH, SUB_ZONES
from src.forecast import (
    alternative,
    best_hour,
    best_window,
    build_features,
    day_curve,
    free_seats,
    is_open,
    next_open,
    peak_hours,
    predict,
    smart_pick,
    status,
    top_slots,
    week_grid,
)


def run_all_tests():
    """Execute forecast-engine checks against the trained local model."""
    print("=" * 60)
    print("STARTING TEST SUITE: test_functions.py")
    print("=" * 60)

    # Load actual trained model bundle
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    bundle = joblib.load(MODEL_PATH)
    assert "models" in bundle, "Missing models in bundle"
    assert "open_profile" in bundle, "Missing open_profile in bundle"
    assert "capacities" in bundle, "Missing capacities in bundle"
    print("✓ Model bundle loaded successfully.")

    # 1. Test build_features
    dt_sample = datetime(2025, 4, 10, 10, 0)
    features_df = build_features(dt_sample, exam=False)
    assert isinstance(features_df, pd.DataFrame), "Features must be a DataFrame"
    assert len(features_df) == 1, "Expected 1 row for single datetime"
    assert set(["hour", "dayofweek", "is_weekend", "is_holiday", "is_exam_period"]).issubset(set(features_df.columns))
    assert features_df.iloc[0]["hour"] == 10
    print("✓ build_features tested successfully.")

    # 2. Test status classification
    assert status(None) == "Closed"
    assert status(np.nan) == "Closed"
    assert status(25.0) == "Free"
    assert status(65.0) == "Busy"
    assert status(85.0) == "Full"
    print("✓ status classification tested successfully.")

    # 3. Test available seats
    assert free_seats(None, "north", bundle) == 0
    assert free_seats(np.nan, "north", bundle) == 0
    seats_50 = free_seats(50.0, "north", bundle)
    # North capacity is 120 -> 50% = 60 seats
    assert seats_50 == 60, f"Expected 60 seats, got {seats_50}"
    seats_100 = free_seats(100.0, "north", bundle)
    assert seats_100 == 0
    print("✓ free_seats tested successfully.")

    # 4. Test is_open and next_open
    open_dt = datetime(2025, 4, 10, 11, 0)  # Thursday 11:00
    closed_dt = datetime(2025, 4, 10, 3, 0)  # Thursday 03:00 (middle of the night)
    assert is_open(bundle, open_dt) is True, "Thursday 11:00 should be open"
    assert is_open(bundle, closed_dt) is False, "Thursday 03:00 should be closed"

    nxt = next_open(bundle, closed_dt)
    assert nxt is not None, "Next open time should not be None"
    assert is_open(bundle, nxt) is True, "Next open time must be an open hour"
    assert nxt > closed_dt, "Next open time must be in future"
    print(f"✓ is_open and next_open tested successfully (next open after 03:00 is {nxt.strftime('%H:%M')}).")

    # 5. Test single prediction (Open and Closed)
    pred_open = predict(bundle, open_dt, "library", exam=False)
    assert isinstance(pred_open, float), "Prediction for open hour must be float"
    assert 0.0 <= pred_open <= 100.0, f"Prediction {pred_open} must be between 0 and 100"

    pred_closed = predict(bundle, closed_dt, "library", exam=False)
    assert math.isnan(pred_closed), "Closed hour prediction must be NaN"
    print(f"✓ Single prediction tested (Open={pred_open:.1f}%, Closed={pred_closed}).")

    # 6. Test multiple predictions
    dts_multi = [datetime(2025, 4, 10, h, 0) for h in range(24)]
    preds_multi = predict(bundle, dts_multi, "library", exam=False)
    assert len(preds_multi) == 24, "Expected 24 predictions"
    assert any(np.isnan(preds_multi)), "Night hours must be NaN"
    assert any(~np.isnan(preds_multi)), "Day hours must be non-NaN"
    print("✓ Multiple predictions tested successfully.")

    # 7. Test day_curve and best_hour
    target_d = date(2025, 4, 10)
    curve = day_curve(bundle, target_d, "south_east", exam=False)
    assert isinstance(curve, pd.DataFrame), "day_curve must return a DataFrame"
    assert len(curve) == 24, "day_curve must have 24 hours"
    assert "occupancy" in curve.columns
    assert "status" in curve.columns
    assert "free_seats" in curve.columns

    best_h = best_hour(curve)
    assert best_h is not None, "best_hour should find a quiet hour"
    assert "hour" in best_h and "occupancy" in best_h
    assert not math.isnan(best_h["occupancy"]), "Best hour must not be NaN"
    print(f"✓ day_curve & best_hour tested successfully (Quiet hour: {best_h['hour']:02d}:00 with {best_h['occupancy']:.1f}%).")

    # 8. Test peak_hours
    peaks = peak_hours(bundle, "library", dayofweek=3, n=3)
    assert len(peaks) == 3, "Expected 3 peak hours"
    assert peaks[0]["occupancy"] >= peaks[1]["occupancy"] >= peaks[2]["occupancy"]
    print(f"✓ peak_hours tested successfully (Top peak at {peaks[0]['hour']}:00 with {peaks[0]['occupancy']:.1f}%).")

    # 9. Test alternative zone
    alt = alternative(bundle, open_dt, "library", exam=False)
    assert alt is not None, "Alternative should return a recommendation"
    assert "zone" in alt and "display_name" in alt
    assert alt["zone"] in SUB_ZONES
    print(f"✓ alternative zone tested successfully (Alternative: {alt['display_name']} with {alt['occupancy']:.1f}%).")

    # 10. Test best_window
    win = best_window(bundle, target_d, "north", hours=2, exam=False)
    assert win is not None, "best_window should return a valid 2-hour window"
    assert win["end_hour"] - win["start_hour"] == 2
    assert "avg_occupancy" in win and "free_seats" in win
    print(f"✓ best_window tested successfully ({win['start_time']} - {win['end_time']}, Avg={win['avg_occupancy']:.1f}%).")

    # 11. Test week_grid
    grid = week_grid(bundle, target_d, "south", days=7, exam=False)
    assert isinstance(grid, pd.DataFrame), "week_grid must return DataFrame"
    assert grid.shape == (7, 24), f"Expected 7x24 grid, got {grid.shape}"
    print("✓ week_grid tested successfully (7x24 matrix generated).")

    # 12. Test top_slots
    slots = top_slots(bundle, target_d, "angdomen", hours=2, days=7, n=3, exam=False)
    assert len(slots) == 3, f"Expected 3 top slots, got {len(slots)}"
    assert slots[0]["avg_occupancy"] <= slots[1]["avg_occupancy"] <= slots[2]["avg_occupancy"]
    print(f"✓ top_slots tested successfully (Best slot: {slots[0]['date_str']} at {slots[0]['start_time']}).")

    # 13. Test smart_pick
    for goal in ["quietest_now", "most_seats", "longest_stay"]:
        rec = smart_pick(bundle, open_dt, goal=goal, exam=False)
        assert "zone" in rec and "zone_name" in rec and "explanation" in rec and "time" in rec
        assert len(rec["explanation"]) > 0
    print("✓ smart_pick tested for all three goals successfully.")

    print("=" * 60)
    print("ALL FORECAST CHECKS PASSED SUCCESSFULLY! (13/13 test groups)")
    print("=" * 60)


def test_forecast_engine():
    """Pytest entrypoint for the forecast engine checks."""
    run_all_tests()


if __name__ == "__main__":
    run_all_tests()

