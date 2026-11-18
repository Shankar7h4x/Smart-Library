"""Comprehensive automated tests for ML pipeline, feature engineering, leakage prevention, and forecasting."""

from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    ALL_ZONES,
    BACKTEST_PATH,
    FEATURES,
    FEATURES_NO_LAG,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    SUB_ZONES,
    ZONE_CAPACITIES,
    ZONE_NAMES,
)
from src.baseline import CalendarMeanBaseline
from src.forecast import (
    build_features,
    day_curve,
    free_seats,
    is_open,
    next_open,
    predict,
    predict_with_ci,
    smart_pick,
    status,
    top_slots,
    week_grid,
)


def test_processed_data_and_features():
    """Verify processed data file exists, has expected columns, and no NaN in key features."""
    assert PROCESSED_DATA_PATH.exists(), "Processed data file must exist"
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    assert len(df) > 1000, f"Expected >1000 rows, got {len(df)}"

    # Check all zones present
    for zone in ALL_ZONES:
        assert zone in df.columns, f"Zone {zone} missing from processed data"

    # Check all features present
    for feat in FEATURES:
        assert feat in df.columns, f"Feature {feat} missing from processed data"

    # Check is_open column
    assert "is_open" in df.columns
    assert set(df["is_open"].unique()).issubset({0, 1})
    assert df["is_open"].sum() > 500, "Should have substantial open hours"


def test_no_data_leakage_in_lags():
    """Ensure lag features strictly refer to past hours (t-1h, t-24h, t-168h)."""
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    # lag_1h at index 1 must equal library value at index 0
    assert np.isclose(df["lag_1h"].iloc[1], df["library"].iloc[0]), "lag_1h must match previous row's library value"
    # lag_24h at index 24 must equal library value at index 0
    assert np.isclose(df["lag_24h"].iloc[24], df["library"].iloc[0]), "lag_24h must match t-24h value"
    # lag_168h at index 168 must equal library value at index 0
    assert np.isclose(df["lag_168h"].iloc[168], df["library"].iloc[0]), "lag_168h must match t-168h value"


def test_cyclical_encodings_bounded():
    """Verify sine and cosine encodings are strictly bounded in [-1, 1]."""
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert col in df.columns
        assert (df[col] >= -1.0001).all() and (df[col] <= 1.0001).all(), f"{col} must be in [-1, 1]"


def test_model_bundle_integrity():
    """Verify trained model bundle loads safely and has all required components."""
    assert MODEL_PATH.exists(), "Trained model bundle must exist"
    bundle = joblib.load(MODEL_PATH)

    assert "models" in bundle, "Bundle must contain models dict"
    assert "resid_std" in bundle, "Bundle must contain resid_std dict"
    assert "open_profile" in bundle, "Bundle must contain open_profile"
    assert "capacities" in bundle, "Bundle must contain capacities"
    assert "features" in bundle, "Bundle must contain features list"

    # All zones must have a trained model
    for zone in ALL_ZONES:
        assert zone in bundle["models"], f"Model for {zone} must be present in bundle"
        model = bundle["models"][zone]
        assert hasattr(model, "predict"), f"Model for {zone} must implement predict()"

        # Residual std must be positive
        assert zone in bundle["resid_std"]
        assert bundle["resid_std"][zone] > 0, f"Residual std for {zone} must be positive"


def test_metrics_json_quality():
    """Verify evaluation metrics show genuine holdout evaluation beating baseline."""
    assert METRICS_PATH.exists(), "metrics.json must exist"
    import json
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    summary = metrics["summary"]
    # Total test samples must be substantial (>100)
    assert summary["test_samples"] > 100, f"Test samples too small: {summary['test_samples']}"
    assert summary["overall_beats_baseline"] is True, "Models must beat baseline overall"

    # Per zone checks
    for zone in ALL_ZONES:
        zm = metrics["zones"][zone]
        assert zm["selected_mae"] > 0
        assert zm["baseline_mae"] > 0
        assert zm["test_samples"] > 20


def test_forecast_predict_single_and_batch():
    """Test single datetime prediction and batch prediction."""
    bundle = joblib.load(MODEL_PATH)
    stockholm = ZoneInfo("Europe/Stockholm")
    test_dt = datetime(2025, 3, 10, 11, 0, tzinfo=stockholm)  # Monday 11:00

    # Single prediction
    pred = predict(bundle, test_dt, "library")
    assert pred is not None, "Library should be open Monday 11:00"
    assert 0.0 <= pred <= 100.0, f"Prediction out of bounds: {pred}"

    # Batch prediction
    dts = [test_dt + timedelta(hours=i) for i in range(5)]
    batch_preds = predict(bundle, dts, "library")
    assert len(batch_preds) == 5
    for p in batch_preds:
        if not np.isnan(p):
            assert 0.0 <= p <= 100.0


def test_prediction_confidence_intervals():
    """Verify confidence intervals are correctly structured (lower <= pred <= upper)."""
    bundle = joblib.load(MODEL_PATH)
    stockholm = ZoneInfo("Europe/Stockholm")
    test_dt = datetime(2025, 3, 12, 14, 0, tzinfo=stockholm)

    pred, lower, upper = predict_with_ci(bundle, test_dt, "library", z_score=1.645)
    if pred is not None:
        assert lower <= pred <= upper, f"CI inverted: {lower} <= {pred} <= {upper}"
        assert 0.0 <= lower <= 100.0
        assert 0.0 <= upper <= 100.0


def test_status_and_free_seats():
    """Verify status categorizations and free seats arithmetic."""
    bundle = joblib.load(MODEL_PATH)
    assert status(20.0) == "Free"
    assert status(65.0) == "Busy"
    assert status(90.0) == "Full"
    assert status(None) == "Closed"
    assert status(float("nan")) == "Closed"

    # 420 capacity for whole library
    assert free_seats(0.0, "library", bundle) == 420
    assert free_seats(100.0, "library", bundle) == 0
    assert free_seats(50.0, "library", bundle) == 210
    assert free_seats(None, "library", bundle) == 0


def test_day_curve_24_hours():
    """Verify 24-hour day curve produces exactly 24 records with proper schema."""
    bundle = joblib.load(MODEL_PATH)
    curve = day_curve(bundle, date(2025, 3, 12), "library")
    assert len(curve) == 24
    assert set(curve.columns) >= {"hour", "datetime", "occupancy", "status", "is_open", "free_seats"}
    assert curve["hour"].tolist() == list(range(24))


def test_smart_pick_recommendations():
    """Verify smart_pick produces a valid recommendation dictionary."""
    bundle = joblib.load(MODEL_PATH)
    stockholm = ZoneInfo("Europe/Stockholm")
    test_dt = datetime(2025, 3, 11, 10, 0, tzinfo=stockholm)

    for goal in ["quietest_now", "most_seats", "longest_stay"]:
        rec = smart_pick(bundle, test_dt, goal)
        assert isinstance(rec, dict)
        assert "zone" in rec
        assert "zone_name" in rec
        assert "explanation" in rec
        assert len(rec["explanation"]) > 0


if __name__ == "__main__":
    print("Running ML Pipeline Test Suite...")
    test_processed_data_and_features()
    print("✓ test_processed_data_and_features passed")
    test_no_data_leakage_in_lags()
    print("✓ test_no_data_leakage_in_lags passed")
    test_cyclical_encodings_bounded()
    print("✓ test_cyclical_encodings_bounded passed")
    test_model_bundle_integrity()
    print("✓ test_model_bundle_integrity passed")
    test_metrics_json_quality()
    print("✓ test_metrics_json_quality passed")
    test_forecast_predict_single_and_batch()
    print("✓ test_forecast_predict_single_and_batch passed")
    test_prediction_confidence_intervals()
    print("✓ test_prediction_confidence_intervals passed")
    test_status_and_free_seats()
    print("✓ test_status_and_free_seats passed")
    test_day_curve_24_hours()
    print("✓ test_day_curve_24_hours passed")
    test_smart_pick_recommendations()
    print("✓ test_smart_pick_recommendations passed")
    print("\nALL TESTS PASSED SUCCESSFULLY! (10/10)")
