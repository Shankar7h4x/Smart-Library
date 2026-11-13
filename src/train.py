"""Train XGBoost occupancy models, evaluate against baseline, and save model bundle."""

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ALL_ZONES,
    BACKTEST_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    ZONE_CAPACITIES,
    ZONE_NAMES,
)



from src.baseline import CalendarMeanBaseline

def compute_open_profile(df: pd.DataFrame) -> dict:
    """Compute open probability for each dayofweek and hour."""
    profile = {}
    grouped = df.groupby(["dayofweek", "hour"])["is_open"].mean()
    for (dow, h), prob in grouped.items():
        profile[f"{dow}_{h}"] = float(prob)
        profile[(int(dow), int(h))] = float(prob)
    return profile


def train_models():
    """Train XGBoost models per zone, evaluate vs baseline, and save artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DATA_PATH.exists():
        print(f"Error: {PROCESSED_DATA_PATH} not found. Run prepare_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(PROCESSED_DATA_PATH)

    # 1. Open profile across all hours
    open_profile = compute_open_profile(df)
    open_df = df[df["is_open"] == 1].copy()
    if open_df.empty:
        raise ValueError("No open-hour records are available for model training.")

    metrics = {
        "zones": {},
        "summary": {},
        "metadata": {
            "training_data_start": str(df.index.min()),
            "training_data_end": str(df.index.max()),
            "target_timezone": "Europe/Stockholm",
            "capacities_are_assumptions": True,
        },
    }
    resid_std = {}
    backtest_records = []
    trained_models_eval = {}
    zone_train_counts, zone_test_counts = {}, {}

    # Per-zone chronological 80/20 holdout. Missing target values are removed
    # per zone so a missing reading cannot enter training or evaluation.
    for zone in ALL_ZONES:
        zone_df = open_df.copy()
        zone_df[zone] = pd.to_numeric(zone_df[zone], errors="coerce")
        zone_df = zone_df[np.isfinite(zone_df[zone])].sort_index()
        if len(zone_df) < 10:
            raise ValueError(
                f"Zone {zone} has only {len(zone_df)} valid open-hour targets; "
                "at least 10 are required for a chronological evaluation."
            )
        split_idx = int(len(zone_df) * 0.8)
        split_idx = min(max(split_idx, 1), len(zone_df) - 1)
        train_df = zone_df.iloc[:split_idx]
        test_df = zone_df.iloc[split_idx:]
        if len(test_df) < 2:
            raise ValueError(f"Zone {zone} has too few holdout samples ({len(test_df)}).")

        X_train, X_test = train_df[FEATURES], test_df[FEATURES]
        y_train, y_test = train_df[zone].to_numpy(), test_df[zone].to_numpy()
        zone_train_counts[zone], zone_test_counts[zone] = len(train_df), len(test_df)

        baseline_group = train_df.groupby(["dayofweek", "hour"])[zone].mean().to_dict()
        overall_mean = float(train_df[zone].mean())
        baseline_model = CalendarMeanBaseline(baseline_group, overall_mean)
        y_base_pred = np.clip(baseline_model.predict(X_test), 0, 100)

        xgb = XGBRegressor(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=1,
        )
        xgb.fit(X_train, y_train)
        y_xgb_pred = np.clip(xgb.predict(X_test), 0, 100)

        mae_xgb = float(mean_absolute_error(y_test, y_xgb_pred))
        rmse_xgb = float(root_mean_squared_error(y_test, y_xgb_pred))
        mae_base = float(mean_absolute_error(y_test, y_base_pred))
        rmse_base = float(root_mean_squared_error(y_test, y_base_pred))
        selected_model = xgb if mae_xgb <= mae_base else baseline_model
        selected_pred = y_xgb_pred if mae_xgb <= mae_base else y_base_pred
        resid_std[zone] = float(np.std(y_test - selected_pred))
        trained_models_eval[zone] = selected_model

        metrics["zones"][zone] = {
            "display_name": ZONE_NAMES[zone],
            "xgb_mae": round(mae_xgb, 2),
            "xgb_rmse": round(rmse_xgb, 2),
            "baseline_mae": round(mae_base, 2),
            "baseline_rmse": round(rmse_base, 2),
            "mae_improvement": round(mae_base - mae_xgb, 2),
            "resid_std": round(resid_std[zone], 2),
            "beats_baseline": bool(mae_xgb <= mae_base),
            "selected_model": "xgboost" if mae_xgb <= mae_base else "weekday_hour_baseline",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        }

        for dt, actual, pred, base in zip(test_df.index, y_test, selected_pred, y_base_pred):
            backtest_records.append({
                "time": dt, "zone": zone, "actual": float(actual),
                "predicted": float(pred), "baseline": float(base),
                "residual": float(actual - pred),
            })

    # Summary metrics
    avg_xgb_mae = float(np.mean([m["xgb_mae"] for m in metrics["zones"].values()]))
    avg_base_mae = float(np.mean([m["baseline_mae"] for m in metrics["zones"].values()]))
    overall_beats = bool(avg_xgb_mae <= avg_base_mae)

    metrics["summary"] = {
        "avg_xgb_mae": round(avg_xgb_mae, 2),
        "avg_baseline_mae": round(avg_base_mae, 2),
        "overall_beats_baseline": overall_beats,
        "train_samples": int(sum(zone_train_counts.values())),
        "test_samples": int(sum(zone_test_counts.values())),
        "test_samples_per_zone": zone_test_counts,
        "evaluation": "per-zone chronological 80/20 holdout",
    }

    # Save metrics.json
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    # Save backtest.parquet
    backtest_df = pd.DataFrame(backtest_records)
    backtest_df.to_parquet(BACKTEST_PATH, index=False)
    print(f"Saved backtest predictions to {BACKTEST_PATH}")

    # Retrain on complete dataset (all open hours)
    print("Retraining all models on complete dataset...")
    X_full = open_df[FEATURES]
    final_models = {}

    for zone in ALL_ZONES:
        zone_full = open_df.copy()
        zone_full[zone] = pd.to_numeric(zone_full[zone], errors="coerce")
        zone_full = zone_full[np.isfinite(zone_full[zone])]
        if zone_full.empty:
            raise ValueError(f"Zone {zone} has no valid targets for final training.")
        xgb_full = XGBRegressor(
            n_estimators=120, max_depth=4, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=1,
        )
        xgb_full.fit(zone_full[FEATURES], zone_full[zone].to_numpy())
        if metrics["zones"][zone]["selected_model"] == "xgboost":
            final_models[zone] = xgb_full
        else:
            grouped = zone_full.groupby(["dayofweek", "hour"])[zone].mean().to_dict()
            final_models[zone] = CalendarMeanBaseline(grouped, float(zone_full[zone].mean()))

    bundle = {
        "models": final_models,
        "resid_std": resid_std,
        "open_profile": open_profile,
        "capacities": ZONE_CAPACITIES,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to {MODEL_PATH}")

    # Self-check prediction for a normal Thursday at 10:00
    # Dayofweek for Thursday = 3, hour = 10, weekend=0, holiday=0, exam=0
    thursday_features = pd.DataFrame([{
        "hour": 10,
        "dayofweek": 3,
        "is_weekend": 0,
        "is_holiday": 0,
        "is_exam_period": 0,
    }])[FEATURES]

    lib_pred = float(final_models["library"].predict(thursday_features)[0])
    lib_pred = float(np.clip(lib_pred, 0, 100))

    print("=" * 60)
    print("MODEL EVALUATION & SELF-CHECK RESULTS")
    print("=" * 60)
    for z, m in metrics["zones"].items():
        comp = "BEATS baseline" if m["beats_baseline"] else "WORSE than baseline"
        print(f"Zone {z:12s} ({m['display_name']:18s}): XGB MAE={m['xgb_mae']:.2f} | Base MAE={m['baseline_mae']:.2f} ({comp})")
    print("-" * 60)
    print(f"Average XGBoost MAE:  {avg_xgb_mae:.2f}")
    print(f"Average Baseline MAE: {avg_base_mae:.2f}")
    print(f"Overall beats baseline: {overall_beats}")
    print("-" * 60)
    print(f"Self-check prediction (Normal Thursday 10:00): {lib_pred:.1f}%")
    if 40.0 <= lib_pred <= 55.0:
        print("Self-check: Predicted whole-library occupancy falls around expected 40-55% range.")
    else:
        print(f"Self-check: Note: Predicted whole-library occupancy is {lib_pred:.1f}% (outside 40-55%).")
    print("=" * 60)

    return bundle, metrics


if __name__ == "__main__":
    train_models()

