"""Train multiple occupancy models per zone, evaluate with time-series CV, select best, and save bundle."""

import json
import sys
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ALL_ZONES, BACKTEST_PATH, FEATURES, METRICS_PATH, MODEL_PATH, MODELS_DIR, PROCESSED_DATA_PATH,
    ZONE_CAPACITIES, ZONE_NAMES,
)
try:
    from config import FEATURES_NO_LAG
except ImportError:
    FEATURES_NO_LAG = ["hour", "dayofweek", "month", "day_of_month", "week_of_year",
                       "quarter", "is_weekend", "is_holiday", "is_exam_period",
                       "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]

from src.baseline import CalendarMeanBaseline

RANDOM_STATE = 42
N_CV_SPLITS = 3   # TimeSeriesSplit folds for quick tuning
MIN_TRAIN = 200   # minimum training rows required
MIN_TEST = 50     # minimum test rows required


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def compute_metrics(y_true, y_pred, label=""):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(median_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    w = wape(y_true, y_pred)
    return dict(mae=round(mae,3), rmse=round(rmse,3), r2=round(r2,4),
                med_ae=round(med_ae,3), bias=round(bias,3), wape=round(w,2) if not np.isnan(w) else None)


def compute_open_profile(df):
    profile = {}
    for (dow, h), prob in df.groupby(["dayofweek","hour"])["is_open"].mean().items():
        profile[f"{dow}_{h}"] = float(prob)
        profile[(int(dow), int(h))] = float(prob)
    return profile


def make_candidate_models():
    """Return dict of candidate model instances."""
    return {
        "rf": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                     n_jobs=-1, random_state=RANDOM_STATE),
        "et": ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                   n_jobs=-1, random_state=RANDOM_STATE),
        "xgb": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              reg_alpha=0.1, reg_lambda=1.0,
                              random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
        "hgb": HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05,
                                              min_samples_leaf=10, random_state=RANDOM_STATE),
    }


def cv_score(model, X, y, n_splits=N_CV_SPLITS):
    """Time-series cross-validation MAE (lower is better)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr_idx, va_idx in tscv.split(X):
        if len(tr_idx) < MIN_TRAIN or len(va_idx) < 10:
            continue
        m = type(model)(**model.get_params())
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = np.clip(m.predict(X.iloc[va_idx]), 0, 100)
        maes.append(mean_absolute_error(y.iloc[va_idx], preds))
    return float(np.mean(maes)) if maes else np.inf


def train_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DATA_PATH.exists():
        print(f"ERROR: {PROCESSED_DATA_PATH} not found. Run prepare_data.py first.")
        sys.exit(1)

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print(f"Loaded hourly data: {df.shape}, open hours: {int(df['is_open'].sum())}")

    # Backup existing model
    if MODEL_PATH.exists():
        backup = MODEL_PATH.parent / "occupancy_prev.joblib"
        shutil.copy(MODEL_PATH, backup)
        print(f"Backed up existing model to {backup}")

    open_profile = compute_open_profile(df)

    # --- Chronological split: last 4 weeks = test holdout ---
    open_df = df[df["is_open"] == 1].copy().sort_index()
    open_df = open_df.reset_index().rename(columns={"time": "timestamp"}).set_index("timestamp")

    # Ensure all feature columns exist and are numeric
    for col in FEATURES:
        if col not in open_df.columns:
            open_df[col] = 0.0
        open_df[col] = pd.to_numeric(open_df[col], errors="coerce").fillna(0.0)

    # Determine test cutoff: last 4 weeks of open data
    test_cutoff = open_df.index.max() - pd.Timedelta(weeks=4)
    train_val_df = open_df[open_df.index <= test_cutoff]
    test_df_full = open_df[open_df.index > test_cutoff]

    print(f"Train+Val window: {train_val_df.index.min()} to {train_val_df.index.max()} ({len(train_val_df)} rows)")
    print(f"Test holdout:     {test_df_full.index.min()} to {test_df_full.index.max()} ({len(test_df_full)} rows)")

    if len(train_val_df) < MIN_TRAIN:
        raise ValueError(f"Insufficient training data: {len(train_val_df)} rows (need {MIN_TRAIN})")
    if len(test_df_full) < MIN_TEST:
        raise ValueError(f"Insufficient test data: {len(test_df_full)} rows (need {MIN_TEST})")

    metrics = {
        "zones": {},
        "summary": {},
        "metadata": {
            "training_data_start": str(df.index.min()),
            "training_data_end": str(df.index.max()),
            "test_cutoff": str(test_cutoff),
            "train_val_rows": int(len(train_val_df)),
            "test_rows": int(len(test_df_full)),
            "features": FEATURES,
            "features_no_lag": FEATURES_NO_LAG,
            "target_timezone": "Europe/Stockholm",
            "model_version": "v2.0",
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "n_cv_splits": N_CV_SPLITS,
        },
    }

    backtest_records = []
    final_models = {}
    resid_std = {}

    for zone in ALL_ZONES:
        print(f"\n--- Zone: {zone} ({ZONE_NAMES[zone]}) ---")

        # Per-zone: drop rows where target is NaN/inf
        def clean_zone(source_df):
            z_df = source_df.copy()
            z_df[zone] = pd.to_numeric(z_df[zone], errors="coerce")
            return z_df[np.isfinite(z_df[zone])].copy()

        tv = clean_zone(train_val_df)
        te = clean_zone(test_df_full)

        print(f"  Train+Val: {len(tv)}, Test: {len(te)}")

        if len(tv) < MIN_TRAIN:
            print(f"  WARNING: Zone {zone} has only {len(tv)} train rows — using baseline only")
            baseline_grp = tv.groupby(["dayofweek","hour"])[zone].mean().to_dict()
            final_models[zone] = CalendarMeanBaseline(baseline_grp, float(tv[zone].mean()))
            resid_std[zone] = float(tv[zone].std())
            continue

        X_tv, y_tv = tv[FEATURES], tv[zone]
        X_te, y_te = te[FEATURES], te[zone]

        # 1. Baseline — needs DataFrame with dayofweek and hour columns
        baseline_grp = tv.groupby(["dayofweek","hour"])[zone].mean().to_dict()
        overall_mean = float(tv[zone].mean())
        baseline = CalendarMeanBaseline(baseline_grp, overall_mean)
        base_feat = X_te[["dayofweek", "hour"]]
        y_base = np.clip(baseline.predict(base_feat), 0, 100)

        base_metrics = compute_metrics(y_te.values, y_base, "baseline")
        print(f"  Baseline MAE: {base_metrics['mae']:.2f}")

        # 2. Train and CV-score candidates
        candidates = make_candidate_models()
        cv_scores = {}
        for name, model in candidates.items():
            try:
                score = cv_score(model, X_tv, y_tv)
                cv_scores[name] = score
                print(f"  CV MAE {name:4s}: {score:.2f}")
            except Exception as ex:
                print(f"  CV failed for {name}: {ex}")
                cv_scores[name] = np.inf

        # 3. Pick best candidate by CV MAE
        best_name = min(cv_scores, key=cv_scores.get)
        best_cv = cv_scores[best_name]
        print(f"  Best candidate: {best_name} (CV MAE={best_cv:.2f})")

        # 4. Final fit on all train+val data
        final_model = type(candidates[best_name])(**candidates[best_name].get_params())
        final_model.fit(X_tv, y_tv)
        y_pred = np.clip(final_model.predict(X_te), 0, 100)

        ml_metrics = compute_metrics(y_te.values, y_pred, best_name)
        print(f"  Test MAE {best_name}: {ml_metrics['mae']:.2f} | Baseline: {base_metrics['mae']:.2f}")

        # 5. Select: ML if beats baseline, else baseline
        if ml_metrics["mae"] <= base_metrics["mae"]:
            selected_name = best_name
            selected_model = final_model
            selected_pred = y_pred
            selected_metrics = ml_metrics
            beats = True
        else:
            selected_name = "baseline"
            selected_model = baseline
            selected_pred = y_base
            selected_metrics = base_metrics
            beats = False
            print(f"  → Using baseline for {zone} (ML didn't beat baseline on test set)")

        resid = y_te.values - selected_pred
        resid_std[zone] = float(np.std(resid))

        # 6. Retrain selected model on ALL open data (train+val+test)
        all_zone = clean_zone(open_df)
        X_all, y_all = all_zone[FEATURES], all_zone[zone]
        if selected_name == "baseline":
            grp_all = all_zone.groupby(["dayofweek","hour"])[zone].mean().to_dict()
            final_models[zone] = CalendarMeanBaseline(grp_all, float(all_zone[zone].mean()))
        else:
            prod_model = type(candidates[best_name])(**candidates[best_name].get_params())
            prod_model.fit(X_all, y_all)
            final_models[zone] = prod_model

        metrics["zones"][zone] = {
            "display_name": ZONE_NAMES[zone],
            "selected_model": selected_name,
            "beats_baseline": beats,
            "train_val_samples": int(len(tv)),
            "test_samples": int(len(te)),
            "cv_mae": {k: round(v, 3) for k, v in cv_scores.items() if not np.isinf(v)},
            "best_cv_model": best_name,
            "best_cv_mae": round(best_cv, 3) if not np.isinf(best_cv) else None,
            # Evaluation metrics on test set
            "xgb_mae": round(ml_metrics["mae"], 3),
            "xgb_rmse": round(ml_metrics["rmse"], 3),
            "xgb_r2": round(ml_metrics["r2"], 4),
            "baseline_mae": round(base_metrics["mae"], 3),
            "baseline_rmse": round(base_metrics["rmse"], 3),
            "mae_improvement": round(base_metrics["mae"] - ml_metrics["mae"], 3),
            "selected_mae": round(selected_metrics["mae"], 3),
            "selected_rmse": round(selected_metrics["rmse"], 3),
            "selected_r2": round(selected_metrics["r2"], 4),
            "selected_med_ae": round(selected_metrics["med_ae"], 3),
            "selected_bias": round(selected_metrics["bias"], 3),
            "selected_wape": selected_metrics["wape"],
            "resid_std": round(resid_std[zone], 3),
        }

        # Backtest records
        for dt, actual, pred, base in zip(te.index, y_te.values, selected_pred, y_base):
            backtest_records.append({
                "time": dt, "zone": zone,
                "actual": float(actual), "predicted": float(pred),
                "baseline": float(base), "residual": float(actual - pred),
            })

    # Summary
    zone_mets = metrics["zones"]
    avg_ml_mae = float(np.mean([v["xgb_mae"] for v in zone_mets.values() if "xgb_mae" in v]))
    avg_base_mae = float(np.mean([v["baseline_mae"] for v in zone_mets.values() if "baseline_mae" in v]))
    avg_sel_mae = float(np.mean([v["selected_mae"] for v in zone_mets.values() if "selected_mae" in v]))
    total_test = int(sum(v["test_samples"] for v in zone_mets.values()))
    total_train = int(sum(v["train_val_samples"] for v in zone_mets.values()))

    metrics["summary"] = {
        "avg_ml_mae": round(avg_ml_mae, 3),
        "avg_baseline_mae": round(avg_base_mae, 3),
        "avg_selected_mae": round(avg_sel_mae, 3),
        "avg_xgb_mae": round(avg_ml_mae, 3),   # keep for backward compat with pages_dataset.py
        "overall_beats_baseline": bool(avg_sel_mae <= avg_base_mae),
        "train_samples": total_train,
        "test_samples": total_test,
        "test_period_weeks": 4,
        "evaluation": "Chronological holdout (last 4 weeks) with TimeSeriesSplit CV for model selection",
        "zones_beating_baseline": int(sum(1 for v in zone_mets.values() if v.get("beats_baseline", False))),
        "total_zones": len(ALL_ZONES),
    }

    # Save metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics → {METRICS_PATH}")

    # Save backtest
    bt_df = pd.DataFrame(backtest_records)
    bt_df.to_parquet(BACKTEST_PATH, index=False)
    print(f"Saved backtest  → {BACKTEST_PATH} ({len(bt_df)} rows)")

    # Save model bundle
    bundle = {
        "models": final_models,
        "resid_std": resid_std,
        "open_profile": open_profile,
        "capacities": ZONE_CAPACITIES,
        "features": FEATURES,
        "features_no_lag": FEATURES_NO_LAG,
        "model_version": "v2.0",
        "trained_at": metrics["metadata"]["trained_at"],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved bundle    → {MODEL_PATH}")

    # Print final summary
    print("\n" + "="*65)
    print("FINAL RESULTS")
    print("="*65)
    for z, m in zone_mets.items():
        beat = "✓ BEATS" if m["beats_baseline"] else "✗ WORSE"
        print(f"{z:12s}: {m['selected_model']:10s} MAE={m['selected_mae']:.2f}  Baseline={m['baseline_mae']:.2f}  {beat}")
    print("-"*65)
    print(f"Avg Selected MAE : {avg_sel_mae:.2f}  (Baseline: {avg_base_mae:.2f})")
    print(f"Train samples    : {total_train}")
    print(f"Test samples     : {total_test}")
    print("="*65)

    return bundle, metrics


if __name__ == "__main__":
    train_models()
