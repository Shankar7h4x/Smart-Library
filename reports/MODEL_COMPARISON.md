# Smart Library — Model Comparison & Selection Report

**Date:** 2026-11-18  
**Author:** Senior Machine Learning Engineer  
**Task:** Comparative evaluation of candidate algorithms for zone-level library occupancy forecasting  

---

## 1. Executive Summary

Previous iterations of the Smart Library relied exclusively on a single `XGBRegressor` trained with 5 calendar features. That setup achieved an average test MAE of 45.70 percentage points, performing worse than the simple weekday-hour mean baseline (43.66 pp).

In this complete rebuild, we evaluated five candidate algorithm families using strict time-series cross-validation (`TimeSeriesSplit`, $k=3$):
1. **Historical Calendar Mean Baseline** (`CalendarMeanBaseline`)
2. **Random Forest Regressor** (`RandomForestRegressor`, 200 trees)
3. **Extra Trees Regressor** (`ExtraTreesRegressor`, 200 trees)
4. **XGBoost Regressor** (`XGBRegressor`, 300 trees, learning rate 0.05, regularization)
5. **Histogram-based Gradient Boosting** (`HistGradientBoostingRegressor`, 300 iterations)

**Outcome:** Every library zone now selects an optimal model that substantially beats the historical baseline. The average selected model MAE dropped from **27.35 pp** (baseline) to **13.65 pp** (a 50.1% overall error reduction).

---

## 2. Model Selection Protocol

To avoid data leakage and prevent lookahead bias:
1. **Chronological Partitioning:**
   - **Train + Validation Window:** 2024-12-09 18:00 to 2025-05-26 13:00 (1,440 open hours)
   - **Held-Out Test Period:** 2025-05-26 14:00 to 2025-06-23 13:00 (181 open hours, 4 weeks)
2. **Time-Series Cross-Validation:**
   - Evaluated candidate models across 3 expanding chronological folds within the training window.
   - Selected candidate algorithm with the lowest mean absolute error across folds.
3. **Baseline Superiority Guardrail:**
   - If the best ML candidate fails to outperform the historical calendar baseline on the untouched holdout period, the zone falls back to the baseline model.
   - All 6 zones successfully outperformed the baseline in this cycle.

---

## 3. Cross-Validation Results by Algorithm (MAE on Train+Val)

The table below reports the mean absolute error across cross-validation folds:

| Library Area | Zone ID | Random Forest | Extra Trees | XGBoost | HistGradientBoost | Selected Candidate |
|---|---|---|---|---|---|---|
| **Whole Library** | `library` | 3.952 | 3.976 | **3.703** | 3.942 | **XGBoost** |
| **South-East Gallery** | `south_east` | 19.239 | **19.118** | 19.296 | 19.775 | **Extra Trees** |
| **North Gallery** | `north` | 7.172 | **6.952** | 7.897 | 8.043 | **Extra Trees** |
| **South Gallery** | `south` | 8.926 | **8.664** | 9.178 | 9.418 | **Extra Trees** |
| **Ångdomen** | `angdomen` | **4.662** | 4.676 | 4.857 | 4.916 | **Random Forest** |
| **Newton** | `newton` | 12.537 | **12.136** | 13.233 | 13.426 | **Extra Trees** |

---

## 4. Final Holdout Evaluation (Untouched 4-Week Test Set)

Performance measured on 1,086 test observations (181 hours × 6 zones):

| Zone | Selected Model | Model MAE (pp) | Baseline MAE (pp) | Improvement (pp) | Relative Error Reduction | Status |
|---|---|---|---|---|---|---|
| **Whole library** | `XGBoost` | **13.01** | 28.42 | +15.41 | **54.2%** | ✅ BEATS |
| **South-East Gallery** | `Extra Trees` | **17.26** | 25.44 | +8.18 | **32.1%** | ✅ BEATS |
| **North Gallery** | `Extra Trees` | **16.49** | 38.09 | +21.60 | **56.7%** | ✅ BEATS |
| **South Gallery** | `Extra Trees` | **15.96** | 29.28 | +13.31 | **45.5%** | ✅ BEATS |
| **Ångdomen** | `Random Forest` | **4.07** | 15.18 | +11.11 | **73.2%** | ✅ BEATS |
| **Newton** | `Extra Trees` | **15.10** | 27.70 | +12.59 | **45.5%** | ✅ BEATS |
| **OVERALL AVERAGE** | — | **13.65** | **27.35** | **+13.70** | **50.1%** | ✅ **BEATS ALL** |

---

## 5. Algorithmic Insights & Discussion

1. **Why Extra Trees Won in Most Sub-Zones:**
   - Sub-zone sensor feeds exhibit higher localized noise (e.g. students moving between Newton and the South Gallery).
   - Extra Trees uses random thresholds for splitting rather than optimal search, yielding higher bias but significantly lower variance on noisy continuous signals.

2. **Why XGBoost Won for the Whole Library:**
   - Whole-library aggregate occupancy is smoother and exhibits stronger non-linear interactions with time and calendar rhythms.
   - XGBoost's gradient boosting and L1/L2 regularization (`reg_alpha=0.1`, `reg_lambda=1.0`) captured complex seasonal patterns effectively without overfitting.

3. **Why Random Forest Won for Ångdomen:**
   - Ångdomen is the quietest sanctuary zone (mean occupancy ~12.8%). Its distribution has a strong floor effect at 0–5%.
   - Standard Random Forest bagging provided well-calibrated bounds, producing an outstanding test MAE of **4.07 pp** (down from 15.18 pp baseline).

4. **Multi-Model Per-Zone Architecture:**
   - Rather than forcing all zones into a single monolithic model or forcing uniform algorithm selection, deploying distinct per-zone models tailored to zone variance achieved optimal accuracy across the entire facility.
