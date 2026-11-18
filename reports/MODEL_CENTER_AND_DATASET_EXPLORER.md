# Smart Library — Model Center & Dataset Explorer Architecture Report

**Date:** 2026-11-18  
**Role:** Senior UI/UX Designer, ML Engineer, and Streamlit Developer  
**Status:** IMPLEMENTED, TESTED, & PRODUCTION-READY  

---

## 1. Overview of Changes

Per project requirements, the previously unified "Data & model" page has been permanently decoupled into two dedicated, professional, comprehensive dashboards:

1. **Model Center (`src/pages_model_center.py`):**
   A complete machine learning operational monitoring, evaluation, and tournament comparison dashboard.
2. **Dataset Explorer (`src/pages_dataset_explorer.py`):**
   A complete time-series data analysis, column schema inspector, data quality audit, and paginated data preview platform.

Both pages strictly adhere to the existing forest-green visual theme (`#00D084`, `#10B981`, `#167A4D`, `#06120B`), support both Dark and Light modes, use responsive containers, and connect directly to real project artifacts without fabricated numbers.

---

## 2. Page 1: Model Center Specification

### Section 1: Model Overview
- Compact overview metric cards displaying:
  - System Model Name: *Spatial Ensemble (Per-Zone Multi-Model)*
  - Active Version: *v2.0-prod (Deployed · Healthy)*
  - Deployed Models: *6 Dedicated Zone Models + 1 Automated Rollback Bundle*
  - Prediction Signals: *22 Features (Temporal, Cyclical & Autoregressive)*
  - Training Timestamp: *Strictly recorded UTC timestamp*
  - Training Dataset Span: *2024-12-09 to 2025-07-31 (233 days)*
  - Model Bundle File Size: *Read dynamically from filesystem*
  - Artifact Location: `models/occupancy.joblib`

### Section 2: Model Performance Benchmarks
- Out-of-sample metrics evaluated on the strict **4-week untouched chronological holdout** (1,086 test samples):
  - **Deployed Models MAE:** **13.65 pp**
  - **Historical Baseline MAE:** **27.35 pp**
  - **Error Reduction:** **+50.1% (+13.70 pp improvement)**
  - **Test Samples:** **1,086 observations** (181 hours × 6 zones)
- Interactive per-zone breakdown table:
  - MAE, Baseline MAE, Improvement, RMSE, $R^2$ Score, Prediction Bias, WAPE, and Test Hours.
- Mathematical definitions expander explaining MAE, RMSE, $R^2$, WAPE, Bias, and Baseline lookup logic.

### Section 3: Model Comparison (Algorithm Tournament)
- Interactive zone selector auditing the cross-validation tournament.
- Comprehensive candidate comparison:
  - `XGBoost` (300 trees, learning rate 0.05, L1/L2 regularization)
  - `Extra Trees` (200 trees, random thresholds, variance reduction)
  - `Random Forest` (200 trees, max depth 12)
  - `HistGradientBoosting` (300 iterations, histogram-binned)
  - `Calendar Baseline` (Weekday-hour mean lookup)
- Displays 3-fold `TimeSeriesSplit` cross-validation MAE alongside final holdout test MAE.
- Highlights the winning deployed model with visual badge and color-coded comparison chart.

### Section 4: Model Prediction & Residual Diagnostics
- Interactive area selector (`Whole library`, `South-East Gallery`, `North Gallery`, `South Gallery`, `Ångdomen`, `Newton`).
- Visualizations across 4 interactive tabs:
  1. **Actual vs. Predicted Timeline:** Dual-trace time-series comparing ground-truth sensor readings against model predictions and baseline.
  2. **Residual Distribution Histogram:** Centered error distribution with zero-error reference line.
  3. **Error by Hour of Day:** Grouped bar chart identifying morning ramp, midday peak, and evening wind-down accuracy.
  4. **Daily Error Trend:** Longitudinal stability across the 4-week test window.

### Section 5: Feature Importance & Signal Weights
- Horizontal interactive Plotly bar chart displaying the top 12 most predictive signals.
- Complete Gini / Mean Decrease in Impurity (MDI) importance table with signal category and semantic description.
- Methodological disclaimer distinguishing predictive importance from physical causality.

### Section 6: Model Architecture & Hyperparameters
- Live inspection of model hyperparameters extracted directly from bundle estimators.
- Input schema (22 features) & output schema (continuous prediction, confidence bounds, status thresholds).
- Documented expanding-window validation and leakage prevention regime.

### Section 7: Model Management & Safe Retraining
- Version registry: Active (`models/occupancy.joblib`), Rollback (`models/occupancy_prev.joblib`), Metrics (`models/metrics.json`), Backtest (`models/backtest.parquet`).
- Safe Retraining Trigger: Manual execution button running `src/train.py` in an isolated subprocess with automatic rollback creation and cache refresh. Never triggers automatically on page load.

---

## 3. Page 2: Dataset Explorer Specification

### Section 1: Dataset Overview
- High-level metric cards:
  - Entity: *KTH Library Time-Series (Hourly Cleaned)*
  - Dimensions: *5,615 rows × 29 columns*
  - Operating Hours: *1,621 open hours (28.9%) vs 3,994 closed hours (71.1%)*
  - Coverage Window: *233 days continuous span*
  - File Size & Format: *Snappy-compressed Apache Parquet*
  - Missingness: *0.00% missing values in open hours*
  - Duplicates: *0 duplicate timestamps*
- External provenance links to Hugging Face, Open Library, and KTH Primo.

### Section 2: Project Dataset Inventory
- Inventory table auditing all project data files:
  - `data/raw/occupancy.parquet` (95,164 rows, 7 cols)
  - `data/processed/hourly.parquet` (5,615 rows, 29 cols)
  - `data/processed/hourly_backup.parquet` (Archive backup)
  - `models/backtest.parquet` (1,086 holdout test predictions)
  - `data/processed/data_quality.json` (JSON audit log)
- Lists format, dimensions, file size, last modified date, and local path.

### Section 3: Dataset Schema & Column Inspector
- Full schema table for all 29 columns:
  - Column name, data type, null count, null %, unique count, example value, and semantic description.
- Deep-dive column inspector: dropdown allowing instant statistical profiling of any column (mean, min, max, std, quantiles).

### Section 4: Interactive Dataset Preview (Paginated & Searchable)
- Memory-safe pagination (`.iloc` slicing) supporting page sizes of 15, 25, 50, or 100 rows.
- Instant search filter matching timestamps and dates.
- Operating hours filter (`Open Hours Only`, `Closed Hours Only`, `All Hours`).
- Column multiselect filter to display custom column subsets.
- CSV download button for the active filtered slice.

### Section 5: Data Quality Analysis & Health Diagnostics
- Quality cards verifying:
  - Timestamp continuity (Stockholm timezone + DST shift handling).
  - Value range boundaries ([0.0, 100.0] bounds, 0 negative values).
  - Target completeness (100% complete across open hours).
- Visual donut chart illustrating operating vs closed night schedule breakdown.

### Section 6: Statistical Analysis & Distributions
- Summary statistics table across all 6 library zones (Mean, Median, Std, Min, 25th %, 75th %, Max, IQR).
- Interactive analytical tabs:
  1. **Zone Occupancy Distributions:** Box plots showing density and quartiles across all zones.
  2. **Timeline Trajectory:** Continuous longitudinal plot of whole-library occupancy.
  3. **Diurnal Rhythm:** Mean occupancy profile by hour of the day (08:00–22:00).
  4. **Cross-Zone Correlation Heatmap:** Pearson correlation matrix between library galleries.

### Section 7: Data Preparation Pipeline Workflow
- Stage-by-stage workflow documentation:
  - Stage 1: Raw Ingest (~3.17 min sensor telemetry)
  - Stage 2: Timezone Normalization (Stockholm DST handling)
  - Stage 3: 1-Hour Aggregation (mean resample)
  - Stage 4: Operational Thresholding (`library > 1.0` -> `is_open`)
  - Stage 5: Feature Engineering (cyclical encodings, shifted lags, rolling stats)
  - Stage 6: Parquet Serialization (`hourly.parquet` + `data_quality.json`)

### Section 8: Dataset Download & Export
- Secure export options:
  - Processed Hourly Time-Series (CSV)
  - Out-of-sample Backtest Results (CSV)
  - Data Quality Audit Report (Markdown)
- Complete isolation: No credentials, API keys, or `.env` files are accessible.

---

## 4. Navigation Architecture Update

In `app.py`, the navigation bar now houses 7 distinct pages:
```python
PAGES = {
    "Study planner": ("📅", "Find your best study window"),
    "Occupancy": ("📊", "Explore library activity"),
    "Book discovery": ("📚", "Search the collection"),
    "Library guide": ("🧭", "Spaces, facilities and access"),
    "Model Center": ("🧠", "ML telemetry & model performance"),
    "Dataset Explorer": ("💾", "Data inspection & pipeline audit"),
    "Settings": ("⚙️", "AI provider and app preferences"),
}
```
The navbar column layout has been updated to proportional widths:
`[1.85, 0.95, 0.88, 1.05, 0.95, 0.98, 1.08, 0.82, 0.70]`
Each page maintains the standard `[0.55, 0.45]` two-column layout with the AI assistant panel mounted in the right container.

---

## 5. Automated Verification Results

All 69 automated tests across all 10 test suites executed and passed:
- `test_app_startup_contract.py`: 1 passed
- `test_functions.py`: 1 passed (13 test groups)
- `test_ml_pipeline.py`: 10 passed
- `test_v7_2_ui_and_api_fixes.py`: 7 passed
- `test_v7_3_fixed_chat_and_search.py`: 9 passed
- `test_v7_4_audit_fixes.py`: 11 passed
- `test_v7_5_equal_columns_and_gemini.py`: 4 passed
- `test_v7_followup_ui_fixes.py`: 4 passed
- `test_v7_regressions.py`: 15 passed
- `test_chatbot_and_resources.py`: 7 passed

**Total: 69 passed, 0 failed (100% pass rate).**
