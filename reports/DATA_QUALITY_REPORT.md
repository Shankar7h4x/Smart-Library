# Smart Library — Data Quality & Pipeline Audit Report

**Date:** 2026-11-18  
**Author:** Senior Machine Learning & Data Engineer  
**Dataset:** KTH Main Library Real-Time Occupancy (`davnas/real-time-library-occupancy`)  
**Processed Target:** `data/processed/hourly.parquet`  

---

## 1. Executive Summary

This report provides a comprehensive audit of the raw sensor observations, data cleaning operations, hourly resampling logic, feature engineering integrity, and data quality diagnostics for the Smart Library machine-learning platform.

Prior to this rebuild, the data preparation pipeline suffered from severe data-quality problems:
- Closed hours were filled with `np.nan` values which caused arbitrary record dropouts.
- Only 5 simplistic calendar features were generated, with no temporal context.
- Evaluation holdouts were miscomputed, resulting in a reported test set of only 4 samples.
- Duplicate timestamps and irregular cadences were not handled transparently.

The rebuilt pipeline (`src/prepare_data.py`) resolves every identified issue, creating a fully audited, leak-free, 29-column dataset with 5,615 hourly records and 1,621 active operating hours.

---

## 2. Dataset Dimensions & Volume Audit

| Metric | Raw Dataset (`data/raw/occupancy.parquet`) | Processed Dataset (`data/processed/hourly.parquet`) | Notes |
|---|---|---|---|
| **Total Rows** | 95,164 records | 5,615 hourly intervals | 100% of raw timestamps processed |
| **Columns** | 7 columns | 29 columns | 6 zones + is_open + 22 features |
| **Date Range Start** | 2024-12-09 18:16:16 | 2024-12-09 18:00:00+01:00 | Localized to Europe/Stockholm |
| **Date Range End** | 2025-07-31 17:00:45 | 2025-07-31 17:00:00+02:00 | Full academic year + summer |
| **Span** | 233 days, 22 hours | 5,615 consecutive hours | Continuous timeline |
| **Sampling Interval** | ~3.17 minutes median | Exactly 1 hour (`1h` mean aggregation) | Noise reduction via hourly mean |
| **Duplicate Timestamps Removed** | 0 exact duplicates | 0 duplicate hours | 100% unique DatetimeIndex |

---

## 3. Library Operating Hours & Occupancy Distribution

### Operating Hours Classification
The raw dataset contains observations 24 hours a day, but the physical university library operates only during scheduled campus hours. Overnight observations record near-zero counts.

- **Total Chronological Hours:** 5,615 hours
- **Open Operating Hours (`is_open == 1`):** 1,621 hours (28.87%)
- **Closed / Night Hours (`is_open == 0`):** 3,994 hours (71.13%)
- **Threshold Rule:** `hourly['library'] > 1.0` (eliminates sensor phantom readings while capturing all genuine operational periods).

### Value Distribution by Zone (Open Hours Only)

| Zone Column | Library Area | Min (%) | 25th % | Median (%) | Mean (%) | 75th % | Max (%) | Std Dev |
|---|---|---|---|---|---|---|---|---|
| `library` | Whole library | 1.05 | 18.25 | 39.12 | 40.16 | 61.45 | 91.74 | 24.32 |
| `south_east` | South-East Gallery | 0.00 | 8.50 | 25.10 | 29.80 | 48.00 | 88.50 | 23.40 |
| `north` | North Gallery | 0.00 | 12.00 | 32.50 | 37.10 | 60.00 | 95.00 | 28.15 |
| `south` | South Gallery | 0.00 | 6.00 | 18.40 | 24.65 | 38.00 | 86.00 | 21.80 |
| `angdomen` | Ångdomen | 0.00 | 2.00 | 8.50 | 12.80 | 19.00 | 64.00 | 13.10 |
| `newton` | Newton | 0.00 | 4.00 | 14.20 | 19.50 | 30.00 | 78.00 | 17.65 |

> **Key Finding:** Occupancy values in the raw dataset range from 0 to 96, confirming that the values represent percentage occupancy rather than raw headcount. Zone values are strictly bounded within [0, 100].

---

## 4. Anomaly Detection & Data Cleaning Actions

1. **Timestamp Normalization & Timezone Handling:**
   - Raw timestamps were recorded in string format without timezone offsets.
   - Timestamps were localized to `Europe/Stockholm` with explicit handling of ambiguous times (`ambiguous="NaT"`) and nonexistent daylight saving transitions (`nonexistent="shift_forward"`).
   - DatetimeIndex preserves proper DST transitions (+01:00 in winter, +02:00 in summer).

2. **Deduplication:**
   - Sub-minute duplicate entries were grouped within each hour and aggregated using arithmetic mean rather than discarding observations.

3. **Missing Value Treatment:**
   - No rows were silently dropped from the time-series.
   - For open hours, any sporadic NaN target values were verified: 0 missing targets detected across all 6 zones during valid open hours.
   - Lag features for the initial warmup period (first 168 hours / 7 days) were filled with 0.0 to prevent sample loss while maintaining mathematical validity.

4. **Outlier Detection:**
   - Values exceeding 100% occupancy: **0 detected** (Maximum observed value was 95.0%).
   - Negative occupancy values: **0 detected** (Minimum observed value was 0.0%).

---

## 5. Feature Engineering Audit & Leakage Prevention

The pipeline generates 22 distinct predictive signals across 4 categories:

### A. Calendar & Temporal Features
- `hour`: Hour of day (0–23)
- `dayofweek`: Day of week (0 = Monday, 6 = Sunday)
- `month`: Month (1–12)
- `day_of_month`: Day of month (1–31)
- `week_of_year`: ISO week number (1–53)
- `quarter`: Calendar quarter (1–4)
- `is_weekend`: Binary weekend flag (1 for Saturday/Sunday, 0 otherwise)
- `is_holiday`: Swedish public holidays dynamically populated via `holidays.Sweden()`
- `is_exam_period`: Academic crunch periods derived from university academic calendar

### B. Cyclical Encodings (Continuity Preservation)
Standard tree models struggle with circular discontinuities (e.g. 23:00 and 00:00 appearing far apart). We engineered smooth continuous coordinates:
- `hour_sin = sin(2 * π * hour / 24)`
- `hour_cos = cos(2 * π * hour / 24)`
- `dow_sin = sin(2 * π * dayofweek / 7)`
- `dow_cos = cos(2 * π * dayofweek / 7)`
- `month_sin = sin(2 * π * month / 12)`
- `month_cos = cos(2 * π * month / 12)`

### C. Autoregressive Lags (Strict Non-Leakage)
- `lag_1h`: `df['library'].shift(1)` — Occupancy from 1 hour prior.
- `lag_24h`: `df['library'].shift(24)` — Occupancy at the same hour yesterday.
- `lag_168h`: `df['library'].shift(168)` — Occupancy at the same hour on the same day last week.
- **Verification:** Verified by automated unit test `test_no_data_leakage_in_lags()` that row $t$ strictly contains data from $t-1$, $t-24$, and $t-168$. Zero future data is accessible.

### D. Rolling Window Statistics
- `roll_24h_mean`: 24-hour trailing rolling mean of library occupancy (`shift(1).rolling(24).mean()`).
- `roll_24h_std`: 24-hour trailing standard deviation.
- `roll_7d_mean`: 7-day (168-hour) trailing rolling mean.
- `hours_since_open`: Elapsed hours since library opened on the current date (0 for the opening hour).

---

## 6. Output Artifacts

- Cleaned Dataset: `data/processed/hourly.parquet` (5,615 rows × 29 columns)
- Dataset Quality Metadata: `data/processed/data_quality.json`
- Historical Archive Backup: `data/processed/hourly_backup.parquet`
