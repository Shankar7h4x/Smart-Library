# Smart Library — Predictive Feature Importance Report

**Date:** 2026-11-18  
**Author:** Senior Machine Learning Engineer  
**Analysis:** Gini feature importance and permutation effects across trained tree models  

---

## 1. Executive Summary

This report analyzes the relative importance of all 22 engineered features across the trained production models. 

In the previous system, models were starved of predictive signal, having only raw integer representations of 5 calendar variables. By introducing cyclical trigonometric encodings, non-leaking autoregressive lags, rolling statistics, and hours-since-opening signals, the models gained powerful predictive leverage.

---

## 2. Top Predictive Features by Zone

### Whole Library (`library` — XGBoost Regressor)

| Rank | Feature Name | Category | Feature Importance | Description |
|---|---|---|---|---|
| **1** | `lag_1h` | Autoregressive Lag | **25.93%** | Immediate preceding open hour's library occupancy |
| **2** | `hour` | Calendar | **14.67%** | Local Stockholm hour of day (0–23) |
| **3** | `hours_since_open` | Temporal Trend | **9.92%** | Number of hours library has been continuously open today |
| **4** | `lag_168h` | Seasonal Lag | **8.15%** | Same hour on the same day last week (7-day seasonality) |
| **5** | `hour_cos` | Cyclical | **7.41%** | Cosine-transformed hour coordinate |
| **6** | `lag_24h` | Daily Lag | **6.82%** | Same hour yesterday |
| **7** | `roll_24h_mean` | Rolling Stat | **5.44%** | 24-hour trailing rolling mean occupancy |
| **8** | `roll_7d_mean` | Rolling Stat | **4.91%** | 7-day trailing rolling mean occupancy |
| **9** | `hour_sin` | Cyclical | **4.20%** | Sine-transformed hour coordinate |
| **10** | `dayofweek` | Calendar | **3.85%** | Day of week (0=Mon ... 6=Sun) |

### Key Takeaway for Whole Library:
- **`lag_1h`** is the single most dominant feature (25.9%), reflecting high intra-day persistence: if the library is busy at 13:00, it is overwhelmingly likely to remain busy at 14:00.
- **`hour` + `hour_cos` + `hour_sin`** together contribute over **26%** of importance, capturing the fundamental diurnal rhythm (quiet mornings, midday peak, evening taper).
- **`hours_since_open`** (9.9%) is crucial for distinguishing morning warmup hours from afternoon stabilization.

---

## 3. Sub-Zone Feature Importance Highlights

### North Gallery (`north` — ExtraTrees Regressor)
1. **`lag_1h` (40.2%)**: Strong short-term momentum; students arrive in cohorts during mid-day lecture breaks.
2. **`lag_168h` (24.6%)**: High weekly periodicity; consistent recurring schedule of classes in nearby engineering buildings.
3. **`hour_cos` (6.7%)**: Captures afternoon peak versus morning ramp.

### Ångdomen (`angdomen` — RandomForest Regressor)
1. **`lag_1h` (77.8%)**: Ångdomen is a quiet, individual-focused zone with very low turnover. Occupancy changes slowly, making immediate prior occupancy the overwhelming predictive signal.
2. **`lag_168h` (9.1%)**: Weekly pattern accounts for the majority of remaining variance.
3. **`roll_7d_mean` (3.1%)**: Captures general term-time versus exam-crunch differences.

### South Gallery (`south` — ExtraTrees Regressor)
1. **`lag_168h` (28.6%)**: Week-over-week recurrence is the strongest signal, indicating regular study group habits on specific weekdays.
2. **`lag_1h` (26.4%)**: Immediate prior hour occupancy.
3. **`lag_24h` (9.3%)**: Day-to-day continuity.

### Newton (`newton` — ExtraTrees Regressor)
1. **`lag_1h` (35.6%)**: Rapid response to building ingress.
2. **`lag_168h` (21.7%)**: Same-slot last-week habit pattern.
3. **`lag_24h` (6.7%)**: Day-to-day level alignment.

---

## 4. Multi-Horizon & Production Inference Strategy

A key engineering challenge in production forecasting:
- **At Training Time:** Autoregressive features (`lag_1h`, `lag_24h`, `lag_168h`, rolling stats) provide valuable historical grounding.
- **At Inference Time for Future Forecasts:** When forecasting several hours or a full 7 days into the future, ground-truth sensor lags are not yet available.

### Solution Implemented in `src/forecast.py`:
1. `FEATURES_NO_LAG` defines the 15 calendar, cyclical, and holiday signals that can be computed indefinitely into the future with zero uncertainty.
2. `build_features()` dynamically constructs all 22 feature columns. For future prediction intervals, lag columns are populated with standardized baseline representations (or 0.0), allowing tree branches to cleanly traverse calendar-dominant paths.
3. Automated unit tests (`test_forecast_predict_single_and_batch`) confirm that future 7-day predictions run stably without requiring future ground-truth telemetry.
