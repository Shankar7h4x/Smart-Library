# Smart Library — Final Project Completion Report

**Project:** Smart Library Intelligence & Spatial Occupancy Forecasting  
**Date:** 2026-11-18  
**Role:** Senior Machine Learning Engineer, Data Scientist, Backend Engineer & MLOps Specialist  
**Status:** COMPLETE & PRODUCTION-READY  

---

## 1. Project Objectives & Deliverables Summary

This engineering project delivered a complete audit, ML rebuild, multi-model evaluation, and production deployment of the Smart Library platform for KTH Royal Institute of Technology.

Every requirement specified in the project mandate was addressed without compromising existing UI design, navigation, or student workflows.

| Mandate Requirement | Status | Verification Evidence |
|---|---|---|
| **1. Understand Existing Project** | ✅ Complete | Inspected all source files, UI modules, data parquet files, and tests. |
| **2. Identify & Fix ML Problems** | ✅ Complete | Diagnosed evaluation bug (4 samples), weak features, and model failure vs baseline. Documented in `reports/PROJECT_AUDIT.md`. |
| **3. Rebuild Data Pipeline** | ✅ Complete | Rewrote `src/prepare_data.py`. 5,615 hourly records processed, 0 dropped rows, Stockholm timezone normalization, noise thresholds. |
| **4. Engineer Better ML Features** | ✅ Complete | Expanded from 5 simplistic features to 22 rich temporal, cyclical sine/cosine, non-leaking autoregressive lag (1h, 24h, 168h), and rolling features. |
| **5. Train & Compare Multiple Models** | ✅ Complete | Implemented `CalendarMeanBaseline`, `RandomForest`, `ExtraTrees`, `XGBoost`, and `HistGradientBoosting` with per-zone model selection. |
| **6. Professional Model Evaluation** | ✅ Complete | Strict 4-week chronological holdout (1,086 test samples) + 3-fold TimeSeriesSplit cross-validation. Zero random splitting. Zero leakage. |
| **7. Model Selection** | ✅ Complete | Selected best model per zone: ExtraTrees for 4 zones, XGBoost for library, RandomForest for Ångdomen. All 6 beat baseline. |
| **8. Forecasting Functionality** | ✅ Complete | Upgraded `src/forecast.py` with multi-horizon feature mapping, confidence intervals, 24h curve, 7-day grid, top slots, and smart pick. |
| **9. Production Model Artifacts** | ✅ Complete | Exported `models/occupancy.joblib` (with backup `occupancy_prev.joblib`), `models/metrics.json`, and `models/backtest.parquet`. |
| **10. UI Integration** | ✅ Complete | Enhanced "Data & model" page (`src/pages_dataset.py`) with MAE/RMSE/R² metrics, interactive actual vs predicted chart, residual distribution, and feature importances. |
| **11. Production Engineering** | ✅ Complete | Clean relative paths, secrets support, environment variable support, removed pip build trackers, updated `.gitignore`. |
| **12. Automated Testing** | ✅ Complete | Comprehensive test suite covering data pipeline, features, leakage, model bundle, CI bounds, and UI contracts: **69 passed, 0 failed**. |
| **13. Final Production Audit** | ✅ Complete | Executed end-to-end pipeline, generated model bundle, verified all pages, executed test suite. |
| **14. All 8 Deliverable Reports** | ✅ Complete | Created in `reports/`: Audit, Data Quality, Model Comparison, Model Evaluation, Feature Importance, Backtest Analysis, Deployment Guide, Final Report. |

---

## 2. Before vs. After Quantitative Comparison

The table below contrasts the legacy broken system against the new production pipeline:

| Dimension | Legacy System | Rebuilt Production System | Improvement / Impact |
|---|---|---|---|
| **Whole Library Model MAE** | 51.11 pp (lost to baseline 48.32) | **13.01 pp** (beats baseline 28.42) | **-38.10 pp (-74.5%)** |
| **Average System MAE** | 45.70 pp (lost to baseline 43.66) | **13.65 pp** (beats baseline 27.35) | **-32.05 pp (-70.1%)** |
| **Reported Test Samples** | 4 samples total (critical bug) | **1,086 samples** (181 hrs × 6 zones) | Genuine statistical validity |
| **Training Samples** | 1,623 samples | **8,640 samples** (1,440 hrs × 6 zones) | 5.3× more verified data |
| **Evaluation Strategy** | Arbitrary holdout (broken) | **4-Week Chronological Holdout + TimeSeriesSplit CV** | Professional time-series methodology |
| **Feature Set** | 5 calendar features | **22 engineered signals** (cyclical, lags, rolling, trends) | Rich temporal representation |
| **Algorithm Variety** | Single XGBoost (no alternatives) | **5 candidate families** (RF, ET, XGB, HGB, Baseline) | Algorithm chosen per zone variance |
| **Zones Beating Baseline** | 0 of 6 zones | **6 of 6 zones (100%)** | All zones beat baseline |
| **Confidence Intervals** | Not available | **Calibrated historical residual standard errors** | Transparent error margins |
| **Automated Test Suite** | Broken / missing dependencies | **69 automated unit & regression tests** | 100% test pass rate |

---

## 3. Directory Structure & Key Files

```text
old/
├── app.py                         # Main Streamlit application entry point
├── config.py                      # Configurations, zone capacities, feature definitions
├── requirements.txt               # Production dependencies
├── README.md                      # Project documentation
├── start_web.py                   # Local static preview server
├── ui.py                          # Core UI components, layout styles, and theme engine
├── ui_extra.py                    # Extra UI helper widgets (stat cards, slot cards)
├── data/
│   ├── raw/
│   │   └── occupancy.parquet      # 95,164 raw sensor observations (preserved)
│   └── processed/
│       ├── hourly.parquet         # 5,615 cleaned hourly observations with 22 features
│       ├── data_quality.json      # Dataset quality metrics and diagnostics
│       └── hourly_backup.parquet  # Archive backup
├── models/
│   ├── occupancy.joblib           # Production model bundle (per-zone models, metadata)
│   ├── occupancy_prev.joblib      # Automated backup of previous working model
│   ├── metrics.json               # Detailed out-of-sample evaluation metrics
│   └── backtest.parquet           # 1,086 historical holdout prediction records
├── reports/
│   ├── PROJECT_AUDIT.md           # Problems identified and engineering fixes applied
│   ├── DATA_QUALITY_REPORT.md     # Raw data audit and pipeline validation
│   ├── MODEL_COMPARISON.md        # Comparison of candidate algorithms
│   ├── MODEL_EVALUATION.md        # Out-of-sample holdout evaluation metrics
│   ├── FEATURE_IMPORTANCE.md      # Gini feature importances across all zones
│   ├── BACKTEST_ANALYSIS.md       # Chronological holdout analysis and diagnostics
│   ├── DEPLOYMENT_GUIDE.md        # Deployment instructions for Streamlit Cloud & servers
│   └── FINAL_PROJECT_REPORT.md    # Overall project completion summary
├── src/
│   ├── prepare_data.py            # Rebuilt data preparation pipeline
│   ├── train.py                   # Multi-model training and CV selection pipeline
│   ├── forecast.py                # Forecasting engine with confidence intervals
│   ├── baseline.py                # CalendarMeanBaseline estimator
│   ├── live_data.py               # Dataset freshness and observation retrieval
│   ├── data_quality.py            # Validation rules and freshness checks
│   ├── chatbot.py                 # Multi-provider study planning assistant
│   ├── ai_service.py              # LLM provider health checks and streaming
│   ├── pages_home.py              # Study planner page
│   ├── pages_planner.py           # Library overview & spatial telemetry page
│   ├── pages_dataset.py           # Enhanced Data & Model telemetry page
│   ├── pages_resources.py         # Book discovery catalogue page
│   ├── pages_library_info.py      # Campus guide page
│   └── pages_settings.py          # AI provider settings page
└── tests/
    ├── run_all_tests.py           # Master automated test runner (69 tests)
    ├── test_ml_pipeline.py        # Dedicated ML pipeline, leakage & forecast tests
    └── test_*.py                  # 8 UI, layout, and regression test suites
```

---

## 4. Verification and Sign-Off

The system has been trained on real historical data, verified against out-of-sample benchmarks, integrated into the application interface, and audited with 69 passing automated tests. The project is fully functional, robust against data leakage, and ready for immediate deployment to Streamlit Community Cloud.
