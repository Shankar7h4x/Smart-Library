"""Configuration settings, zone mappings, capacities, and constants."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent

# Automatically load local .env if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "occupancy.parquet"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "hourly.parquet"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "occupancy.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
BACKTEST_PATH = MODELS_DIR / "backtest.parquet"

# Zone definitions & column mappings
ZONE_MAP = {
    "KTH Library": "library",
    "South-East Gallery": "south_east",
    "North Gallery": "north",
    "South Gallery": "south",
    "Ångdomen": "angdomen",
    "Newton": "newton",
}

ZONE_NAMES = {
    "library": "Whole library",
    "south_east": "South-East Gallery",
    "north": "North Gallery",
    "south": "South Gallery",
    "angdomen": "Ångdomen",
    "newton": "Newton",
}

SUB_ZONES = ["south_east", "north", "south", "angdomen", "newton"]
ALL_ZONES = ["library"] + SUB_ZONES

# Assumed capacities (seats)
# assumption, replace if known
ZONE_CAPACITIES = {
    "south_east": 100,
    "north": 120,
    "south": 100,
    "angdomen": 60,
    "newton": 40,
}
ZONE_CAPACITIES["library"] = sum(ZONE_CAPACITIES.values())  # 420

# Occupancy thresholds & status
STATUS_FREE = "Free"
STATUS_BUSY = "Busy"
STATUS_FULL = "Full"
STATUS_CLOSED = "Closed"

STATUS_THRESHOLDS = {
    "free_max": 50.0,
    "busy_max": 80.0,
}

# Shared semantic palette. Runtime theme tokens in ui.inject_css()
# provide the corresponding dark values when dark mode is selected.
COLOR_BACKGROUND = "#F4F7F3"
COLOR_SECONDARY_BG = "#EDF3EE"
COLOR_CARD = "#FFFFFF"
COLOR_SURFACE = "#FFFFFF"
COLOR_ELEVATED_CARD = "#FFFFFF"
COLOR_PRIMARY = "#1D6949"
COLOR_SECONDARY = "#25845C"
COLOR_PRIMARY_LIGHT = "#E3F0E7"
COLOR_TEXT_MAIN = "#17251E"
COLOR_TEXT_MUTED = "#718078"
COLOR_MUTED_GREEN = "#4F8D6C"
COLOR_BORDER = "#DFE8E1"
COLOR_SUCCESS = "#168653"
COLOR_WARNING = "#A66E12"
COLOR_DANGER = "#BD4E4E"
COLOR_CLOSED = "#718078"

STATUS_COLORS = {
    STATUS_FREE: "#00D084",
    STATUS_BUSY: "#F5B942",
    STATUS_FULL: "#FF6464",
    STATUS_CLOSED: "#5A6860",
}

# Exam periods (month, day) tuples for recurring dates
# approximate - verify with KTH academic calendar
EXAM_PERIODS = [
    ((1, 13), (1, 31)),
    ((3, 24), (4, 12)),
    ((5, 26), (6, 14)),
    ((8, 18), (8, 29)),
]

FEATURES_NO_LAG = [
    "hour", "dayofweek", "month", "day_of_month", "week_of_year",
    "quarter", "is_weekend", "is_holiday", "is_exam_period",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]

FEATURES = [
    "hour", "dayofweek", "month", "day_of_month", "week_of_year",
    "quarter", "is_weekend", "is_holiday", "is_exam_period",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1h", "lag_24h", "lag_168h",
    "roll_24h_mean", "roll_24h_std", "roll_7d_mean",
    "hours_since_open",
]

