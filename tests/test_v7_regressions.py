"""Dependency-light regression tests for v7 core correctness."""
from datetime import datetime
from pathlib import Path
import sys
import types

import numpy as np
try:
    import pytest
except ImportError:
    class _PytestStub:
        class mark:
            @staticmethod
            def parametrize(argnames, argvalues):
                def decorator(fn):
                    def wrapper(*args, **kwargs):
                        for val in argvalues:
                            if isinstance(val, (tuple, list)):
                                fn(*val)
                            else:
                                fn(val)
                    return wrapper
                return decorator
    pytest = _PytestStub()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Holidays are optional for these tests; provide a deterministic empty calendar
# so the feature builder can be tested without network or extra packages.
holidays_stub = types.ModuleType("holidays")
holidays_stub.Sweden = lambda years=None: set()
sys.modules.setdefault("holidays", holidays_stub)

from config import ZONE_CAPACITIES
from src.forecast import build_features, free_seats, status
from src.env_settings import write_env_preserving_unmanaged
from src.data_quality import is_fresh_timestamp, valid_occupancy
from datetime import timedelta


def test_build_features_normalizes_aware_datetime_to_stockholm():
    features = build_features(datetime.fromisoformat("2025-01-15T09:00:00+00:00"))
    assert int(features.iloc[0]["hour"]) == 10


def test_build_features_keeps_naive_local_wall_time():
    features = build_features(datetime(2025, 1, 15, 10, 0))
    assert int(features.iloc[0]["hour"]) == 10


def test_unknown_occupancy_is_not_reported_as_free_seats():
    assert free_seats(None, "north") == 0
    assert free_seats(np.nan, "north") == 0


def test_free_seat_estimate_is_bounded():
    assert free_seats(-20, "north") == ZONE_CAPACITIES["north"]
    assert free_seats(120, "north") == 0


def test_status_thresholds_are_consistent():
    assert status(49.9) == "Free"
    assert status(50) == "Busy"
    assert status(80) == "Busy"
    assert status(80.1) == "Full"
    assert status(np.nan) == "Closed"


def test_env_writer_preserves_unmanaged_settings_and_comments(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# custom config\\nDATABASE_URL=sqlite:///app.db\\nAI_PROVIDER=old\\n", encoding="utf-8")
    write_env_preserving_unmanaged(env_file, {"AI_PROVIDER": "openai", "OPENAI_MODEL": "gpt-4o-mini"})
    contents = env_file.read_text(encoding="utf-8")
    assert "# custom config" in contents
    assert "DATABASE_URL=sqlite:///app.db" in contents
    assert "AI_PROVIDER=openai" in contents
    assert "OPENAI_MODEL=gpt-4o-mini" in contents


def test_future_observation_is_not_fresh():
    now = datetime(2026, 1, 1, 12, 0)
    assert not is_fresh_timestamp(now.replace(hour=13), now)


def test_recent_and_stale_observation_freshness():
    now = datetime(2026, 1, 1, 12, 0)
    assert is_fresh_timestamp(now - timedelta(minutes=10), now)
    assert not is_fresh_timestamp(now - timedelta(hours=2), now)


@pytest.mark.parametrize("value", [0, 25.5, 100])
def test_valid_occupancy_accepts_range(value):
    assert valid_occupancy(value)


@pytest.mark.parametrize("value", [-1, 101, float("nan"), float("inf"), "bad", None])
def test_valid_occupancy_rejects_invalid_values(value):
    assert not valid_occupancy(value)


def test_static_preview_server_is_localhost_only():
    source = (ROOT / "start_web.py").read_text(encoding="utf-8")
    assert 'HOST = "127.0.0.1"' in source
    assert 'directory=str(WEB_ROOT)' in source


def test_streamlit_security_protections_enabled():
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "enableCORS = true" in config
    assert "enableXsrfProtection = true" in config


def test_all_six_book_categories_are_present_in_catalogue_and_shelves():
    import json
    catalogue = json.loads((ROOT / "data" / "netflix_books_catalog.json").read_text(encoding="utf-8"))
    expected = {
        "Computer Science & AI", "Agriculture & Biosystems",
        "Mathematics & Statistics", "Physics & Quantum",
        "Economics & Management", "Robotics & Engineering",
    }
    assert expected.issubset(set(catalogue))
    resource_page = (ROOT / "src" / "pages_resources.py").read_text(encoding="utf-8")
    assert "Agriculture & Biosystems" in resource_page


def test_theme_css_uses_shared_tokens_without_inline_color_hacks():
    css_source = (ROOT / "ui.py").read_text(encoding="utf-8")
    assert "--paper:" in css_source
    assert "--surface:" in css_source
    assert "*[style*=" not in css_source


def test_static_preview_only_serves_web_directory():
    source = (ROOT / "start_web.py").read_text(encoding="utf-8")
    assert "WEB_ROOT = ROOT / \"web\"" in source
    assert 'directory=str(WEB_ROOT)' in source
