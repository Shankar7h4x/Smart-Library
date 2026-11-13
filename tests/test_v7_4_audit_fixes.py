"""Regression tests for v7.4 audit and bug fixes across UI, layout, and logic."""

from datetime import date, datetime
from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import MODEL_PATH
from src.chatbot import (
    evaluate_recommendations,
    format_concise_recommendation,
    render_recommendation_card_html,
)
from src.forecast import top_slots
from ui import gauge, zone_card
from ui_extra import stat_card, slot_card, tip_card, hour_strip, legend
import streamlit as st


def test_chatbot_imports_date():
    """Ensure date is imported and available in src/chatbot."""
    import src.chatbot as cb
    assert hasattr(cb, "date")


def test_format_concise_recommendation_handles_group_and_window_preferences():
    """Ensure seat_pref variable is defined and works for non-quiet/collab environments."""
    dummy_top = [{
        "key": "south_east",
        "name": "South-East Gallery",
        "level": "Level 2",
        "occupancy": 35.0,
        "free_seats": 65,
        "capacity": 100,
        "crowd_level": "Low",
        "badge_class": "green",
        "description": "Flexible study zone",
    }]
    # Test group seating with moderate activity
    form_group = {
        "environment": "Moderate activity",
        "seating": "Group seating",
        "arrival_hour": 14,
    }
    res_group = format_concise_recommendation(dummy_top, form_group)
    assert "group seating" in res_group["text"]

    # Test window seating with moderate activity
    form_window = {
        "environment": "Moderate activity",
        "seating": "Near windows",
        "arrival_hour": 14,
    }
    res_window = format_concise_recommendation(dummy_top, form_window)
    assert "window-seat preference" in res_window["text"]


def test_render_recommendation_card_html_handles_none_gracefully():
    """Ensure None or invalid card dictionary returns empty string instead of raising TypeError."""
    assert render_recommendation_card_html(None) == ""
    assert render_recommendation_card_html({}) == ""
    assert render_recommendation_card_html("invalid") == ""


def test_format_concise_recommendation_empty_results():
    """Ensure empty recommendations produce card=None and helpful guidance."""
    res = format_concise_recommendation([], {})
    assert res["card"] is None
    assert "Check the opening schedule" in res["text"]


def test_top_slots_works_for_custom_dates():
    """Ensure top_slots returns candidate slots for target dates."""
    bundle = joblib.load(MODEL_PATH)
    target_d = date(2025, 4, 10)
    slots = top_slots(bundle, target_d, "angdomen", hours=2, days=7, n=3, exam=False)
    assert len(slots) == 3
    for s in slots:
        assert "start_time" in s
        assert "duration_hours" in s
        assert "avg_occupancy" in s
        assert "free_seats" in s


def test_ui_components_render_without_raising():
    """Ensure UI cards and gauges render safely."""
    # Test gauge
    gauge(42.0)
    gauge(None)
    gauge(88.0)

    # Test zone_card
    zone_card("Ångdomen", 25.0, "Free", 45, "angdomen")
    zone_card("Closed Wing", None, "Closed", 0, "library")

    # Test ui_extra
    stat_card("Test Metric", "95%")
    slot_card("Mon Apr 14", "10:00", 2, 35.0, 65)
    tip_card("Pro Tip", "Arrive early for quiet spots", "Tip")
    hour_strip([{"hour": 10, "occupancy": 30.0, "is_open": True, "status": "Free"}])
    legend()


def test_navbar_visibility_and_chrome_reset_styles():
    """Ensure Streamlit header chrome is suppressed and navbar is edge-to-edge sticky."""
    ui_source = (ROOT / "ui.py").read_text(encoding="utf-8")
    # Chrome reset
    assert 'header[data-testid="stHeader"]' in ui_source
    assert "display:none !important;" in ui_source
    # Navbar sticky positioning
    assert 'div[data-testid="stElementContainer"]:has([class*="st-key-main_navigation"])' in ui_source
    assert "z-index:1000 !important;" in ui_source
    assert "width:100vw !important" in ui_source
    assert "margin:0 0 0 calc(50% - 50vw)" in ui_source


def test_chatbot_unified_border_styling():
    """Ensure chatbot container has a unified theme-aware border enclosing the whole panel."""
    ui_source = (ROOT / "ui.py").read_text(encoding="utf-8")
    # Theme tokens
    assert "--chatbot-border:" in ui_source
    assert "--chatbot-glow:" in ui_source
    # Unified border container targeting Streamlit 1.66 and wrapper elements
    assert 'div[data-testid="stVerticalBlock"][class*="st-key-chatbot_panel_"]' in ui_source
    assert "border:2px solid var(--chatbot-border) !important;" in ui_source
    assert "border-radius:18px !important;" in ui_source
    # Messages scrolling within container
    assert 'height:clamp(260px, calc(100vh - 510px), 480px)' in ui_source
    assert "overflow-y:auto !important" in ui_source


def test_google_ai_studio_provider_configured_and_accepted():
    """Ensure Google AI Studio provider and model are configured and verified."""
    from src.ai_service import get_configured_providers, check_provider_api
    providers = get_configured_providers()
    assert len(providers) > 0
    # First provider should be gemini
    p_name, p_key, p_model = providers[0]
    assert p_name == "gemini"
    assert "gemini-2.5-flash" in p_model or "gemini-flash" in p_model
    # Check that Google AI Studio key is accepted
    healthy, detail = check_provider_api(p_name, p_key)
    assert healthy is True
    assert "Google AI Studio" in detail


def test_book_discovery_scroll_token_and_anchor():
    """Ensure Book Discovery defines stable results anchor and auto-scrolling trigger."""
    resources_source = (ROOT / "src" / "pages_resources.py").read_text(encoding="utf-8")
    assert "book-search-results-section" in resources_source
    assert "scroll_to_results_token" in resources_source
    assert "scrollIntoView" in resources_source


def test_consistent_container_box_styling():
    """Ensure work panels, cards, and shelves follow standardized 18px radius and balanced padding."""
    ui_source = (ROOT / "ui.py").read_text(encoding="utf-8")
    assert "border-radius:18px !important;" in ui_source
    assert "st-key-resources_search_panel" in ui_source
    assert "st-key-resources_books_panel" in ui_source
    assert "scroll-margin-top:5.5rem !important;" in ui_source
