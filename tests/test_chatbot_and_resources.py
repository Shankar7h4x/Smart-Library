"""Automated test script for Global Chatbot, Toy Mascot, Page Introductions, and Resources."""

from datetime import date, datetime
from pathlib import Path
import sys
from threading import Event
from zoneinfo import ZoneInfo
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_PATH
from src.books import kth_primo_search_url, search_books
from src.book_data import get_curated_books
from src.chatbot import (
    PAGE_INTRODUCTIONS,
    PAGE_SUGGESTIONS,
    evaluate_recommendations,
    format_concise_recommendation,
    handle_deterministic_query,
    render_toy_mascot,
)
from src import live_data
from src.ai_service import get_configured_providers

KTH_TZ = ZoneInfo("Europe/Stockholm")


def test_page_introductions_and_suggestions():
    pages = ["Home", "Overview", "Resources", "Library Info"]
    for p in pages:
        assert p in PAGE_INTRODUCTIONS, f"Missing intro for {p}"
        assert len(PAGE_INTRODUCTIONS[p]) > 20, f"Intro too short for {p}"
        assert p in PAGE_SUGGESTIONS, f"Missing suggestions for {p}"
        assert len(PAGE_SUGGESTIONS[p]) >= 3, f"Too few suggestions for {p}"


def test_toy_mascot_markup():
    for state in ["idle", "thinking", "excited", "welcome"]:
        mascot_markup = render_toy_mascot(state)
        assert f"toy-mascot-wrapper {state}" in mascot_markup
        assert "mascot-screen" in mascot_markup
        assert "mascot-eye" in mascot_markup
        assert "mascot-antenna" in mascot_markup


def test_concise_recommendations():
    bundle = joblib.load(MODEL_PATH)
    form_data = {
        "date": date(2026, 4, 15),
        "arrival_hour": 14,
        "duration": 3,
        "duration_str": "3 hours",
        "environment": "Quiet environment",
        "seating": "Individual seating",
        "purpose": "Exam preparation",
        "crowd": "Avoid crowded areas",
        "notes": "Need quiet desk",
        "exam": True,
    }
    recs = evaluate_recommendations(bundle, form_data)
    assert len(recs) == 5, f"Expected 5 zone recommendations, got {len(recs)}"
    assert recs[0]["score"] >= recs[-1]["score"], "Recommendations not ranked by score"
    concise_rec = format_concise_recommendation(recs, form_data)
    assert "Based on your preferences, I recommend:" in concise_rec["text"]
    assert "**Why?**" in concise_rec["text"]
    assert "**Alternative:**" in concise_rec["text"]
    word_count = len(concise_rec["text"].split())
    assert word_count < 150, f"Recommendation too verbose: {word_count} words"


def test_chatbot_queries_with_ml_data():
    bundle = joblib.load(MODEL_PATH)
    test_queries = [
        ("Which area is least crowded?", "Overview"),
        ("How many seats are available?", "Home"),
        ("Find books about AI", "Resources"),
        ("Where is the library?", "Library Info"),
    ]
    for q, p in test_queries:
        ans = handle_deterministic_query(q, p, bundle)
        assert ans and "text" in ans and len(ans["text"]) > 20, f"Empty response for '{q}'"


def test_resources_catalog_and_search():
    books = get_curated_books()
    assert len(books) >= 50, f"Expected 50+ books, got {len(books)}"

    primo_url = kth_primo_search_url("Algorithms and Data Structures")
    assert "kth-ch.primo.exlibrisgroup.com" in primo_url
    assert "Algorithms" in primo_url

    search_results, err = search_books("Physics")
    assert len(search_results) > 0, "No books returned for 'Physics'"


def test_data_status_refresh_does_not_block_page_render(monkeypatch):
    started = Event()
    release = Event()
    finished = Event()

    def slow_archive_fetch():
        started.set()
        release.wait(timeout=2)
        finished.set()
        return None, "archive unavailable"

    monkeypatch.setattr(live_data, "_fetch_latest_occupancy", slow_archive_fetch)
    with live_data._status_lock:
        live_data._status_snapshot = None
        live_data._status_error = None
        live_data._status_refreshing = False
        live_data._status_checked_at = 0.0

    data_status = live_data.get_data_status_info()
    assert data_status["last_updated"] == "Checking archive"
    assert "background" in data_status["status_text"]
    assert started.wait(timeout=1)
    release.set()
    assert finished.wait(timeout=1)


def test_ai_provider_configuration_manager():
    providers = get_configured_providers()
    assert isinstance(providers, list)
