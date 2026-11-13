from pathlib import Path
from src.ai_service import check_provider_api, get_configured_providers, DEFAULT_GEMINI_MODEL

def test_gemini_provider_configuration_and_default_model():
    assert DEFAULT_GEMINI_MODEL == "gemini-2.5-flash"
    providers = get_configured_providers()
    assert any(p[0] == "gemini" for p in providers)
    # Key check should gracefully report status
    healthy, detail = check_provider_api("gemini", "AQ.invalid_dummy_key_for_testing")
    assert healthy is False
    assert "AQ.invalid" not in detail  # Never leak raw key in error detail

def test_six_pages_standardized_two_column_structure():
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    resources_source = (Path(__file__).resolve().parents[1] / "src" / "pages_resources.py").read_text(encoding="utf-8")
    
    # Verify 0.55 / 0.45 split used in app.py and resources page
    assert 'st.columns([0.55, 0.45]' in app_source
    assert 'st.columns([0.55, 0.45]' in resources_source
    # Verify brand lockup in navbar
    assert 'nav-brand-lockup' in app_source
    # Verify chatbot panel invocation on all pages
    assert 'render_chatbot_panel(bundle, "Home")' in app_source
    assert 'render_chatbot_panel(bundle, "Overview")' in app_source
    assert 'render_chatbot_panel(bundle, "Library Info")' in app_source
    assert 'render_chatbot_panel(bundle, "Dataset")' in app_source
    assert 'render_chatbot_panel(bundle, "Settings")' in app_source
    assert 'render_chatbot_panel(bundle, "Resources")' in resources_source

def test_css_equal_column_heights_and_natural_scrolling():
    ui_source = (Path(__file__).resolve().parents[1] / "ui.py").read_text(encoding="utf-8")
    
    # Column stretch in CSS Grid / Flexbox
    assert "align-items:stretch !important;" in ui_source
    assert "align-self:stretch !important;" in ui_source
    assert "gap:0 !important;" in ui_source
    assert "border-right:none !important;" in ui_source
    
    # Removal of arbitrary max-height and inner scrolling on primary_work_panel
    assert "max-height:none !important;" in ui_source
    assert "overflow:visible !important;" in ui_source
    
    # Chatbot panel full border and pinned composer
    assert "border:2px solid var(--chatbot-border) !important;" in ui_source
    assert "margin-top:auto !important;" in ui_source
    assert "min-height:max(580px, calc(100vh - 90px)) !important;" in ui_source
    assert "gap:.75rem !important;" in ui_source
    assert "calc(55% - .375rem)" in ui_source
    assert "calc(45% - .375rem)" in ui_source
    assert '[data-testid="stLayoutWrapper"]:has([class*="st-key-chat_messages_"])' in ui_source
    assert "flex:1 1 0 !important;" in ui_source

def test_book_discovery_scroll_target_rendered():
    resources_source = (Path(__file__).resolve().parents[1] / "src" / "pages_resources.py").read_text(encoding="utf-8")
    
    assert 'id="book-search-results-section"' in resources_source
    assert 'scrollIntoView' in resources_source
    assert 'scroll_to_results_token' in resources_source
