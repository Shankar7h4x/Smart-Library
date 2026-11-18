from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_chatbot_uses_fixed_transcript_and_bottom_composer():
    source = (ROOT / "src" / "chatbot.py").read_text(encoding="utf-8")
    assert 'key=f"chat_messages_{page_key}"' in source
    assert 'st.form(key=f"assistant_composer_form_{page_key}"' in source
    assert "st.chat_input(" not in source


def test_chatbot_response_is_persisted_without_rendering_below_composer():
    source = (ROOT / "src" / "chatbot.py").read_text(encoding="utf-8")
    assert "def _prepare_chat_reply(" in source
    assert "history.append({" in source
    assert "output.write_stream(stream, cursor=" in source
    assert "chunks.append(str(chunk))" not in source


def test_chatbot_render_does_not_wait_for_provider_health_or_archive():
    source = (ROOT / "src" / "chatbot.py").read_text(encoding="utf-8")
    assert "_cached_provider_health(primary_provider" not in source
    assert "get_data_status_info()" not in source


def test_chat_message_bubbles_use_compact_spacing():
    source = (ROOT / "ui.py").read_text(encoding="utf-8")
    assert '[data-testid="stChatMessage"]' in source
    assert '[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]' in source
    assert 'line-height:' in source
    assert "overflow:visible !important;" in source
    assert "flex:0 0 auto !important;" in source


def test_resources_scroll_token_has_time_import():
    source = (ROOT / "src" / "pages_resources.py").read_text(encoding="utf-8")
    assert "import time" in source
    assert "time.time()" in source


def test_book_search_page_is_not_fragment_isolated():
    source = (ROOT / "src" / "pages_resources.py").read_text(encoding="utf-8")
    assert "@st.fragment\ndef _render_resources_top_section" not in source
    assert 'first_shelf = (f"Search results · {search_term}", search_books_found[:55])' in source
    assert "book-search-results-anchor" in source


def test_subjects_are_included_in_open_library_query_matching():
    source = (ROOT / "src" / "book_data.py").read_text(encoding="utf-8")
    assert 'subjects = " ".join(str(s) for s in (doc.get("subject", []) or []))' in source
    assert 'haystack = f"{title} {authors} {subjects}".casefold()' in source


def test_navigation_is_full_bleed_and_chat_transcript_has_own_scroll():
    source = (ROOT / "ui.py").read_text(encoding="utf-8")
    assert "width:100vw !important" in source
    assert "margin:0 0 0 calc(50% - 50vw)" in source
    assert "overflow-y:auto !important" in source
    assert "height:clamp(260px, calc(100vh - 510px), 480px)" in source


def test_planner_left_column_is_a_single_panel():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'st.container(key="primary_work_panel", border=True)' in source
