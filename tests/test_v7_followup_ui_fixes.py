from pathlib import Path

from src.ai_service import check_provider_api
from src.env_settings import write_env_preserving_unmanaged


def test_provider_health_marks_missing_key_unhealthy():
    healthy, detail = check_provider_api("openai", "")
    assert healthy is False
    assert detail == "API key missing"


def test_env_writer_preserves_unmanaged_lines_and_real_newlines(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("# keep this comment\nOTHER_SETTING=preserve-me\nAI_PROVIDER=gemini\n", encoding="utf-8")
    write_env_preserving_unmanaged(env_path, {"AI_PROVIDER": "openai", "OPENAI_MODEL": "gpt-4o-mini"})
    content = env_path.read_text(encoding="utf-8")
    assert "\\n" not in content
    assert "# keep this comment" in content
    assert "OTHER_SETTING=preserve-me" in content
    assert "AI_PROVIDER=openai" in content
    assert "OPENAI_MODEL=gpt-4o-mini" in content


def test_catalogue_status_labels_are_explicitly_unverified():
    source = Path(__file__).resolve().parents[1] / "src" / "pages_resources.py"
    content = source.read_text(encoding="utf-8")
    assert "Dataset tag · Checked out (unverified)" in content
    assert "Verify live holdings" in content


def test_resources_page_has_search_scroll_anchor():
    source = Path(__file__).resolve().parents[1] / "src" / "pages_resources.py"
    assert 'book-search-results-anchor' in source.read_text(encoding="utf-8")
