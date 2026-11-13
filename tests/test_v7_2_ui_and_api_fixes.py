from pathlib import Path
from types import SimpleNamespace

from src import ai_service
from src.env_settings import write_env_preserving_unmanaged


def test_chatbot_render_does_not_probe_provider_health():
    source = (Path(__file__).resolve().parents[1] / "src" / "chatbot.py").read_text(encoding="utf-8")
    render_source = source[source.index("def render_chatbot_panel("):]
    assert "_cached_provider_health(" not in render_source
    assert 'provider_text = f"{primary_provider.capitalize()} configured' in render_source


def test_provider_health_accepts_valid_key(monkeypatch):
    monkeypatch.setattr(ai_service.requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200))
    healthy, detail = ai_service.check_provider_api("openai", "test-key")
    assert healthy is True
    assert detail == "API key accepted"


def test_provider_health_rejects_invalid_key(monkeypatch):
    monkeypatch.setattr(ai_service.requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=401))
    healthy, detail = ai_service.check_provider_api("openai", "bad-key")
    assert healthy is False
    assert "rejected" in detail


def test_env_writer_persists_keys_and_keeps_other_settings(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# preserve comment\nOTHER=value\n", encoding="utf-8")
    values = {
        "AI_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-secret",
        "OPENAI_MODEL": "gpt-4o-mini",
    }
    write_env_preserving_unmanaged(env_file, values)
    saved = env_file.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=test-secret" in saved
    assert "AI_PROVIDER=openai" in saved
    assert "OTHER=value" in saved
    assert "# preserve comment" in saved


def test_settings_page_has_write_readback_verification():
    source = (Path(__file__).resolve().parents[1] / "src" / "pages_settings.py").read_text(encoding="utf-8")
    assert "def _write_env(values):" in source
    assert "Could not verify the saved setting" in source
    assert "Save and verify AI settings" in source


def test_chat_messages_have_right_aligned_user_lane():
    source = (Path(__file__).resolve().parents[1] / "src" / "chatbot.py").read_text(encoding="utf-8")
    assert "def _render_aligned_chat_message(" in source
    assert "st.columns([1.05, 4.0]" in source


def test_full_width_and_nav_refresh_styles_exist():
    source = (Path(__file__).resolve().parents[1] / "ui.py").read_text(encoding="utf-8")
    assert "max-width:none !important" in source
    assert "linear-gradient(135deg,var(--green),var(--green-hover))" in source
    assert "grid-template-columns:2.15rem minmax(0,1fr)" in source
