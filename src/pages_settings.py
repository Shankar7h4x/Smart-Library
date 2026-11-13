"""Local AI provider configuration with verified local persistence."""
from pathlib import Path
import os
import streamlit as st

from src.ai_service import check_provider_api, load_project_env
from src.env_settings import write_env_preserving_unmanaged

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

PROVIDERS = ("openai", "groq", "gemini")


def _read_env():
    """Read local settings, then fill missing values from the process environment."""
    values = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    for key in (
        "AI_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL",
        "GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL",
    ):
        if not values.get(key) and os.getenv(key):
            values[key] = os.getenv(key, "")
    return values


def _write_env(values):
    """Write and verify the project's local .env file."""
    write_env_preserving_unmanaged(ENV_FILE, values)
    if not ENV_FILE.exists():
        raise OSError("The .env file was not created.")
    saved = _read_env()
    for key, value in values.items():
        if saved.get(key, "") != value:
            raise OSError(f"Could not verify the saved setting: {key}")


def render_settings_page():
    st.markdown(
        """
        <div style="margin-bottom: 1.15rem;">
          <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
            <span style="background: rgba(0,208,132,0.15); color: var(--green); font-size: 0.78rem; font-weight: 800; padding: 2px 8px; border-radius: 9999px; text-transform: uppercase; border: 1px solid var(--green);">
              Preferences & AI Setup
            </span>
            <span style="font-size: 0.82rem; color: var(--green);">Secure Local Environment</span>
          </div>
          <h1 style="font-size: 2.1rem; font-weight: 800; color: var(--ink); margin: 0 0 0.35rem 0; letter-spacing: -0.025em; line-height: 1.25;">
            Settings & AI Provider Setup
          </h1>
          <p style="font-size: 1rem; color: var(--muted); margin: 0; line-height: 1.55;">
            Connect your AI provider (Google AI Studio Gemini, OpenAI, or Groq). Keys are stored safely in your local .env file.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    current = _read_env()
    provider = st.selectbox(
        "Active AI provider",
        list(PROVIDERS),
        index=PROVIDERS.index(current.get("AI_PROVIDER", "gemini").lower())
        if current.get("AI_PROVIDER", "gemini").lower() in PROVIDERS else 2,
        key="settings_active_provider",
    )

    st.markdown("### Google Gemini (Google AI Studio)")
    gemini_saved = bool(current.get("GEMINI_API_KEY"))
    st.caption("Saved Google AI Studio key detected · hidden" if gemini_saved else "No saved Gemini key detected")
    gemini_key = st.text_input(
        "Gemini API key", value="", type="password",
        placeholder="Enter a Google AI Studio key or leave blank to keep saved key",
        help="Generated in Google AI Studio. Stored privately in local .env.",
        key="settings_gemini_api_key",
    )
    gemini_model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest", "gemini-2.5-flash-lite"]
    saved_gemini_model = current.get("GEMINI_MODEL", "gemini-2.5-flash")
    if saved_gemini_model not in gemini_model_options:
        gemini_model_options.insert(0, saved_gemini_model)
    gemini_model = st.selectbox(
        "Gemini model",
        gemini_model_options,
        index=gemini_model_options.index(saved_gemini_model),
        help="gemini-2.5-flash is recommended for fast, high-quality academic assistant responses.",
        key="settings_gemini_model",
    )

    st.markdown("### OpenAI")
    openai_saved = bool(current.get("OPENAI_API_KEY"))
    st.caption("Saved key detected · hidden" if openai_saved else "No saved OpenAI key detected")
    openai_key = st.text_input(
        "OpenAI API key", value="", type="password",
        placeholder="Enter a new key to save or leave blank to keep the saved key",
        help="The key is saved locally in .env. Never share it in screenshots.",
        key="settings_openai_api_key",
    )
    openai_model = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"],
        index=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"].index(
            current.get("OPENAI_MODEL", "gpt-4o-mini")
        ) if current.get("OPENAI_MODEL", "gpt-4o-mini") in
        ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o"] else 0,
        key="settings_openai_model",
    )

    with st.expander("Other AI providers (Groq)"):
        st.markdown("**Groq**")
        st.caption("Saved key detected · hidden" if current.get("GROQ_API_KEY") else "No saved Groq key detected")
        groq_key = st.text_input(
            "Groq API key", value="", type="password",
            placeholder="Enter a new key or leave blank to keep the saved key",
            key="settings_groq_api_key",
        )
        groq_model = st.text_input(
            "Groq model", value=current.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            key="settings_groq_model",
        )

    if st.button("Save and verify AI settings", type="primary", use_container_width=True):
        values = {
            "AI_PROVIDER": provider,
            "OPENAI_API_KEY": openai_key.strip() or current.get("OPENAI_API_KEY", ""),
            "OPENAI_MODEL": openai_model,
            "GROQ_API_KEY": groq_key.strip() or current.get("GROQ_API_KEY", ""),
            "GROQ_MODEL": groq_model.strip() or "llama-3.3-70b-versatile",
            "GEMINI_API_KEY": gemini_key.strip() or current.get("GEMINI_API_KEY", ""),
            "GEMINI_MODEL": gemini_model.strip() or "gemini-2.5-flash",
        }
        try:
            _write_env(values)
            # Refresh this process immediately so the chatbot sees the saved values.
            for key, value in values.items():
                if value:
                    os.environ[key] = value
                else:
                    os.environ.pop(key, None)
            load_project_env()
            st.success(f"Settings saved and verified in: {ENV_FILE}")
            active_key = values.get(f"{provider.upper()}_API_KEY", "")
            if active_key:
                with st.spinner(f"Checking {provider.capitalize()} API credentials..."):
                    healthy, detail = check_provider_api(provider, active_key)
                if healthy:
                    st.success(f"{provider.capitalize()} API accepted the key. The chatbot can use this provider.")
                else:
                    st.error(f"Key was saved, but the provider check failed: {detail}")
            else:
                st.warning(f"Settings were saved, but no {provider.capitalize()} API key is configured.")
        except (OSError, ValueError) as exc:
            st.error(f"Could not save AI settings: {exc}")

    st.caption(
        f"Local settings file: {ENV_FILE}. The .env file is excluded from the project ZIP and Git. "
        "Keep it private and do not upload it or share screenshots containing the key."
    )
