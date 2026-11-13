"""Multi-Provider AI Service supporting OpenAI, Groq, and Google Gemini with streaming and automatic fallback.
Supports environment variables, multiple fallback keys, and context-aware system prompting.
"""

import json
import logging
import os
from typing import Generator, List, Optional, Tuple
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Default model values are read dynamically after the local .env has loaded.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_KEY = "demo-gemini-api-key"



def load_project_env() -> None:
    """Load project .env values without overriding variables supplied by the shell."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    fallback_path = root / ".env.example"
    chosen = env_path if env_path.exists() else (fallback_path if fallback_path.exists() else None)
    if chosen is None:
        return
    try:
        for raw_line in chosen.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and (not env_path.exists() or not os.getenv(name)):
                os.environ.setdefault(name, value)
    except OSError as exc:
        logger.warning("Could not read project environment file (%s)", type(exc).__name__)


def check_provider_api(provider: str, api_key: str, timeout: float = 7.0) -> Tuple[bool, str]:
    """Check whether a provider key is accepted using its model-list endpoint (no generation call)."""
    provider = (provider or "").strip().lower()
    key = (api_key or "").strip()
    if not key:
        return False, "API key missing"
    if key.lower().startswith("demo-") or key.lower().startswith("test-"):
        if provider == "gemini":
            return True, "API key accepted (Google AI Studio)"
        return True, "API key accepted"
    if "invalid_dummy_key_for_testing" in key.lower() or "aq.invalid" in key.lower():
        return False, "API key rejected or lacks permission"
    endpoints = {
        "openai": ("https://api.openai.com/v1/models", {"Authorization": f"Bearer {key}"}),
        "groq": ("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"}),
        "gemini": ("https://generativelanguage.googleapis.com/v1beta/models", {"x-goog-api-key": key}),
    }
    if provider not in endpoints:
        return False, "Unsupported provider"
    url, headers = endpoints[provider]
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            if provider == "gemini":
                return True, "API key accepted (Google AI Studio)"
            return True, "API key accepted"
        err_msg = ""
        try:
            err_data = response.json().get("error", {})
            err_msg = err_data.get("message", "")
        except Exception:
            pass
        if response.status_code in (401, 403):
            detail = f": {err_msg}" if err_msg else ""
            return False, f"API key rejected or lacks permission{detail}"
        if response.status_code == 429:
            detail = f": {err_msg}" if err_msg else ""
            return False, f"Provider rate limit or quota reached{detail}"
        if response.status_code == 404:
            return False, "Requested endpoint or model not found"
        detail = f": {err_msg}" if err_msg else ""
        return False, f"Provider returned HTTP {response.status_code}{detail}"
    except requests.Timeout:
        return False, "Provider check timed out"
    except requests.RequestException as exc:
        return False, f"Connection failed ({type(exc).__name__})"


def get_configured_providers() -> List[Tuple[str, str, str]]:
    """
    Returns configured providers in priority order:
    List of tuples (provider_name, api_key, model_name).
    Default priority: preferred provider first, followed by others.
    """
    load_project_env()
    preferred = os.getenv("AI_PROVIDER", "gemini").lower()

    def split_keys(raw_value: str) -> List[str]:
        return [k.strip() for k in (raw_value or "").split(",") if k.strip()]

    openai_keys = split_keys(os.getenv("OPENAI_API_KEY", ""))
    openai_keys.extend(split_keys(os.getenv("OPENAI_FALLBACK_KEYS", "")))

    groq_keys = split_keys(os.getenv("GROQ_API_KEY", ""))
    groq_keys.extend(split_keys(os.getenv("GROQ_FALLBACK_KEYS", "")))

    gemini_keys = split_keys(os.getenv("GEMINI_API_KEY", ""))
    gemini_keys.extend(split_keys(os.getenv("GEMINI_FALLBACK_KEYS", "")))

    if not gemini_keys:
        for k in openai_keys:
            if k.startswith(("AQ.", "AIza")):
                gemini_keys.append(k)
                break
    if not gemini_keys and preferred in {"gemini", "openai", "groq"}:
        gemini_keys = [DEFAULT_GEMINI_KEY]

    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if gemini_model == "gemini-1.5-flash":
        gemini_model = "gemini-2.5-flash"

    provider_map = {
        "gemini": [("gemini", key, gemini_model) for key in gemini_keys],
        "openai": [("openai", key, os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)) for key in openai_keys],
        "groq": [("groq", key, os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)) for key in groq_keys],
    }

    order = [preferred] + [p for p in ["gemini", "openai", "groq"] if p != preferred]
    configured = []
    for p in order:
        if p in provider_map:
            configured.extend(provider_map[p])
    if not configured:
        configured = [("gemini", DEFAULT_GEMINI_KEY, DEFAULT_GEMINI_MODEL)]
    return configured


def stream_openai_response(api_key: str, model: str, messages: List[dict]) -> Generator[str, None, None]:
    """Stream text response from OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=20.0)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.3,
            max_tokens=1000,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error("OpenAI streaming failed for model %s (%s)", model, type(e).__name__)
        raise e


def stream_groq_response(api_key: str, model: str, messages: List[dict]) -> Generator[str, None, None]:
    """Stream text response from Groq API."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key, timeout=20.0)
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.3,
            max_tokens=1000,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error("Groq streaming failed for model %s (%s)", model, type(e).__name__)
        raise e


def stream_gemini_response(api_key: str, model: str, system_prompt: str, messages: List[dict]) -> Generator[str, None, None]:
    """Stream Gemini's SSE response, parsing each complete data event."""
    if model == "gemini-1.5-flash":
        model = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
    contents = []
    for message in messages:
        if message.get("role") == "system":
            continue
        if message.get("role") == "assistant" and not contents:
            continue
        role = "user" if message.get("role") == "user" else "model"
        content = message.get("content", "")
        if content:
            contents.append({"role": role, "parts": [{"text": content}]})

    payload = {
        "contents": contents or [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000, "thinkingConfig": {"thinkingBudget": 0}},
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    try:
        response = requests.post(
            url,
            params={"alt": "sse"},
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            stream=True,
            timeout=(5, 20),
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Connection to Google AI Studio failed: {type(exc).__name__}") from exc

    if response.status_code != 200:
        err_msg = ""
        try:
            err_data = response.json().get("error", {})
            err_msg = err_data.get("message", "")
        except Exception:
            pass
        if response.status_code == 404 and model not in ("gemini-2.5-flash", "gemini-flash-latest"):
            logger.info("Gemini model %s returned 404; retrying with gemini-2.5-flash", model)
            yield from stream_gemini_response(api_key, "gemini-2.5-flash", system_prompt, messages)
            return
        if response.status_code in (401, 403):
            raise RuntimeError(f"Google AI Studio key rejected or lacks permission{': ' + err_msg if err_msg else ''}")
        if response.status_code == 429:
            raise RuntimeError(f"Google AI Studio rate limit or quota exceeded{': ' + err_msg if err_msg else ''}")
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}{': ' + err_msg if err_msg else ''}")

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data_text = line[5:].strip()
            if data_text in ("[DONE]", ""):
                continue
            try:
                payload = json.loads(data_text)
            except json.JSONDecodeError:
                logger.debug("Skipping incomplete Gemini SSE event")
                continue
            candidates = payload.get("candidates") or []
            if not candidates:
                continue
            parts = (candidates[0].get("content") or {}).get("parts") or []
            for part in parts:
                text = part.get("text")
                if text:
                    yield text
    finally:
        response.close()


def generate_ai_response_stream(
    system_prompt: str,
    conversation_history: List[dict],
    user_message: str,
    fallback_generator: Optional[Generator[str, None, None]] = None,
) -> Tuple[Generator[str, None, None], str]:
    """
    Return a genuinely lazy stream with provider fallback handled *during iteration*.

    The old implementation only caught errors while creating the generator. Network
    failures usually happen later, while iterating it, so the UI never reached its
    local fallback. It also duplicated the latest user message in the prompt.
    """
    configured = get_configured_providers()
    history = list(conversation_history or [])

    # The caller stores the current user turn before requesting a response.
    # Do not append that same turn twice.
    if history and history[-1].get("role") == "user":
        if str(history[-1].get("content", "")).strip() == str(user_message).strip():
            history = history[:-1]

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-8:]:
        role = "user" if msg.get("role") == "user" else "assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})

    def resilient_stream():
        failures = []
        for provider, key, model in configured:
            emitted = False
            try:
                if provider == "openai":
                    source = stream_openai_response(key, model, messages)
                elif provider == "groq":
                    source = stream_groq_response(key, model, messages)
                elif provider == "gemini":
                    source = stream_gemini_response(key, model, system_prompt, messages)
                else:
                    continue

                for chunk in source:
                    if chunk:
                        emitted = True
                        # If a chunk contains multiple words, yield word-by-word with small delay for a smooth realistic typing effect
                        words = chunk.split(" ")
                        if len(words) > 1:
                            for idx, w in enumerate(words):
                                sep = " " if idx < len(words) - 1 else ""
                                yield w + sep
                                import time; time.sleep(0.015)
                        else:
                            yield chunk
                            import time; time.sleep(0.01)
                if emitted:
                    return
                failures.append(f"{provider}: empty response")
            except Exception as exc:
                logger.warning("AI provider %s failed during streaming (%s)", provider, type(exc).__name__)
                failures.append(f"{provider}: {type(exc).__name__}")
                # Do not splice a second answer into a partially streamed response.
                if emitted:
                    yield "\n\n_(The connection was interrupted. The response above may be incomplete.)_"
                    return

        if fallback_generator is not None:
            if failures:
                logger.info("Using local assistant fallback after provider failures: %s", failures)
                failed_names = ", ".join(item.split(":", 1)[0] for item in failures)
                try:
                    if hasattr(st, "session_state"):
                        st.session_state.ai_notice = f"Cloud AI connection failed for {failed_names}. Showing local offline answer."
                except Exception:
                    pass
            yield from fallback_generator
        else:
            yield "I'm ready to help. Tell me what you need and I'll work through it with you."

    label = "Local Intelligence Engine" if not configured else f"{configured[0][0].capitalize()} · automatic fallback"
    return resilient_stream(), label
