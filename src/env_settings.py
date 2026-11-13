"""Safe .env updates for the managed AI-provider settings."""
from pathlib import Path

MANAGED_AI_KEYS = {
    "AI_PROVIDER", "OPENAI_API_KEY", "OPENAI_MODEL",
    "GROQ_API_KEY", "GROQ_MODEL", "GEMINI_API_KEY", "GEMINI_MODEL",
}


def write_env_preserving_unmanaged(path: Path, values: dict) -> None:
    """Update managed keys atomically without deleting comments or unrelated settings."""
    path = Path(path)
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    updated, written = [], set()
    for line in existing:
        stripped = line.strip()
        key = stripped.split("=", 1)[0].strip() if stripped and not stripped.startswith("#") and "=" in stripped else None
        if key in MANAGED_AI_KEYS:
            if key in values:
                updated.append(f"{key}={values[key]}")
                written.add(key)
            continue
        updated.append(line)
    for key in sorted(MANAGED_AI_KEYS - written):
        if key in values:
            updated.append(f"{key}={values[key]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)
