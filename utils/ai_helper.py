import os
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI

    OPENAI_OK = True
except ImportError:
    OpenAI = None
    OPENAI_OK = False


APP_ROOT = Path(__file__).resolve().parents[1]
_ENV_LOADED = False

AI_PROVIDERS = {
    "NVIDIA": {
        "env_keys": ("NVIDIA_API_KEY",),
        "secret_keys": ("NVIDIA_API_KEY",),
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "meta/llama-3.1-8b-instruct",
    },
    "OpenAI": {
        "env_keys": ("OPENAI_API_KEY",),
        "secret_keys": ("OPENAI_API_KEY",),
        "base_url": None,
        "default_model": "gpt-4o-mini",
    },
}
DEFAULT_PROVIDER = "NVIDIA"


def _load_env_files():
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    if load_dotenv is not None:
        env_candidates = [APP_ROOT / ".env", Path.cwd() / ".env"]
        seen = set()
        for path in env_candidates:
            resolved = str(path.resolve())
            if path.exists() and resolved not in seen:
                load_dotenv(path, override=False)
                seen.add(resolved)
        load_dotenv(override=False)

    _ENV_LOADED = True


def _safe_session_state_get(key, default=None):
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def _safe_secrets_get(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        return None
    return None


def _provider_names():
    return list(AI_PROVIDERS.keys())


def get_api_key(provider=None, with_source=False):
    """
    Retrieves the API key for a given provider.
    Priority: environment variables > streamlit secrets > session overrides.
    """
    _load_env_files()
    provider = provider or get_selected_provider()
    provider_cfg = AI_PROVIDERS.get(provider, AI_PROVIDERS[DEFAULT_PROVIDER])

    # 1. Check environment variables
    for env_key in provider_cfg["env_keys"]:
        env_value = os.getenv(env_key, "").strip()
        if env_value:
            return (env_value, f"environment:{env_key}") if with_source else env_value

    # 2. Check Streamlit secrets
    for secret_key in provider_cfg["secret_keys"]:
        secret_value = _safe_secrets_get(secret_key)
        if isinstance(secret_value, str) and secret_value.strip():
            cleaned = secret_value.strip()
            return (cleaned, f"streamlit-secrets:{secret_key}") if with_source else cleaned

    # 3. Check generic 'ai' secret block
    ai_secret = _safe_secrets_get("ai")
    if isinstance(ai_secret, dict):
        for candidate in (provider.lower(), f"{provider.lower()}_api_key", "api_key", "key"):
            secret_value = ai_secret.get(candidate)
            if isinstance(secret_value, str) and secret_value.strip():
                cleaned = secret_value.strip()
                return (cleaned, f"streamlit-secrets:ai.{candidate}") if with_source else cleaned

    # 4. Check session override (Fallback only)
    session_key = (_safe_session_state_get("ai_api_key_override") or "").strip()
    if session_key:
        return (session_key, "session-override") if with_source else session_key

    return (None, "missing") if with_source else None


def get_selected_model(provider=None, explicit_model=None):
    provider = provider or get_selected_provider()
    if explicit_model:
        return explicit_model

    # Priority 1: Environment variable
    env_model = os.getenv("AI_MODEL", "").strip()
    if env_model:
        return env_model

    # Priority 2: Session override
    session_model = (_safe_session_state_get("ai_model_override") or "").strip()
    if session_model:
        return session_model

    return AI_PROVIDERS.get(provider, AI_PROVIDERS[DEFAULT_PROVIDER])["default_model"]


def get_ai_status(provider=None, explicit_model=None):
    """Returns a status dictionary for the current AI configuration."""
    provider = provider or get_selected_provider()
    api_key, source = get_api_key(provider=provider, with_source=True)
    model = get_selected_model(provider=provider, explicit_model=explicit_model)
    return {
        "provider": provider,
        "model": model,
        "source": source,
        "available": bool(api_key and OPENAI_OK),
        "api_key": api_key,
        "openai_sdk_ready": OPENAI_OK,
        "last_error": _safe_session_state_get("last_ai_error"),
    }


def get_selected_provider():
    """
    Automatically selects an AI provider based on available API keys.
    Prioritizes NVIDIA, then OpenAI. Checks session overrides, then environment, then secrets.
    """
    _load_env_files()
    
    # 1. Check which keys are actually available
    nvidia_key = get_api_key("NVIDIA")
    openai_key = get_api_key("OpenAI")
    
    # 2. Return the first one that has a valid key
    if nvidia_key:
        return "NVIDIA"
    if openai_key:
        return "OpenAI"
    
    # 3. Fallback to default or environment-defined
    env_provider = os.getenv("AI_PROVIDER", "").strip()
    if env_provider in AI_PROVIDERS:
        return env_provider
    
    return DEFAULT_PROVIDER


def _build_client(provider, api_key):
    provider_cfg = AI_PROVIDERS.get(provider, AI_PROVIDERS[DEFAULT_PROVIDER])
    base_url = provider_cfg.get("base_url")
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key)
    return OpenAI(api_key=api_key)


def get_ai_explanation(
    prompt,
    system_prompt=(
        "You are an expert AI assistant. Keep explanations concise and professional."
    ),
    model=None,
    max_tokens=1000,
    timeout=30,
):
    """
    Sends a prompt to the configured LLM provider and returns the textual explanation.
    Returns None if the API is unavailable or the request fails.
    """
    status = get_ai_status(explicit_model=model)
    if not status["available"]:
        return None

    try:
        client = _build_client(status["provider"], status["api_key"])
        resp = client.chat.completions.create(
            model=status["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        st.session_state["last_ai_error"] = ""
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        st.session_state["last_ai_error"] = str(exc)
        print(f"AI Helper Error: {exc}")
        return None
