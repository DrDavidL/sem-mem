"""
Centralized configuration and secret loading for sem-mem.

Supports multiple secret sources with priority:
1. Environment variables (highest priority)
2. .env file in project root
3. Streamlit secrets (for Streamlit apps)
4. Interactive prompt (lowest priority, optional)
"""

import os
from pathlib import Path
from typing import Optional, Literal

# Available chat models
CHAT_MODELS = {
    "gpt-5.1": {
        "is_reasoning": True,
        "supports_temperature": False,
        "default_reasoning_effort": "low",
    },
    "gpt-4.1": {
        "is_reasoning": False,
        "supports_temperature": True,
        "default_reasoning_effort": None,
    },
}

DEFAULT_CHAT_MODEL = "gpt-5.1"
REASONING_EFFORTS = ["low", "medium", "high"]

# Query expansion model - use a fast/cheap model for reformulating queries
QUERY_EXPANSION_MODEL = "gpt-4.1-mini"


def _load_dotenv() -> None:
    """Load .env file if it exists and python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        # Try project root first, then current directory
        for path in [Path(__file__).parent.parent / ".env", Path(".env")]:
            if path.exists():
                load_dotenv(path)
                return
    except ImportError:
        pass  # python-dotenv not installed


def _get_streamlit_secret(key: str) -> Optional[str]:
    """Get secret from Streamlit if available."""
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


def get_api_key(
    api_key: Optional[str] = None,
    prompt_if_missing: bool = False,
) -> Optional[str]:
    """
    Get OpenAI API key from various sources.

    Priority:
    1. Passed api_key argument
    2. OPENAI_API_KEY environment variable
    3. .env file
    4. Streamlit secrets
    5. Interactive prompt (if prompt_if_missing=True)

    Args:
        api_key: Explicit API key (highest priority)
        prompt_if_missing: If True, prompt user interactively

    Returns:
        API key string or None
    """
    # 1. Explicit argument
    if api_key:
        return api_key

    # 2. Load .env file
    _load_dotenv()

    # 3. Environment variable
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # 4. Streamlit secrets
    st_key = _get_streamlit_secret("OPENAI_API_KEY")
    if st_key:
        return st_key

    # 5. Interactive prompt
    if prompt_if_missing:
        import getpass
        return getpass.getpass("Enter OpenAI API Key: ")

    return None


def get_config() -> dict:
    """
    Get all configuration values.

    Returns:
        Dict with configuration keys and values
    """
    _load_dotenv()

    chat_model = os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
    reasoning_effort = os.getenv("REASONING_EFFORT", "low")

    # Validate
    if chat_model not in CHAT_MODELS:
        chat_model = DEFAULT_CHAT_MODEL
    if reasoning_effort not in REASONING_EFFORTS:
        reasoning_effort = "low"

    return {
        "api_key": get_api_key(),
        "storage_dir": os.getenv("MEMORY_STORAGE_DIR", "./local_memory"),
        "cache_size": int(os.getenv("CACHE_SIZE", "20")),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "chat_model": chat_model,
        "reasoning_effort": reasoning_effort,
        "api_url": os.getenv("SEMMEM_API_URL", "http://localhost:8000"),
    }


def get_model_config(model: str) -> dict:
    """Get configuration for a specific model."""
    return CHAT_MODELS.get(model, CHAT_MODELS[DEFAULT_CHAT_MODEL])


def is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model (like gpt-5.1, o3)."""
    config = get_model_config(model)
    return config.get("is_reasoning", False)


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Basic validation of API key format.
    Does NOT verify the key is valid with OpenAI.
    """
    if not api_key:
        return False

    # OpenAI keys start with 'sk-'
    if not api_key.startswith("sk-"):
        return False

    # Should be reasonably long
    if len(api_key) < 20:
        return False

    return True
