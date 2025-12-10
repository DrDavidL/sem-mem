"""
Centralized configuration and secret loading for sem-mem.

Supports multiple secret sources with priority:
1. Environment variables (highest priority)
2. .env file in project root
3. Streamlit secrets (for Streamlit apps)
4. Interactive prompt (lowest priority, optional)

Provider Configuration:
- SEMMEM_CHAT_PROVIDER: Chat provider (openai, azure, anthropic, google, ollama, openrouter)
- SEMMEM_EMBEDDING_PROVIDER: Embedding provider (openai, azure, google, ollama)
- Provider-specific keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
"""

import os
from pathlib import Path
from typing import Optional, Literal

# =============================================================================
# Provider Configuration
# =============================================================================

# Default providers
DEFAULT_CHAT_PROVIDER = "openai"
DEFAULT_EMBEDDING_PROVIDER = "openai"

# Provider-specific environment variable names for API keys
PROVIDER_API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "azure": "AZURE_OPENAI_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,  # Ollama doesn't require an API key
}

# =============================================================================
# Chat Models (OpenAI-specific, other providers have their own model handling)
# =============================================================================

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

# --- Auto Thread Rename ---
# Automatically generate descriptive titles for conversation threads
AUTO_THREAD_RENAME_ENABLED = True
AUTO_THREAD_RENAME_MIN_USER_MESSAGES = 3  # Trigger after N user messages
AUTO_THREAD_RENAME_MODEL = "gpt-4.1-mini"  # Fast/cheap model for title generation
AUTO_THREAD_RENAME_MAX_WORDS = 8  # Maximum words in generated title

# --- Conversation Summarization ---
# Windowed summaries for long conversations, stored in L2 for durable semantic history
CONVERSATION_SUMMARY_TOKEN_THRESHOLD = 8000  # Approx tokens before summarizing
CONVERSATION_SUMMARY_MIN_MESSAGES = 10  # Don't summarize tiny threads
CONVERSATION_SUMMARY_MODEL = "gpt-4.1-mini"  # Cheap but capable summarizer
CONVERSATION_SUMMARY_MAX_CHARS = 8000  # Safety cap for prompt content
CONVERSATION_SUMMARY_LEAVE_RECENT = 6  # Raw turns kept after last summary window
CONVERSATION_SUMMARY_MAX_WINDOWS_PER_THREAD = 10  # Soft guardrail

# --- Thread Deletion ---
# Behavior when deleting a thread
ON_DELETE_THREAD_BEHAVIOR = "prompt"  # "prompt", "always_save", or "never_save"
FAREWELL_SUMMARY_MODEL = "gpt-4.1-mini"  # Model for farewell summaries
FAREWELL_SUMMARY_MAX_CHARS = 8000  # Safety cap for prompt content

# =============================================================================
# Outcome-Based Learning
# =============================================================================
# Track which memories actually help users, not just which are semantically similar.
# Memories accumulate utility scores based on success/failure feedback.

OUTCOME_LEARNING_ENABLED = True  # Master switch for outcome-based scoring

# EWMA (Exponentially Weighted Moving Average) smoothing factor for utility updates
# Higher = more weight on recent outcomes, faster adaptation
# Lower = more stable scores, slower to change
OUTCOME_EWMA_ALPHA = 0.3

# Weight of utility score in final retrieval ranking
# final_score = sim_score + OUTCOME_RETRIEVAL_ALPHA * (utility_score - 0.5)
# At 0.2, a utility of 1.0 adds +0.1 to score, utility of 0.0 subtracts -0.1
OUTCOME_RETRIEVAL_ALPHA = 0.2

# Pattern promotion thresholds
# A memory becomes a "pattern" (proven useful) when it reaches both thresholds
PATTERN_MIN_SUCCESSES = 3  # Minimum successful retrievals
PATTERN_MIN_UTILITY = 0.9  # Minimum utility score

# Outcome value mapping for EWMA calculation
OUTCOME_VALUES = {
    "success": 1.0,   # Memory was helpful
    "neutral": 0.5,   # Unknown/no feedback
    "failure": 0.0,   # Memory was not helpful or misleading
}

# =============================================================================
# Time-Aware Retrieval (Phase 3)
# =============================================================================
# Apply time decay to retrieval scores, preferring recent memories while
# preserving old high-signal ones (identity facts, stable preferences).

TIME_DECAY_ENABLED = True  # Master switch for time-decay scoring

# Half-life in days: after this many days, decay factor drops to 50%
# 30 days means a 30-day-old memory has 50% time weight
# 60 days means 25% time weight, etc.
TIME_DECAY_HALF_LIFE_DAYS = 30

# Alpha floor: minimum time weight for very old memories
# At 0.3, even ancient memories retain 30% of their original score if highly relevant
# This prevents stable identity facts from disappearing
TIME_DECAY_ALPHA = 0.3

# =============================================================================
# Memory Consolidation
# =============================================================================
# Offline routine that reviews memories to create patterns and reduce redundancy.
# Runs periodically (external scheduling) to distill stable preferences/principles.

CONSOLIDATION_ENABLED = True
CONSOLIDATION_DRY_RUN = True  # Analyze but don't write (flip to False after tuning)

# Memory selection limits
CONSOLIDATION_RECENT_LIMIT = 50    # How many recent memories to review
CONSOLIDATION_COLD_SAMPLE = 50     # How many older memories to sample randomly

# Output limits per run
CONSOLIDATION_MAX_NEW_PATTERNS = 5  # Cap pattern creation per run

# Model for consolidation analysis
CONSOLIDATION_MODEL = "gpt-4.1-mini"

# Scheduling (advisory; actual scheduling is external via cron/script)
CONSOLIDATION_FREQUENCY = "daily"  # "hourly" | "daily" | "manual"

# Objectives passed to LLM (explicit, designer-defined, not self-generated)
CONSOLIDATION_OBJECTIVES = [
    "reduce redundancy in stored memories",
    "promote stable preferences and principles that help future interactions",
    "highlight contradictions or inconsistencies for human review (do not auto-resolve)",
]


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
    provider: str = "openai",
) -> Optional[str]:
    """
    Get API key for a provider from various sources.

    Priority:
    1. Passed api_key argument
    2. Provider-specific environment variable (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
    3. .env file
    4. Streamlit secrets
    5. Interactive prompt (if prompt_if_missing=True)

    Args:
        api_key: Explicit API key (highest priority)
        prompt_if_missing: If True, prompt user interactively
        provider: Provider name to get key for (default: "openai")

    Returns:
        API key string or None
    """
    # 1. Explicit argument
    if api_key:
        return api_key

    # 2. Load .env file
    _load_dotenv()

    # Get the environment variable name for this provider
    env_var = PROVIDER_API_KEY_ENV_VARS.get(provider.lower(), f"{provider.upper()}_API_KEY")
    if env_var is None:
        # Provider doesn't need an API key (e.g., Ollama)
        return None

    # 3. Environment variable
    env_key = os.getenv(env_var)
    if env_key:
        return env_key

    # 4. Streamlit secrets
    st_key = _get_streamlit_secret(env_var)
    if st_key:
        return st_key

    # 5. Interactive prompt
    if prompt_if_missing:
        import getpass
        return getpass.getpass(f"Enter {provider} API Key: ")

    return None


def get_provider_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Get API key for a specific provider.

    Convenience function that uses get_api_key with provider-specific defaults.

    Args:
        provider: Provider name ("openai", "anthropic", "google", etc.)
        api_key: Explicit API key (overrides env vars)

    Returns:
        API key string or None
    """
    return get_api_key(api_key=api_key, provider=provider)


def get_config() -> dict:
    """
    Get all configuration values.

    Returns:
        Dict with configuration keys and values
    """
    _load_dotenv()

    # Provider configuration
    chat_provider = os.getenv("SEMMEM_CHAT_PROVIDER", DEFAULT_CHAT_PROVIDER)
    embedding_provider = os.getenv("SEMMEM_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)

    # Model configuration
    chat_model = os.getenv("SEMMEM_CHAT_MODEL", os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL))
    embedding_model = os.getenv("SEMMEM_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    reasoning_effort = os.getenv("REASONING_EFFORT", "low")

    # Validate reasoning effort
    if reasoning_effort not in REASONING_EFFORTS:
        reasoning_effort = "low"

    return {
        # Provider settings
        "chat_provider": chat_provider,
        "embedding_provider": embedding_provider,

        # API keys (per provider)
        "api_key": get_api_key(provider=chat_provider),  # Legacy, for backward compat
        "chat_api_key": get_api_key(provider=chat_provider),
        "embedding_api_key": get_api_key(provider=embedding_provider),

        # Model settings
        "chat_model": chat_model,
        "embedding_model": embedding_model,
        "reasoning_effort": reasoning_effort,

        # Storage settings
        "storage_dir": os.getenv("MEMORY_STORAGE_DIR", "./local_memory"),
        "cache_size": int(os.getenv("CACHE_SIZE", "20")),

        # API server settings
        "api_url": os.getenv("SEMMEM_API_URL", "http://localhost:8000"),

        # Provider-specific settings
        "provider_kwargs": get_provider_kwargs(),
    }


def get_provider_kwargs() -> dict:
    """
    Get provider-specific configuration from environment variables.

    Returns:
        Dict with provider-specific settings (azure_endpoint, ollama_base_url, etc.)
    """
    _load_dotenv()

    kwargs = {}

    # Azure OpenAI settings
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if azure_endpoint:
        kwargs["azure_endpoint"] = azure_endpoint
    azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
    if azure_api_version:
        kwargs["azure_api_version"] = azure_api_version

    # Ollama settings
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    if ollama_base_url:
        kwargs["ollama_base_url"] = ollama_base_url

    # OpenRouter settings
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    if openrouter_base_url:
        kwargs["openrouter_base_url"] = openrouter_base_url

    return kwargs


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
