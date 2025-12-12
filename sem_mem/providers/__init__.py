"""
Provider registry and factory functions.

Use get_chat_provider() and get_embedding_provider() to get provider instances.
Chat and embedding providers can be different (e.g., Claude for chat, OpenAI for embeddings).

Provider Tiers:
- Tier 1 (First-class): OpenAI, Azure OpenAI
- Tier 2 (Supported): Anthropic, Google, Ollama
- Tier 3 (Experimental): OpenRouter
"""

from typing import Optional, Dict, Type, Set

from .base import BaseChatProvider, BaseEmbeddingProvider, ChatResponse, EmbeddingResponse, ToolCall
from ..exceptions import ProviderNotFoundError, ProviderConfigError


# Provider registries (populated on first access to allow lazy imports)
_CHAT_PROVIDERS: Dict[str, Type[BaseChatProvider]] = {}
_EMBEDDING_PROVIDERS: Dict[str, Type[BaseEmbeddingProvider]] = {}
_PROVIDERS_INITIALIZED = False


def _init_providers():
    """Lazy initialization of provider registries."""
    global _PROVIDERS_INITIALIZED, _CHAT_PROVIDERS, _EMBEDDING_PROVIDERS

    if _PROVIDERS_INITIALIZED:
        return

    # Import Tier 1 providers (always available)
    from .openai_provider import OpenAIChatProvider, OpenAIEmbeddingProvider, OpenAIProvider
    from .azure import AzureChatProvider, AzureEmbeddingProvider, AzureProvider

    # Register Tier 1 chat providers
    _CHAT_PROVIDERS.update({
        "openai": OpenAIChatProvider,
        "azure": AzureChatProvider,
    })

    # Register Tier 1 embedding providers
    _EMBEDDING_PROVIDERS.update({
        "openai": OpenAIEmbeddingProvider,
        "azure": AzureEmbeddingProvider,
    })

    # Import Tier 2 providers (may not be installed)
    try:
        from .anthropic import AnthropicChatProvider
        _CHAT_PROVIDERS["anthropic"] = AnthropicChatProvider
    except ImportError:
        pass  # anthropic package not installed

    try:
        from .ollama import OllamaChatProvider, OllamaEmbeddingProvider
        _CHAT_PROVIDERS["ollama"] = OllamaChatProvider
        _EMBEDDING_PROVIDERS["ollama"] = OllamaEmbeddingProvider
    except ImportError:
        pass  # requests package not installed

    try:
        from .google import GoogleChatProvider, GoogleEmbeddingProvider
        _CHAT_PROVIDERS["google"] = GoogleChatProvider
        _EMBEDDING_PROVIDERS["google"] = GoogleEmbeddingProvider
    except ImportError:
        pass  # google-generativeai package not installed

    # Import Tier 3 providers (experimental)
    try:
        from .openrouter import OpenRouterChatProvider
        _CHAT_PROVIDERS["openrouter"] = OpenRouterChatProvider
    except ImportError:
        pass  # Should not fail (uses openai SDK)

    _PROVIDERS_INITIALIZED = True


def get_available_chat_providers() -> Set[str]:
    """Get set of available chat provider names."""
    _init_providers()
    return set(_CHAT_PROVIDERS.keys())


def get_available_embedding_providers() -> Set[str]:
    """Get set of available embedding provider names."""
    _init_providers()
    return set(_EMBEDDING_PROVIDERS.keys())


def get_chat_provider(
    provider: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseChatProvider:
    """
    Get a chat provider instance.

    Args:
        provider: Provider name ("openai", "azure", "anthropic", etc.)
        api_key: API key for the provider. If not provided, uses env vars.
        **kwargs: Provider-specific configuration (azure_endpoint, ollama_base_url, etc.)

    Returns:
        Chat provider instance.

    Raises:
        ProviderNotFoundError: If provider is not available.
        ProviderConfigError: If configuration is invalid.
    """
    _init_providers()

    provider = provider.lower()

    if provider not in _CHAT_PROVIDERS:
        raise ProviderNotFoundError(
            provider=provider,
            provider_type="chat",
            available=list(_CHAT_PROVIDERS.keys()),
        )

    provider_class = _CHAT_PROVIDERS[provider]

    try:
        return provider_class(api_key=api_key, **kwargs)
    except Exception as e:
        raise ProviderConfigError(provider, str(e)) from e


def get_embedding_provider(
    provider: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseEmbeddingProvider:
    """
    Get an embedding provider instance.

    Args:
        provider: Provider name ("openai", "azure", "google", "ollama")
        api_key: API key for the provider. If not provided, uses env vars.
        **kwargs: Provider-specific configuration.

    Returns:
        Embedding provider instance.

    Raises:
        ProviderNotFoundError: If provider is not available.
        ProviderConfigError: If configuration is invalid.
    """
    _init_providers()

    provider = provider.lower()

    if provider not in _EMBEDDING_PROVIDERS:
        raise ProviderNotFoundError(
            provider=provider,
            provider_type="embedding",
            available=list(_EMBEDDING_PROVIDERS.keys()),
        )

    provider_class = _EMBEDDING_PROVIDERS[provider]

    try:
        return provider_class(api_key=api_key, **kwargs)
    except Exception as e:
        raise ProviderConfigError(provider, str(e)) from e


def register_chat_provider(name: str, provider_class: Type[BaseChatProvider]):
    """
    Register a custom chat provider.

    Args:
        name: Provider name to register.
        provider_class: Provider class implementing BaseChatProvider.
    """
    _init_providers()
    _CHAT_PROVIDERS[name.lower()] = provider_class


def register_embedding_provider(name: str, provider_class: Type[BaseEmbeddingProvider]):
    """
    Register a custom embedding provider.

    Args:
        name: Provider name to register.
        provider_class: Provider class implementing BaseEmbeddingProvider.
    """
    _init_providers()
    _EMBEDDING_PROVIDERS[name.lower()] = provider_class


__all__ = [
    # Base classes
    "BaseChatProvider",
    "BaseEmbeddingProvider",
    "ChatResponse",
    "EmbeddingResponse",
    "ToolCall",
    # Factory functions
    "get_chat_provider",
    "get_embedding_provider",
    "get_available_chat_providers",
    "get_available_embedding_providers",
    # Registration
    "register_chat_provider",
    "register_embedding_provider",
]
