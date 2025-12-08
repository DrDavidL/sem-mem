"""
OpenRouter provider implementation.

This is a Tier 3 (experimental) provider for chat only.
OpenRouter provides access to many models via an OpenAI-compatible API.

Key differences from OpenAI:
- Uses OpenAI SDK with different base URL
- Different model naming (e.g., "anthropic/claude-3-opus")
- No embeddings support
- Some models may have different behaviors
"""

from typing import List, Dict, Optional, Any

from openai import OpenAI

from .base import BaseChatProvider, ChatResponse


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterChatProvider(BaseChatProvider):
    """
    OpenRouter chat provider using OpenAI-compatible API.

    Note: OpenRouter does not provide embeddings. Use a separate
    embedding provider (OpenAI, Google, Ollama) for embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        openrouter_base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenRouter chat provider.

        Args:
            api_key: OpenRouter API key. If not provided, uses OPENROUTER_API_KEY env var.
            openrouter_base_url: OpenRouter API URL. Defaults to https://openrouter.ai/api/v1.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        import os

        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key required. Provide api_key parameter or "
                "set OPENROUTER_API_KEY environment variable."
            )

        base_url = (
            openrouter_base_url
            or os.getenv("OPENROUTER_BASE_URL")
            or DEFAULT_OPENROUTER_BASE_URL
        )

        # Use OpenAI client with OpenRouter base URL
        self._client = OpenAI(
            api_key=key,
            base_url=base_url,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def supports_responses_api(self) -> bool:
        # OpenRouter uses OpenAI-compatible Chat Completions
        return False

    @property
    def supports_reasoning(self) -> bool:
        # Some models on OpenRouter support reasoning, but behavior varies
        return False

    @property
    def client(self) -> OpenAI:
        """Access underlying OpenAI client for advanced use cases."""
        return self._client

    def chat(
        self,
        messages: Optional[List[Dict[str, str]]],
        model: str,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat request to OpenRouter.

        Args:
            messages: List of message dicts (required).
            model: Model name (e.g., "anthropic/claude-3-opus", "openai/gpt-4").
            instructions: System instructions.
            previous_response_id: Ignored (OpenRouter doesn't support this).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional parameters:
                - transforms: OpenRouter-specific transforms
                - route: OpenRouter routing preferences

        Returns:
            ChatResponse with generated text.
        """
        if not messages:
            raise ValueError("messages required for OpenRouter")

        # Build messages list with optional system message
        api_messages = []
        if instructions:
            api_messages.append({"role": "system", "content": instructions})
        api_messages.extend(messages)

        params: Dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # OpenRouter-specific parameters
        transforms = kwargs.pop("transforms", None)
        if transforms:
            params["transforms"] = transforms

        route = kwargs.pop("route", None)
        if route:
            params["route"] = route

        # Add HTTP-Referer and X-Title headers via extra_headers
        # These help OpenRouter track usage and may be required for some features
        extra_headers = kwargs.pop("extra_headers", {})
        if "HTTP-Referer" not in extra_headers:
            extra_headers["HTTP-Referer"] = "https://github.com/anthropics/sem-mem"
        if "X-Title" not in extra_headers:
            extra_headers["X-Title"] = "Sem-Mem"

        response = self._client.chat.completions.create(
            **params,
            extra_headers=extra_headers,
        )

        return ChatResponse(
            text=response.choices[0].message.content or "",
            response_id=None,
            raw_response=response,
        )
