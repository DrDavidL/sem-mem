"""
Anthropic (Claude) provider implementation.

This is a Tier 2 provider for chat only - Anthropic does not provide embeddings.
Use this with a separate embedding provider (e.g., OpenAI, Google).

Key differences from OpenAI:
- Uses anthropic SDK (Messages API)
- No embeddings support
- Different message format (system is separate param)
- Supports extended thinking for claude-3-5-sonnet and newer
"""

from typing import List, Dict, Optional, Any

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

from .base import BaseChatProvider, ChatResponse


# Claude models with extended thinking support
ANTHROPIC_REASONING_MODELS = {
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
}


class AnthropicChatProvider(BaseChatProvider):
    """
    Anthropic Claude chat provider using the Messages API.

    Note: Anthropic does not provide an embeddings API. Use a separate
    embedding provider (OpenAI, Google, Ollama) for embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Anthropic chat provider.

        Args:
            api_key: Anthropic API key. If not provided, uses ANTHROPIC_API_KEY env var.
            **kwargs: Additional arguments (ignored, for compatibility).

        Raises:
            ImportError: If anthropic package is not installed.
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            )

        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supports_responses_api(self) -> bool:
        # Anthropic doesn't have an equivalent to Responses API
        return False

    @property
    def supports_reasoning(self) -> bool:
        # Claude has extended thinking for some models
        return True

    @property
    def client(self) -> "anthropic.Anthropic":
        """Access underlying Anthropic client for advanced use cases."""
        return self._client

    def is_reasoning_model(self, model: str) -> bool:
        """Check if model supports extended thinking."""
        return model in ANTHROPIC_REASONING_MODELS

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
        Send a chat request using Anthropic Messages API.

        Args:
            messages: List of message dicts (required).
            model: Model name (e.g., "claude-sonnet-4-20250514").
            instructions: System instructions (mapped to 'system' parameter).
            previous_response_id: Ignored (Anthropic doesn't support this).
            temperature: Sampling temperature (0-1 for Anthropic).
            max_tokens: Maximum tokens in response (required by Anthropic).
            **kwargs: Additional parameters:
                - extended_thinking: Enable extended thinking for supported models
                - thinking_budget: Token budget for thinking (default 10000)

        Returns:
            ChatResponse with generated text (response_id is always None).
        """
        if not messages:
            raise ValueError("messages required for Anthropic")

        # Anthropic uses a separate 'system' parameter instead of system message
        # Convert messages to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                if not instructions:
                    instructions = content
                continue

            # Anthropic uses "user" and "assistant" roles
            if role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})

        if not anthropic_messages:
            raise ValueError("At least one user or assistant message required")

        # Build parameters
        params: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
        }

        if instructions:
            params["system"] = instructions

        # Handle extended thinking for supported models
        extended_thinking = kwargs.pop("extended_thinking", False)
        thinking_budget = kwargs.pop("thinking_budget", 10000)

        if extended_thinking and self.is_reasoning_model(model):
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Extended thinking doesn't support temperature
        else:
            if temperature is not None:
                params["temperature"] = temperature

        response = self._client.messages.create(**params)

        # Extract text from response
        # Anthropic returns content blocks, we need to extract text
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)

        return ChatResponse(
            text="".join(text_parts),
            response_id=None,  # Anthropic doesn't have response IDs
            raw_response=response,
        )
