"""
Abstract base classes for LLM providers.

Each provider is a thin adapter over its respective SDK, mapping parameters
and responses to a common interface. This enables Sem-Mem to work with
multiple LLM backends while keeping the core logic provider-agnostic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class ChatResponse:
    """Unified response format across all chat providers."""

    text: str
    """The generated text response."""

    response_id: Optional[str] = None
    """Conversation continuity ID (OpenAI Responses API only)."""

    raw_response: Any = None
    """Original provider response for debugging/advanced use."""


@dataclass
class EmbeddingResponse:
    """Unified embedding format across all embedding providers."""

    embeddings: List[np.ndarray]
    """List of embedding vectors."""

    model: str
    """Model used to generate embeddings."""

    dimension: int
    """Dimension of each embedding vector."""


class BaseChatProvider(ABC):
    """
    Abstract base for chat/completion providers.

    Implementations are thin shims that:
    1. Map common parameters to provider-specific format
    2. Call the provider's SDK
    3. Transform responses to ChatResponse

    The `chat()` method handles both stateless requests (messages-based)
    and stateful requests (previous_response_id for OpenAI Responses API).
    Non-OpenAI providers simply ignore previous_response_id.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and error messages."""
        pass

    @property
    @abstractmethod
    def supports_responses_api(self) -> bool:
        """Whether provider supports OpenAI Responses API for conversation continuity."""
        pass

    @property
    @abstractmethod
    def supports_reasoning(self) -> bool:
        """Whether provider has reasoning/thinking models (e.g., o1, Claude extended thinking)."""
        pass

    @property
    def max_context_tokens(self) -> Optional[int]:
        """Maximum context window size. Override per model if needed."""
        return None

    @property
    def max_output_tokens(self) -> Optional[int]:
        """Maximum output tokens. Override per model if needed."""
        return None

    @abstractmethod
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
        Send a chat request.

        Args:
            messages: List of message dicts with 'role' and 'content'.
                     Can be None if using previous_response_id (OpenAI only).
            model: Model name/ID to use.
            instructions: System instructions (maps to system message or instructions param).
            previous_response_id: For conversation continuity (OpenAI Responses API only).
                                 Non-OpenAI providers ignore this.
            temperature: Sampling temperature (0-2). Some reasoning models don't support this.
            max_tokens: Maximum tokens in response.
            **kwargs: Provider-specific parameters (e.g., reasoning_effort, tools).

        Returns:
            ChatResponse with text, optional response_id, and raw response.

        Behavior by provider:
        - OpenAI with previous_response_id: Uses Responses API
        - OpenAI without previous_response_id: Uses Chat Completions API
        - Others: Uses messages + instructions, ignores previous_response_id
        """
        pass

    def is_reasoning_model(self, model: str) -> bool:
        """
        Check if a specific model is a reasoning model.

        Override in subclasses that have reasoning models.
        Default returns False.
        """
        return False


class BaseEmbeddingProvider(ABC):
    """
    Abstract base for embedding providers.

    Implementations provide:
    1. Model-specific dimension information (single source of truth)
    2. Text-to-embedding conversion via the provider's SDK
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and error messages."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default embedding model name."""
        pass

    @property
    @abstractmethod
    def default_dimension(self) -> int:
        """Default embedding dimension for the default model."""
        pass

    @abstractmethod
    def model_dimension(self, model: str) -> int:
        """
        Get embedding dimension for a specific model.

        This is the single source of truth for dimensions.
        Raises ValueError if model is unknown.
        """
        pass

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            model: Model to use. Defaults to default_model.

        Returns:
            EmbeddingResponse with embeddings list, model name, and dimension.
        """
        pass

    @abstractmethod
    def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Convenience method that calls embed() and returns first result.

        Args:
            text: Text to embed.
            model: Model to use. Defaults to default_model.

        Returns:
            Embedding vector as numpy array.
        """
        pass
