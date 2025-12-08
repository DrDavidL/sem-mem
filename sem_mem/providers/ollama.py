"""
Ollama provider implementation.

This is a Tier 2 provider for local inference via Ollama.
Supports both chat and embeddings with models running locally.

Key differences from cloud providers:
- No API key required
- Uses local HTTP API (default: http://localhost:11434)
- Model names depend on what's installed locally
- Embedding dimensions depend on the model used
"""

from typing import List, Dict, Optional, Any
import json
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

from .base import (
    BaseChatProvider,
    BaseEmbeddingProvider,
    ChatResponse,
    EmbeddingResponse,
)


# Common Ollama embedding model dimensions
# Users can add more by subclassing or we can expand this list
OLLAMA_EMBEDDING_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
    "bge-m3": 1024,
    "bge-large": 1024,
}

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaChatProvider(BaseChatProvider):
    """
    Ollama chat provider for local LLM inference.

    Requires Ollama running locally with desired models pulled.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,  # Ignored, Ollama doesn't need API key
        ollama_base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Ollama chat provider.

        Args:
            api_key: Ignored (Ollama doesn't require API key).
            ollama_base_url: Ollama server URL. Defaults to http://localhost:11434.
            **kwargs: Additional arguments (ignored, for compatibility).

        Raises:
            ImportError: If requests package is not installed.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "Ollama provider requires the 'requests' package. "
                "Install it with: pip install requests"
            )

        import os
        self._base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        )

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def supports_responses_api(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        # Some Ollama models may support reasoning, but we don't track them
        return False

    @property
    def base_url(self) -> str:
        """Get the Ollama server base URL."""
        return self._base_url

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
        Send a chat request to Ollama.

        Args:
            messages: List of message dicts (required).
            model: Model name (e.g., "llama3.2", "mistral", "codellama").
            instructions: System instructions.
            previous_response_id: Ignored (Ollama doesn't support this).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response (maps to num_predict).
            **kwargs: Additional options passed to Ollama.

        Returns:
            ChatResponse with generated text.
        """
        if not messages:
            raise ValueError("messages required for Ollama")

        # Build messages with optional system message
        ollama_messages = []
        if instructions:
            ollama_messages.append({"role": "system", "content": instructions})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            ollama_messages.append({"role": role, "content": content})

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,  # We don't support streaming in this implementation
        }

        # Build options
        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        if options:
            payload["options"] = options

        # Make request
        url = f"{self._base_url}/api/chat"
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        text = result.get("message", {}).get("content", "")

        return ChatResponse(
            text=text,
            response_id=None,
            raw_response=result,
        )


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """
    Ollama embedding provider for local embedding generation.

    Requires Ollama running locally with an embedding model pulled
    (e.g., nomic-embed-text, mxbai-embed-large).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,  # Ignored
        ollama_base_url: Optional[str] = None,
        default_embedding_model: str = "nomic-embed-text",
        **kwargs,
    ):
        """
        Initialize Ollama embedding provider.

        Args:
            api_key: Ignored (Ollama doesn't require API key).
            ollama_base_url: Ollama server URL. Defaults to http://localhost:11434.
            default_embedding_model: Default model to use for embeddings.
            **kwargs: Additional arguments (ignored).

        Raises:
            ImportError: If requests package is not installed.
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "Ollama provider requires the 'requests' package. "
                "Install it with: pip install requests"
            )

        import os
        self._base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        )
        self._default_model = default_embedding_model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return self._default_model

    @property
    def default_dimension(self) -> int:
        return self.model_dimension(self._default_model)

    @property
    def base_url(self) -> str:
        """Get the Ollama server base URL."""
        return self._base_url

    def model_dimension(self, model: str) -> int:
        """
        Get embedding dimension for a model.

        For unknown models, we need to query Ollama to get the actual dimension.
        This is done lazily on first embed call for efficiency.
        """
        if model in OLLAMA_EMBEDDING_DIMENSIONS:
            return OLLAMA_EMBEDDING_DIMENSIONS[model]

        # For unknown models, we need to make a test embedding call
        # This is expensive but necessary for custom/new models
        try:
            test_embedding = self._embed_single_raw("test", model)
            dimension = len(test_embedding)
            # Cache for future calls
            OLLAMA_EMBEDDING_DIMENSIONS[model] = dimension
            return dimension
        except Exception as e:
            raise ValueError(
                f"Unknown Ollama embedding model: {model}. "
                f"Known models: {list(OLLAMA_EMBEDDING_DIMENSIONS.keys())}. "
                f"Error probing model: {e}"
            )

    def _embed_single_raw(self, text: str, model: str) -> List[float]:
        """Make raw embedding request to Ollama."""
        url = f"{self._base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text,
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("embedding", [])

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for multiple texts."""
        model = model or self._default_model

        # Ollama doesn't support batch embeddings, so we need to make multiple calls
        embeddings = []
        for text in texts:
            embedding = self._embed_single_raw(text, model)
            embeddings.append(np.array(embedding))

        dimension = len(embeddings[0]) if embeddings else self.model_dimension(model)

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            dimension=dimension,
        )

    def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """Generate embedding for a single text."""
        model = model or self._default_model
        embedding = self._embed_single_raw(text, model)
        return np.array(embedding)


class OllamaProvider(OllamaChatProvider, OllamaEmbeddingProvider):
    """
    Combined Ollama provider for both chat and embeddings.

    Convenience class that provides both interfaces.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        default_embedding_model: str = "nomic-embed-text",
        **kwargs,
    ):
        """
        Initialize combined Ollama provider.

        Args:
            api_key: Ignored (Ollama doesn't require API key).
            ollama_base_url: Ollama server URL.
            default_embedding_model: Default model for embeddings.
            **kwargs: Additional arguments (ignored).
        """
        if not HAS_REQUESTS:
            raise ImportError(
                "Ollama provider requires the 'requests' package. "
                "Install it with: pip install requests"
            )

        import os
        self._base_url = (
            ollama_base_url
            or os.getenv("OLLAMA_BASE_URL")
            or DEFAULT_OLLAMA_BASE_URL
        )
        self._default_model = default_embedding_model
