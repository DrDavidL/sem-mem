"""
Google (Gemini) provider implementation.

This is a Tier 2 provider supporting both chat and embeddings.

Key differences from OpenAI:
- Uses google-generativeai SDK
- Different message format (Content objects)
- text-embedding-004 is the default embedding model (768 dimensions)
- Supports Gemini 2.0 and other models
"""

from typing import List, Dict, Optional, Any
import numpy as np

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False
    genai = None

from .base import (
    BaseChatProvider,
    BaseEmbeddingProvider,
    ChatResponse,
    EmbeddingResponse,
)


# Google embedding model dimensions
GOOGLE_EMBEDDING_DIMENSIONS = {
    "text-embedding-004": 768,
    "text-embedding-005": 768,
    "embedding-001": 768,
}


class GoogleChatProvider(BaseChatProvider):
    """
    Google Gemini chat provider using the generativeai SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Google Gemini chat provider.

        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            **kwargs: Additional arguments (ignored, for compatibility).

        Raises:
            ImportError: If google-generativeai package is not installed.
        """
        if not HAS_GOOGLE:
            raise ImportError(
                "Google provider requires the 'google-generativeai' package. "
                "Install it with: pip install google-generativeai"
            )

        import os
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Google API key required. Provide api_key parameter or "
                "set GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=key)
        self._api_key = key

    @property
    def name(self) -> str:
        return "google"

    @property
    def supports_responses_api(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        # Gemini has some reasoning capabilities but not like o1/o3
        return False

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
        Send a chat request to Google Gemini.

        Args:
            messages: List of message dicts (required).
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro").
            instructions: System instructions.
            previous_response_id: Ignored (Google doesn't support this).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional parameters.

        Returns:
            ChatResponse with generated text.
        """
        if not messages:
            raise ValueError("messages required for Google Gemini")

        # Create model with system instruction if provided
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens

        model_kwargs = {}
        if generation_config:
            model_kwargs["generation_config"] = generation_config
        if instructions:
            model_kwargs["system_instruction"] = instructions

        gemini_model = genai.GenerativeModel(model, **model_kwargs)

        # Convert messages to Gemini format
        # Gemini uses "user" and "model" roles
        history = []
        current_message = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Skip system messages (handled via system_instruction)
            if role == "system":
                continue

            # Map roles
            gemini_role = "model" if role == "assistant" else "user"

            if msg == messages[-1]:
                # Last message is sent separately
                current_message = content
            else:
                history.append({"role": gemini_role, "parts": [content]})

        if current_message is None:
            # If there's only one message, use it
            if messages:
                current_message = messages[-1].get("content", "")
            else:
                raise ValueError("At least one message required")

        # Start chat with history if any
        chat = gemini_model.start_chat(history=history)
        response = chat.send_message(current_message)

        return ChatResponse(
            text=response.text,
            response_id=None,
            raw_response=response,
        )


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Google embedding provider using text-embedding-004.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Google embedding provider.

        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            **kwargs: Additional arguments (ignored).

        Raises:
            ImportError: If google-generativeai package is not installed.
        """
        if not HAS_GOOGLE:
            raise ImportError(
                "Google provider requires the 'google-generativeai' package. "
                "Install it with: pip install google-generativeai"
            )

        import os
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Google API key required. Provide api_key parameter or "
                "set GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=key)
        self._api_key = key

    @property
    def name(self) -> str:
        return "google"

    @property
    def default_model(self) -> str:
        return "text-embedding-004"

    @property
    def default_dimension(self) -> int:
        return GOOGLE_EMBEDDING_DIMENSIONS[self.default_model]

    def model_dimension(self, model: str) -> int:
        """Get embedding dimension for a model."""
        if model not in GOOGLE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unknown Google embedding model: {model}. "
                f"Known models: {list(GOOGLE_EMBEDDING_DIMENSIONS.keys())}"
            )
        return GOOGLE_EMBEDDING_DIMENSIONS[model]

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for multiple texts."""
        model = model or self.default_model
        dimension = self.model_dimension(model)

        # Google's embed_content supports batching
        result = genai.embed_content(
            model=f"models/{model}",
            content=texts,
            task_type="retrieval_document",
        )

        # Handle both single and batch results
        if isinstance(result["embedding"][0], list):
            embeddings = [np.array(emb) for emb in result["embedding"]]
        else:
            # Single text returns a single embedding
            embeddings = [np.array(result["embedding"])]

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
        model = model or self.default_model
        result = genai.embed_content(
            model=f"models/{model}",
            content=text,
            task_type="retrieval_document",
        )
        return np.array(result["embedding"])


class GoogleProvider(GoogleChatProvider, GoogleEmbeddingProvider):
    """
    Combined Google provider for both chat and embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize combined Google provider.

        Args:
            api_key: Google API key.
            **kwargs: Additional arguments (ignored).
        """
        if not HAS_GOOGLE:
            raise ImportError(
                "Google provider requires the 'google-generativeai' package. "
                "Install it with: pip install google-generativeai"
            )

        import os
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Google API key required. Provide api_key parameter or "
                "set GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=key)
        self._api_key = key
