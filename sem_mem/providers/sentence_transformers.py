"""
Sentence Transformers embedding provider.

Local embedding generation using Hugging Face sentence-transformers models.
No API key required - runs entirely on CPU/GPU locally.

Recommended models:
- Qwen/Qwen3-Embedding-0.6B (1024 dim, multilingual, 32k context)
- sentence-transformers/all-MiniLM-L6-v2 (384 dim, fast, English-focused)
- BAAI/bge-small-en-v1.5 (384 dim, fast, English)
- BAAI/bge-large-en-v1.5 (1024 dim, high quality, English)
"""

import logging
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None

from .base import BaseEmbeddingProvider, EmbeddingResponse

logger = logging.getLogger(__name__)


# Model dimensions (can auto-detect, but caching known values is faster)
SENTENCE_TRANSFORMER_DIMENSIONS = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-m3": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "thenlper/gte-large": 1024,
    "thenlper/gte-base": 768,
    "thenlper/gte-small": 384,
    "intfloat/e5-large-v2": 1024,
    "intfloat/e5-base-v2": 768,
    "intfloat/e5-small-v2": 384,
}

# Default model - good balance of quality, size, and multilingual support
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Runs entirely locally - no API calls, no rate limits, no costs.
    Supports CPU and GPU inference.

    Example:
        >>> from sem_mem.providers import get_embedding_provider
        >>> provider = get_embedding_provider("sentence-transformers")
        >>> embedding = provider.embed_single("Hello, world!")
        >>> print(embedding.shape)  # (1024,) for Qwen3-Embedding-0.6B
    """

    def __init__(
        self,
        api_key: Optional[str] = None,  # Ignored, not needed for local models
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """
        Initialize sentence-transformers embedding provider.

        Args:
            api_key: Ignored (local models don't need API keys).
            model_name: Model to use. Defaults to Qwen/Qwen3-Embedding-0.6B.
            device: Device for inference ('cpu', 'cuda', 'mps', or None for auto).
            trust_remote_code: Whether to trust remote code for models like Qwen.
            **kwargs: Additional arguments (ignored for compatibility with other providers).

        Raises:
            ImportError: If sentence-transformers package is not installed.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "SentenceTransformer provider requires the 'sentence-transformers' package. "
                "Install it with: pip install sentence-transformers"
            )

        self._model_name = model_name or DEFAULT_MODEL
        self._device = device
        self._model: Optional[SentenceTransformer] = None
        self._trust_remote_code = trust_remote_code
        # Ignore kwargs from other providers (azure_endpoint, ollama_base_url, etc.)
        self._kwargs = {}

        # Dimension (lazy-load if not known)
        self._dimension: Optional[int] = SENTENCE_TRANSFORMER_DIMENSIONS.get(self._model_name)

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info(f"Loading sentence-transformer model: {self._model_name}")
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                trust_remote_code=self._trust_remote_code,
                **self._kwargs,
            )
            # Get actual dimension from model if not known
            if self._dimension is None:
                self._dimension = self._model.get_sentence_embedding_dimension()
                SENTENCE_TRANSFORMER_DIMENSIONS[self._model_name] = self._dimension
            logger.info(f"Model loaded. Dimension: {self._dimension}, Device: {self._model.device}")
        return self._model

    @property
    def name(self) -> str:
        return "sentence-transformers"

    @property
    def default_model(self) -> str:
        return self._model_name

    @property
    def default_dimension(self) -> int:
        if self._dimension is None:
            # Force model load to get dimension
            self._load_model()
        return self._dimension

    def model_dimension(self, model: str) -> int:
        """Get embedding dimension for a model."""
        # If it's our current model, return cached dimension
        if model == self._model_name:
            return self.default_dimension

        # Check known dimensions
        if model in SENTENCE_TRANSFORMER_DIMENSIONS:
            return SENTENCE_TRANSFORMER_DIMENSIONS[model]

        # Unknown model - would need to load to check
        raise ValueError(
            f"Unknown dimension for model: {model}. "
            f"Known models: {list(SENTENCE_TRANSFORMER_DIMENSIONS.keys())}. "
            f"You can add it to SENTENCE_TRANSFORMER_DIMENSIONS or load the model to auto-detect."
        )

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            model: Model to use. If different from initialized model, raises error.

        Returns:
            EmbeddingResponse with embeddings list, model name, and dimension.
        """
        if model and model != self._model_name:
            raise ValueError(
                f"Cannot switch models on the fly. Initialized with {self._model_name}, "
                f"but requested {model}. Create a new provider instance for a different model."
            )

        model = self._model_name
        st_model = self._load_model()

        # Generate embeddings (batch processing)
        embeddings_array = st_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=len(texts) > 10,
        )

        # Convert to list of numpy arrays
        embeddings = [np.array(emb) for emb in embeddings_array]

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            dimension=self._dimension,
        )

    def embed_single(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.
            model: Model to use. If different from initialized model, raises error.

        Returns:
            Embedding vector as numpy array.
        """
        if model and model != self._model_name:
            raise ValueError(
                f"Cannot switch models on the fly. Initialized with {self._model_name}, "
                f"but requested {model}. Create a new provider instance for a different model."
            )

        st_model = self._load_model()

        # Generate single embedding
        embedding = st_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.array(embedding)
