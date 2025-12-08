"""
Custom exceptions for sem-mem.

These exceptions provide helpful error messages with actionable next steps.
"""


class SemMemError(Exception):
    """Base exception for sem-mem errors."""

    pass


class EmbeddingMismatchError(SemMemError):
    """
    Raised when the configured embedding provider/model doesn't match the index.

    The HNSW index is locked to the embedding provider and model used to create it.
    Changing embeddings requires either deleting the index or migrating.
    """

    def __init__(
        self,
        stored_provider: str,
        stored_model: str,
        current_provider: str,
        current_model: str,
        storage_dir: str,
    ):
        self.stored_provider = stored_provider
        self.stored_model = stored_model
        self.current_provider = current_provider
        self.current_model = current_model
        self.storage_dir = storage_dir

        message = (
            f"Embedding mismatch detected!\n\n"
            f"Index was created with: {stored_provider}/{stored_model}\n"
            f"Current configuration: {current_provider}/{current_model}\n\n"
            f"To switch embedding providers, either:\n"
            f"  1. Delete the index:\n"
            f"     rm -rf {storage_dir}/hnsw_index.bin {storage_dir}/hnsw_metadata.json\n\n"
            f"  2. Migrate embeddings (preserves data):\n"
            f"     python -m sem_mem.migrate --storage-dir {storage_dir} "
            f"--provider {current_provider} --model {current_model}"
        )
        super().__init__(message)


class ProviderNotFoundError(SemMemError):
    """Raised when a requested provider is not available."""

    def __init__(self, provider: str, provider_type: str, available: list):
        self.provider = provider
        self.provider_type = provider_type
        self.available = available

        message = (
            f"Unknown {provider_type} provider: '{provider}'\n"
            f"Available providers: {', '.join(available)}"
        )
        super().__init__(message)


class ProviderConfigError(SemMemError):
    """Raised when provider configuration is invalid or missing."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        full_message = f"Configuration error for {provider} provider: {message}"
        super().__init__(full_message)


class EmbeddingDimensionError(SemMemError):
    """Raised when embedding dimensions don't match expected values."""

    def __init__(self, expected: int, actual: int, context: str = ""):
        self.expected = expected
        self.actual = actual
        self.context = context

        message = f"Embedding dimension mismatch: expected {expected}, got {actual}"
        if context:
            message += f" ({context})"
        super().__init__(message)


# =============================================================================
# Provider API Errors (normalized across providers)
# =============================================================================


class ProviderAPIError(SemMemError):
    """Base class for provider API errors."""

    def __init__(self, provider: str, message: str, original_error: Exception = None):
        self.provider = provider
        self.original_error = original_error
        full_message = f"[{provider}] {message}"
        super().__init__(full_message)


class ProviderAuthError(ProviderAPIError):
    """Raised when authentication fails (invalid API key, expired token, etc.)."""

    def __init__(self, provider: str, message: str = None, original_error: Exception = None):
        message = message or "Authentication failed. Check your API key."
        super().__init__(provider, message, original_error)


class ProviderRateLimitError(ProviderAPIError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        provider: str,
        message: str = None,
        retry_after: float = None,
        original_error: Exception = None,
    ):
        self.retry_after = retry_after
        message = message or "Rate limit exceeded."
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        super().__init__(provider, message, original_error)


class ProviderBadRequestError(ProviderAPIError):
    """Raised when the request is malformed or invalid."""

    def __init__(self, provider: str, message: str = None, original_error: Exception = None):
        message = message or "Bad request. Check your input parameters."
        super().__init__(provider, message, original_error)
