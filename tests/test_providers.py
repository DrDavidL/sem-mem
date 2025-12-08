"""
Tests for provider abstraction layer.

Tests provider error normalization and basic provider functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from sem_mem.providers import (
    get_chat_provider,
    get_embedding_provider,
    get_available_chat_providers,
    get_available_embedding_providers,
    BaseChatProvider,
    BaseEmbeddingProvider,
    ChatResponse,
    EmbeddingResponse,
)
from sem_mem.exceptions import (
    ProviderNotFoundError,
    ProviderConfigError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderBadRequestError,
)


# =============================================================================
# Provider Registry Tests
# =============================================================================


class TestProviderRegistry:
    """Tests for provider discovery and registration."""

    def test_openai_always_available(self):
        """OpenAI should always be in available providers."""
        chat_providers = get_available_chat_providers()
        embedding_providers = get_available_embedding_providers()

        assert "openai" in chat_providers
        assert "openai" in embedding_providers

    def test_azure_always_available(self):
        """Azure should always be available (uses openai SDK)."""
        chat_providers = get_available_chat_providers()
        embedding_providers = get_available_embedding_providers()

        assert "azure" in chat_providers
        assert "azure" in embedding_providers

    def test_unknown_provider_raises_error(self):
        """Unknown provider should raise ProviderNotFoundError."""
        with pytest.raises(ProviderNotFoundError) as exc_info:
            get_chat_provider("nonexistent_provider")

        assert "nonexistent_provider" in str(exc_info.value)
        assert "chat" in str(exc_info.value)

    def test_unknown_embedding_provider_raises_error(self):
        """Unknown embedding provider should raise ProviderNotFoundError."""
        with pytest.raises(ProviderNotFoundError) as exc_info:
            get_embedding_provider("nonexistent_provider")

        assert "nonexistent_provider" in str(exc_info.value)
        assert "embedding" in str(exc_info.value)


# =============================================================================
# Error Normalization Tests
# =============================================================================


class TestProviderErrorNormalization:
    """Tests for normalized provider error types."""

    def test_provider_api_error_includes_provider_name(self):
        """ProviderAPIError should include provider name in message."""
        error = ProviderAPIError("openai", "Something went wrong")
        assert "[openai]" in str(error)
        assert "Something went wrong" in str(error)

    def test_provider_auth_error_default_message(self):
        """ProviderAuthError should have helpful default message."""
        error = ProviderAuthError("anthropic")
        assert "[anthropic]" in str(error)
        assert "API key" in str(error)

    def test_provider_auth_error_custom_message(self):
        """ProviderAuthError should accept custom message."""
        error = ProviderAuthError("google", "Token expired")
        assert "[google]" in str(error)
        assert "Token expired" in str(error)

    def test_rate_limit_error_without_retry_after(self):
        """ProviderRateLimitError should work without retry_after."""
        error = ProviderRateLimitError("openai")
        assert "[openai]" in str(error)
        assert "Rate limit" in str(error)
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """ProviderRateLimitError should include retry_after in message."""
        error = ProviderRateLimitError("openai", retry_after=30.5)
        assert "[openai]" in str(error)
        assert "30.5 seconds" in str(error)
        assert error.retry_after == 30.5

    def test_bad_request_error_default_message(self):
        """ProviderBadRequestError should have helpful default message."""
        error = ProviderBadRequestError("azure")
        assert "[azure]" in str(error)
        assert "Bad request" in str(error)

    def test_error_preserves_original_exception(self):
        """Provider errors should preserve original exception."""
        original = ValueError("Original error")
        error = ProviderAPIError("openai", "Wrapped error", original_error=original)
        assert error.original_error is original

    def test_all_errors_inherit_from_base(self):
        """All provider errors should inherit from ProviderAPIError."""
        assert issubclass(ProviderAuthError, ProviderAPIError)
        assert issubclass(ProviderRateLimitError, ProviderAPIError)
        assert issubclass(ProviderBadRequestError, ProviderAPIError)


class TestOpenAIErrorNormalization:
    """Tests for OpenAI-specific error handling."""

    @patch("sem_mem.providers.openai_provider.OpenAI")
    def test_openai_auth_error_normalized(self, mock_openai_class):
        """OpenAI authentication error should be catchable."""
        from openai import AuthenticationError

        # Create a mock that raises auth error on chat
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Get provider
        provider = get_chat_provider("openai", api_key="test-key")

        # Simulate auth error from OpenAI
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=Mock(status_code=401),
            body=None,
        )

        # The raw OpenAI error should propagate (providers are thin shims)
        # Users can catch and convert if needed
        with pytest.raises(AuthenticationError):
            provider.chat(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4",
            )

    @patch("sem_mem.providers.openai_provider.OpenAI")
    def test_openai_rate_limit_error_propagates(self, mock_openai_class):
        """OpenAI rate limit error should propagate."""
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        provider = get_chat_provider("openai", api_key="test-key")

        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=Mock(status_code=429, headers={"retry-after": "30"}),
            body=None,
        )

        with pytest.raises(RateLimitError):
            provider.chat(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4",
            )


# =============================================================================
# Provider Interface Tests
# =============================================================================


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_chat_response_with_all_fields(self):
        """ChatResponse should accept all fields."""
        response = ChatResponse(
            text="Hello world",
            response_id="resp_123",
            raw_response={"raw": "data"},
        )
        assert response.text == "Hello world"
        assert response.response_id == "resp_123"
        assert response.raw_response == {"raw": "data"}

    def test_chat_response_minimal(self):
        """ChatResponse should work with just text."""
        response = ChatResponse(text="Hello")
        assert response.text == "Hello"
        assert response.response_id is None
        assert response.raw_response is None


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse dataclass."""

    def test_embedding_response_fields(self):
        """EmbeddingResponse should store all fields correctly."""
        embeddings = [np.array([1.0, 2.0, 3.0])]
        response = EmbeddingResponse(
            embeddings=embeddings,
            model="text-embedding-3-small",
            dimension=3,
        )
        assert len(response.embeddings) == 1
        assert response.model == "text-embedding-3-small"
        assert response.dimension == 3


# =============================================================================
# Provider Dimension Tests
# =============================================================================


class TestEmbeddingDimensions:
    """Tests for provider-driven embedding dimensions."""

    @patch("sem_mem.providers.openai_provider.OpenAI")
    def test_openai_dimension_lookup(self, mock_openai_class):
        """OpenAI provider should know its model dimensions."""
        mock_openai_class.return_value = MagicMock()

        provider = get_embedding_provider("openai", api_key="test-key")

        assert provider.model_dimension("text-embedding-3-small") == 1536
        assert provider.model_dimension("text-embedding-3-large") == 3072
        assert provider.model_dimension("text-embedding-ada-002") == 1536

    @patch("sem_mem.providers.openai_provider.OpenAI")
    def test_openai_unknown_model_raises(self, mock_openai_class):
        """OpenAI provider should raise for unknown models."""
        mock_openai_class.return_value = MagicMock()

        provider = get_embedding_provider("openai", api_key="test-key")

        with pytest.raises(ValueError) as exc_info:
            provider.model_dimension("unknown-model")

        assert "unknown-model" in str(exc_info.value).lower()

    @patch("sem_mem.providers.openai_provider.OpenAI")
    def test_openai_default_model(self, mock_openai_class):
        """OpenAI provider should have correct default model."""
        mock_openai_class.return_value = MagicMock()

        provider = get_embedding_provider("openai", api_key="test-key")

        assert provider.default_model == "text-embedding-3-small"
        assert provider.default_dimension == 1536


# =============================================================================
# Azure Provider Tests
# =============================================================================


class TestAzureProvider:
    """Tests specific to Azure OpenAI provider."""

    def test_azure_requires_endpoint(self):
        """Azure provider should require endpoint configuration."""
        # Clear any env vars that might be set
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises((ValueError, ProviderConfigError)):
                get_chat_provider("azure", api_key="test-key")

    @patch("sem_mem.providers.azure.AzureOpenAI")
    @patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com"})
    def test_azure_with_endpoint(self, mock_azure_class):
        """Azure provider should work with endpoint configured."""
        mock_azure_class.return_value = MagicMock()

        provider = get_chat_provider(
            "azure",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        assert provider.name == "azure"
        assert provider.supports_responses_api is False  # Azure doesn't support Responses API


# =============================================================================
# Chat Provider Contract Tests
# =============================================================================


class TestChatProviderContract:
    """Tests that all chat providers follow the same contract."""

    @patch("sem_mem.providers.openai_provider.OpenAI")
    def test_openai_provider_properties(self, mock_openai_class):
        """OpenAI provider should implement all required properties."""
        mock_openai_class.return_value = MagicMock()

        provider = get_chat_provider("openai", api_key="test-key")

        # Required properties
        assert isinstance(provider.name, str)
        assert isinstance(provider.supports_responses_api, bool)
        assert isinstance(provider.supports_reasoning, bool)

        # OpenAI-specific
        assert provider.name == "openai"
        assert provider.supports_responses_api is True
        assert provider.supports_reasoning is True

    @patch("sem_mem.providers.azure.AzureOpenAI")
    def test_azure_provider_properties(self, mock_azure_class):
        """Azure provider should implement all required properties."""
        mock_azure_class.return_value = MagicMock()

        provider = get_chat_provider(
            "azure",
            api_key="test-key",
            azure_endpoint="https://test.openai.azure.com",
        )

        assert provider.name == "azure"
        assert provider.supports_responses_api is False
        assert provider.supports_reasoning is True
