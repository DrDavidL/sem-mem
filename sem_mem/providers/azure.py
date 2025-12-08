"""
Azure OpenAI provider implementation.

This is a Tier 1 provider that uses the same OpenAI SDK with Azure-specific
configuration. Key differences from standard OpenAI:
- Uses AzureOpenAI client instead of OpenAI
- Uses deployment names instead of model names
- Requires azure_endpoint and api_version
- Does NOT support Responses API (uses Chat Completions only)
"""

from typing import List, Dict, Optional, Any
import numpy as np

from openai import AzureOpenAI

from .base import (
    BaseChatProvider,
    BaseEmbeddingProvider,
    ChatResponse,
    EmbeddingResponse,
)


# Azure OpenAI supports the same embedding models as OpenAI
# Dimensions depend on what you deploy
AZURE_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Reasoning models available on Azure
AZURE_REASONING_MODELS = {"gpt-5.1", "o1", "o3", "o1-preview", "o1-mini"}


class AzureChatProvider(BaseChatProvider):
    """
    Azure OpenAI chat provider using Chat Completions API.

    Azure OpenAI does NOT support the Responses API, so all requests use
    Chat Completions with explicit message history.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-02-01",
        **kwargs,
    ):
        """
        Initialize Azure OpenAI chat provider.

        Args:
            api_key: Azure OpenAI API key. If not provided, uses AZURE_OPENAI_KEY env var.
            azure_endpoint: Azure OpenAI endpoint URL (e.g., https://xxx.openai.azure.com).
                           If not provided, uses AZURE_OPENAI_ENDPOINT env var.
            azure_api_version: API version to use (default: 2024-02-01).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        import os

        # Azure requires endpoint
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure endpoint required. Provide azure_endpoint parameter or "
                "set AZURE_OPENAI_ENDPOINT environment variable."
            )

        # Get API key from env if not provided
        key = api_key or os.getenv("AZURE_OPENAI_KEY")
        if not key:
            raise ValueError(
                "Azure API key required. Provide api_key parameter or "
                "set AZURE_OPENAI_KEY environment variable."
            )

        self._client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=azure_api_version,
        )
        self._api_version = azure_api_version

    @property
    def name(self) -> str:
        return "azure"

    @property
    def supports_responses_api(self) -> bool:
        # Azure does NOT support the Responses API
        return False

    @property
    def supports_reasoning(self) -> bool:
        return True

    @property
    def client(self) -> AzureOpenAI:
        """Access underlying Azure OpenAI client for advanced use cases."""
        return self._client

    def is_reasoning_model(self, model: str) -> bool:
        """Check if model is a reasoning model (o1, o3, gpt-5.1)."""
        return model in AZURE_REASONING_MODELS

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
        Send a chat request using Azure OpenAI Chat Completions.

        Note: previous_response_id is ignored since Azure doesn't support
        the Responses API. Callers must manage conversation state explicitly.

        Args:
            messages: List of message dicts (required).
            model: Deployment name in Azure.
            instructions: System instructions.
            previous_response_id: Ignored (Azure doesn't support Responses API).
            temperature: Sampling temperature (ignored for reasoning models).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional parameters:
                - reasoning_effort: "low", "medium", "high" for reasoning models
                - response_format: For structured output

        Returns:
            ChatResponse with generated text (response_id is always None).
        """
        if not messages:
            raise ValueError("messages required for Azure OpenAI")

        # Build messages list with optional system message
        api_messages = []
        if instructions:
            # Use "developer" role for reasoning models, "system" for others
            if self.is_reasoning_model(model):
                api_messages.append({"role": "developer", "content": instructions})
            else:
                api_messages.append({"role": "system", "content": instructions})
        api_messages.extend(messages)

        params: Dict[str, Any] = {
            "model": model,  # This is the deployment name in Azure
            "messages": api_messages,
        }

        # Handle reasoning models vs standard models
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if self.is_reasoning_model(model):
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            # Use max_completion_tokens for reasoning models
            if max_tokens is not None:
                params["max_completion_tokens"] = max_tokens
            # Don't set temperature for reasoning models
        else:
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens

        # Pass through additional params
        response_format = kwargs.pop("response_format", None)
        if response_format:
            params["response_format"] = response_format

        response = self._client.chat.completions.create(**params)

        return ChatResponse(
            text=response.choices[0].message.content or "",
            response_id=None,  # Azure doesn't have response IDs
            raw_response=response,
        )


class AzureEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-02-01",
        **kwargs,
    ):
        """
        Initialize Azure OpenAI embedding provider.

        Args:
            api_key: Azure OpenAI API key.
            azure_endpoint: Azure OpenAI endpoint URL.
            azure_api_version: API version to use.
            **kwargs: Additional arguments (ignored).
        """
        import os

        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure endpoint required. Provide azure_endpoint parameter or "
                "set AZURE_OPENAI_ENDPOINT environment variable."
            )

        key = api_key or os.getenv("AZURE_OPENAI_KEY")
        if not key:
            raise ValueError(
                "Azure API key required. Provide api_key parameter or "
                "set AZURE_OPENAI_KEY environment variable."
            )

        self._client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=azure_api_version,
        )

    @property
    def name(self) -> str:
        return "azure"

    @property
    def default_model(self) -> str:
        return "text-embedding-3-small"

    @property
    def default_dimension(self) -> int:
        return AZURE_EMBEDDING_DIMENSIONS[self.default_model]

    @property
    def client(self) -> AzureOpenAI:
        """Access underlying Azure OpenAI client for advanced use cases."""
        return self._client

    def model_dimension(self, model: str) -> int:
        """Get embedding dimension for a model (deployment)."""
        if model not in AZURE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unknown Azure embedding model: {model}. "
                f"Known models: {list(AZURE_EMBEDDING_DIMENSIONS.keys())}"
            )
        return AZURE_EMBEDDING_DIMENSIONS[model]

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for multiple texts."""
        model = model or self.default_model
        dimension = self.model_dimension(model)

        # model is the deployment name in Azure
        response = self._client.embeddings.create(input=texts, model=model)

        embeddings = [np.array(item.embedding) for item in response.data]

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
        response = self._client.embeddings.create(input=text, model=model)
        return np.array(response.data[0].embedding)


class AzureProvider(AzureChatProvider, AzureEmbeddingProvider):
    """
    Combined Azure OpenAI provider for both chat and embeddings.

    Convenience class that provides both interfaces with a single client.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_version: str = "2024-02-01",
        **kwargs,
    ):
        """
        Initialize combined Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key.
            azure_endpoint: Azure OpenAI endpoint URL.
            azure_api_version: API version to use.
            **kwargs: Additional arguments (ignored).
        """
        import os

        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise ValueError(
                "Azure endpoint required. Provide azure_endpoint parameter or "
                "set AZURE_OPENAI_ENDPOINT environment variable."
            )

        key = api_key or os.getenv("AZURE_OPENAI_KEY")
        if not key:
            raise ValueError(
                "Azure API key required. Provide api_key parameter or "
                "set AZURE_OPENAI_KEY environment variable."
            )

        # Initialize single client
        self._client = AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=azure_api_version,
        )
        self._api_version = azure_api_version
