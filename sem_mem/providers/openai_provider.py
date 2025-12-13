"""
OpenAI provider implementation.

This is the reference ("golden path") implementation that supports:
- Responses API for conversation continuity
- Chat Completions API for utility calls
- Embeddings API for vector generation
- Special handling for reasoning models (gpt-5.1, o1, o3)
- Automatic retry with exponential backoff for rate limits
"""

import logging
import time
from typing import List, Dict, Optional, Any, Callable, TypeVar

import numpy as np

from openai import OpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError

from .base import (
    BaseChatProvider,
    BaseEmbeddingProvider,
    ChatResponse,
    EmbeddingResponse,
    ToolCall,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_errors: tuple = (RateLimitError, APIConnectionError, APITimeoutError),
) -> T:
    """
    Execute a function with exponential backoff retry on specific errors.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for delay after each retry
        retryable_errors: Tuple of exception types to retry on

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_errors as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise

            # Check for specific rate limit info in the error
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = min(float(retry_after), max_delay)
                    except ValueError:
                        pass

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed with {type(e).__name__}: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
        except APIError as e:
            # For other API errors, check if it's a server error (5xx) worth retrying
            if hasattr(e, "status_code") and e.status_code and 500 <= e.status_code < 600:
                last_exception = e
                if attempt == max_retries:
                    logger.error(f"All {max_retries + 1} attempts failed: {e}")
                    raise
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed with server error: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                raise

    raise last_exception


# Embedding model dimensions (single source of truth)
OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Reasoning models that require special handling
OPENAI_REASONING_MODELS = {"gpt-5.1", "o1", "o3", "o1-preview", "o1-mini"}


class OpenAIChatProvider(BaseChatProvider):
    """
    OpenAI chat provider supporting both Responses API and Chat Completions.

    When previous_response_id is provided, uses Responses API for conversation
    continuity. Otherwise, falls back to Chat Completions API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI chat provider.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        self._client = OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supports_responses_api(self) -> bool:
        return True

    @property
    def supports_reasoning(self) -> bool:
        return True

    @property
    def client(self) -> OpenAI:
        """Access underlying OpenAI client for advanced use cases."""
        return self._client

    def is_reasoning_model(self, model: str) -> bool:
        """Check if model is a reasoning model (o1, o3, gpt-5.1)."""
        return model in OPENAI_REASONING_MODELS

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
        Send a chat request using either Responses API or Chat Completions.

        Uses Responses API when:
        - previous_response_id is provided, OR
        - use_responses_api=True in kwargs

        Uses Chat Completions otherwise.

        Args:
            messages: List of message dicts (required for Chat Completions).
            model: Model name.
            instructions: System instructions.
            previous_response_id: For conversation continuity (Responses API).
            temperature: Sampling temperature (ignored for reasoning models).
            max_tokens: Maximum tokens in response.
            **kwargs: Additional parameters:
                - reasoning_effort: "low", "medium", "high" for reasoning models
                - tools: List of tool definitions
                - response_format: For structured output
                - use_responses_api: Force Responses API even without previous_response_id

        Returns:
            ChatResponse with generated text and optional response_id.
        """
        use_responses_api = kwargs.pop("use_responses_api", False)

        # Decide which API to use
        if previous_response_id or use_responses_api:
            return self._chat_responses_api(
                messages=messages,
                model=model,
                instructions=instructions,
                previous_response_id=previous_response_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            return self._chat_completions_api(
                messages=messages,
                model=model,
                instructions=instructions,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

    def _chat_responses_api(
        self,
        messages: Optional[List[Dict[str, str]]],
        model: str,
        instructions: Optional[str] = None,
        previous_response_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """Use OpenAI Responses API for stateful conversations."""
        params: Dict[str, Any] = {"model": model}

        if instructions:
            params["instructions"] = instructions

        # Build input from messages if provided
        if messages:
            # Extract user message content for input
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                params["input"] = user_messages[-1].get("content", "")
        elif previous_response_id:
            # Must have either messages or previous_response_id
            pass
        else:
            raise ValueError("Either messages or previous_response_id required")

        if previous_response_id:
            params["previous_response_id"] = previous_response_id

        # Handle reasoning models vs standard models
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if self.is_reasoning_model(model):
            if reasoning_effort:
                params["reasoning"] = {"effort": reasoning_effort}
            # Don't set temperature for reasoning models
        else:
            if temperature is not None:
                params["temperature"] = temperature

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Pass through additional params (tools, etc.)
        tools = kwargs.pop("tools", None)
        if tools:
            params["tools"] = tools

        def _call():
            return self._client.responses.create(**params)

        response = retry_with_backoff(_call)

        # Check for tool calls in the response output
        tool_calls = None
        if hasattr(response, "output") and response.output:
            parsed_tool_calls = []
            for item in response.output:
                if hasattr(item, "type") and item.type == "function_call":
                    import json
                    try:
                        args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": item.arguments}
                    parsed_tool_calls.append(ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=args,
                    ))
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls

        return ChatResponse(
            text=response.output_text or "",
            response_id=response.id,
            raw_response=response,
            tool_calls=tool_calls,
        )

    def submit_tool_results(
        self,
        previous_response_id: str,
        tool_results: List[Dict[str, Any]],
        model: str,
        instructions: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Submit tool results to continue a conversation after tool calls.

        This is used when the model requests tool calls and we need to
        provide the results back to get a final response.

        Args:
            previous_response_id: The response ID from the tool-calling response.
            tool_results: List of dicts with 'call_id' and 'output' keys.
            model: Model name.
            instructions: System instructions.
            **kwargs: Additional parameters.

        Returns:
            ChatResponse with the model's final response.
        """
        params: Dict[str, Any] = {
            "model": model,
            "previous_response_id": previous_response_id,
            "input": tool_results,  # Tool results as input
        }

        if instructions:
            params["instructions"] = instructions

        # Pass through tools in case model needs another round
        tools = kwargs.pop("tools", None)
        if tools:
            params["tools"] = tools

        # Handle reasoning models
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if self.is_reasoning_model(model):
            if reasoning_effort:
                params["reasoning"] = {"effort": reasoning_effort}
        else:
            temperature = kwargs.pop("temperature", None)
            if temperature is not None:
                params["temperature"] = temperature

        def _call():
            return self._client.responses.create(**params)

        response = retry_with_backoff(_call)

        # Check for additional tool calls
        tool_calls = None
        if hasattr(response, "output") and response.output:
            import json
            parsed_tool_calls = []
            for item in response.output:
                if hasattr(item, "type") and item.type == "function_call":
                    try:
                        args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": item.arguments}
                    parsed_tool_calls.append(ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=args,
                    ))
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls

        return ChatResponse(
            text=response.output_text or "",
            response_id=response.id,
            raw_response=response,
            tool_calls=tool_calls,
        )

    def _chat_completions_api(
        self,
        messages: Optional[List[Dict[str, str]]],
        model: str,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> ChatResponse:
        """Use OpenAI Chat Completions API for stateless requests."""
        if not messages:
            raise ValueError("messages required for Chat Completions API")

        # Build messages list with optional system message
        api_messages = []
        if instructions:
            api_messages.append({"role": "system", "content": instructions})
        api_messages.extend(messages)

        params: Dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }

        # Handle reasoning models vs standard models
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if self.is_reasoning_model(model):
            if reasoning_effort:
                # Note: reasoning param syntax may differ in Chat Completions
                params["reasoning"] = {"effort": reasoning_effort}
        else:
            if temperature is not None:
                params["temperature"] = temperature

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Pass through additional params
        response_format = kwargs.pop("response_format", None)
        if response_format:
            params["response_format"] = response_format

        def _call():
            return self._client.chat.completions.create(**params)

        response = retry_with_backoff(_call)

        return ChatResponse(
            text=response.choices[0].message.content or "",
            response_id=None,  # Chat Completions doesn't have response IDs
            raw_response=response,
        )


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small by default."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        self._client = OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "text-embedding-3-small"

    @property
    def default_dimension(self) -> int:
        return OPENAI_EMBEDDING_DIMENSIONS[self.default_model]

    @property
    def client(self) -> OpenAI:
        """Access underlying OpenAI client for advanced use cases."""
        return self._client

    def model_dimension(self, model: str) -> int:
        """Get embedding dimension for a model."""
        if model not in OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unknown OpenAI embedding model: {model}. "
                f"Known models: {list(OPENAI_EMBEDDING_DIMENSIONS.keys())}"
            )
        return OPENAI_EMBEDDING_DIMENSIONS[model]

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> EmbeddingResponse:
        """Generate embeddings for multiple texts with retry on rate limits."""
        model = model or self.default_model
        dimension = self.model_dimension(model)

        def _call():
            return self._client.embeddings.create(input=texts, model=model)

        response = retry_with_backoff(_call)

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
        """Generate embedding for a single text with retry on rate limits."""
        model = model or self.default_model

        def _call():
            return self._client.embeddings.create(input=text, model=model)

        response = retry_with_backoff(_call)
        return np.array(response.data[0].embedding)


class OpenAIProvider(OpenAIChatProvider, OpenAIEmbeddingProvider):
    """
    Combined OpenAI provider for both chat and embeddings.

    This is a convenience class that provides both interfaces with a single
    client instance.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize combined OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        # Initialize single client
        self._client = OpenAI(api_key=api_key)
