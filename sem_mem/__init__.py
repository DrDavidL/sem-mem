"""
Sem-Mem: Tiered Semantic Memory for AI Agents

A decorator-based API for adding semantic memory to chatbots.

Usage:
    from sem_mem import SemanticMemory, with_memory

    # Option 1: Decorator
    memory = SemanticMemory(api_key="...")

    @with_memory(memory)
    def chat(user_input: str) -> str:
        # Your chatbot logic here
        return response

    # Option 2: Direct usage
    memory = SemanticMemory(api_key="...")
    response, response_id, memories, logs = memory.chat_with_memory(
        "What is X?",
        previous_response_id=prev_id
    )
"""

from .core import SemanticMemory, SmartCache, MEMORY_SYSTEM_CONTEXT, get_memory_system_context
from .decorators import with_memory, with_instructions, with_rag, MemoryChat
from .config import get_api_key, get_config, get_provider_api_key, get_provider_kwargs
from .vector_index import HNSWIndex, migrate_lsh_to_hnsw
from .thread_storage import ThreadStorage
from .backup import MemoryBackup
from .auto_memory import AutoMemory, AsyncAutoMemory, MemorySignal, compute_quick_salience
from .thread_utils import (
    generate_thread_title,
    estimate_message_tokens,
    select_summary_window,
    summarize_conversation_window,
    summarize_deleted_thread,
)
from .exceptions import (
    SemMemError,
    EmbeddingMismatchError,
    ProviderNotFoundError,
    ProviderConfigError,
    ProviderAPIError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderBadRequestError,
)
from .providers import (
    get_chat_provider,
    get_embedding_provider,
    get_available_chat_providers,
    get_available_embedding_providers,
    register_chat_provider,
    register_embedding_provider,
    BaseChatProvider,
    BaseEmbeddingProvider,
    ChatResponse,
    EmbeddingResponse,
)

# Async imports are optional (require aiofiles)
try:
    from .async_core import AsyncSemanticMemory
    _has_async = True
except ImportError:
    AsyncSemanticMemory = None
    _has_async = False

__all__ = [
    # Core
    "SemanticMemory",
    "SmartCache",
    "MEMORY_SYSTEM_CONTEXT",
    "get_memory_system_context",
    # Decorators
    "MemoryChat",
    "with_memory",
    "with_instructions",
    "with_rag",
    # Config
    "get_api_key",
    "get_config",
    "get_provider_api_key",
    "get_provider_kwargs",
    # Index
    "HNSWIndex",
    "migrate_lsh_to_hnsw",
    # Backup and Thread Storage
    "ThreadStorage",
    "MemoryBackup",
    # Auto-memory
    "AutoMemory",
    "AsyncAutoMemory",
    "MemorySignal",
    "compute_quick_salience",
    # Thread utils
    "generate_thread_title",
    "estimate_message_tokens",
    "select_summary_window",
    "summarize_conversation_window",
    "summarize_deleted_thread",
    # Exceptions
    "SemMemError",
    "EmbeddingMismatchError",
    "ProviderNotFoundError",
    "ProviderConfigError",
    "ProviderAPIError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderBadRequestError",
    # Providers
    "get_chat_provider",
    "get_embedding_provider",
    "get_available_chat_providers",
    "get_available_embedding_providers",
    "register_chat_provider",
    "register_embedding_provider",
    "BaseChatProvider",
    "BaseEmbeddingProvider",
    "ChatResponse",
    "EmbeddingResponse",
]

if _has_async:
    __all__.append("AsyncSemanticMemory")

__version__ = "0.1.0"
