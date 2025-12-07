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
from .config import get_api_key, get_config
from .vector_index import HNSWIndex, migrate_lsh_to_hnsw
from .auto_memory import AutoMemory, AsyncAutoMemory, MemorySignal, compute_quick_salience
from .thread_utils import (
    generate_thread_title,
    estimate_message_tokens,
    select_summary_window,
    summarize_conversation_window,
    summarize_deleted_thread,
)

# Async imports are optional (require aiofiles)
try:
    from .async_core import AsyncSemanticMemory
    _has_async = True
except ImportError:
    AsyncSemanticMemory = None
    _has_async = False

__all__ = [
    "SemanticMemory",
    "SmartCache",
    "MEMORY_SYSTEM_CONTEXT",
    "get_memory_system_context",
    "MemoryChat",
    "with_memory",
    "with_instructions",
    "with_rag",
    "get_api_key",
    "get_config",
    "HNSWIndex",
    "migrate_lsh_to_hnsw",
    "AutoMemory",
    "AsyncAutoMemory",
    "MemorySignal",
    "compute_quick_salience",
    "generate_thread_title",
    "estimate_message_tokens",
    "select_summary_window",
    "summarize_conversation_window",
    "summarize_deleted_thread",
]

if _has_async:
    __all__.append("AsyncSemanticMemory")

__version__ = "0.1.0"
