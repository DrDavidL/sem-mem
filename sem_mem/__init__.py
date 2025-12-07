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

from .core import SemanticMemory, SmartCache
from .decorators import with_memory, with_instructions, with_rag, MemoryChat
from .config import get_api_key, get_config
from .vector_index import HNSWIndex, migrate_lsh_to_hnsw

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
    "MemoryChat",
    "with_memory",
    "with_instructions",
    "with_rag",
    "get_api_key",
    "get_config",
]

if _has_async:
    __all__.append("AsyncSemanticMemory")

__version__ = "0.1.0"
