"""
Decorator-based API for adding semantic memory to chatbots.

Provides three levels of integration:
- @with_rag: Just RAG retrieval (adds context to prompts)
- @with_instructions: Just persistent instructions
- @with_memory: Full memory system (RAG + instructions + conversation state)
"""

from functools import wraps
from typing import Callable, Optional, Any
from .core import SemanticMemory


def with_rag(
    memory: SemanticMemory,
    limit: int = 3,
    threshold: float = 0.40,
    context_prefix: str = "Relevant context:\n",
    expand_query: bool = True,
):
    """
    Decorator that adds RAG (retrieval-augmented generation) to a chat function.

    The decorated function receives the original user input, but the memory
    system will have already retrieved relevant context and made it available.

    Args:
        memory: SemanticMemory instance
        limit: Max number of memories to retrieve
        threshold: Similarity threshold for retrieval
        context_prefix: Text to prepend to retrieved context
        expand_query: Use LLM to generate alternative query phrasings

    Example:
        memory = SemanticMemory(api_key="...")

        @with_rag(memory)
        def chat(user_input: str, context: str = "") -> str:
            # context contains retrieved memories
            prompt = f"{context}\n\nUser: {user_input}" if context else user_input
            return llm.complete(prompt)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(user_input: str, *args, **kwargs) -> Any:
            # Retrieve relevant memories
            memories, logs = memory.recall(user_input, limit=limit, threshold=threshold, expand_query=expand_query)

            # Build context string
            if memories:
                context = context_prefix + "\n".join(f"- {m}" for m in memories)
            else:
                context = ""

            # Inject context into kwargs
            kwargs["context"] = context
            kwargs["_memories"] = memories
            kwargs["_logs"] = logs

            return func(user_input, *args, **kwargs)
        return wrapper
    return decorator


def with_instructions(
    memory: SemanticMemory,
    default_instructions: str = "You are a helpful assistant.",
):
    """
    Decorator that adds persistent instructions to a chat function.

    Args:
        memory: SemanticMemory instance
        default_instructions: Fallback if no instructions file exists

    Example:
        memory = SemanticMemory(api_key="...")

        @with_instructions(memory)
        def chat(user_input: str, instructions: str = "") -> str:
            return llm.complete(user_input, system=instructions)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(user_input: str, *args, **kwargs) -> Any:
            # Load instructions
            instructions = memory.load_instructions() or default_instructions
            kwargs["instructions"] = instructions
            return func(user_input, *args, **kwargs)
        return wrapper
    return decorator


def with_memory(
    memory: SemanticMemory,
    limit: int = 3,
    threshold: float = 0.40,
    default_instructions: str = "You are a helpful assistant.",
    auto_remember: bool = False,
    expand_query: bool = True,
):
    """
    Full-featured decorator combining RAG, instructions, and conversation state.

    This decorator handles:
    - Retrieving relevant memories (RAG)
    - Loading persistent instructions
    - Managing conversation state via previous_response_id
    - Optionally auto-saving responses to memory

    Args:
        memory: SemanticMemory instance
        limit: Max memories to retrieve
        threshold: Similarity threshold
        default_instructions: Fallback instructions
        auto_remember: If True, automatically save each exchange to L2
        expand_query: Use LLM to generate alternative query phrasings

    Example:
        memory = SemanticMemory(api_key="...")

        @with_memory(memory)
        def chat(user_input: str, **mem) -> str:
            # mem contains: instructions, context, memories, logs
            # Use memory.chat_with_memory() for full Responses API integration
            response, resp_id, mems, logs = memory.chat_with_memory(
                user_input,
                previous_response_id=mem.get("previous_response_id")
            )
            return response

        # Or for simple use cases, just use the injected context:
        @with_memory(memory)
        def simple_chat(user_input: str, context: str = "", instructions: str = "", **_) -> str:
            prompt = f"{instructions}\n\n{context}\n\nUser: {user_input}"
            return my_llm(prompt)
    """
    def decorator(func: Callable) -> Callable:
        # Track conversation state per-session
        conversation_state = {"previous_response_id": None}

        @wraps(func)
        def wrapper(user_input: str, *args, **kwargs) -> Any:
            # Retrieve relevant memories
            memories, logs = memory.recall(user_input, limit=limit, threshold=threshold, expand_query=expand_query)

            # Build context string
            if memories:
                context = "Relevant context:\n" + "\n".join(f"- {m}" for m in memories)
            else:
                context = ""

            # Load instructions
            instructions = memory.load_instructions() or default_instructions

            # Inject all memory-related data
            kwargs["context"] = context
            kwargs["instructions"] = instructions
            kwargs["memories"] = memories
            kwargs["logs"] = logs
            kwargs["previous_response_id"] = conversation_state["previous_response_id"]

            # Helper to update conversation state
            def set_response_id(resp_id: str):
                conversation_state["previous_response_id"] = resp_id
            kwargs["_set_response_id"] = set_response_id

            # Helper to remember something
            kwargs["_remember"] = memory.remember

            result = func(user_input, *args, **kwargs)

            # Auto-remember if enabled
            if auto_remember and result:
                exchange = f"User: {user_input}\n\nAssistant: {result}"
                memory.remember(exchange, metadata={"source": "auto_remember"})

            return result
        return wrapper
    return decorator


class MemoryChat:
    """
    Class-based alternative to decorators for stateful chat sessions.

    Example:
        memory = SemanticMemory(api_key="...")
        chat = MemoryChat(memory)

        response = chat.send("Hello!")
        response = chat.send("What did I just say?")  # Has conversation context

        chat.remember("Important fact to save")
        chat.add_instruction("Always be concise")
        chat.save_thread()  # Save entire conversation to L2
    """

    def __init__(
        self,
        memory: SemanticMemory,
        default_instructions: str = "You are a helpful assistant.",
    ):
        self.memory = memory
        self.default_instructions = default_instructions
        self.previous_response_id: Optional[str] = None
        self.messages: list = []

    def send(self, user_input: str) -> str:
        """Send a message and get a response."""
        response_text, response_id, memories, logs = self.memory.chat_with_memory(
            user_input,
            previous_response_id=self.previous_response_id
        )
        self.previous_response_id = response_id
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text

    def remember(self, fact: str, metadata: dict = None) -> str:
        """Save a fact to L2 memory."""
        return self.memory.remember(fact, metadata)

    def add_instruction(self, instruction: str):
        """Add a permanent instruction."""
        self.memory.add_instruction(instruction)

    def save_thread(self, thread_name: str = "chat") -> int:
        """Save the current conversation to L2 memory."""
        return self.memory.save_thread_to_memory(self.messages, thread_name)

    def new_thread(self):
        """Start a new conversation (clears state but keeps memory)."""
        self.previous_response_id = None
        self.messages = []

    @property
    def instructions(self) -> str:
        """Get current instructions."""
        return self.memory.load_instructions() or self.default_instructions
