import os
import threading
from datetime import datetime
import numpy as np
from openai import OpenAI
from collections import OrderedDict
from pypdf import PdfReader
from typing import List, Dict, Optional
from .config import QUERY_EXPANSION_MODEL
from .vector_index import HNSWIndex

# Template for system context (use get_memory_system_context() to get with current time)
_MEMORY_SYSTEM_CONTEXT_TEMPLATE = """Current date and time: {datetime}

You have access to a semantic memory system that helps you remember and recall information across conversations.

How your memory works:
- **Retrieved Context**: When relevant memories are found, they appear at the start of the user's message as "Relevant context from memory"
- **Auto-Memory**: Important facts from our conversations (like personal details, preferences, decisions) are automatically saved for future reference
- **Instructions**: You have persistent instructions that guide your behavior

When you see retrieved context:
- Treat it as reliable information from previous conversations
- Use it naturally without explicitly mentioning "my memory says..."
- If context seems outdated or contradicted by the user, trust the user's current input

You can help the user manage their memory by suggesting they use:
- "remember: <fact>" to explicitly save important information
- "instruct: <guideline>" to add persistent behavioral instructions
"""


def get_memory_system_context() -> str:
    """Get the memory system context with current date/time."""
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return _MEMORY_SYSTEM_CONTEXT_TEMPLATE.format(datetime=now)


# For backwards compatibility, provide a static version (without timestamp)
MEMORY_SYSTEM_CONTEXT = _MEMORY_SYSTEM_CONTEXT_TEMPLATE.format(datetime="(timestamp not available)")


class SmartCache:
    def __init__(self, capacity=20, protected_ratio=0.8, persist_threshold=6):
        self.capacity = capacity
        self.persist_threshold = persist_threshold  # Hits before auto-save to L2
        # Split memory into two segments
        self.protected_cap = int(capacity * protected_ratio)
        self.probation_cap = capacity - self.protected_cap

        # OrderedDict allows us to move items to 'end' (Newest) easily
        self.protected = OrderedDict()
        self.probation = OrderedDict()
        # Track hit counts per item (by text key)
        self.hit_counts = {}
        # Track items that need to be persisted to L2
        self.pending_persist = []
        # Thread safety for concurrent access
        self._lock = threading.RLock()

    def get(self, key_text):
        """
        Retrieves an item and updates its status (The "Reset" Logic).
        Returns (item, status, should_persist).
        """
        with self._lock:
            # 1. Check Protected (VIP)
            if key_text in self.protected:
                # HIT in Protected: "Reset" it to the newest position
                item = self.protected.pop(key_text)
                self.protected[key_text] = item
                should_persist = self._increment_hits(key_text, item)
                return item, "Protected", should_persist

            # 2. Check Probation (New/Transient)
            if key_text in self.probation:
                # HIT in Probation: PROMOTE to Protected
                item = self.probation.pop(key_text)
                self._add_to_protected(key_text, item)
                should_persist = self._increment_hits(key_text, item)
                return item, "Promoted to Protected", should_persist

            return None, "Miss", False

    def _increment_hits(self, key_text, item):
        """Increment hit count and check if item should be persisted."""
        self.hit_counts[key_text] = self.hit_counts.get(key_text, 0) + 1
        if self.hit_counts[key_text] == self.persist_threshold:
            self.pending_persist.append(item)
            return True
        return False

    def get_pending_persist(self):
        """Get and clear items pending persistence to L2."""
        with self._lock:
            items = self.pending_persist[:]
            self.pending_persist = []
            return items

    def add(self, item):
        """
        New items always start in Probation.
        """
        key = item['text']
        with self._lock:
            # If it's already known, just refresh it
            if key in self.protected or key in self.probation:
                # Release lock before calling get() which also acquires it
                pass
            else:
                # Add to Probation (Newest)
                self.probation[key] = item

                # If Probation is full, evict the oldest (FIFO behavior for new stuff)
                if len(self.probation) > self.probation_cap:
                    self.probation.popitem(last=False)  # last=False pops the OLDEST
                return
        # Refresh existing item (outside lock since get() acquires it)
        self.get(key)

    def _add_to_protected(self, key, item):
        """
        Handles the logic of adding to the VIP section.
        """
        self.protected[key] = item

        # If Protected is full, we don't delete. We DEMOTE the oldest VIP back to Probation.
        # This gives it one last chance to be used before dying.
        if len(self.protected) > self.protected_cap:
            demoted_key, demoted_val = self.protected.popitem(last=False)
            self.add(demoted_val) # Send back to Probation

    def list_items(self):
        """Helper to visualize the segments."""
        with self._lock:
            return {
                "Protected (Sticky)": list(reversed(self.protected.values())),
                "Probation (Transient)": list(reversed(self.probation.values()))
            }

    def __iter__(self):
        """Iterate over all cached items (Protected first, then Probation)."""
        with self._lock:
            # Create a snapshot to iterate over
            items = list(reversed(list(self.protected.values()))) + \
                    list(reversed(list(self.probation.values())))
        yield from items

    def __contains__(self, item):
        """Check if an item is in the cache by its text key."""
        key = item['text'] if isinstance(item, dict) else item
        with self._lock:
            return key in self.protected or key in self.probation


class SemanticMemory:
    def __init__(
        self,
        api_key: Optional[str] = None,
        storage_dir: str = "./local_memory",
        cache_size: int = 20,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        chat_model: str = "gpt-5.1",
        reasoning_effort: str = "low",
        auto_memory: bool = True,
        auto_memory_threshold: float = 0.5,
        include_memory_context: bool = True,
        web_search: bool = False,
    ):
        self.client = OpenAI(api_key=api_key)
        self.storage_dir = storage_dir
        self.instructions_file = os.path.join(storage_dir, "instructions.txt")
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.chat_model = chat_model
        self.reasoning_effort = reasoning_effort
        self.auto_memory_enabled = auto_memory
        self.auto_memory_threshold = auto_memory_threshold
        self.include_memory_context = include_memory_context
        self.web_search_enabled = web_search

        # Segmented LRU cache for hot items (L1)
        self.local_cache = SmartCache(capacity=cache_size)

        # HNSW index for L2 storage
        self.vector_index = HNSWIndex(
            storage_dir=storage_dir,
            embedding_dim=embedding_dim,
        )

        # Auto-memory evaluator (lazy init to avoid import if disabled)
        self._auto_memory = None

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    @property
    def auto_memory(self):
        """Lazy-initialize auto-memory evaluator."""
        if self._auto_memory is None and self.auto_memory_enabled:
            from .auto_memory import AutoMemory
            self._auto_memory = AutoMemory(
                client=self.client,
                salience_threshold=self.auto_memory_threshold,
            )
        return self._auto_memory

    def _is_reasoning_model(self) -> bool:
        """Check if current model is a reasoning model."""
        return self.chat_model in ("gpt-5.1", "o1", "o3")

    def _expand_query(self, query: str) -> List[str]:
        """
        Use a fast/cheap model to generate alternative search queries.
        This improves recall by searching with multiple phrasings.

        Returns list of queries including the original.
        """
        try:
            response = self.client.chat.completions.create(
                model=QUERY_EXPANSION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate 2-3 alternative phrasings of the user's question "
                            "that would match stored facts. Focus on:\n"
                            "- Converting questions to statements (e.g., 'Where do I live?' -> 'I live in')\n"
                            "- Extracting key entities and topics\n"
                            "- Using synonyms\n\n"
                            "Return ONLY the alternative queries, one per line. No numbering or explanations."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                temperature=0.3,
                max_tokens=100,
            )
            content = response.choices[0].message.content or ""
            alternatives = content.strip().split("\n")
            # Clean and filter empty lines
            alternatives = [q.strip() for q in alternatives if q.strip()]
            # Always include original query first
            return [query] + alternatives[:3]  # Limit to 3 alternatives
        except Exception:
            # On any error, just use original query
            return [query]

    def load_instructions(self) -> str:
        """Load instructions from file."""
        if os.path.exists(self.instructions_file):
            with open(self.instructions_file, 'r') as f:
                return f.read()
        return ""

    def save_instructions(self, text: str):
        """Save instructions to file."""
        with open(self.instructions_file, 'w') as f:
            f.write(text)

    def add_instruction(self, text: str):
        """Append a new instruction."""
        current = self.load_instructions()
        if current:
            new_text = f"{current}\n{text}"
        else:
            new_text = text
        self.save_instructions(new_text)

    def _get_embedding(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(input=text, model=self.embedding_model)
        return np.array(resp.data[0].embedding)

    def remember(self, text: str, metadata: Dict = None):
        """
        Store a memory in L2 (HNSW index) and promote to L1 (cache).
        """
        vector = self._get_embedding(text)

        entry = {
            "text": text,
            "vector": vector.tolist(),
            "metadata": metadata or {}
        }

        # Add to HNSW index (L2)
        entry_id, is_new = self.vector_index.add(text, vector, metadata)

        if is_new:
            self.vector_index.save()
            msg = f"Stored in HNSW index (id={entry_id})"
        else:
            msg = "Memory already exists."

        # Promote to L1 (RAM cache)
        if entry not in self.local_cache:
            self.local_cache.add(entry)
            msg += " & Promoted to Hot Cache."

        return msg

    def _persist_hot_items(self):
        """Save frequently accessed L1 items to L2 for long-term storage."""
        items = self.local_cache.get_pending_persist()
        for item in items:
            text = item['text']
            vector = np.array(item['vector'])
            metadata = item.get('metadata', {})

            # Add to HNSW if not already there
            _, is_new = self.vector_index.add(text, vector, metadata)
            if is_new:
                self.vector_index.save()

    def recall(
        self,
        query: str,
        limit: int = 3,
        threshold: float = 0.40,
        expand_query: bool = True,
    ):
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0-1)
            expand_query: If True, use LLM to generate alternative query phrasings
        """
        logs = []

        # --- Query Expansion ---
        if expand_query:
            queries = self._expand_query(query)
            if len(queries) > 1:
                logs.append(f"🔍 Expanded to {len(queries)} queries")
        else:
            queries = [query]

        # Get embeddings for all query variants
        query_vecs = [self._get_embedding(q) for q in queries]

        # --- TIER 1: Check Smart Cache ---
        all_cache_items = list(self.local_cache)

        l1_hits = []
        seen_texts = set()
        for query_vec in query_vecs:
            for item in all_cache_items:
                if item['text'] in seen_texts:
                    continue
                score = np.dot(query_vec, np.array(item['vector']))
                if score > threshold:
                    seen_texts.add(item['text'])
                    l1_hits.append((score, item))

        if l1_hits:
            l1_hits.sort(key=lambda x: x[0], reverse=True)
            best_hit = l1_hits[0][1]

            # Trigger the "Reset/Promote" logic on the SmartCache
            _, status, _ = self.local_cache.get(best_hit['text'])

            logs.append(f"⚡ L1 HIT ({status}) | Conf: {l1_hits[0][0]:.2f}")

            # Auto-persist frequently accessed items to L2
            self._persist_hot_items()

            return [x[1]['text'] for x in l1_hits[:limit]], logs

        # --- TIER 2: Search HNSW Index ---
        logs.append("🔍 L1 Miss... Searching HNSW index...")

        # Search HNSW with all query variants
        results = self.vector_index.search(
            query_vectors=query_vecs,
            k=limit * 2,  # Get extra candidates for filtering
            threshold=threshold,
        )

        if not results:
            return [], logs

        # Limit results
        top_results = results[:limit]

        # Promote to L1 cache
        promoted = 0
        for score, item in top_results:
            self.local_cache.add(item)
            promoted += 1

        if promoted:
            logs.append(f"🔼 Promoted {promoted} to Probation.")

        return [item['text'] for _, item in top_results], logs

    def bulk_learn_pdf(self, pdf_file, chunk_size=500):
        """Reads PDF, chunks it, and saves to L2."""
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if len(chunk.strip()) > 20:
                chunks.append(chunk)

        count = 0
        for c in chunks:
            self.remember(c, metadata={"source": "pdf_upload"})
            count += 1
        return count

    def import_memories(self, entries: List[Dict]) -> int:
        """
        Import memory entries (e.g., from a JSON export).

        Args:
            entries: List of dicts with 'text', 'vector', and optional 'metadata'

        Returns:
            Number of new entries added
        """
        added = 0
        for entry in entries:
            text = entry.get('text', '')
            vector = np.array(entry.get('vector', []))
            metadata = entry.get('metadata', {})

            if text and len(vector) == self.embedding_dim:
                _, is_new = self.vector_index.add(text, vector, metadata)
                if is_new:
                    added += 1

        if added:
            self.vector_index.save()

        return added

    def save_thread_to_memory(self, messages: List[Dict], thread_name: str = "thread"):
        """
        Save a conversation thread to L2 memory.
        Creates a summary of the conversation for semantic retrieval.
        """
        if not messages:
            return 0

        # Build conversation text
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Strip system logs and citations from assistant messages
            if role == "assistant":
                # Remove everything after **System Logs:** or **Retrieved Context:**
                if "**System Logs:**" in content:
                    content = content.split("**System Logs:**")[0].strip()
                if "**Retrieved Context:**" in content:
                    content = content.split("**Retrieved Context:**")[0].strip()
            conversation_parts.append(f"{role.upper()}: {content}")

        full_conversation = "\n\n".join(conversation_parts)

        # Chunk long conversations (max ~1500 chars per chunk for good embeddings)
        chunk_size = 1500
        chunks = []
        if len(full_conversation) <= chunk_size:
            chunks.append(full_conversation)
        else:
            # Split by message boundaries when possible
            current_chunk = ""
            for part in conversation_parts:
                if len(current_chunk) + len(part) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = part
                else:
                    current_chunk += "\n\n" + part if current_chunk else part
            if current_chunk:
                chunks.append(current_chunk.strip())

        # Save each chunk to L2
        count = 0
        for chunk in chunks:
            self.remember(chunk, metadata={"source": "thread", "thread_name": thread_name})
            count += 1

        return count

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "l2_memories": self.vector_index.get_entry_count(),
            "l1_cache_size": len(list(self.local_cache)),
            "instructions_length": len(self.load_instructions()),
        }

    def chat_with_memory(
        self,
        user_query: str,
        previous_response_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        auto_remember: Optional[bool] = None,
        web_search: Optional[bool] = None,
    ):
        """
        Agentic Wrapper using Responses API.

        Args:
            user_query: The user's question
            previous_response_id: For conversation continuity
            model: Override the default chat model (gpt-5.1, gpt-4.1)
            reasoning_effort: For reasoning models: "low", "medium", "high"
            auto_remember: Override auto-memory setting for this call
            web_search: Override web search setting for this call

        Returns:
            Tuple of (response_text, response_id, memories, logs)
        """
        memories, logs = self.recall(user_query)

        # Build instructions: memory context (with current timestamp) + user instructions
        user_instructions = self.load_instructions() or "You are a helpful assistant."
        if self.include_memory_context:
            instructions = f"{get_memory_system_context()}\n\n{user_instructions}"
        else:
            instructions = user_instructions

        # Build input with memory context
        if memories:
            context_block = "\n".join([f"- {m}" for m in memories])
            input_text = f"Relevant context from memory:\n{context_block}\n\nUser question: {user_query}"
        else:
            input_text = user_query

        # Use provided model or fall back to instance default
        use_model = model or self.chat_model
        use_reasoning = reasoning_effort or self.reasoning_effort
        use_web_search = web_search if web_search is not None else self.web_search_enabled

        # Build base parameters
        create_params: Dict = {
            "model": use_model,
            "instructions": instructions,
            "input": input_text,
        }

        if previous_response_id:
            create_params["previous_response_id"] = previous_response_id

        # Add web search tool if enabled
        if use_web_search:
            create_params["tools"] = [{"type": "web_search_preview"}]
            logs.append("🌐 Web search enabled")

        # Reasoning models (gpt-5.1, o1, o3) require special handling
        if use_model in ("gpt-5.1", "o1", "o3"):
            # Reasoning models don't support temperature
            # Use reasoning parameter for effort level
            create_params["reasoning"] = {"effort": use_reasoning}
        else:
            # Standard models support temperature
            create_params["temperature"] = 0.7

        response = self.client.responses.create(**create_params)  # type: ignore
        response_text = response.output_text

        # Auto-memory: evaluate and store salient exchanges
        should_auto_remember = auto_remember if auto_remember is not None else self.auto_memory_enabled
        if should_auto_remember and self.auto_memory:
            signal = self.auto_memory.evaluate(user_query, response_text)
            if signal.should_remember and signal.memory_text:
                self.remember(signal.memory_text, metadata={
                    "source": "auto_memory",
                    "salience": signal.salience,
                    "reason": signal.reason,
                })
                logs.append(f"💾 Auto-saved: {signal.reason} (salience: {signal.salience:.2f})")

        return response_text, response.id, memories, logs
