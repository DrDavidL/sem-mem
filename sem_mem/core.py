import os
import threading
from datetime import datetime
from math import exp, log
import numpy as np
from collections import OrderedDict
from pypdf import PdfReader
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from .config import (
    QUERY_EXPANSION_MODEL,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBEDDING_PROVIDER,
    get_api_key,
    get_provider_kwargs,
    # Outcome-based learning
    OUTCOME_LEARNING_ENABLED,
    OUTCOME_EWMA_ALPHA,
    OUTCOME_RETRIEVAL_ALPHA,
    PATTERN_MIN_SUCCESSES,
    PATTERN_MIN_UTILITY,
    OUTCOME_VALUES,
    # Time-aware retrieval
    TIME_DECAY_ENABLED,
    TIME_DECAY_HALF_LIFE_DAYS,
    TIME_DECAY_ALPHA,
)
from .vector_index import HNSWIndex
from .thread_storage import ThreadStorage
from .backup import MemoryBackup
from .lexical_index import LexicalIndex, looks_like_identifier, merge_search_results
from .providers import (
    get_chat_provider,
    get_embedding_provider,
    BaseChatProvider,
    BaseEmbeddingProvider,
)
from .web_search import WebSearchManager, format_search_results_for_context

if TYPE_CHECKING:
    from openai import OpenAI


# =============================================================================
# Time-Aware Scoring (Phase 3)
# =============================================================================

def time_adjusted_score(
    sim_score: float,
    timestamp: Optional[str],
    half_life_days: float = TIME_DECAY_HALF_LIFE_DAYS,
    alpha: float = TIME_DECAY_ALPHA,
) -> float:
    """
    Apply time decay to a similarity score.

    Uses exponential decay with a half-life model:
    - After half_life_days, the time weight drops to 50%
    - After 2*half_life_days, the time weight drops to 25%
    - The alpha floor ensures old memories don't disappear completely

    Formula: sim_score * (alpha + (1 - alpha) * decay_factor)
    Where: decay_factor = exp(-decay_rate * age_days)
           decay_rate = ln(2) / half_life_days

    Args:
        sim_score: Raw similarity score (typically 0-1)
        timestamp: ISO format timestamp string (e.g., "2024-01-15T10:30:00")
        half_life_days: Days until decay reaches 50%
        alpha: Floor weight (0-1). Old memories retain at least alpha * sim_score

    Returns:
        Time-adjusted score. For recent memories, close to sim_score.
        For old memories, approaches alpha * sim_score.

    Examples:
        >>> # Fresh memory (today) - no decay
        >>> time_adjusted_score(0.8, datetime.now().isoformat())  # ≈ 0.8

        >>> # 30-day-old memory with default half_life=30
        >>> time_adjusted_score(0.8, "...")  # ≈ 0.8 * (0.3 + 0.7 * 0.5) = 0.52

        >>> # Very old memory - approaches floor
        >>> time_adjusted_score(0.8, "2020-01-01T00:00:00")  # ≈ 0.8 * 0.3 = 0.24
    """
    if not timestamp:
        # No timestamp - don't apply decay
        return sim_score

    try:
        created_at = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        age_days = (datetime.now() - created_at.replace(tzinfo=None)).days
    except (ValueError, TypeError):
        # Invalid timestamp - don't apply decay
        return sim_score

    if age_days <= 0:
        # Future or same-day memory - no decay
        return sim_score

    # Exponential decay: half-life model
    decay_rate = log(2) / half_life_days
    decay_factor = exp(-decay_rate * age_days)

    # Apply floor: alpha + (1 - alpha) * decay
    time_weight = alpha + (1 - alpha) * decay_factor

    return sim_score * time_weight


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

{file_access_context}"""

# File access context template (included when files are whitelisted)
_FILE_ACCESS_CONTEXT_TEMPLATE = """
File Access:
You have access to whitelisted files from the codebase. When files are available:
- **Available Files**: Listed below with path, size, and type (text/pdf/word)
- You can reference file contents when discussing code, documentation, or project details
- Use this knowledge to give accurate, context-aware answers about the project

{file_list}"""


def get_memory_system_context(include_files: bool = False) -> str:
    """Get the memory system context with current date/time.

    Args:
        include_files: If True, include whitelisted file list in context

    Returns:
        System context string for the agent
    """
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    file_access_context = ""
    if include_files:
        try:
            from .file_access import load_whitelist, get_file_info
            allowed = load_whitelist()
            if allowed:
                files = get_file_info(allowed)
                if files:
                    # Format file list compactly
                    file_lines = []
                    for f in files[:100]:  # Limit to first 100 files
                        size_kb = f["size"] / 1024
                        file_lines.append(f"  - {f['path']} ({size_kb:.1f}KB, {f['content_type']})")

                    file_list = "\n".join(file_lines)
                    if len(files) > 100:
                        file_list += f"\n  ... and {len(files) - 100} more files"

                    file_access_context = _FILE_ACCESS_CONTEXT_TEMPLATE.format(file_list=file_list)
        except ImportError:
            pass  # file_access module not available

    return _MEMORY_SYSTEM_CONTEXT_TEMPLATE.format(
        datetime=now,
        file_access_context=file_access_context
    )


# For backwards compatibility, provide a static version (without timestamp)
MEMORY_SYSTEM_CONTEXT = _MEMORY_SYSTEM_CONTEXT_TEMPLATE.format(
    datetime="(timestamp not available)",
    file_access_context=""
)


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
        # Legacy param (still works for backward compat)
        api_key: Optional[str] = None,
        # Provider selection (defaults from config/env)
        chat_provider: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        chat_api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        # Provider-specific settings
        provider_kwargs: Optional[Dict] = None,
        # Core params
        storage_dir: str = "./local_memory",
        cache_size: int = 20,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        chat_model: str = "gpt-5.1",
        reasoning_effort: str = "low",
        auto_memory: bool = True,
        auto_memory_threshold: float = 0.5,
        include_memory_context: bool = True,
        include_file_access: bool = False,
        web_search: bool = False,
    ):
        """
        Initialize SemanticMemory with provider abstraction.

        Args:
            api_key: Legacy OpenAI API key (for backward compat). Use chat_api_key/embedding_api_key instead.
            chat_provider: Chat provider name ("openai", "azure", "anthropic", etc.). Default from env.
            embedding_provider: Embedding provider name ("openai", "azure", "google", "ollama"). Default from env.
            chat_api_key: API key for chat provider. Falls back to api_key or env var.
            embedding_api_key: API key for embedding provider. Falls back to api_key or env var.
            provider_kwargs: Provider-specific settings (azure_endpoint, ollama_base_url, etc.).
            storage_dir: Directory for HNSW index and instructions.
            cache_size: L1 SmartCache capacity.
            embedding_model: Model for embeddings. Default from provider.
            embedding_dim: Embedding dimension. Auto-detected from provider if not specified.
            chat_model: Default chat model.
            reasoning_effort: For reasoning models: "low", "medium", "high".
            auto_memory: Enable auto-memory salience detection.
            auto_memory_threshold: Salience threshold for auto-save (0-1).
            include_memory_context: Include memory system context in instructions.
            include_file_access: Include file access context.
            web_search: Enable web search tool.
        """
        # Resolve provider names from env if not specified
        chat_provider = chat_provider or os.getenv("SEMMEM_CHAT_PROVIDER", DEFAULT_CHAT_PROVIDER)
        embedding_provider = embedding_provider or os.getenv("SEMMEM_EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER)

        # Resolve API keys
        resolved_chat_key = chat_api_key or api_key or get_api_key(provider=chat_provider)
        resolved_embedding_key = embedding_api_key or api_key or get_api_key(provider=embedding_provider)

        # Merge provider kwargs from env
        all_provider_kwargs = get_provider_kwargs()
        if provider_kwargs:
            all_provider_kwargs.update(provider_kwargs)

        # Initialize providers
        self._chat_provider: BaseChatProvider = get_chat_provider(
            chat_provider,
            api_key=resolved_chat_key,
            **all_provider_kwargs,
        )
        self._embedding_provider: BaseEmbeddingProvider = get_embedding_provider(
            embedding_provider,
            api_key=resolved_embedding_key,
            **all_provider_kwargs,
        )

        # Store provider names for metadata
        self._chat_provider_name = chat_provider
        self._embedding_provider_name = embedding_provider

        # Get embedding model and dimension from provider (single source of truth)
        self.embedding_model = embedding_model or self._embedding_provider.default_model
        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = self._embedding_provider.model_dimension(self.embedding_model)

        # Store other settings
        self.storage_dir = storage_dir
        self.instructions_file = os.path.join(storage_dir, "instructions.txt")
        self.chat_model = chat_model
        self.reasoning_effort = reasoning_effort
        self.auto_memory_enabled = auto_memory
        self.auto_memory_threshold = auto_memory_threshold
        self.include_memory_context = include_memory_context
        self.include_file_access = include_file_access
        self.web_search_enabled = web_search

        # Web search manager (auto-detects Google PSE or falls back to OpenAI)
        self._web_search = WebSearchManager()

        # Segmented LRU cache for hot items (L1)
        self.local_cache = SmartCache(capacity=cache_size)

        # HNSW index for L2 storage (with provider metadata)
        self.vector_index = HNSWIndex(
            storage_dir=storage_dir,
            embedding_dim=self.embedding_dim,
            embedding_provider=embedding_provider,
            embedding_model=self.embedding_model,
        )

        # Lexical index for hybrid search (Phase 2 enhancement)
        self._lexical_index_path = os.path.join(storage_dir, "lexical_index.json")
        self.lexical_index = LexicalIndex()
        self._load_lexical_index()

        # Auto-memory evaluator (lazy init to avoid import if disabled)
        self._auto_memory = None

        # Thread storage and backup manager
        self._thread_storage = ThreadStorage(storage_dir=storage_dir)
        self._backup = MemoryBackup(
            storage_dir=storage_dir,
            vector_index=self.vector_index,
            thread_storage=self._thread_storage,
            embedding_provider=self._embedding_provider,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
        )

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    @property
    def client(self) -> "OpenAI":
        """
        Legacy access to OpenAI client.

        For backward compatibility with code that accesses memory.client directly.
        Only available when using OpenAI provider.
        """
        if hasattr(self._chat_provider, 'client'):
            return self._chat_provider.client
        raise AttributeError(
            f"Provider '{self._chat_provider_name}' does not expose a client attribute. "
            f"Use the provider abstraction methods instead."
        )

    @property
    def auto_memory(self):
        """Lazy-initialize auto-memory evaluator."""
        if self._auto_memory is None and self.auto_memory_enabled:
            from .auto_memory import AutoMemory
            self._auto_memory = AutoMemory(
                chat_provider=self._chat_provider,
                salience_threshold=self.auto_memory_threshold,
            )
        return self._auto_memory

    def _is_reasoning_model(self, model: Optional[str] = None) -> bool:
        """Check if model is a reasoning model."""
        model = model or self.chat_model
        return self._chat_provider.is_reasoning_model(model)

    @property
    def is_exa_available(self) -> bool:
        """Check if Exa is configured for web search."""
        return self._web_search.is_exa_available()

    @property
    def is_google_pse_available(self) -> bool:
        """Check if Google PSE is configured for web search."""
        return self._web_search.is_google_pse_available()

    @property
    def web_search_backend(self) -> Optional[str]:
        """Get the active web search backend name."""
        return self._web_search.get_available_backend()

    def _load_lexical_index(self) -> None:
        """Load lexical index from disk, rebuilding if needed."""
        if self.lexical_index.load(self._lexical_index_path):
            return  # Loaded successfully

        # If no lexical index exists, rebuild from HNSW entries
        self._rebuild_lexical_index()

    def _rebuild_lexical_index(self) -> None:
        """Rebuild lexical index from all HNSW entries."""
        self.lexical_index.clear()
        for entry in self.vector_index.get_all_entries():
            text = entry.get("text", "")
            if text:
                # Look up the ID for this text
                result = self.vector_index.get_entry_by_text(text)
                if result:
                    memory_id, _ = result
                    self.lexical_index.add(str(memory_id), text)
        self.lexical_index.save(self._lexical_index_path)

    def _save_lexical_index(self) -> None:
        """Save lexical index to disk."""
        self.lexical_index.save(self._lexical_index_path)

    def _expand_query(self, query: str) -> List[str]:
        """
        Use a fast/cheap model to generate alternative search queries.
        This improves recall by searching with multiple phrasings.

        Returns list of queries including the original.
        """
        try:
            response = self._chat_provider.chat(
                messages=[{"role": "user", "content": query}],
                model=QUERY_EXPANSION_MODEL,
                instructions=(
                    "Generate 2-3 alternative phrasings of the user's question "
                    "that would match stored facts. Focus on:\n"
                    "- Converting questions to statements (e.g., 'Where do I live?' -> 'I live in')\n"
                    "- Extracting key entities and topics\n"
                    "- Using synonyms\n\n"
                    "Return ONLY the alternative queries, one per line. No numbering or explanations."
                ),
                temperature=0.3,
                max_tokens=100,
            )
            content = response.text or ""
            alternatives = content.strip().split("\n")
            # Clean and filter empty lines
            alternatives = [q.strip() for q in alternatives if q.strip()]
            # Always include original query first
            return [query] + alternatives[:3]  # Limit to 3 alternatives
        except Exception:
            # On any error, just use original query
            return [query]

    def load_instructions(self) -> str:
        """Load user instructions from file.

        If no instructions file exists, copies from instructions.example.txt
        if available in the project root.

        Note: This returns only user-defined instructions. System context
        (memory capabilities, file access, timestamps) is added separately
        in chat_with_memory() and is not user-editable.
        """
        if os.path.exists(self.instructions_file):
            with open(self.instructions_file, 'r') as f:
                return f.read()

        # Try to copy from example file on first run
        example_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "instructions.example.txt"
        )
        if os.path.exists(example_file):
            import shutil
            shutil.copy(example_file, self.instructions_file)
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
        """Generate embedding using the configured embedding provider."""
        return self._embedding_provider.embed_single(text, self.embedding_model)

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
            # Also add to lexical index for hybrid search
            self.lexical_index.add(str(entry_id), text)
            self._save_lexical_index()
            msg = f"Stored in HNSW index (id={entry_id})"
        else:
            msg = "Memory already exists."

        # Promote to L1 (RAM cache)
        if entry not in self.local_cache:
            self.local_cache.add(entry)
            msg += " & Promoted to Hot Cache."

        return msg

    def save_memory(
        self,
        text: str,
        kind: str = "fact",
        metadata: Optional[Dict] = None,
    ) -> Tuple[int, bool]:
        """
        Save a memory to L2 with specified kind.

        This is the unified entry point for storing memories, used by
        consolidation, auto-memory, and other subsystems.

        Args:
            text: Memory text content
            kind: Memory type ("fact", "pattern", "impression", "correction", etc.)
            metadata: Additional metadata

        Returns:
            Tuple of (memory_id, is_new) where is_new is False if text already existed
        """
        combined_metadata = dict(metadata or {})
        combined_metadata["kind"] = kind

        vector = self._get_embedding(text)
        memory_id, is_new = self.vector_index.add(text, vector, combined_metadata)

        if is_new:
            self.vector_index.save()
            # Also add to lexical index for hybrid search
            self.lexical_index.add(str(memory_id), text)
            self._save_lexical_index()

        return memory_id, is_new

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
        use_outcomes: bool = True,
        include_metadata: bool = False,
    ):
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0-1)
            expand_query: If True, use LLM to generate alternative query phrasings
            use_outcomes: If True, adjust scores based on utility_score (default True)
            include_metadata: If True, return rich dicts with id, scores, etc.
                             If False (default), return list of text strings.

        Returns:
            If include_metadata=False (default):
                Tuple of (memories, logs) where memories is list of text strings.
            If include_metadata=True:
                Tuple of (memories, logs) where memories is list of dicts with:
                    - id: memory_id for use with record_outcome()
                    - text: memory text
                    - sim_score: raw similarity score
                    - utility_score: outcome-based utility (0-1)
                    - final_score: combined score used for ranking
                    - is_pattern: whether this memory is a "proven useful" pattern

        Invariant: if both an older fact and a newer correction are retrieved,
        the model should see the correction after the older fact, treating it as overriding.
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

            if include_metadata:
                # For L1 hits, we need to look up memory_id from index
                result_dicts = []
                for score, item in l1_hits[:limit]:
                    entry_info = self.vector_index.get_entry_by_text(item['text'])
                    memory_id = entry_info[0] if entry_info else None
                    utility = item.get('utility_score', 0.5)
                    result_dicts.append({
                        "id": memory_id,
                        "text": item['text'],
                        "sim_score": round(score, 4),
                        "utility_score": round(utility, 4),
                        "final_score": round(score, 4),  # No adjustment for L1 hits
                        "is_pattern": item.get('is_pattern', False),
                    })
                return result_dicts, logs
            else:
                raw_memories = [x[1]['text'] for x in l1_hits[:limit]]
                return self._format_memories_for_display(raw_memories), logs

        # --- TIER 2: Search HNSW Index (with optional lexical hybrid) ---
        logs.append("🔍 L1 Miss... Searching HNSW index...")

        # Search HNSW with all query variants - returns (score, memory_id, entry) tuples
        results = self.vector_index.search(
            query_vectors=query_vecs,
            k=limit * 2,  # Get extra candidates for filtering
            threshold=threshold,
        )

        # Check if query looks like an identifier (file names, table names, etc.)
        # If so, also try lexical search and merge results
        if looks_like_identifier(query) and len(self.lexical_index) > 0:
            logs.append("🔤 Query looks like identifier, adding lexical search...")
            lexical_results = self.lexical_index.search(query, k=limit * 2)

            if lexical_results:
                # Convert lexical results to same format as vector results
                # lexical_results: [(doc_id, score), ...]
                lexical_entries = []
                for doc_id, lex_score in lexical_results:
                    entry = self.vector_index.get_entry(int(doc_id))
                    if entry:
                        lexical_entries.append((lex_score, int(doc_id), entry))

                if lexical_entries:
                    # Merge vector and lexical results
                    # Convert to dict for merging: {memory_id: (score, entry)}
                    vector_dict = {mem_id: (score, entry) for score, mem_id, entry in results}
                    lexical_dict = {mem_id: (score, entry) for score, mem_id, entry in lexical_entries}

                    # Combine scores with alpha=0.7 for vector, 0.3 for lexical
                    all_ids = set(vector_dict.keys()) | set(lexical_dict.keys())
                    merged = []
                    for mem_id in all_ids:
                        v_score, v_entry = vector_dict.get(mem_id, (0.0, None))
                        l_score, l_entry = lexical_dict.get(mem_id, (0.0, None))
                        entry = v_entry or l_entry
                        combined_score = 0.7 * v_score + 0.3 * l_score
                        if combined_score >= threshold:
                            merged.append((combined_score, mem_id, entry))

                    merged.sort(key=lambda x: x[0], reverse=True)
                    results = merged
                    logs.append(f"🔀 Merged {len(vector_dict)} vector + {len(lexical_dict)} lexical results")

        if not results:
            if include_metadata:
                return [], logs
            return [], logs

        # Apply outcome-based scoring adjustment
        if use_outcomes and OUTCOME_LEARNING_ENABLED:
            adjusted = []
            for sim_score, memory_id, entry in results:
                utility = entry.get("utility_score", 0.5)
                # final_score = sim_score + alpha * (utility - 0.5)
                # This adds up to +0.1 for utility=1.0 and subtracts up to -0.1 for utility=0.0
                final_score = sim_score + OUTCOME_RETRIEVAL_ALPHA * (utility - 0.5)
                adjusted.append((final_score, sim_score, memory_id, entry))
            adjusted.sort(key=lambda x: x[0], reverse=True)
            logs.append("📊 Applied outcome scoring")
        else:
            # No adjustment - keep original scores
            adjusted = [(sim_score, sim_score, memory_id, entry) for sim_score, memory_id, entry in results]

        # Apply time-decay scoring (Phase 3)
        if TIME_DECAY_ENABLED:
            time_adjusted = []
            for final_score, sim_score, memory_id, entry in adjusted:
                timestamp = entry.get("created_at")
                # Apply time decay to the final score
                decayed_score = time_adjusted_score(final_score, timestamp)
                time_adjusted.append((decayed_score, sim_score, memory_id, entry))
            time_adjusted.sort(key=lambda x: x[0], reverse=True)
            adjusted = time_adjusted
            logs.append("⏱️ Applied time-decay scoring")

        # Limit results
        top_results = adjusted[:limit]

        # Promote to L1 cache
        promoted = 0
        for _, _, _, entry in top_results:
            self.local_cache.add(entry)
            promoted += 1

        if promoted:
            logs.append(f"🔼 Promoted {promoted} to Probation.")

        # Format output
        if include_metadata:
            return [
                {
                    "id": memory_id,
                    "text": entry["text"],
                    "sim_score": round(sim_score, 4),
                    "utility_score": round(entry.get("utility_score", 0.5), 4),
                    "final_score": round(final_score, 4),
                    "is_pattern": entry.get("is_pattern", False),
                }
                for final_score, sim_score, memory_id, entry in top_results
            ], logs
        else:
            raw_memories = [entry["text"] for _, _, _, entry in top_results]
            return self._format_memories_for_display(raw_memories), logs

    def _format_memories_for_display(self, memories: List[str]) -> List[str]:
        """
        Format memories for display, parsing structured memories and
        ordering so corrections appear last (most recent).

        Args:
            memories: List of raw memory strings (may be JSON-structured or plain text)

        Returns:
            List of formatted memory strings, with corrections at the end
        """
        from .auto_memory import parse_structured_memory, MEMORY_KIND_CORRECTION

        regular_memories = []
        corrections = []

        for mem in memories:
            text, metadata = parse_structured_memory(mem)
            kind = metadata.get("kind", "fact")

            # Format with kind prefix for clarity when it's a special type
            if kind == MEMORY_KIND_CORRECTION:
                corrections.append(f"[CORRECTION] {text}")
            elif kind in ("identity", "preference", "decision"):
                regular_memories.append(text)
            else:
                regular_memories.append(text)

        # Return regular memories first, then corrections (so model sees corrections last)
        return regular_memories + corrections

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

    def record_outcome(
        self,
        memory_id: int,
        outcome: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Record whether a retrieved memory was helpful.

        This updates the memory's utility_score using EWMA (Exponentially Weighted
        Moving Average), allowing the system to learn which memories actually help
        users rather than just which are semantically similar.

        Args:
            memory_id: Internal ID from recall results (with include_metadata=True)
            outcome: "success" (memory helped), "failure" (memory was unhelpful),
                    or "neutral" (no clear signal)
            metadata: Optional context about the outcome (reserved for future use)

        Returns:
            Updated entry dict with new utility scores:
                - utility_score: EWMA-smoothed utility (0-1)
                - success_count: Total successful retrievals
                - failure_count: Total failed retrievals
                - last_outcome_at: ISO timestamp of this update
                - is_pattern: True if promoted to "proven useful" tier

        Raises:
            ValueError: If outcome is not "success", "failure", or "neutral"
            KeyError: If memory_id not found in index

        Example:
            # Get memories with IDs
            memories, logs = memory.recall("my query", include_metadata=True)

            # After user feedback or implicit signal
            for mem in memories:
                if user_found_helpful(mem):
                    memory.record_outcome(mem["id"], "success")
                else:
                    memory.record_outcome(mem["id"], "failure")
        """
        if outcome not in OUTCOME_VALUES:
            raise ValueError(
                f"Invalid outcome: {outcome}. Must be one of {list(OUTCOME_VALUES.keys())}"
            )

        entry = self.vector_index.get_entry(memory_id)
        if not entry:
            raise KeyError(f"Memory not found: {memory_id}")

        # EWMA update: new = alpha * value + (1 - alpha) * old
        old_utility = entry.get("utility_score", 0.5)
        value = OUTCOME_VALUES[outcome]
        new_utility = OUTCOME_EWMA_ALPHA * value + (1 - OUTCOME_EWMA_ALPHA) * old_utility

        # Update counts
        success_count = entry.get("success_count", 0)
        failure_count = entry.get("failure_count", 0)
        if outcome == "success":
            success_count += 1
        elif outcome == "failure":
            failure_count += 1

        # Check for pattern promotion
        # A memory becomes a "pattern" when it has proven consistently useful
        is_pattern = entry.get("is_pattern", False)
        if not is_pattern:
            if success_count >= PATTERN_MIN_SUCCESSES and new_utility >= PATTERN_MIN_UTILITY:
                is_pattern = True

        # Apply updates
        updates = {
            "utility_score": new_utility,
            "success_count": success_count,
            "failure_count": failure_count,
            "last_outcome_at": datetime.now().isoformat(),
            "is_pattern": is_pattern,
        }
        self.vector_index.update_entry(memory_id, **updates)
        self.vector_index.save()  # Persist changes

        return {**entry, **updates}

    def chat_with_memory(
        self,
        user_query: str,
        previous_response_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        auto_remember: Optional[bool] = None,
        web_search: Optional[bool] = None,
        instructions: Optional[str] = None,
    ):
        """
        Chat with memory context using the configured provider.

        Uses the provider's chat() method, which handles:
        - OpenAI Responses API when previous_response_id is provided
        - OpenAI Chat Completions otherwise
        - Other providers use their native APIs

        Args:
            user_query: The user's question
            previous_response_id: For conversation continuity (OpenAI only)
            model: Override the default chat model
            reasoning_effort: For reasoning models: "low", "medium", "high"
            auto_remember: Override auto-memory setting for this call
            web_search: Override web search setting for this call
            instructions: Override instructions (None = use global instructions.txt)

        Returns:
            Tuple of (response_text, response_id, memories, logs)
        """
        memories, logs = self.recall(user_query)

        # Build instructions: memory context (with current timestamp) + user instructions
        # Use provided instructions, or fall back to global instructions.txt
        user_instructions = instructions if instructions is not None else (self.load_instructions() or "You are a helpful assistant.")
        if self.include_memory_context:
            system_context = get_memory_system_context(include_files=self.include_file_access)
            full_instructions = f"{system_context}\n\n{user_instructions}"
        else:
            full_instructions = user_instructions

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

        # Build provider-specific kwargs
        kwargs: Dict = {}

        # Web search: use best available backend (Exa > Google PSE > OpenAI)
        web_search_context = ""
        if use_web_search:
            backend = self._web_search.get_available_backend()
            if backend in ("exa", "google_pse"):
                # Use Exa or Google PSE (inject results as context)
                try:
                    search_response = self._web_search.search(user_query, num_results=5)
                    if search_response and search_response.results:
                        web_search_context = format_search_results_for_context(search_response)
                        backend_name = "Exa" if search_response.backend == "exa" else "Google PSE"
                        logs.append(f"🔍 {backend_name}: {len(search_response.results)} results")
                except Exception as e:
                    logs.append(f"⚠️ {backend} error: {e}")
                    # Fall back to OpenAI web search
                    kwargs["tools"] = [{"type": "web_search_preview"}]
                    logs.append("🌐 Fallback to OpenAI web search")
            else:
                # Use OpenAI's web_search_preview tool
                kwargs["tools"] = [{"type": "web_search_preview"}]
                logs.append("🌐 OpenAI web search enabled")

        # Reasoning models require special handling
        if self._is_reasoning_model(use_model):
            kwargs["reasoning_effort"] = use_reasoning
            # Don't set temperature for reasoning models
        else:
            kwargs["temperature"] = 0.7

        # Inject web search results into context (for Google PSE)
        if web_search_context:
            input_text = f"{web_search_context}\n\n{input_text}"

        # For Responses API, we pass input_text as messages with one user message
        # The provider will handle Responses API vs Chat Completions based on previous_response_id
        messages = [{"role": "user", "content": input_text}]

        # Call the provider
        response = self._chat_provider.chat(
            messages=messages,
            model=use_model,
            instructions=full_instructions,
            previous_response_id=previous_response_id,
            use_responses_api=True,  # Prefer Responses API when available
            **kwargs,
        )

        response_text = response.text
        response_id = response.response_id

        # Auto-memory: evaluate and store salient exchanges
        should_auto_remember = auto_remember if auto_remember is not None else self.auto_memory_enabled
        if should_auto_remember and self.auto_memory:
            signal = self.auto_memory.evaluate(user_query, response_text)
            if signal.should_remember and signal.memory_text:
                # Use structured memory format for better retrieval handling
                from .auto_memory import format_structured_memory
                structured_memory = format_structured_memory(
                    text=signal.memory_text,
                    kind=signal.kind,
                    meta={
                        "source": "auto_memory",
                        "salience": signal.salience,
                        "reason": signal.reason,
                    }
                )
                self.remember(structured_memory, metadata={
                    "source": "auto_memory",
                    "kind": signal.kind,
                    "salience": signal.salience,
                })
                logs.append(f"💾 Auto-saved ({signal.kind}): {signal.reason} (salience: {signal.salience:.2f})")

        return response_text, response_id, memories, logs

    # ==========================================================================
    # Thread Persistence Methods
    # ==========================================================================

    def save_threads(self, threads: Dict[str, Dict]) -> None:
        """
        Save all conversation threads to disk.

        Args:
            threads: Dict mapping thread names to thread data
        """
        self._thread_storage.save_threads(threads)

    def load_threads(self) -> Dict[str, Dict]:
        """
        Load conversation threads from disk.

        Returns:
            Dict mapping thread names to thread data.
            Returns empty dict if no threads file exists.
        """
        return self._thread_storage.load_threads()

    def save_thread(self, name: str, thread: Dict) -> None:
        """
        Save a single conversation thread.

        Args:
            name: Thread name/identifier
            thread: Thread data dict
        """
        self._thread_storage.save_thread(name, thread)

    def get_thread(self, name: str) -> Optional[Dict]:
        """
        Get a single thread by name.

        Args:
            name: Thread name/identifier

        Returns:
            Thread data dict, or None if not found
        """
        return self._thread_storage.get_thread(name)

    def delete_thread(self, name: str) -> bool:
        """
        Delete a conversation thread from persistent storage.

        Note: This only removes from disk storage, not from any in-memory
        session state.

        Args:
            name: Thread name to delete

        Returns:
            True if deleted, False if thread didn't exist
        """
        return self._thread_storage.delete_thread(name)

    # ==========================================================================
    # Backup and Export Methods
    # ==========================================================================

    def export_all(
        self,
        include_vectors: bool = True,
        include_threads: bool = True,
    ) -> Dict:
        """
        Export all memory data to a JSON-serializable dict.

        Args:
            include_vectors: Include embedding vectors (larger but fully portable)
            include_threads: Include conversation threads

        Returns:
            Complete backup dict with instructions, memories, threads, and metadata
        """
        return self._backup.export_all(
            include_vectors=include_vectors,
            include_threads=include_threads,
        )

    def backup(self, backup_name: Optional[str] = None) -> Dict:
        """
        Create a timestamped backup of all memory data.

        Args:
            backup_name: Optional custom name (without .json).
                        If None, uses timestamp.

        Returns:
            {"path": "backups/...", "stats": {"memory_count": N, ...}}
        """
        return self._backup.create_backup(backup_name=backup_name)

    def restore(
        self,
        backup_name: str,
        merge: bool = False,
        re_embed: bool = False,
    ) -> Dict:
        """
        Restore memory from a backup file.

        Args:
            backup_name: Backup filename (with or without .json)
            merge: If True, merge with existing data. If False, replace all.
            re_embed: If True, recompute vectors using current embedding model.

        Returns:
            {
                "memories_added": N,
                "memories_skipped": N,
                "threads_restored": N,
                "re_embedded": N,
                "instructions_action": "replaced" | "kept"
            }

        Raises:
            FileNotFoundError: If backup doesn't exist
            EmbeddingMismatchError: If embedding model mismatch with re_embed=False
        """
        # Update backup's reference to vector_index in case it was recreated
        self._backup.vector_index = self.vector_index
        result = self._backup.restore_backup(
            backup_name=backup_name,
            merge=merge,
            re_embed=re_embed,
        )
        # Update our reference to the potentially new vector_index
        self.vector_index = self._backup.vector_index
        return result

    def list_backups(self) -> List[Dict]:
        """
        List available backups.

        Returns:
            List of backup info dicts with name, path, created_at,
            memory_count, thread_count
        """
        return self._backup.list_backups()

    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup file.

        Args:
            backup_name: Backup filename (with or without .json)

        Returns:
            True if deleted, False if didn't exist
        """
        return self._backup.delete_backup(backup_name)
