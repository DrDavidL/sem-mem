"""
Async version of SemanticMemory for use with FastAPI and other async frameworks.
"""

import os
import json
import asyncio
import aiofiles
import numpy as np
from openai import AsyncOpenAI
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .core import SmartCache, get_memory_system_context
from .thread_storage import ThreadStorage
from .backup import MemoryBackup
from .config import (
    QUERY_EXPANSION_MODEL,
    OUTCOME_LEARNING_ENABLED,
    OUTCOME_EWMA_ALPHA,
    OUTCOME_RETRIEVAL_ALPHA,
    PATTERN_MIN_SUCCESSES,
    PATTERN_MIN_UTILITY,
    OUTCOME_VALUES,
    get_api_key,
    DEFAULT_CHAT_PROVIDER,
)
from .vector_index import HNSWIndex
from .providers import get_chat_provider


class AsyncSemanticMemory:
    """
    Async-native semantic memory with the same features as SemanticMemory.

    Use this with FastAPI, async web frameworks, or when processing
    multiple documents concurrently.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        storage_dir: str = "./local_memory",
        cache_size: int = 20,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        max_concurrent_embeddings: int = 10,
        chat_model: str = "gpt-5.1",
        reasoning_effort: str = "low",
        auto_memory: bool = True,
        auto_memory_threshold: float = 0.5,
        include_memory_context: bool = True,
        include_file_access: bool = False,
        web_search: bool = False,
    ):
        # Resolve API key
        resolved_api_key = api_key or get_api_key(provider=DEFAULT_CHAT_PROVIDER)

        self.client = AsyncOpenAI(api_key=resolved_api_key)
        self.storage_dir = storage_dir
        self.instructions_file = os.path.join(storage_dir, "instructions.txt")
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.chat_model = chat_model
        self.reasoning_effort = reasoning_effort
        self.auto_memory_enabled = auto_memory
        self.auto_memory_threshold = auto_memory_threshold
        self.include_memory_context = include_memory_context
        self.include_file_access = include_file_access
        self.web_search_enabled = web_search

        # Initialize sync chat provider for consolidator and other sync operations
        self._chat_provider = get_chat_provider(
            DEFAULT_CHAT_PROVIDER,
            api_key=resolved_api_key,
        )

        # Semaphore to limit concurrent API calls
        self._embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)

        # Reuse thread-safe SmartCache (sync operations are fast enough)
        self.local_cache = SmartCache(capacity=cache_size)

        # HNSW index for L2 storage (sync - operations are fast)
        self.vector_index = HNSWIndex(
            storage_dir=storage_dir,
            embedding_dim=embedding_dim,
        )

        # Auto-memory evaluator (lazy init)
        self._auto_memory = None

        # Thread storage and backup manager (sync operations, fast)
        self._thread_storage = ThreadStorage(storage_dir=storage_dir)
        self._backup = MemoryBackup(
            storage_dir=storage_dir,
            vector_index=self.vector_index,
            thread_storage=self._thread_storage,
            embedding_provider=None,  # Async uses its own embedding via _get_embedding
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

        # Ensure storage directory exists (sync is fine for init)
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    @property
    def auto_memory(self):
        """Lazy-initialize async auto-memory evaluator."""
        if self._auto_memory is None and self.auto_memory_enabled:
            from .auto_memory import AsyncAutoMemory
            self._auto_memory = AsyncAutoMemory(
                client=self.client,
                salience_threshold=self.auto_memory_threshold,
            )
        return self._auto_memory

    async def load_instructions(self) -> str:
        """Load user instructions from file.

        If no instructions file exists, copies from instructions.example.txt
        if available in the project root.

        Note: This returns only user-defined instructions. System context
        (memory capabilities, file access, timestamps) is added separately
        in chat_with_memory() and is not user-editable.
        """
        if os.path.exists(self.instructions_file):
            async with aiofiles.open(self.instructions_file, 'r') as f:
                return await f.read()

        # Try to copy from example file on first run
        example_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "instructions.example.txt"
        )
        if os.path.exists(example_file):
            import shutil
            shutil.copy(example_file, self.instructions_file)
            async with aiofiles.open(self.instructions_file, 'r') as f:
                return await f.read()

        return ""

    async def save_instructions(self, text: str):
        """Save instructions to file."""
        async with aiofiles.open(self.instructions_file, 'w') as f:
            await f.write(text)

    async def add_instruction(self, text: str):
        """Append a new instruction."""
        current = await self.load_instructions()
        new_text = f"{current}\n{text}" if current else text
        await self.save_instructions(new_text)

    async def _expand_query(self, query: str) -> List[str]:
        """
        Use a fast/cheap model to generate alternative search queries.
        This improves recall by searching with multiple phrasings.
        """
        try:
            response = await self.client.chat.completions.create(
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
            alternatives = [q.strip() for q in alternatives if q.strip()]
            return [query] + alternatives[:3]
        except Exception:
            return [query]

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with rate limiting."""
        async with self._embedding_semaphore:
            resp = await self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return np.array(resp.data[0].embedding)

    async def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get multiple embeddings concurrently."""
        tasks = [self._get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def remember(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a memory in L2 (HNSW index) and promote to L1 (cache).
        """
        vector = await self._get_embedding(text)

        entry = {
            "text": text,
            "vector": vector.tolist(),
            "metadata": metadata or {}
        }

        # Add to HNSW index (L2) - sync operation is fast
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

    async def remember_batch(
        self,
        texts: List[str],
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Store multiple memories concurrently.
        Returns count of new memories added.
        """
        # Get all embeddings concurrently
        vectors = await self._get_embeddings_batch(texts)

        added = 0
        for text, vector in zip(texts, vectors):
            entry = {
                "text": text,
                "vector": vector.tolist(),
                "metadata": metadata or {}
            }

            # Add to HNSW index
            _, is_new = self.vector_index.add(text, vector, metadata)
            if is_new:
                added += 1
                # Add to L1
                self.local_cache.add(entry)

        if added:
            self.vector_index.save()

        return added

    def save_memory(
        self,
        text: str,
        kind: str = "fact",
        metadata: Optional[Dict] = None,
    ) -> Tuple[int, bool]:
        """
        Save a memory to L2 with specified kind.

        This is a sync method used by consolidation and other subsystems.
        For async usage, use remember() instead.

        Args:
            text: Memory text content
            kind: Memory type ("fact", "pattern", "impression", "correction", etc.)
            metadata: Additional metadata

        Returns:
            Tuple of (memory_id, is_new) where is_new is False if text already existed
        """
        combined_metadata = dict(metadata or {})
        combined_metadata["kind"] = kind

        # Use sync embedding via a sync OpenAI client
        # This is acceptable for consolidation which runs offline
        from openai import OpenAI
        sync_client = OpenAI(api_key=self.client.api_key)
        response = sync_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        vector = np.array(response.data[0].embedding)

        memory_id, is_new = self.vector_index.add(text, vector, combined_metadata)

        if is_new:
            self.vector_index.save()

        return memory_id, is_new

    def _persist_hot_items(self):
        """Save frequently accessed L1 items to L2."""
        items = self.local_cache.get_pending_persist()
        for item in items:
            text = item['text']
            vector = np.array(item['vector'])
            metadata = item.get('metadata', {})

            # Add to HNSW if not already there
            _, is_new = self.vector_index.add(text, vector, metadata)
            if is_new:
                self.vector_index.save()

    async def recall(
        self,
        query: str,
        limit: int = 3,
        threshold: float = 0.40,
        expand_query: bool = True,
        use_outcomes: bool = True,
        include_metadata: bool = False,
    ) -> Tuple[List, List[str]]:
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
            queries = await self._expand_query(query)
            if len(queries) > 1:
                logs.append(f"🔍 Expanded to {len(queries)} queries")
        else:
            queries = [query]

        # Get embeddings for all query variants concurrently
        query_vecs = await self._get_embeddings_batch(queries)

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
            _, status, _ = self.local_cache.get(best_hit['text'])
            logs.append(f"⚡ L1 HIT ({status}) | Conf: {l1_hits[0][0]:.2f}")
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

        # --- TIER 2: Search HNSW Index ---
        logs.append("🔍 L1 Miss... Searching HNSW index...")

        # Search HNSW with all query variants - returns (score, memory_id, entry) tuples
        results = self.vector_index.search(
            query_vectors=query_vecs,
            k=limit * 2,
            threshold=threshold,
        )

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
                final_score = sim_score + OUTCOME_RETRIEVAL_ALPHA * (utility - 0.5)
                adjusted.append((final_score, sim_score, memory_id, entry))
            adjusted.sort(key=lambda x: x[0], reverse=True)
            logs.append("📊 Applied outcome scoring")
        else:
            # No adjustment - keep original scores
            adjusted = [(sim_score, sim_score, memory_id, entry) for sim_score, memory_id, entry in results]

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

    async def chat_with_memory(
        self,
        user_query: str,
        previous_response_id: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        auto_remember: Optional[bool] = None,
        web_search: Optional[bool] = None,
    ) -> Tuple[str, str, List[str], List[str]]:
        """
        Chat with RAG using Responses API.

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
        memories, logs = await self.recall(user_query)

        # Build instructions: memory context (with current timestamp) + user instructions
        user_instructions = await self.load_instructions() or "You are a helpful assistant."
        if self.include_memory_context:
            system_context = get_memory_system_context(include_files=self.include_file_access)
            instructions = f"{system_context}\n\n{user_instructions}"
        else:
            instructions = user_instructions

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
            create_params["reasoning"] = {"effort": use_reasoning}
        else:
            create_params["temperature"] = 0.7

        response = await self.client.responses.create(**create_params)  # type: ignore
        response_text = response.output_text

        # Auto-memory: evaluate and store salient exchanges
        should_auto_remember = auto_remember if auto_remember is not None else self.auto_memory_enabled
        if should_auto_remember and self.auto_memory:
            signal = await self.auto_memory.evaluate(user_query, response_text)
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
                await self.remember(structured_memory, metadata={
                    "source": "auto_memory",
                    "kind": signal.kind,
                    "salience": signal.salience,
                })
                logs.append(f"💾 Auto-saved ({signal.kind}): {signal.reason} (salience: {signal.salience:.2f})")

        return response_text, response.id, memories, logs

    async def bulk_learn_texts(
        self,
        texts: List[str],
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Learn multiple text chunks concurrently.
        More efficient than calling remember() in a loop.
        """
        return await self.remember_batch(texts, metadata)

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "l2_memories": self.vector_index.get_entry_count(),
            "l1_cache_size": len(list(self.local_cache)),
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

        Note: This is a sync method since the index operations are fast.

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
            memories, logs = await memory.recall("my query", include_metadata=True)

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

    # ==========================================================================
    # Thread Persistence Methods (async file I/O)
    # ==========================================================================

    async def save_threads(self, threads: Dict[str, Dict]) -> None:
        """
        Save all conversation threads to disk.

        Args:
            threads: Dict mapping thread names to thread data
        """
        data = {"version": self._thread_storage.VERSION, "threads": threads}
        temp_path = self._thread_storage.threads_file + ".tmp"
        async with aiofiles.open(temp_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))
        os.replace(temp_path, self._thread_storage.threads_file)

    async def load_threads(self) -> Dict[str, Dict]:
        """
        Load conversation threads from disk.

        Returns:
            Dict mapping thread names to thread data.
        """
        if not os.path.exists(self._thread_storage.threads_file):
            return {}
        try:
            async with aiofiles.open(self._thread_storage.threads_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return data.get("threads", {})
        except (json.JSONDecodeError, IOError):
            return {}

    async def save_thread(self, name: str, thread: Dict) -> None:
        """
        Save a single conversation thread.

        Args:
            name: Thread name/identifier
            thread: Thread data dict
        """
        threads = await self.load_threads()
        threads[name] = thread
        await self.save_threads(threads)

    async def get_thread(self, name: str) -> Optional[Dict]:
        """
        Get a single thread by name.

        Args:
            name: Thread name/identifier

        Returns:
            Thread data dict, or None if not found
        """
        threads = await self.load_threads()
        return threads.get(name)

    async def delete_thread(self, name: str) -> bool:
        """
        Delete a conversation thread from persistent storage.

        Args:
            name: Thread name to delete

        Returns:
            True if deleted, False if thread didn't exist
        """
        threads = await self.load_threads()
        if name in threads:
            del threads[name]
            await self.save_threads(threads)
            return True
        return False

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

        Note: This is sync because the actual work is fast (in-memory).

        Args:
            include_vectors: Include embedding vectors
            include_threads: Include conversation threads

        Returns:
            Complete backup dict
        """
        return self._backup.export_all(
            include_vectors=include_vectors,
            include_threads=include_threads,
        )

    async def backup(self, backup_name: Optional[str] = None) -> Dict:
        """
        Create a timestamped backup of all memory data.

        Args:
            backup_name: Optional custom name (without .json)

        Returns:
            {"path": "backups/...", "stats": {...}}
        """
        # Export data (sync, fast)
        data = self._backup.export_all(include_vectors=True, include_threads=True)

        # Generate filename
        if backup_name:
            safe_name = "".join(c for c in backup_name if c.isalnum() or c in "-_")
            filename = f"{safe_name}.json"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backup_{timestamp}.json"

        backup_path = os.path.join(self._backup.backups_dir, filename)
        relative_path = os.path.join("backups", filename)

        # Write async
        async with aiofiles.open(backup_path, 'w') as f:
            await f.write(json.dumps(data, indent=2))

        return {
            "path": relative_path,
            "stats": {
                "memory_count": data["metadata"]["memory_count"],
                "thread_count": data["metadata"]["thread_count"],
                "created_at": data["created_at"],
            }
        }

    async def restore(
        self,
        backup_name: str,
        merge: bool = False,
        re_embed: bool = False,
    ) -> Dict:
        """
        Restore memory from a backup file.

        Args:
            backup_name: Backup filename (with or without .json)
            merge: If True, merge with existing data. If False, replace.
            re_embed: If True, recompute vectors.

        Returns:
            Restore statistics dict

        Raises:
            FileNotFoundError: If backup doesn't exist
        """
        backup_path = self._backup.get_backup_path(backup_name)

        # Read async
        async with aiofiles.open(backup_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)

        # Most restore logic is sync (fast index operations)
        # Only re-embedding needs async
        if re_embed:
            # Re-embed memories async
            memories = data.get("memories", [])
            for memory in memories:
                vector = await self._get_embedding(memory["text"])
                memory["vector"] = vector.tolist()

        # Now do sync restore with pre-computed vectors
        self._backup.vector_index = self.vector_index
        # Temporarily mark as having vectors
        if "metadata" in data:
            data["metadata"]["include_vectors"] = True

        # Write modified data to temp file and restore
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            # Rename to backup dir temporarily
            temp_backup_name = f"_temp_restore_{os.getpid()}.json"
            final_temp_path = os.path.join(self._backup.backups_dir, temp_backup_name)
            os.rename(temp_path, final_temp_path)

            result = self._backup.restore_backup(
                backup_name=temp_backup_name,
                merge=merge,
                re_embed=False,  # Already re-embedded above
            )

            # Clean up temp backup
            os.remove(final_temp_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        self.vector_index = self._backup.vector_index
        if re_embed:
            result["re_embedded"] = len(data.get("memories", []))

        return result

    def list_backups(self) -> List[Dict]:
        """
        List available backups.

        Returns:
            List of backup info dicts
        """
        return self._backup.list_backups()

    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup file.

        Args:
            backup_name: Backup filename

        Returns:
            True if deleted, False if didn't exist
        """
        return self._backup.delete_backup(backup_name)
