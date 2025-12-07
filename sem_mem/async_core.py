"""
Async version of SemanticMemory for use with FastAPI and other async frameworks.
"""

import os
import asyncio
import aiofiles
import numpy as np
from openai import AsyncOpenAI
from typing import List, Dict, Optional, Tuple
from .core import SmartCache, get_memory_system_context
from .config import QUERY_EXPANSION_MODEL
from .vector_index import HNSWIndex


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
        web_search: bool = False,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
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
        """Load instructions from file."""
        if os.path.exists(self.instructions_file):
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
    ) -> Tuple[List[str], List[str]]:
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0-1)
            expand_query: If True, use LLM to generate alternative query phrasings

        Returns:
            Tuple of (memories, logs)
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
            return [x[1]['text'] for x in l1_hits[:limit]], logs

        # --- TIER 2: Search HNSW Index ---
        logs.append("🔍 L1 Miss... Searching HNSW index...")

        # Search HNSW with all query variants (sync - very fast)
        results = self.vector_index.search(
            query_vectors=query_vecs,
            k=limit * 2,
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
            instructions = f"{get_memory_system_context()}\n\n{user_instructions}"
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
                await self.remember(signal.memory_text, metadata={
                    "source": "auto_memory",
                    "salience": signal.salience,
                    "reason": signal.reason,
                })
                logs.append(f"💾 Auto-saved: {signal.reason} (salience: {signal.salience:.2f})")

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
