"""
HNSW-based vector index for semantic memory.

Replaces LSH with hnswlib for better recall and performance at scale.
"""

import os
import json
import threading
import warnings
import numpy as np
import hnswlib
from typing import List, Dict, Optional, Tuple

from .exceptions import EmbeddingMismatchError


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) index for vector similarity search.

    Provides O(log n) approximate nearest neighbor search with high recall.
    Persists to disk as index.bin + metadata.json files.
    """

    def __init__(
        self,
        storage_dir: str = "./local_memory",
        embedding_dim: int = 1536,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize HNSW index.

        Args:
            storage_dir: Directory for persistence
            embedding_dim: Dimension of embedding vectors
            max_elements: Maximum capacity (can be resized)
            ef_construction: Controls index quality (higher = better but slower build)
            M: Number of connections per layer (higher = better recall, more memory)
            embedding_provider: Provider name for embeddings (stored in metadata)
            embedding_model: Model name for embeddings (stored in metadata)

        Raises:
            EmbeddingMismatchError: If existing index was created with different provider/model
        """
        self.storage_dir = storage_dir
        self.embedding_dim = embedding_dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        self.index_path = os.path.join(storage_dir, "hnsw_index.bin")
        self.metadata_path = os.path.join(storage_dir, "hnsw_metadata.json")

        # Thread safety
        self._lock = threading.RLock()

        # ID -> memory entry mapping
        self._id_to_entry: Dict[int, Dict] = {}
        self._text_to_id: Dict[str, int] = {}
        self._next_id: int = 0

        # Initialize or load index
        self._index: Optional[hnswlib.Index] = None
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index or create new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load()
        else:
            self._create_new()

    def _create_new(self):
        """Create a fresh HNSW index."""
        self._index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )
        # Set ef (search quality) to a reasonable default
        self._index.set_ef(50)
        self._id_to_entry = {}
        self._text_to_id = {}
        self._next_id = 0

    def _ensure_outcome_defaults(self, entry: dict) -> dict:
        """Ensure outcome fields have defaults for backward compatibility."""
        entry.setdefault("utility_score", 0.5)
        entry.setdefault("success_count", 0)
        entry.setdefault("failure_count", 0)
        entry.setdefault("last_outcome_at", None)
        entry.setdefault("is_pattern", False)
        return entry

    def _load(self):
        """Load index and metadata from disk."""
        with self._lock:
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            # Validate embedding provider/model compatibility
            stored_provider = metadata.get('embedding_provider')
            stored_model = metadata.get('embedding_model')

            # Only validate if both the stored and current values are specified
            if stored_provider and self.embedding_provider:
                if stored_provider != self.embedding_provider or stored_model != self.embedding_model:
                    raise EmbeddingMismatchError(
                        stored_provider=stored_provider,
                        stored_model=stored_model,
                        current_provider=self.embedding_provider,
                        current_model=self.embedding_model,
                        storage_dir=self.storage_dir,
                    )
            elif stored_provider and not self.embedding_provider:
                # Use stored values if current not specified (migration path)
                self.embedding_provider = stored_provider
                self.embedding_model = stored_model
            elif self.embedding_provider and not stored_provider:
                # Old index without provider info - warn but continue
                warnings.warn(
                    f"Existing index has no provider metadata. "
                    f"Assuming it was created with {self.embedding_provider}/{self.embedding_model}. "
                    f"Delete the index if this is incorrect.",
                    UserWarning,
                )

            self._id_to_entry = {
                int(k): self._ensure_outcome_defaults(v)
                for k, v in metadata['entries'].items()
            }
            self._text_to_id = metadata['text_to_id']
            self._next_id = metadata['next_id']

            # Load HNSW index
            self._index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
            self._index.load_index(self.index_path, max_elements=self.max_elements)
            self._index.set_ef(50)

    def save(self):
        """Persist index and metadata to disk."""
        with self._lock:
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)

            # Save metadata with embedding provider info
            metadata = {
                'entries': self._id_to_entry,
                'text_to_id': self._text_to_id,
                'next_id': self._next_id,
                # Embedding provider info for validation on reload
                'embedding_provider': self.embedding_provider,
                'embedding_model': self.embedding_model,
                'embedding_dim': self.embedding_dim,
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)

            # Save HNSW index
            self._index.save_index(self.index_path)

    def add(self, text: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> Tuple[int, bool]:
        """
        Add a memory to the index.

        Args:
            text: The memory text
            vector: Embedding vector
            metadata: Optional metadata dict

        Returns:
            Tuple of (id, is_new) - is_new is False if text already existed
        """
        with self._lock:
            # Check for duplicate
            if text in self._text_to_id:
                return self._text_to_id[text], False

            # Assign ID
            entry_id = self._next_id
            self._next_id += 1

            # Resize if needed
            if entry_id >= self._index.get_max_elements():
                new_size = self._index.get_max_elements() * 2
                self._index.resize_index(new_size)

            # Store entry with outcome defaults
            entry = {
                'text': text,
                'vector': vector.tolist(),
                'metadata': metadata or {}
            }
            # Apply outcome field defaults for new entries
            self._ensure_outcome_defaults(entry)
            self._id_to_entry[entry_id] = entry
            self._text_to_id[text] = entry_id

            # Add to HNSW index
            self._index.add_items(vector.reshape(1, -1), np.array([entry_id]))

            return entry_id, True

    def search(
        self,
        query_vectors: List[np.ndarray],
        k: int = 10,
        threshold: float = 0.40,
    ) -> List[Tuple[float, int, Dict]]:
        """
        Search for similar memories using multiple query vectors.

        Args:
            query_vectors: List of query embedding vectors
            k: Number of candidates to retrieve per query
            threshold: Minimum similarity score (0-1, cosine similarity)

        Returns:
            List of (score, memory_id, entry) tuples, sorted by score descending
        """
        with self._lock:
            current_count = self._index.get_current_count()
            if current_count == 0:
                return []

            # Limit k to number of elements in index
            effective_k = min(k, current_count)

            # Search with all query vectors
            all_results: Dict[int, float] = {}  # id -> best score

            for query_vec in query_vectors:
                # hnswlib returns (ids, distances) where distance is 1 - cosine_similarity
                labels, distances = self._index.knn_query(query_vec.reshape(1, -1), k=effective_k)

                for label, dist in zip(labels[0], distances[0]):
                    # Convert distance to similarity (cosine space: similarity = 1 - distance)
                    similarity = 1 - dist
                    if similarity > threshold:
                        if label not in all_results or similarity > all_results[label]:
                            all_results[label] = similarity

            # Build results list
            results = []
            for entry_id, score in all_results.items():
                if entry_id in self._id_to_entry:
                    results.append((score, entry_id, self._id_to_entry[entry_id]))

            # Sort by score descending
            results.sort(key=lambda x: x[0], reverse=True)
            return results

    def get_all_entries(self) -> List[Dict]:
        """Get all stored entries."""
        with self._lock:
            return list(self._id_to_entry.values())

    def get_entry_count(self) -> int:
        """Get number of stored entries."""
        with self._lock:
            return len(self._id_to_entry)

    def contains(self, text: str) -> bool:
        """Check if text is already in index."""
        with self._lock:
            return text in self._text_to_id

    def get_entry(self, memory_id: int) -> Optional[Dict]:
        """Get entry by internal ID."""
        with self._lock:
            return self._id_to_entry.get(memory_id)

    def get_entry_by_text(self, text: str) -> Optional[Tuple[int, Dict]]:
        """Get entry by text content. Returns (id, entry) or None."""
        with self._lock:
            memory_id = self._text_to_id.get(text)
            if memory_id is not None:
                return memory_id, self._id_to_entry.get(memory_id)
            return None

    def update_entry(self, memory_id: int, **fields) -> bool:
        """Update specific fields on an entry. Returns True if found."""
        with self._lock:
            if memory_id not in self._id_to_entry:
                return False
            self._id_to_entry[memory_id].update(fields)
            return True

    def delete(self, text: str) -> bool:
        """
        Mark an entry as deleted.
        Note: hnswlib doesn't support true deletion, but we remove from metadata.
        The vector remains in the index but won't be returned in searches.
        """
        with self._lock:
            if text not in self._text_to_id:
                return False

            entry_id = self._text_to_id[text]
            del self._text_to_id[text]
            del self._id_to_entry[entry_id]
            # Note: Can't remove from hnswlib, but search won't return it
            return True


def migrate_lsh_to_hnsw(storage_dir: str, embedding_dim: int = 1536) -> int:
    """
    Migrate existing LSH bucket files to HNSW index.

    Args:
        storage_dir: Directory containing bucket_*.json files
        embedding_dim: Dimension of embeddings

    Returns:
        Number of entries migrated
    """
    import glob

    # Create new HNSW index
    index = HNSWIndex(storage_dir=storage_dir, embedding_dim=embedding_dim)

    # Find all bucket files
    bucket_pattern = os.path.join(storage_dir, "bucket_*.json")
    bucket_files = glob.glob(bucket_pattern)

    migrated = 0
    for bucket_file in bucket_files:
        with open(bucket_file, 'r') as f:
            entries = json.load(f)

        for entry in entries:
            text = entry.get('text', '')
            vector = np.array(entry.get('vector', []))
            metadata = entry.get('metadata', {})

            if text and len(vector) == embedding_dim:
                _, is_new = index.add(text, vector, metadata)
                if is_new:
                    migrated += 1

    # Save the new index
    index.save()

    return migrated
