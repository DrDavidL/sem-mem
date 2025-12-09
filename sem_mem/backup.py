"""
Backup and restore module for sem-mem.

Provides full backup/restore functionality including:
- Memory vectors (HNSW index)
- Instructions
- Conversation threads
"""

import os
import json
import hashlib
from glob import glob
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .thread_storage import ThreadStorage
from .exceptions import EmbeddingMismatchError

if TYPE_CHECKING:
    from .vector_index import HNSWIndex
    from .providers import BaseEmbeddingProvider


class MemoryBackup:
    """
    Full backup/restore for sem-mem.

    Backup format (sem-mem-backup v1.0):
    {
        "format": "sem-mem-backup",
        "version": "1.0",
        "created_at": "2025-12-08T10:30:00Z",
        "instructions": "You are a helpful assistant...",
        "memories": [
            {"id": "abc123...", "text": "...", "vector": [...], "metadata": {...}}
        ],
        "threads": {
            "Thread 1": {"messages": [...], "title": "...", ...}
        },
        "metadata": {
            "memory_count": 42,
            "thread_count": 3,
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "include_vectors": true
        }
    }
    """

    FORMAT = "sem-mem-backup"
    VERSION = "1.0"

    def __init__(
        self,
        storage_dir: str = "./local_memory",
        vector_index: Optional["HNSWIndex"] = None,
        thread_storage: Optional["ThreadStorage"] = None,
        embedding_provider: Optional["BaseEmbeddingProvider"] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: int = 1536,
    ):
        """
        Initialize backup manager.

        Args:
            storage_dir: Base directory for memory storage
            vector_index: HNSWIndex instance for memory vectors
            thread_storage: ThreadStorage instance for conversation threads
            embedding_provider: Embedding provider for re-embedding on restore
            embedding_model: Embedding model name (for metadata)
            embedding_dim: Embedding dimension (for validation)
        """
        self.storage_dir = storage_dir
        self.backups_dir = os.path.join(storage_dir, "backups")
        self.instructions_file = os.path.join(storage_dir, "instructions.txt")

        self.vector_index = vector_index
        self.thread_storage = thread_storage or ThreadStorage(storage_dir)
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

        # Ensure backups directory exists
        if not os.path.exists(self.backups_dir):
            os.makedirs(self.backups_dir)

    @staticmethod
    def compute_memory_id(text: str, metadata: Optional[Dict] = None) -> str:
        """
        Compute deterministic SHA256 ID for deduplication.

        Args:
            text: Memory text content
            metadata: Optional metadata dict

        Returns:
            16-character hex string ID
        """
        content = text + json.dumps(metadata or {}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_instructions(self) -> str:
        """Load instructions from file."""
        if os.path.exists(self.instructions_file):
            with open(self.instructions_file, 'r') as f:
                return f.read()
        return ""

    def _save_instructions(self, text: str) -> None:
        """Save instructions to file."""
        with open(self.instructions_file, 'w') as f:
            f.write(text)

    def export_all(
        self,
        include_vectors: bool = True,
        include_threads: bool = True,
    ) -> Dict:
        """
        Export all data to JSON-serializable dict.

        Args:
            include_vectors: Include embedding vectors (larger but portable)
            include_threads: Include conversation threads

        Returns:
            Complete backup dict ready for JSON serialization
        """
        # Get all memory entries
        memories = []
        if self.vector_index:
            entries = self.vector_index.get_all_entries()
            for entry in entries:
                memory = {
                    "id": self.compute_memory_id(entry["text"], entry.get("metadata")),
                    "text": entry["text"],
                    "metadata": entry.get("metadata", {}),
                }
                if include_vectors and "vector" in entry:
                    memory["vector"] = entry["vector"]
                memories.append(memory)

        # Get threads
        threads = {}
        if include_threads:
            threads = self.thread_storage.load_threads()

        # Build export
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "created_at": now,
            "instructions": self._load_instructions(),
            "memories": memories,
            "threads": threads,
            "metadata": {
                "memory_count": len(memories),
                "thread_count": len(threads),
                "embedding_provider": getattr(self.embedding_provider, 'name', None) if self.embedding_provider else None,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "include_vectors": include_vectors,
            }
        }

    def create_backup(
        self,
        backup_name: Optional[str] = None,
        include_vectors: bool = True,
        include_threads: bool = True,
    ) -> Dict:
        """
        Create timestamped backup file.

        Args:
            backup_name: Optional custom name (without .json extension).
                        If None, uses timestamp.
            include_vectors: Include embedding vectors
            include_threads: Include conversation threads

        Returns:
            {"path": "backups/...", "stats": {...}}
        """
        # Generate filename
        if backup_name:
            # Sanitize name
            safe_name = "".join(c for c in backup_name if c.isalnum() or c in "-_")
            filename = f"{safe_name}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backup_{timestamp}.json"

        backup_path = os.path.join(self.backups_dir, filename)
        relative_path = os.path.join("backups", filename)

        # Export data
        data = self.export_all(
            include_vectors=include_vectors,
            include_threads=include_threads,
        )

        # Write to file
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)

        return {
            "path": relative_path,
            "stats": {
                "memory_count": data["metadata"]["memory_count"],
                "thread_count": data["metadata"]["thread_count"],
                "created_at": data["created_at"],
            }
        }

    def list_backups(self) -> List[Dict]:
        """
        List available backups with metadata.

        Returns:
            List of backup info dicts sorted by creation time (newest first)
        """
        backups = []
        pattern = os.path.join(self.backups_dir, "*.json")

        for filepath in glob(pattern):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Validate it's a sem-mem backup
                if data.get("format") != self.FORMAT:
                    continue

                backups.append({
                    "name": filename,
                    "path": os.path.join("backups", filename),
                    "created_at": data.get("created_at", ""),
                    "memory_count": data.get("metadata", {}).get("memory_count", 0),
                    "thread_count": data.get("metadata", {}).get("thread_count", 0),
                    "version": data.get("version", "unknown"),
                })
            except (json.JSONDecodeError, IOError):
                # Skip corrupted files
                continue

        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups

    def get_backup_path(self, backup_name: str) -> str:
        """
        Resolve backup name to full path.

        Args:
            backup_name: Backup filename (with or without .json)

        Returns:
            Full path to backup file

        Raises:
            FileNotFoundError: If backup doesn't exist
        """
        if not backup_name.endswith(".json"):
            backup_name = f"{backup_name}.json"

        backup_path = os.path.join(self.backups_dir, backup_name)
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_name}")

        return backup_path

    def restore_backup(
        self,
        backup_name: str,
        merge: bool = False,
        re_embed: bool = False,
    ) -> Dict:
        """
        Restore from backup.

        Args:
            backup_name: Backup filename (with or without .json)
            merge: If True, merge with existing data. If False, replace.
            re_embed: If True, recompute vectors from text.

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
            EmbeddingMismatchError: If model mismatch with re_embed=False
            ValueError: If vectors missing with re_embed=False
        """
        backup_path = self.get_backup_path(backup_name)

        with open(backup_path, 'r') as f:
            data = json.load(f)

        # Validate format
        if data.get("format") != self.FORMAT:
            raise ValueError(f"Invalid backup format: {data.get('format')}")

        backup_metadata = data.get("metadata", {})
        backup_model = backup_metadata.get("embedding_model")
        backup_dim = backup_metadata.get("embedding_dim")
        include_vectors = backup_metadata.get("include_vectors", True)

        # Check embedding compatibility
        if not re_embed:
            if not include_vectors:
                raise ValueError(
                    "Backup has no vectors (include_vectors=False). "
                    "Use re_embed=True to regenerate embeddings."
                )
            if backup_model and self.embedding_model and backup_model != self.embedding_model:
                raise EmbeddingMismatchError(
                    stored_provider=backup_metadata.get("embedding_provider"),
                    stored_model=backup_model,
                    current_provider=getattr(self.embedding_provider, 'name', None) if self.embedding_provider else None,
                    current_model=self.embedding_model,
                    storage_dir=self.storage_dir,
                )
            if backup_dim and self.embedding_dim and backup_dim != self.embedding_dim:
                raise EmbeddingMismatchError(
                    stored_provider=backup_metadata.get("embedding_provider"),
                    stored_model=backup_model,
                    current_provider=getattr(self.embedding_provider, 'name', None) if self.embedding_provider else None,
                    current_model=self.embedding_model,
                    storage_dir=self.storage_dir,
                )

        stats = {
            "memories_added": 0,
            "memories_skipped": 0,
            "threads_restored": 0,
            "re_embedded": 0,
            "instructions_action": "kept",
        }

        # --- Restore Instructions ---
        backup_instructions = data.get("instructions")
        if backup_instructions:
            if merge:
                # Keep current instructions in merge mode
                stats["instructions_action"] = "kept"
            else:
                # Replace in non-merge mode
                self._save_instructions(backup_instructions)
                stats["instructions_action"] = "replaced"

        # --- Restore Threads ---
        backup_threads = data.get("threads", {})
        if backup_threads:
            if merge:
                # Merge: add/overwrite threads by name
                current_threads = self.thread_storage.load_threads()
                current_threads.update(backup_threads)
                self.thread_storage.save_threads(current_threads)
                stats["threads_restored"] = len(backup_threads)
            else:
                # Replace all threads
                self.thread_storage.save_threads(backup_threads)
                stats["threads_restored"] = len(backup_threads)
        elif not merge:
            # Clear threads if backup has none and we're replacing
            self.thread_storage.save_threads({})

        # --- Restore Memories ---
        backup_memories = data.get("memories", [])
        if self.vector_index and backup_memories:
            # Get existing IDs for dedup in merge mode
            existing_ids = set()
            if merge:
                for entry in self.vector_index.get_all_entries():
                    mem_id = self.compute_memory_id(entry["text"], entry.get("metadata"))
                    existing_ids.add(mem_id)
            else:
                # Clear index for replace mode
                # We need to recreate the index
                from .vector_index import HNSWIndex
                self.vector_index = HNSWIndex(
                    storage_dir=self.storage_dir,
                    embedding_dim=self.embedding_dim,
                    embedding_provider=getattr(self.embedding_provider, 'name', None) if self.embedding_provider else None,
                    embedding_model=self.embedding_model,
                )

            # Add memories
            for memory in backup_memories:
                mem_id = memory.get("id") or self.compute_memory_id(
                    memory["text"], memory.get("metadata")
                )

                # Skip duplicates in merge mode
                if merge and mem_id in existing_ids:
                    stats["memories_skipped"] += 1
                    continue

                # Get or compute vector
                if re_embed:
                    if not self.embedding_provider:
                        raise ValueError(
                            "Cannot re-embed: no embedding provider configured"
                        )
                    vector = self.embedding_provider.embed_single(
                        memory["text"],
                        self.embedding_model,
                    )
                    stats["re_embedded"] += 1
                else:
                    vector = np.array(memory["vector"])

                # Add to index
                _, is_new = self.vector_index.add(
                    memory["text"],
                    vector,
                    memory.get("metadata"),
                )

                if is_new:
                    stats["memories_added"] += 1
                else:
                    stats["memories_skipped"] += 1

            # Save index
            self.vector_index.save()
        elif not merge and self.vector_index:
            # Clear index if no memories in backup and replacing
            from .vector_index import HNSWIndex
            self.vector_index = HNSWIndex(
                storage_dir=self.storage_dir,
                embedding_dim=self.embedding_dim,
                embedding_provider=getattr(self.embedding_provider, 'name', None) if self.embedding_provider else None,
                embedding_model=self.embedding_model,
            )
            self.vector_index.save()

        return stats

    def delete_backup(self, backup_name: str) -> bool:
        """
        Delete a backup file.

        Args:
            backup_name: Backup filename (with or without .json)

        Returns:
            True if deleted, False if didn't exist
        """
        try:
            backup_path = self.get_backup_path(backup_name)
            os.remove(backup_path)
            return True
        except FileNotFoundError:
            return False
