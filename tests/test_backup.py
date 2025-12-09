"""
Tests for backup and thread persistence functionality.
"""

import os
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from sem_mem.thread_storage import ThreadStorage
from sem_mem.backup import MemoryBackup


# =============================================================================
# ThreadStorage Tests
# =============================================================================


class TestThreadStorage:
    """Tests for ThreadStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create a ThreadStorage instance."""
        return ThreadStorage(storage_dir=temp_dir)

    def test_save_and_load_threads(self, storage):
        """Should save and load threads correctly."""
        threads = {
            "Thread 1": {
                "messages": [{"role": "user", "content": "Hello"}],
                "title": "Test Thread",
            },
            "Thread 2": {
                "messages": [],
                "title": "Empty Thread",
            },
        }

        storage.save_threads(threads)
        loaded = storage.load_threads()

        assert loaded == threads

    def test_load_empty_returns_empty_dict(self, storage):
        """Should return empty dict if no file exists."""
        loaded = storage.load_threads()
        assert loaded == {}

    def test_save_single_thread(self, storage):
        """Should save and update single threads."""
        # Save first thread
        storage.save_thread("Thread 1", {"messages": [], "title": "First"})
        assert storage.get_thread("Thread 1")["title"] == "First"

        # Save second thread
        storage.save_thread("Thread 2", {"messages": [], "title": "Second"})
        threads = storage.load_threads()
        assert len(threads) == 2
        assert threads["Thread 1"]["title"] == "First"
        assert threads["Thread 2"]["title"] == "Second"

        # Update first thread
        storage.save_thread("Thread 1", {"messages": [], "title": "Updated"})
        assert storage.get_thread("Thread 1")["title"] == "Updated"

    def test_get_thread_not_found(self, storage):
        """Should return None for non-existent thread."""
        assert storage.get_thread("NonExistent") is None

    def test_delete_thread(self, storage):
        """Should delete a thread."""
        storage.save_threads({
            "Thread 1": {"messages": []},
            "Thread 2": {"messages": []},
        })

        result = storage.delete_thread("Thread 1")
        assert result is True
        assert storage.get_thread("Thread 1") is None
        assert storage.get_thread("Thread 2") is not None

    def test_delete_nonexistent_thread(self, storage):
        """Should return False when deleting non-existent thread."""
        result = storage.delete_thread("NonExistent")
        assert result is False

    def test_rename_thread(self, storage):
        """Should rename a thread."""
        storage.save_thread("OldName", {"messages": [], "title": "Test"})

        result = storage.rename_thread("OldName", "NewName")
        assert result is True
        assert storage.get_thread("OldName") is None
        assert storage.get_thread("NewName")["title"] == "Test"

    def test_rename_nonexistent_thread(self, storage):
        """Should return False when renaming non-existent thread."""
        result = storage.rename_thread("NonExistent", "NewName")
        assert result is False

    def test_list_thread_names(self, storage):
        """Should list all thread names."""
        storage.save_threads({
            "Alpha": {"messages": []},
            "Beta": {"messages": []},
            "Gamma": {"messages": []},
        })

        names = storage.list_thread_names()
        assert set(names) == {"Alpha", "Beta", "Gamma"}

    def test_get_thread_count(self, storage):
        """Should return correct thread count."""
        assert storage.get_thread_count() == 0

        storage.save_threads({"A": {}, "B": {}, "C": {}})
        assert storage.get_thread_count() == 3

    def test_clear_all(self, storage):
        """Should clear all threads."""
        storage.save_threads({"A": {}, "B": {}})
        count = storage.clear_all()

        assert count == 2
        assert storage.load_threads() == {}

    def test_atomic_write_creates_temp_file(self, storage, temp_dir):
        """Should use atomic writes (temp file + replace)."""
        storage.save_threads({"Test": {"messages": []}})

        # Verify no temp file remains
        assert not os.path.exists(os.path.join(temp_dir, "threads.json.tmp"))

    def test_corrupted_file_returns_empty(self, storage, temp_dir):
        """Should handle corrupted JSON gracefully."""
        # Write invalid JSON
        with open(os.path.join(temp_dir, "threads.json"), 'w') as f:
            f.write("{invalid json")

        loaded = storage.load_threads()
        assert loaded == {}

    def test_versioned_format(self, storage, temp_dir):
        """Should save with version information."""
        storage.save_threads({"Test": {}})

        with open(os.path.join(temp_dir, "threads.json"), 'r') as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == "1.0"
        assert "threads" in data


# =============================================================================
# MemoryBackup Tests
# =============================================================================


class TestMemoryBackup:
    """Tests for MemoryBackup class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mock_vector_index(self):
        """Create a mock HNSW index."""
        mock = MagicMock()
        mock.get_all_entries.return_value = [
            {
                "text": "Memory 1",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {"source": "test"},
            },
            {
                "text": "Memory 2",
                "vector": [0.4, 0.5, 0.6],
                "metadata": {},
            },
        ]
        return mock

    @pytest.fixture
    def backup(self, temp_dir, mock_vector_index):
        """Create a MemoryBackup instance."""
        thread_storage = ThreadStorage(storage_dir=temp_dir)

        # Create instructions file
        instructions_file = os.path.join(temp_dir, "instructions.txt")
        with open(instructions_file, 'w') as f:
            f.write("Test instructions")

        return MemoryBackup(
            storage_dir=temp_dir,
            vector_index=mock_vector_index,
            thread_storage=thread_storage,
            embedding_model="test-model",
            embedding_dim=3,
        )

    def test_compute_memory_id_deterministic(self):
        """Memory ID should be deterministic."""
        id1 = MemoryBackup.compute_memory_id("test", {"key": "value"})
        id2 = MemoryBackup.compute_memory_id("test", {"key": "value"})
        assert id1 == id2

    def test_compute_memory_id_different_for_different_input(self):
        """Different inputs should produce different IDs."""
        id1 = MemoryBackup.compute_memory_id("test1", {})
        id2 = MemoryBackup.compute_memory_id("test2", {})
        assert id1 != id2

    def test_compute_memory_id_metadata_order_independent(self):
        """Metadata key order shouldn't affect ID."""
        id1 = MemoryBackup.compute_memory_id("test", {"a": 1, "b": 2})
        id2 = MemoryBackup.compute_memory_id("test", {"b": 2, "a": 1})
        assert id1 == id2

    def test_export_all_with_vectors(self, backup):
        """Should export all data with vectors."""
        backup.thread_storage.save_threads({"Thread 1": {"messages": []}})

        data = backup.export_all(include_vectors=True, include_threads=True)

        assert data["format"] == "sem-mem-backup"
        assert data["version"] == "1.0"
        assert "created_at" in data
        assert data["instructions"] == "Test instructions"
        assert len(data["memories"]) == 2
        assert "vector" in data["memories"][0]
        assert "Thread 1" in data["threads"]
        assert data["metadata"]["memory_count"] == 2
        assert data["metadata"]["thread_count"] == 1
        assert data["metadata"]["include_vectors"] is True

    def test_export_all_without_vectors(self, backup):
        """Should export data without vectors when requested."""
        data = backup.export_all(include_vectors=False, include_threads=True)

        assert len(data["memories"]) == 2
        assert "vector" not in data["memories"][0]
        assert data["metadata"]["include_vectors"] is False

    def test_export_all_without_threads(self, backup):
        """Should export data without threads when requested."""
        backup.thread_storage.save_threads({"Thread 1": {"messages": []}})

        data = backup.export_all(include_vectors=True, include_threads=False)

        assert data["threads"] == {}
        assert data["metadata"]["thread_count"] == 0

    def test_create_backup_timestamped(self, backup, temp_dir):
        """Should create backup with timestamp."""
        result = backup.create_backup()

        assert "path" in result
        assert result["path"].startswith("backups/backup_")
        assert result["path"].endswith(".json")
        assert "stats" in result
        assert result["stats"]["memory_count"] == 2

        # Verify file exists
        full_path = os.path.join(temp_dir, result["path"])
        assert os.path.exists(full_path)

    def test_create_backup_custom_name(self, backup, temp_dir):
        """Should create backup with custom name."""
        result = backup.create_backup(backup_name="my_custom_backup")

        assert result["path"] == "backups/my_custom_backup.json"

        full_path = os.path.join(temp_dir, result["path"])
        assert os.path.exists(full_path)

    def test_create_backup_sanitizes_name(self, backup):
        """Should sanitize custom names."""
        result = backup.create_backup(backup_name="my/backup/../test")

        # Should remove path separators and special chars
        assert ".." not in result["path"]
        assert "/" not in result["path"].replace("backups/", "")

    def test_list_backups(self, backup, temp_dir):
        """Should list all backups."""
        # Create multiple backups
        backup.create_backup(backup_name="backup1")
        backup.create_backup(backup_name="backup2")

        backups = backup.list_backups()

        assert len(backups) == 2
        names = [b["name"] for b in backups]
        assert "backup1.json" in names
        assert "backup2.json" in names

    def test_list_backups_ignores_invalid_files(self, backup, temp_dir):
        """Should ignore non-backup JSON files."""
        backup.create_backup(backup_name="valid")

        # Create invalid file
        invalid_path = os.path.join(temp_dir, "backups", "invalid.json")
        with open(invalid_path, 'w') as f:
            json.dump({"not": "a backup"}, f)

        backups = backup.list_backups()
        assert len(backups) == 1
        assert backups[0]["name"] == "valid.json"

    def test_delete_backup(self, backup, temp_dir):
        """Should delete a backup."""
        backup.create_backup(backup_name="to_delete")
        assert len(backup.list_backups()) == 1

        result = backup.delete_backup("to_delete")
        assert result is True
        assert len(backup.list_backups()) == 0

    def test_delete_nonexistent_backup(self, backup):
        """Should return False for non-existent backup."""
        result = backup.delete_backup("nonexistent")
        assert result is False


# =============================================================================
# Restore Tests
# =============================================================================


class TestBackupRestore:
    """Tests for backup restore functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def setup_backup_with_data(self, temp_dir):
        """Create a backup file with test data."""
        backups_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backups_dir)

        backup_data = {
            "format": "sem-mem-backup",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "instructions": "Backup instructions",
            "memories": [
                {
                    "id": "mem1",
                    "text": "Memory 1",
                    "vector": [0.1, 0.2, 0.3],
                    "metadata": {"source": "backup"},
                },
                {
                    "id": "mem2",
                    "text": "Memory 2",
                    "vector": [0.4, 0.5, 0.6],
                    "metadata": {},
                },
            ],
            "threads": {
                "Thread 1": {"messages": [{"role": "user", "content": "Hi"}]},
            },
            "metadata": {
                "memory_count": 2,
                "thread_count": 1,
                "embedding_model": "test-model",
                "embedding_dim": 3,
                "include_vectors": True,
            },
        }

        backup_path = os.path.join(backups_dir, "test_backup.json")
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f)

        return temp_dir, backup_data

    def test_restore_replace_mode(self, setup_backup_with_data):
        """Should replace all data in replace mode."""
        temp_dir, backup_data = setup_backup_with_data

        # Create mock index
        mock_index = MagicMock()
        mock_index.get_all_entries.return_value = []
        mock_index.add.return_value = (0, True)

        thread_storage = ThreadStorage(storage_dir=temp_dir)
        thread_storage.save_threads({"Old Thread": {"messages": []}})

        # Write existing instructions
        with open(os.path.join(temp_dir, "instructions.txt"), 'w') as f:
            f.write("Old instructions")

        backup = MemoryBackup(
            storage_dir=temp_dir,
            vector_index=mock_index,
            thread_storage=thread_storage,
            embedding_model="test-model",
            embedding_dim=3,
        )

        # Mock vector index recreation
        with patch.object(backup, 'vector_index', mock_index):
            result = backup.restore_backup("test_backup", merge=False)

        assert result["memories_added"] == 2
        assert result["threads_restored"] == 1
        assert result["instructions_action"] == "replaced"

        # Verify instructions replaced
        with open(os.path.join(temp_dir, "instructions.txt"), 'r') as f:
            assert f.read() == "Backup instructions"

        # Verify threads replaced
        threads = thread_storage.load_threads()
        assert "Old Thread" not in threads
        assert "Thread 1" in threads

    def test_restore_merge_mode(self, setup_backup_with_data):
        """Should merge data in merge mode."""
        temp_dir, backup_data = setup_backup_with_data

        mock_index = MagicMock()
        # Return existing entry with different ID
        mock_index.get_all_entries.return_value = [
            {"text": "Existing", "metadata": {}},
        ]
        mock_index.add.return_value = (0, True)

        thread_storage = ThreadStorage(storage_dir=temp_dir)
        thread_storage.save_threads({"Existing Thread": {"messages": []}})

        with open(os.path.join(temp_dir, "instructions.txt"), 'w') as f:
            f.write("Existing instructions")

        backup = MemoryBackup(
            storage_dir=temp_dir,
            vector_index=mock_index,
            thread_storage=thread_storage,
            embedding_model="test-model",
            embedding_dim=3,
        )

        result = backup.restore_backup("test_backup", merge=True)

        assert result["instructions_action"] == "kept"

        # Verify instructions kept
        with open(os.path.join(temp_dir, "instructions.txt"), 'r') as f:
            assert f.read() == "Existing instructions"

        # Verify threads merged
        threads = thread_storage.load_threads()
        assert "Existing Thread" in threads
        assert "Thread 1" in threads

    def test_restore_not_found(self, temp_dir):
        """Should raise FileNotFoundError for missing backup."""
        backup = MemoryBackup(storage_dir=temp_dir)

        with pytest.raises(FileNotFoundError):
            backup.restore_backup("nonexistent")

    def test_restore_missing_vectors_error(self, temp_dir):
        """Should raise error if vectors missing and re_embed=False."""
        backups_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backups_dir)

        # Create backup without vectors
        backup_data = {
            "format": "sem-mem-backup",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "instructions": "",
            "memories": [{"id": "1", "text": "test", "metadata": {}}],
            "threads": {},
            "metadata": {"include_vectors": False},
        }

        with open(os.path.join(backups_dir, "no_vectors.json"), 'w') as f:
            json.dump(backup_data, f)

        mock_index = MagicMock()
        backup = MemoryBackup(
            storage_dir=temp_dir,
            vector_index=mock_index,
            embedding_model="test-model",
            embedding_dim=3,
        )

        with pytest.raises(ValueError, match="no vectors"):
            backup.restore_backup("no_vectors", re_embed=False)


# =============================================================================
# Merge/Dedup Tests
# =============================================================================


class TestMergeSemantics:
    """Tests for merge and deduplication behavior."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_merge_skips_duplicates(self, temp_dir):
        """Should skip memories with same ID during merge."""
        backups_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backups_dir)

        # Compute ID for test memory
        test_id = MemoryBackup.compute_memory_id("Same text", {"key": "value"})

        backup_data = {
            "format": "sem-mem-backup",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "instructions": "",
            "memories": [
                {"id": test_id, "text": "Same text", "vector": [0.1, 0.2], "metadata": {"key": "value"}},
            ],
            "threads": {},
            "metadata": {"include_vectors": True, "embedding_dim": 2},
        }

        with open(os.path.join(backups_dir, "dupe.json"), 'w') as f:
            json.dump(backup_data, f)

        mock_index = MagicMock()
        # Return entry with same ID
        mock_index.get_all_entries.return_value = [
            {"text": "Same text", "metadata": {"key": "value"}},
        ]
        mock_index.add.return_value = (0, False)  # Not new

        backup = MemoryBackup(
            storage_dir=temp_dir,
            vector_index=mock_index,
            embedding_model="test",
            embedding_dim=2,
        )

        result = backup.restore_backup("dupe", merge=True)

        assert result["memories_skipped"] == 1
        assert result["memories_added"] == 0


# =============================================================================
# Missing Data Tests
# =============================================================================


class TestMissingData:
    """Tests for handling backups with missing data."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_missing_instructions_keeps_current(self, temp_dir):
        """Should keep current instructions if backup has none."""
        backups_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backups_dir)

        backup_data = {
            "format": "sem-mem-backup",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            # No instructions key
            "memories": [],
            "threads": {},
            "metadata": {"include_vectors": True},
        }

        with open(os.path.join(backups_dir, "no_instr.json"), 'w') as f:
            json.dump(backup_data, f)

        # Write current instructions
        with open(os.path.join(temp_dir, "instructions.txt"), 'w') as f:
            f.write("Current instructions")

        mock_index = MagicMock()
        backup = MemoryBackup(storage_dir=temp_dir, vector_index=mock_index)

        # Avoid index recreation by mocking it
        with patch.object(backup, 'vector_index', mock_index):
            result = backup.restore_backup("no_instr", merge=False)

        # Instructions should be kept
        with open(os.path.join(temp_dir, "instructions.txt"), 'r') as f:
            assert f.read() == "Current instructions"

    def test_missing_threads_clears_in_replace_mode(self, temp_dir):
        """Should clear threads if backup has none in replace mode."""
        backups_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backups_dir)

        backup_data = {
            "format": "sem-mem-backup",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "instructions": "test",
            "memories": [],
            # No threads key or empty
            "metadata": {"include_vectors": True},
        }

        with open(os.path.join(backups_dir, "no_threads.json"), 'w') as f:
            json.dump(backup_data, f)

        thread_storage = ThreadStorage(storage_dir=temp_dir)
        thread_storage.save_threads({"Old": {"messages": []}})

        mock_index = MagicMock()
        backup = MemoryBackup(
            storage_dir=temp_dir,
            vector_index=mock_index,
            thread_storage=thread_storage,
        )

        with patch.object(backup, 'vector_index', mock_index):
            backup.restore_backup("no_threads", merge=False)

        assert thread_storage.load_threads() == {}

    def test_missing_threads_keeps_current_in_merge_mode(self, temp_dir):
        """Should keep current threads if backup has none in merge mode."""
        backups_dir = os.path.join(temp_dir, "backups")
        os.makedirs(backups_dir)

        backup_data = {
            "format": "sem-mem-backup",
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "instructions": "",
            "memories": [],
            "threads": {},
            "metadata": {"include_vectors": True},
        }

        with open(os.path.join(backups_dir, "empty.json"), 'w') as f:
            json.dump(backup_data, f)

        thread_storage = ThreadStorage(storage_dir=temp_dir)
        thread_storage.save_threads({"Existing": {"messages": []}})

        mock_index = MagicMock()
        backup = MemoryBackup(
            storage_dir=temp_dir,
            vector_index=mock_index,
            thread_storage=thread_storage,
        )

        backup.restore_backup("empty", merge=True)

        threads = thread_storage.load_threads()
        assert "Existing" in threads
