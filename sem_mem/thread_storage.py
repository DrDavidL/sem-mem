"""
Thread persistence module for sem-mem.

Provides atomic file-based storage for conversation threads.
"""

import os
import json
from typing import Dict, Optional


class ThreadStorage:
    """
    Persist conversation threads to disk with atomic writes.

    Thread format (threads.json):
    {
        "version": "1.0",
        "threads": {
            "Thread 1": {
                "messages": [{"role": "user", "content": "..."}, ...],
                "response_id": "resp_xxx",
                "title": "Discussion about X",
                "title_user_overridden": false,
                "summary_windows": [...]
            }
        }
    }
    """

    VERSION = "1.0"

    def __init__(self, storage_dir: str = "./local_memory"):
        """
        Initialize thread storage.

        Args:
            storage_dir: Directory for storing threads.json
        """
        self.storage_dir = storage_dir
        self.threads_file = os.path.join(storage_dir, "threads.json")

        # Ensure storage directory exists
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def save_threads(self, threads: Dict[str, Dict]) -> None:
        """
        Save all threads to disk (atomic write).

        Args:
            threads: Dict mapping thread names to thread data
        """
        data = {"version": self.VERSION, "threads": threads}
        self._atomic_write(data)

    def load_threads(self) -> Dict[str, Dict]:
        """
        Load threads from disk.

        Returns:
            Dict mapping thread names to thread data.
            Returns empty dict if file doesn't exist.
        """
        raw = self._load_raw()
        return raw.get("threads", {})

    def save_thread(self, name: str, thread: Dict) -> None:
        """
        Save/update a single thread (load-modify-atomic-write).

        Args:
            name: Thread name/key
            thread: Thread data dict
        """
        data = self._load_raw()
        data["threads"][name] = thread
        self._atomic_write(data)

    def get_thread(self, name: str) -> Optional[Dict]:
        """
        Get a single thread by name.

        Args:
            name: Thread name/key

        Returns:
            Thread data dict, or None if not found
        """
        threads = self.load_threads()
        return threads.get(name)

    def delete_thread(self, name: str) -> bool:
        """
        Delete a thread from disk.

        Args:
            name: Thread name/key to delete

        Returns:
            True if thread was deleted, False if it didn't exist
        """
        data = self._load_raw()
        if name in data["threads"]:
            del data["threads"][name]
            self._atomic_write(data)
            return True
        return False

    def rename_thread(self, old_name: str, new_name: str) -> bool:
        """
        Rename a thread.

        Args:
            old_name: Current thread name
            new_name: New thread name

        Returns:
            True if renamed, False if old_name didn't exist
        """
        data = self._load_raw()
        if old_name in data["threads"]:
            data["threads"][new_name] = data["threads"].pop(old_name)
            self._atomic_write(data)
            return True
        return False

    def list_thread_names(self) -> list:
        """
        Get list of all thread names.

        Returns:
            List of thread name strings
        """
        return list(self.load_threads().keys())

    def get_thread_count(self) -> int:
        """
        Get number of stored threads.

        Returns:
            Count of threads
        """
        return len(self.load_threads())

    def _atomic_write(self, data: Dict) -> None:
        """
        Write to temp file then os.replace for atomicity.

        This prevents corruption from interrupted writes.

        Args:
            data: Dict to write as JSON
        """
        temp_path = self.threads_file + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, self.threads_file)

    def _load_raw(self) -> Dict:
        """
        Load raw file data with version.

        Returns:
            Full file structure with version and threads keys.
            Returns default structure if file doesn't exist.
        """
        if not os.path.exists(self.threads_file):
            return {"version": self.VERSION, "threads": {}}

        try:
            with open(self.threads_file, 'r') as f:
                data = json.load(f)
                # Ensure required keys exist
                if "threads" not in data:
                    data["threads"] = {}
                if "version" not in data:
                    data["version"] = self.VERSION
                return data
        except json.JSONDecodeError:
            # Corrupted file, return empty
            return {"version": self.VERSION, "threads": {}}

    def clear_all(self) -> int:
        """
        Delete all threads.

        Returns:
            Number of threads deleted
        """
        data = self._load_raw()
        count = len(data["threads"])
        data["threads"] = {}
        self._atomic_write(data)
        return count
