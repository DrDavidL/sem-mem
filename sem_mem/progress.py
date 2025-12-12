"""
Progress log module for sem-mem.

Provides structured JSONL logging for long-running background processes
to leave auditable "what changed + what's next" artifacts.

Each log entry is a single JSON object on one line in local_memory/progress_log.jsonl:
{
    "timestamp": "2025-01-15T10:30:00Z",
    "component": "consolidation" | "ingestion" | "manual" | "backup" | "restore",
    "summary": "Short human-readable description",
    "details": {...}
}
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Any


# Progress log component types
ProgressComponent = Literal["consolidation", "ingestion", "manual", "backup", "restore"]


class ProgressLog:
    """
    Append-only JSONL progress log for background processes.

    Each entry is a single JSON object on one line.
    """

    DEFAULT_FILENAME = "progress_log.jsonl"

    def __init__(self, storage_dir: str = "./local_memory"):
        """
        Initialize progress log.

        Args:
            storage_dir: Directory for log file
        """
        self.storage_dir = storage_dir
        self.log_path = os.path.join(storage_dir, self.DEFAULT_FILENAME)

        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def append(
        self,
        component: ProgressComponent,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Append a progress entry.

        Args:
            component: Which subsystem produced this entry
            summary: Human-readable 1-2 sentence summary
            details: Optional structured details (JSON-serializable)

        Returns:
            The entry that was written
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "component": component,
            "summary": summary,
            "details": details or {},
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def read_recent(self, limit: int = 50) -> List[Dict]:
        """
        Read recent progress entries.

        Args:
            limit: Maximum entries to return

        Returns:
            List of entries, most recent first
        """
        if not os.path.exists(self.log_path):
            return []

        entries = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except IOError:
            return []

        return list(reversed(entries[-limit:]))

    def read_by_component(
        self,
        component: ProgressComponent,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Read entries for a specific component.

        Args:
            component: Filter by component
            limit: Maximum entries to return

        Returns:
            Matching entries, most recent first
        """
        all_entries = self.read_recent(limit=limit * 2)
        filtered = [e for e in all_entries if e.get("component") == component]
        return filtered[:limit]

    def get_last_entry(self, component: Optional[ProgressComponent] = None) -> Optional[Dict]:
        """
        Get the most recent entry.

        Args:
            component: Optional component filter

        Returns:
            Most recent entry or None
        """
        if component:
            entries = self.read_by_component(component, limit=1)
        else:
            entries = self.read_recent(limit=1)
        return entries[0] if entries else None

    def clear(self) -> bool:
        """
        Clear the progress log.

        Returns:
            True if cleared, False if didn't exist
        """
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
            return True
        return False
