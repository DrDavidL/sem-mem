"""
Memory consolidation worker.

Periodically reviews memories to:
- Create patterns from recurring themes
- Demote redundant memories
- Flag contradictions for human review

Key principle: Transparent objectives defined by the system designer,
not self-generated goals.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

from .config import (
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_DRY_RUN,
    CONSOLIDATION_RECENT_LIMIT,
    CONSOLIDATION_COLD_SAMPLE,
    CONSOLIDATION_MAX_NEW_PATTERNS,
    CONSOLIDATION_MODEL,
    CONSOLIDATION_OBJECTIVES,
)

if TYPE_CHECKING:
    from .core import SemanticMemory

logger = logging.getLogger(__name__)


CONSOLIDATION_SYSTEM_PROMPT = """You are a consolidation process for a long-term memory system that supports an AI assistant helping a specific person.

You will receive a list of memories (facts, preferences, decisions, corrections). Your goals:

1. Identify a small number (<= {max_patterns}) of patterns or principles that:
   - Seem stable
   - Would be helpful for future conversations
   - Generalize across multiple memories

2. Optionally flag memories that appear redundant or clearly superseded by others.

3. Identify any contradictions between memories that should be reviewed by a human.

Your consolidation objectives (defined by the system designer):
{objectives}

Return a JSON object with this exact structure:
{{
  "patterns": [
    {{"text": "Pattern description...", "source_ids": [1, 2], "reasoning": "Why this is a stable pattern"}}
  ],
  "demotions": [
    {{"memory_id": 3, "reason": "Superseded by memory 5"}}
  ],
  "contradictions": [
    {{"ids": [4, 6], "summary": "These memories conflict about X"}}
  ]
}}

Keep patterns concise and actionable. Only flag clear contradictions, not subtle differences.
Return valid JSON only, no additional text."""


class Consolidator:
    """
    Memory consolidation worker.

    Performs offline analysis of memories to create patterns,
    reduce redundancy, and flag contradictions.
    """

    def __init__(
        self,
        memory: "SemanticMemory",
        config: Optional[Dict] = None,
    ):
        """
        Initialize consolidator.

        Args:
            memory: SemanticMemory instance to consolidate
            config: Optional config overrides
        """
        self.memory = memory
        self.config = config or {}

        # Allow config overrides
        self.dry_run = self.config.get("dry_run", CONSOLIDATION_DRY_RUN)
        self.recent_limit = self.config.get("recent_limit", CONSOLIDATION_RECENT_LIMIT)
        self.cold_sample = self.config.get("cold_sample", CONSOLIDATION_COLD_SAMPLE)
        self.max_patterns = self.config.get("max_patterns", CONSOLIDATION_MAX_NEW_PATTERNS)
        self.model = self.config.get("model", CONSOLIDATION_MODEL)
        self.objectives = self.config.get("objectives", CONSOLIDATION_OBJECTIVES)

    def run_once(self) -> Dict:
        """
        Perform a single consolidation pass.

        Returns:
            Stats about what changed (or would change in dry_run mode).
        """
        if not CONSOLIDATION_ENABLED:
            return {"skipped": True, "reason": "consolidation disabled"}

        # Step 1: Select memories
        memories = self._select_memories()
        if not memories:
            return {
                "patterns_created": 0,
                "demotions": 0,
                "contradictions_flagged": 0,
                "memories_reviewed": 0,
            }

        logger.info(f"Consolidator reviewing {len(memories)} memories")

        # Step 2: Call LLM for analysis
        llm_input = self._format_for_llm(memories)
        llm_output = self._call_llm(llm_input)

        if not llm_output:
            logger.warning("Consolidator: LLM returned no output")
            return {
                "patterns_created": 0,
                "demotions": 0,
                "contradictions_flagged": 0,
                "memories_reviewed": len(memories),
                "error": "LLM returned no output",
            }

        # Step 3: Parse output
        result = self._parse_output(llm_output)

        # Step 4: Apply changes (or just log in dry_run)
        stats = self._apply_changes(result, memories)
        stats["memories_reviewed"] = len(memories)
        stats["dry_run"] = self.dry_run

        return stats

    def _select_memories(self) -> List[Dict]:
        """Select memories for consolidation review."""
        recent = self.memory.vector_index.get_recent_memories(limit=self.recent_limit)
        recent_ids = [m["id"] for m in recent]

        cold = self.memory.vector_index.sample_memories(
            limit=self.cold_sample,
            exclude_ids=recent_ids,
        )

        return recent + cold

    def _format_for_llm(self, memories: List[Dict]) -> str:
        """Format memories for LLM input."""
        lines = []
        for mem in memories:
            mem_id = mem["id"]
            text = mem.get("text", "")
            kind = mem.get("metadata", {}).get("kind", "fact")
            utility = mem.get("utility_score", 0.5)
            is_pattern = mem.get("is_pattern", False)

            status = "PATTERN" if is_pattern else f"utility={utility:.2f}"
            lines.append(f"[ID={mem_id}] ({kind}, {status}) {text}")

        return "\n".join(lines)

    def _call_llm(self, memories_text: str) -> Optional[str]:
        """Call LLM for consolidation analysis."""
        system_prompt = CONSOLIDATION_SYSTEM_PROMPT.format(
            max_patterns=self.max_patterns,
            objectives="\n".join(f"- {obj}" for obj in self.objectives),
        )

        user_prompt = f"Here are the memories to analyze:\n\n{memories_text}"

        try:
            # Use the memory's chat provider
            response = self.memory._chat_provider.chat(
                messages=[{"role": "user", "content": user_prompt}],
                model=self.model,
                instructions=system_prompt,
            )
            # Return the text content from the ChatResponse
            return response.text if response else None
        except Exception as e:
            logger.error(f"Consolidator LLM call failed: {e}")
            return None

    def _parse_output(self, llm_output: str) -> Dict:
        """Parse LLM JSON output safely."""
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in llm_output:
                start = llm_output.index("```json") + 7
                end = llm_output.index("```", start)
                llm_output = llm_output[start:end].strip()
            elif "```" in llm_output:
                start = llm_output.index("```") + 3
                end = llm_output.index("```", start)
                llm_output = llm_output[start:end].strip()

            result = json.loads(llm_output)
            return {
                "patterns": result.get("patterns", []),
                "demotions": result.get("demotions", []),
                "contradictions": result.get("contradictions", []),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse consolidator output: {e}")
            logger.debug(f"Raw output: {llm_output[:500]}...")
            return {"patterns": [], "demotions": [], "contradictions": []}

    def _apply_changes(self, result: Dict, memories: List[Dict]) -> Dict:
        """Apply consolidation changes (or log in dry_run mode)."""
        patterns_created = 0
        demotions = 0
        contradictions_flagged = 0

        # Build ID lookup for validation
        valid_ids = {m["id"] for m in memories}

        # Process patterns
        for p in result["patterns"][:self.max_patterns]:
            pattern_text = p.get("text", "").strip()
            if not pattern_text:
                continue

            # Check for existing similar pattern (deduplication)
            if self._pattern_exists(pattern_text):
                logger.info(f"Consolidator: Skipping duplicate pattern: {pattern_text[:50]}...")
                continue

            if self.dry_run:
                logger.info(f"[DRY RUN] Would create pattern: {pattern_text[:100]}...")
            else:
                self.memory.save_memory(
                    text=pattern_text,
                    kind="pattern",
                    metadata={
                        "source_ids": p.get("source_ids", []),
                        "created_by": "consolidator",
                        "reasoning": p.get("reasoning", ""),
                    },
                )
                logger.info(f"Created pattern: {pattern_text[:50]}...")

            patterns_created += 1

        # Process demotions
        for d in result["demotions"]:
            memory_id = d.get("memory_id")
            if memory_id not in valid_ids:
                continue

            if self.dry_run:
                logger.info(f"[DRY RUN] Would demote memory {memory_id}: {d.get('reason', '')}")
            else:
                self._demote_memory(memory_id)
                logger.info(f"Demoted memory {memory_id}: {d.get('reason', '')}")

            demotions += 1

        # Process contradictions
        contradictions = result.get("contradictions", [])
        if contradictions:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would flag {len(contradictions)} contradictions")
            else:
                self._store_contradictions(contradictions)

            contradictions_flagged = len(contradictions)

        return {
            "patterns_created": patterns_created,
            "demotions": demotions,
            "contradictions_flagged": contradictions_flagged,
        }

    def _pattern_exists(self, pattern_text: str) -> bool:
        """Check if a semantically similar pattern already exists."""
        try:
            results, _ = self.memory.recall(
                query=pattern_text,
                limit=3,
                threshold=0.85,
                expand_query=False,
                include_metadata=True,
            )

            # Check if any result is a pattern
            for r in results:
                if r.get("metadata", {}).get("kind") == "pattern":
                    # Reinforce existing pattern instead of creating duplicate
                    if not self.dry_run:
                        self.memory.record_outcome(
                            r["id"], "success",
                            metadata={"source": "consolidator_dedup"}
                        )
                    return True

            return False
        except Exception as e:
            logger.warning(f"Error checking for existing pattern: {e}")
            return False

    def _demote_memory(self, memory_id: int):
        """Demote a memory by recording a synthetic failure outcome."""
        try:
            self.memory.record_outcome(
                memory_id,
                "failure",
                metadata={"source": "consolidator"},
            )
        except KeyError:
            logger.warning(f"Cannot demote unknown memory: {memory_id}")

    def _store_contradictions(self, contradictions: List[Dict]):
        """Store contradictions for human review."""
        storage_dir = self.memory.vector_index.storage_dir
        contradictions_file = os.path.join(storage_dir, "contradictions.json")

        # Load existing
        existing = []
        if os.path.exists(contradictions_file):
            try:
                with open(contradictions_file, 'r') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        # Append new with timestamp
        for c in contradictions:
            existing.append({
                **c,
                "flagged_at": datetime.now().isoformat(),
                "status": "pending_review",
            })

        # Save
        with open(contradictions_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Stored {len(contradictions)} contradictions for review")
