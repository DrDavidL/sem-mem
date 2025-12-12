"""Tests for memory consolidation."""

import pytest
import tempfile
import json
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from sem_mem.vector_index import HNSWIndex
from sem_mem.consolidation import Consolidator, CONSOLIDATION_SYSTEM_PROMPT
from sem_mem.auto_memory import MEMORY_KIND_CORRECTION, MEMORY_KIND_FACT


class TestCreatedAtField:
    """Test created_at field in data model."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield td

    def test_new_entry_gets_created_at(self, temp_dir):
        """New entries should have created_at timestamp."""
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        vector = np.array([1.0, 0.0, 0.0, 0.0])

        entry_id, is_new = index.add("test memory", vector, {})

        assert is_new
        entry = index.get_entry(entry_id)
        assert "created_at" in entry
        # Should be a valid ISO timestamp
        datetime.fromisoformat(entry["created_at"])

    def test_loaded_entry_gets_created_at_default(self, temp_dir):
        """Loaded entries without created_at should get default."""
        # Create index with an entry
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = index.add("test memory", vector, {})
        index.save()

        # Manually remove created_at from the metadata
        with open(index.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Remove created_at
        if str(entry_id) in metadata['entries']:
            del metadata['entries'][str(entry_id)]['created_at']
        with open(index.metadata_path, 'w') as f:
            json.dump(metadata, f)

        # Reload and check that default is applied
        index2 = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        entry = index2.get_entry(entry_id)
        assert "created_at" in entry
        # Should be valid ISO timestamp
        datetime.fromisoformat(entry["created_at"])


class TestMemorySelectionHelpers:
    """Test get_recent_memories and sample_memories."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield td

    @pytest.fixture
    def populated_index(self, temp_dir):
        """Create index with several memories."""
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        vectors = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0, 0.0]),
        ]
        entry_ids = []
        for i, vec in enumerate(vectors):
            entry_id, _ = index.add(f"memory {i}", vec, {"index": i})
            entry_ids.append(entry_id)
        return index, entry_ids

    def test_get_recent_memories_returns_dicts_with_id(self, populated_index):
        """get_recent_memories should return dicts with 'id' field."""
        index, _ = populated_index
        recent = index.get_recent_memories(limit=3)

        assert len(recent) == 3
        for mem in recent:
            assert "id" in mem
            assert "text" in mem
            assert "created_at" in mem
            assert isinstance(mem["id"], int)

    def test_get_recent_memories_sorted_by_created_at(self, temp_dir):
        """Recent memories should be sorted by created_at descending."""
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)

        # Add memories with explicit created_at timestamps
        for i in range(5):
            vector = np.array([float(i), 0.0, 0.0, 0.0])
            entry_id, _ = index.add(f"memory {i}", vector, {})
            # Manually set created_at to control order
            index.update_entry(entry_id, created_at=f"2025-01-0{i+1}T00:00:00")

        recent = index.get_recent_memories(limit=5)

        # Should be in descending order (most recent first)
        timestamps = [m["created_at"] for m in recent]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_sample_memories_returns_dicts_with_id(self, populated_index):
        """sample_memories should return dicts with 'id' field."""
        index, _ = populated_index
        sample = index.sample_memories(limit=3)

        assert len(sample) == 3
        for mem in sample:
            assert "id" in mem
            assert "text" in mem
            assert isinstance(mem["id"], int)

    def test_sample_memories_excludes_ids(self, populated_index):
        """sample_memories should exclude specified IDs."""
        index, entry_ids = populated_index
        exclude = entry_ids[:2]

        sample = index.sample_memories(limit=10, exclude_ids=exclude)

        sampled_ids = {m["id"] for m in sample}
        for excluded_id in exclude:
            assert excluded_id not in sampled_ids

    def test_sample_memories_empty_index(self, temp_dir):
        """sample_memories should return empty list for empty index."""
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        sample = index.sample_memories(limit=10)
        assert sample == []

    def test_get_recent_memories_empty_index(self, temp_dir):
        """get_recent_memories should return empty list for empty index."""
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        recent = index.get_recent_memories(limit=10)
        assert recent == []


class TestConsolidator:
    """Test Consolidator class."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock SemanticMemory for testing."""
        memory = Mock()

        # Mock vector_index
        memory.vector_index = Mock()
        memory.vector_index.storage_dir = tempfile.mkdtemp()
        memory.vector_index.get_recent_memories = Mock(return_value=[
            {"id": 1, "text": "I prefer morning meetings", "metadata": {"kind": "fact"}, "utility_score": 0.5, "is_pattern": False},
            {"id": 2, "text": "I like coffee", "metadata": {"kind": "fact"}, "utility_score": 0.7, "is_pattern": False},
        ])
        memory.vector_index.sample_memories = Mock(return_value=[
            {"id": 3, "text": "Python is my favorite language", "metadata": {"kind": "fact"}, "utility_score": 0.6, "is_pattern": False},
        ])

        # Mock other methods
        memory.save_memory = Mock(return_value=(10, True))
        memory.record_outcome = Mock()
        memory.recall = Mock(return_value=([], []))

        # Mock chat provider - return a mock ChatResponse with .text attribute
        mock_response = Mock()
        mock_response.text = '{"patterns": [], "demotions": [], "contradictions": []}'
        memory._chat_provider = Mock()
        memory._chat_provider.chat = Mock(return_value=mock_response)

        return memory

    def test_run_once_no_memories_returns_zeros(self, mock_memory):
        """run_once with no memories should return zero stats."""
        mock_memory.vector_index.get_recent_memories.return_value = []
        mock_memory.vector_index.sample_memories.return_value = []

        consolidator = Consolidator(mock_memory)
        stats = consolidator.run_once()

        assert stats["patterns_created"] == 0
        assert stats["demotions"] == 0
        assert stats["contradictions_flagged"] == 0
        assert stats["memories_reviewed"] == 0

    def test_run_once_counts_memories_reviewed(self, mock_memory):
        """run_once should count total memories reviewed."""
        consolidator = Consolidator(mock_memory)
        stats = consolidator.run_once()

        # 2 recent + 1 cold = 3 total
        assert stats["memories_reviewed"] == 3

    def _mock_chat_response(self, mock_memory, text: str):
        """Helper to set up mock chat provider response with .text attribute."""
        mock_response = Mock()
        mock_response.text = text
        mock_memory._chat_provider.chat.return_value = mock_response

    def test_dry_run_does_not_call_save(self, mock_memory):
        """In dry run mode, no saves should occur."""
        self._mock_chat_response(mock_memory, '''
        {
            "patterns": [{"text": "User prefers mornings", "source_ids": [1], "reasoning": "test"}],
            "demotions": [],
            "contradictions": []
        }
        ''')

        consolidator = Consolidator(mock_memory, config={"dry_run": True})
        stats = consolidator.run_once()

        # Pattern was proposed
        assert stats["patterns_created"] == 1
        # But save_memory should NOT be called
        mock_memory.save_memory.assert_not_called()

    def test_non_dry_run_calls_save(self, mock_memory):
        """With dry_run=False, saves should occur."""
        self._mock_chat_response(mock_memory, '''
        {
            "patterns": [{"text": "User prefers mornings", "source_ids": [1], "reasoning": "test"}],
            "demotions": [],
            "contradictions": []
        }
        ''')

        consolidator = Consolidator(mock_memory, config={"dry_run": False})
        stats = consolidator.run_once()

        assert stats["patterns_created"] == 1
        mock_memory.save_memory.assert_called_once()
        call_kwargs = mock_memory.save_memory.call_args
        assert call_kwargs[1]["kind"] == "pattern"

    def test_demotion_calls_record_outcome_failure(self, mock_memory):
        """Demotions should call record_outcome with 'failure'."""
        self._mock_chat_response(mock_memory, '''
        {
            "patterns": [],
            "demotions": [{"memory_id": 1, "reason": "superseded"}],
            "contradictions": []
        }
        ''')

        consolidator = Consolidator(mock_memory, config={"dry_run": False})
        stats = consolidator.run_once()

        assert stats["demotions"] == 1
        mock_memory.record_outcome.assert_called_once()
        call_args = mock_memory.record_outcome.call_args
        assert call_args[0][0] == 1  # memory_id
        assert call_args[0][1] == "failure"  # outcome

    def test_dry_run_demotion_does_not_call_record_outcome(self, mock_memory):
        """In dry run, demotions should not call record_outcome."""
        self._mock_chat_response(mock_memory, '''
        {
            "patterns": [],
            "demotions": [{"memory_id": 1, "reason": "superseded"}],
            "contradictions": []
        }
        ''')

        consolidator = Consolidator(mock_memory, config={"dry_run": True})
        stats = consolidator.run_once()

        assert stats["demotions"] == 1
        mock_memory.record_outcome.assert_not_called()

    def test_pattern_dedup_skips_existing(self, mock_memory):
        """Should skip pattern creation if similar pattern exists."""
        self._mock_chat_response(mock_memory, '''
        {
            "patterns": [{"text": "User likes mornings", "source_ids": [1], "reasoning": "test"}],
            "demotions": [],
            "contradictions": []
        }
        ''')
        # Mock recall to return an existing pattern
        mock_memory.recall.return_value = (
            [{"id": 99, "text": "User prefers mornings", "metadata": {"kind": "pattern"}}],
            []
        )

        consolidator = Consolidator(mock_memory, config={"dry_run": False})
        stats = consolidator.run_once()

        # Pattern was proposed but should be deduplicated
        assert stats["patterns_created"] == 0
        mock_memory.save_memory.assert_not_called()
        # Should reinforce existing pattern
        mock_memory.record_outcome.assert_called()

    def test_contradictions_stored_to_file(self, mock_memory):
        """Contradictions should be stored to contradictions.json."""
        self._mock_chat_response(mock_memory, '''
        {
            "patterns": [],
            "demotions": [],
            "contradictions": [{"ids": [1, 2], "summary": "Conflicting preferences"}]
        }
        ''')

        consolidator = Consolidator(mock_memory, config={"dry_run": False})
        stats = consolidator.run_once()

        assert stats["contradictions_flagged"] == 1

        # Check file was created
        contradictions_file = os.path.join(mock_memory.vector_index.storage_dir, "contradictions.json")
        assert os.path.exists(contradictions_file)

        with open(contradictions_file, 'r') as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["summary"] == "Conflicting preferences"
        assert data[0]["status"] == "pending_review"


class TestConsolidatorParsing:
    """Test JSON parsing in consolidator."""

    @pytest.fixture
    def consolidator(self):
        """Create a consolidator with mocked memory."""
        memory = Mock()
        memory.vector_index = Mock()
        memory.vector_index.storage_dir = tempfile.mkdtemp()
        return Consolidator(memory)

    def test_parse_plain_json(self, consolidator):
        """Should parse plain JSON."""
        output = '{"patterns": [], "demotions": [], "contradictions": []}'
        result = consolidator._parse_output(output)

        assert result["patterns"] == []
        assert result["demotions"] == []
        assert result["contradictions"] == []

    def test_parse_json_in_markdown_block(self, consolidator):
        """Should extract JSON from markdown code block."""
        output = '''Here's my analysis:

```json
{"patterns": [{"text": "test"}], "demotions": [], "contradictions": []}
```

That's all!'''
        result = consolidator._parse_output(output)

        assert len(result["patterns"]) == 1
        assert result["patterns"][0]["text"] == "test"

    def test_parse_json_in_plain_code_block(self, consolidator):
        """Should extract JSON from plain code block."""
        output = '''```
{"patterns": [], "demotions": [{"memory_id": 1, "reason": "old"}], "contradictions": []}
```'''
        result = consolidator._parse_output(output)

        assert len(result["demotions"]) == 1

    def test_parse_invalid_json_returns_empty(self, consolidator):
        """Invalid JSON should return empty results."""
        output = "This is not JSON at all"
        result = consolidator._parse_output(output)

        assert result["patterns"] == []
        assert result["demotions"] == []
        assert result["contradictions"] == []


class TestFormatForLLM:
    """Test memory formatting for LLM input."""

    @pytest.fixture
    def consolidator(self):
        memory = Mock()
        memory.vector_index = Mock()
        memory.vector_index.storage_dir = tempfile.mkdtemp()
        return Consolidator(memory)

    def test_format_includes_id_and_text(self, consolidator):
        """Format should include ID and text."""
        memories = [
            {"id": 1, "text": "I like coffee", "metadata": {}, "utility_score": 0.5, "is_pattern": False},
        ]
        output = consolidator._format_for_llm(memories)

        assert "[ID=1]" in output
        assert "I like coffee" in output

    def test_format_includes_kind(self, consolidator):
        """Format should include memory kind."""
        memories = [
            {"id": 1, "text": "test", "metadata": {"kind": "pattern"}, "utility_score": 0.9, "is_pattern": True},
        ]
        output = consolidator._format_for_llm(memories)

        assert "(pattern," in output

    def test_format_shows_pattern_status(self, consolidator):
        """Patterns should be marked as PATTERN."""
        memories = [
            {"id": 1, "text": "test", "metadata": {}, "utility_score": 0.9, "is_pattern": True},
        ]
        output = consolidator._format_for_llm(memories)

        assert "PATTERN" in output

    def test_format_shows_utility_for_non_patterns(self, consolidator):
        """Non-patterns should show utility score."""
        memories = [
            {"id": 1, "text": "test", "metadata": {}, "utility_score": 0.75, "is_pattern": False},
        ]
        output = consolidator._format_for_llm(memories)

        assert "utility=0.75" in output


class TestCorrectionOrdering:
    """Test that corrections appear after regular facts in formatted output.

    This ensures the model sees corrections last, treating them as authoritative.
    See CLAUDE.md "Conflict Handling & Precedence" for design rationale.
    """

    def test_corrections_appear_last_in_formatted_output(self):
        """_format_memories_for_display should put corrections at the end."""
        from sem_mem.core import SemanticMemory

        # Create structured memory strings (as they would appear in L2)
        fact_memory = json.dumps({
            "text": "My favorite color is blue",
            "kind": MEMORY_KIND_FACT,
        })
        correction_memory = json.dumps({
            "text": "My favorite color is green",
            "kind": MEMORY_KIND_CORRECTION,
        })
        identity_memory = json.dumps({
            "text": "I am a physician",
            "kind": "identity",
        })

        # Input has mixed order: correction in the middle
        memories = [fact_memory, correction_memory, identity_memory]

        # Mock the minimal SemanticMemory needed
        with patch.object(SemanticMemory, '__init__', lambda self, **kw: None):
            mem = SemanticMemory()
            # Call the private method directly
            result = mem._format_memories_for_display(memories)

        # Corrections should be last, with [CORRECTION] prefix
        assert len(result) == 3
        assert "[CORRECTION]" in result[-1]
        assert "green" in result[-1]
        # Regular memories appear first
        assert "[CORRECTION]" not in result[0]
        assert "[CORRECTION]" not in result[1]

    def test_multiple_corrections_all_at_end(self):
        """Multiple corrections should all appear at the end."""
        from sem_mem.core import SemanticMemory

        memories = [
            json.dumps({"text": "Original fact 1", "kind": MEMORY_KIND_FACT}),
            json.dumps({"text": "Correction 1", "kind": MEMORY_KIND_CORRECTION}),
            json.dumps({"text": "Original fact 2", "kind": MEMORY_KIND_FACT}),
            json.dumps({"text": "Correction 2", "kind": MEMORY_KIND_CORRECTION}),
        ]

        with patch.object(SemanticMemory, '__init__', lambda self, **kw: None):
            mem = SemanticMemory()
            result = mem._format_memories_for_display(memories)

        # First two should be regular facts
        assert "[CORRECTION]" not in result[0]
        assert "[CORRECTION]" not in result[1]
        # Last two should be corrections
        assert "[CORRECTION]" in result[2]
        assert "[CORRECTION]" in result[3]

    def test_plain_text_memories_treated_as_facts(self):
        """Plain text (non-JSON) memories should be treated as facts."""
        from sem_mem.core import SemanticMemory

        memories = [
            "Plain text memory",
            json.dumps({"text": "A correction", "kind": MEMORY_KIND_CORRECTION}),
        ]

        with patch.object(SemanticMemory, '__init__', lambda self, **kw: None):
            mem = SemanticMemory()
            result = mem._format_memories_for_display(memories)

        assert result[0] == "Plain text memory"
        assert "[CORRECTION]" in result[1]

    def test_no_corrections_returns_original_order(self):
        """Without corrections, order should be preserved."""
        from sem_mem.core import SemanticMemory

        memories = [
            json.dumps({"text": "Fact A", "kind": MEMORY_KIND_FACT}),
            json.dumps({"text": "Identity B", "kind": "identity"}),
            json.dumps({"text": "Decision C", "kind": "decision"}),
        ]

        with patch.object(SemanticMemory, '__init__', lambda self, **kw: None):
            mem = SemanticMemory()
            result = mem._format_memories_for_display(memories)

        assert "Fact A" in result[0]
        assert "Identity B" in result[1]
        assert "Decision C" in result[2]
        for r in result:
            assert "[CORRECTION]" not in r
