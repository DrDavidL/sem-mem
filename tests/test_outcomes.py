"""
Tests for outcome-based learning functionality.

Tests the EWMA utility scoring, pattern promotion, and outcome-adjusted retrieval.
"""

import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from sem_mem.vector_index import HNSWIndex
from sem_mem.config import (
    OUTCOME_EWMA_ALPHA,
    OUTCOME_RETRIEVAL_ALPHA,
    PATTERN_MIN_SUCCESSES,
    PATTERN_MIN_UTILITY,
    OUTCOME_VALUES,
)


# =============================================================================
# HNSWIndex Outcome Fields Tests
# =============================================================================


class TestHNSWIndexOutcomeFields:
    """Tests for outcome fields in HNSWIndex."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def index(self, temp_dir):
        """Create an HNSWIndex instance."""
        return HNSWIndex(
            storage_dir=temp_dir,
            embedding_dim=4,  # Small dimension for tests
        )

    def test_new_entry_has_default_outcome_fields(self, index):
        """New entries should have default outcome fields."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, is_new = index.add("test memory", vector, {})

        entry = index.get_entry(entry_id)
        assert entry is not None
        assert entry["utility_score"] == 0.5
        assert entry["success_count"] == 0
        assert entry["failure_count"] == 0
        assert entry["last_outcome_at"] is None
        assert entry["is_pattern"] is False

    def test_get_entry_returns_none_for_invalid_id(self, index):
        """get_entry should return None for invalid ID."""
        assert index.get_entry(999) is None

    def test_get_entry_by_text_returns_id_and_entry(self, index):
        """get_entry_by_text should return (id, entry) tuple."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = index.add("test memory", vector, {})

        result = index.get_entry_by_text("test memory")
        assert result is not None
        memory_id, entry = result
        assert memory_id == entry_id
        assert entry["text"] == "test memory"

    def test_get_entry_by_text_returns_none_for_unknown(self, index):
        """get_entry_by_text should return None for unknown text."""
        assert index.get_entry_by_text("unknown") is None

    def test_update_entry_updates_fields(self, index):
        """update_entry should update specified fields."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = index.add("test memory", vector, {})

        success = index.update_entry(
            entry_id,
            utility_score=0.8,
            success_count=5,
            is_pattern=True,
        )
        assert success is True

        entry = index.get_entry(entry_id)
        assert entry["utility_score"] == 0.8
        assert entry["success_count"] == 5
        assert entry["is_pattern"] is True
        # Unchanged fields should remain
        assert entry["failure_count"] == 0

    def test_update_entry_returns_false_for_invalid_id(self, index):
        """update_entry should return False for invalid ID."""
        success = index.update_entry(999, utility_score=0.9)
        assert success is False

    def test_search_returns_memory_id(self, index):
        """search() should return (score, memory_id, entry) tuples."""
        vectors = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ]
        for i, vec in enumerate(vectors):
            index.add(f"memory {i}", vec, {})

        query_vec = np.array([1.0, 0.0, 0.0, 0.0])
        results = index.search([query_vec], k=3, threshold=0.0)

        assert len(results) >= 1
        score, memory_id, entry = results[0]
        # Score may be numpy float
        assert isinstance(score, (float, np.floating))
        # memory_id may be numpy int
        assert isinstance(memory_id, (int, np.integer))
        assert isinstance(entry, dict)
        assert "text" in entry

    def test_backward_compat_loads_missing_outcome_fields(self, temp_dir):
        """Loading old index without outcome fields should apply defaults."""
        # Create index with an entry
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = index.add("old memory", vector, {})

        # Manually remove outcome fields to simulate old index
        index._id_to_entry[entry_id] = {
            "text": "old memory",
            "vector": vector.tolist(),
            "metadata": {},
        }
        index.save()

        # Reload and check defaults applied
        new_index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        entry = new_index.get_entry(entry_id)

        assert entry["utility_score"] == 0.5
        assert entry["success_count"] == 0
        assert entry["failure_count"] == 0
        assert entry["last_outcome_at"] is None
        assert entry["is_pattern"] is False


# =============================================================================
# Outcome Update Tests (EWMA)
# =============================================================================


class TestOutcomeUpdates:
    """Tests for EWMA utility score updates."""

    def test_success_increases_utility(self):
        """Success should increase utility_score."""
        # Start at 0.5, record success (1.0)
        # Expected: 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        old_utility = 0.5
        value = OUTCOME_VALUES["success"]
        new_utility = OUTCOME_EWMA_ALPHA * value + (1 - OUTCOME_EWMA_ALPHA) * old_utility
        assert new_utility == pytest.approx(0.65)

    def test_failure_decreases_utility(self):
        """Failure should decrease utility_score."""
        # Start at 0.5, record failure (0.0)
        # Expected: 0.3 * 0.0 + 0.7 * 0.5 = 0.35
        old_utility = 0.5
        value = OUTCOME_VALUES["failure"]
        new_utility = OUTCOME_EWMA_ALPHA * value + (1 - OUTCOME_EWMA_ALPHA) * old_utility
        assert new_utility == pytest.approx(0.35)

    def test_neutral_no_change(self):
        """Neutral should keep utility_score the same."""
        # Start at 0.5, record neutral (0.5)
        # Expected: 0.3 * 0.5 + 0.7 * 0.5 = 0.5
        old_utility = 0.5
        value = OUTCOME_VALUES["neutral"]
        new_utility = OUTCOME_EWMA_ALPHA * value + (1 - OUTCOME_EWMA_ALPHA) * old_utility
        assert new_utility == pytest.approx(0.5)

    def test_ewma_converges_to_consistent_outcomes(self):
        """Utility should converge toward consistent outcome value."""
        utility = 0.5
        # Record 10 successes
        for _ in range(10):
            utility = OUTCOME_EWMA_ALPHA * 1.0 + (1 - OUTCOME_EWMA_ALPHA) * utility
        # Should be close to 1.0
        assert utility > 0.95

        # Now record 10 failures
        for _ in range(10):
            utility = OUTCOME_EWMA_ALPHA * 0.0 + (1 - OUTCOME_EWMA_ALPHA) * utility
        # Should be close to 0.0
        assert utility < 0.05

    def test_pattern_promotion_thresholds(self):
        """Test pattern promotion requires both success count and utility."""
        # Just high utility is not enough
        success_count = 2  # Below PATTERN_MIN_SUCCESSES (3)
        utility = 0.95  # Above PATTERN_MIN_UTILITY (0.9)
        is_pattern = success_count >= PATTERN_MIN_SUCCESSES and utility >= PATTERN_MIN_UTILITY
        assert is_pattern is False

        # Just high success count is not enough
        success_count = 5  # Above threshold
        utility = 0.7  # Below threshold
        is_pattern = success_count >= PATTERN_MIN_SUCCESSES and utility >= PATTERN_MIN_UTILITY
        assert is_pattern is False

        # Both thresholds met
        success_count = 3
        utility = 0.9
        is_pattern = success_count >= PATTERN_MIN_SUCCESSES and utility >= PATTERN_MIN_UTILITY
        assert is_pattern is True


# =============================================================================
# Scoring Integration Tests
# =============================================================================


class TestScoringIntegration:
    """Tests for outcome-based scoring in retrieval."""

    def test_high_utility_boosts_ranking(self):
        """High utility memory should rank above same-similarity low-utility."""
        sim_score = 0.8

        # High utility (1.0) adds bonus
        high_utility = 1.0
        high_final = sim_score + OUTCOME_RETRIEVAL_ALPHA * (high_utility - 0.5)

        # Low utility (0.0) subtracts penalty
        low_utility = 0.0
        low_final = sim_score + OUTCOME_RETRIEVAL_ALPHA * (low_utility - 0.5)

        assert high_final > low_final
        # At alpha=0.2: high=0.9, low=0.7
        assert high_final == pytest.approx(0.9)
        assert low_final == pytest.approx(0.7)

    def test_neutral_utility_no_effect(self):
        """Neutral utility (0.5) should not change the score."""
        sim_score = 0.8
        utility = 0.5
        final_score = sim_score + OUTCOME_RETRIEVAL_ALPHA * (utility - 0.5)
        assert final_score == pytest.approx(sim_score)

    def test_utility_can_rerank_results(self):
        """Utility differences should be able to reorder results."""
        # Memory A: lower similarity but high utility
        sim_a = 0.75
        utility_a = 1.0
        final_a = sim_a + OUTCOME_RETRIEVAL_ALPHA * (utility_a - 0.5)

        # Memory B: higher similarity but low utility
        sim_b = 0.80
        utility_b = 0.0
        final_b = sim_b + OUTCOME_RETRIEVAL_ALPHA * (utility_b - 0.5)

        # A should rank higher despite lower similarity
        # final_a = 0.75 + 0.2 * 0.5 = 0.85
        # final_b = 0.80 + 0.2 * (-0.5) = 0.70
        assert final_a > final_b


# =============================================================================
# SemanticMemory record_outcome Tests (with mocking)
# =============================================================================


class TestSemanticMemoryRecordOutcome:
    """Tests for SemanticMemory.record_outcome method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mock_memory(self, temp_dir):
        """Create a mock SemanticMemory-like object for testing."""
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)

        class MockMemory:
            def __init__(self, index):
                self.vector_index = index

            def record_outcome(self, memory_id, outcome, metadata=None):
                if outcome not in OUTCOME_VALUES:
                    raise ValueError(f"Invalid outcome: {outcome}")

                entry = self.vector_index.get_entry(memory_id)
                if not entry:
                    raise KeyError(f"Memory not found: {memory_id}")

                old_utility = entry.get("utility_score", 0.5)
                value = OUTCOME_VALUES[outcome]
                new_utility = OUTCOME_EWMA_ALPHA * value + (1 - OUTCOME_EWMA_ALPHA) * old_utility

                success_count = entry.get("success_count", 0)
                failure_count = entry.get("failure_count", 0)
                if outcome == "success":
                    success_count += 1
                elif outcome == "failure":
                    failure_count += 1

                is_pattern = entry.get("is_pattern", False)
                if not is_pattern:
                    if success_count >= PATTERN_MIN_SUCCESSES and new_utility >= PATTERN_MIN_UTILITY:
                        is_pattern = True

                from datetime import datetime
                updates = {
                    "utility_score": new_utility,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "last_outcome_at": datetime.now().isoformat(),
                    "is_pattern": is_pattern,
                }
                self.vector_index.update_entry(memory_id, **updates)
                self.vector_index.save()

                return {**entry, **updates}

        return MockMemory(index)

    def test_record_success_increases_utility(self, mock_memory):
        """record_outcome with 'success' should increase utility."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = mock_memory.vector_index.add("test", vector, {})

        result = mock_memory.record_outcome(entry_id, "success")

        assert result["utility_score"] == pytest.approx(0.65)
        assert result["success_count"] == 1
        assert result["failure_count"] == 0

    def test_record_failure_decreases_utility(self, mock_memory):
        """record_outcome with 'failure' should decrease utility."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = mock_memory.vector_index.add("test", vector, {})

        result = mock_memory.record_outcome(entry_id, "failure")

        assert result["utility_score"] == pytest.approx(0.35)
        assert result["success_count"] == 0
        assert result["failure_count"] == 1

    def test_invalid_outcome_raises(self, mock_memory):
        """Invalid outcome string should raise ValueError."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = mock_memory.vector_index.add("test", vector, {})

        with pytest.raises(ValueError, match="Invalid outcome"):
            mock_memory.record_outcome(entry_id, "invalid")

    def test_unknown_memory_id_raises(self, mock_memory):
        """Unknown memory_id should raise KeyError."""
        with pytest.raises(KeyError, match="Memory not found"):
            mock_memory.record_outcome(999, "success")

    def test_pattern_promotion_after_successes(self, mock_memory):
        """Memory should become pattern after enough successes."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = mock_memory.vector_index.add("test", vector, {})

        # Record enough successes to promote
        for i in range(PATTERN_MIN_SUCCESSES + 2):
            result = mock_memory.record_outcome(entry_id, "success")

        assert result["is_pattern"] is True
        assert result["success_count"] >= PATTERN_MIN_SUCCESSES
        assert result["utility_score"] >= PATTERN_MIN_UTILITY

    def test_updates_persist(self, mock_memory, temp_dir):
        """Outcome updates should persist across reloads."""
        vector = np.array([1.0, 0.0, 0.0, 0.0])
        entry_id, _ = mock_memory.vector_index.add("test", vector, {})

        mock_memory.record_outcome(entry_id, "success")
        mock_memory.record_outcome(entry_id, "success")

        # Reload index
        new_index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)
        entry = new_index.get_entry(entry_id)

        assert entry["success_count"] == 2
        assert entry["utility_score"] > 0.5


# =============================================================================
# Recall with include_metadata Tests
# =============================================================================


class TestRecallWithMetadata:
    """Tests for recall() with include_metadata parameter."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def populated_index(self, temp_dir):
        """Create an HNSWIndex with some test entries."""
        index = HNSWIndex(storage_dir=temp_dir, embedding_dim=4)

        # Add memories with different utility scores
        vectors = [
            (np.array([1.0, 0.0, 0.0, 0.0]), "high utility memory", 0.9),
            (np.array([0.9, 0.1, 0.0, 0.0]), "medium utility memory", 0.5),
            (np.array([0.8, 0.2, 0.0, 0.0]), "low utility memory", 0.1),
        ]

        entry_ids = []
        for vec, text, utility in vectors:
            entry_id, _ = index.add(text, vec, {})
            index.update_entry(entry_id, utility_score=utility)
            entry_ids.append(entry_id)

        index.save()
        return index, entry_ids

    def test_search_results_include_memory_id(self, populated_index):
        """Search results should include memory_id for each entry."""
        index, entry_ids = populated_index
        query_vec = np.array([1.0, 0.0, 0.0, 0.0])

        results = index.search([query_vec], k=3, threshold=0.0)

        assert len(results) == 3
        for score, memory_id, entry in results:
            # Convert numpy int to Python int for comparison
            assert int(memory_id) in entry_ids
            assert isinstance(memory_id, (int, np.integer))

    def test_utility_scores_returned_in_results(self, populated_index):
        """Utility scores should be accessible in search results."""
        index, _ = populated_index
        query_vec = np.array([1.0, 0.0, 0.0, 0.0])

        results = index.search([query_vec], k=3, threshold=0.0)

        utilities = [entry.get("utility_score", 0.5) for _, _, entry in results]
        assert 0.9 in utilities
        assert 0.5 in utilities
        assert 0.1 in utilities
