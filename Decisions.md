# Architectural Decisions

This document captures key architectural decisions and potential improvements for sem-mem, including lessons learned from external research.

---

## Analysis: "Beyond Basic RAG" Article Lessons

**Source:** Discussion post on advanced RAG architectures

### Summary of Article Patterns

The article describes three architectural patterns to address common RAG failures:

| Pattern | Problem Solved | Key Technique |
|---------|----------------|---------------|
| **Mini-RAG** | Latency & privacy for static data | Client-side vector search with pre-computed embeddings |
| **Hybrid Graph** | Exactness (SQL names) + temporal context | Trigrams + temporal graphs |
| **Noise Filter** | Signal-to-noise in chat logs | Heuristic gates before LLM evaluation |

---

## Relevance to Sem-Mem

### Already Implemented (Strengths)

1. **Heuristic Signal Filtering** - The article's "Slack Brain" pattern is *already implemented* in sem-mem's [auto_memory.py](sem_mem/auto_memory.py):
   - Cheap heuristic patterns before LLM evaluation
   - Salience scoring (0-1) with thresholds
   - LLM judge only for ambiguous cases (0.3-0.7)
   - **Takeaway:** Our approach aligns with industry best practice

2. **Local-First Storage** - The "Mini-RAG" principle of shipping embeddings locally matches our architecture:
   - HNSW index stored locally (`hnsw_index.bin`)
   - No cloud vector DB dependency for retrieval
   - L1 cache for instant recall of hot memories

3. **Tiered Memory** - Our L1/L2 architecture is more sophisticated than basic RAG:
   - SmartCache (L1) → instant in-memory access
   - HNSW (L2) → persistent semantic search
   - Promotion/demotion between tiers

### Gaps & Opportunities

| Gap | Article Pattern | Current State | Potential Improvement |
|-----|-----------------|---------------|----------------------|
| **Exact match retrieval** | Trigrams (pg_trgm) | Vector-only search | Add BM25/lexical fallback for exact queries |
| **Temporal context** | Temporal graphs | Timestamps in metadata but not used for retrieval | Time-aware retrieval ranking |
| **Thread normalization** | Aggressive normalization to `.md`/`.csv` | Conversations stored as-is | Summarize/normalize before L2 storage |
| **Conversation patterns** | Q&A pattern detection | N/A | Add heuristic: "ends with thanks/fixed" = completed thread |

---

## Proposed Plan: Phase-Based Improvements

*Refined based on Laura's guidance (Dec 2024)*

### Phase 1: Enhanced Heuristic Filtering (Low Effort, High Value)

**Goal:** Better decide which interactions are worth turning into memories.

**Files to modify:**
- [sem_mem/auto_memory.py](sem_mem/auto_memory.py) - Core heuristic logic
- [sem_mem/thread_utils.py](sem_mem/thread_utils.py) - Thread-level detection

#### 1.1 Q&A Completion Patterns

Detect resolved problem-solving episodes:

```python
# Question-like starters
QUESTION_PATTERNS = [
    r"\?$",
    r"^how\s+(do|can|should)\s+i\b",
    r"^why\s+(is|does|did|are)\b",
    r"^can\s+you\s+help\b",
    r"^what\s+(is|are|does)\b",
]

# Completion tokens (thread-final position)
COMPLETION_PATTERNS = [
    r"\bthanks?\b",
    r"\bthank\s+you\b",
    r"\bthat\s+worked\b",
    r"\bgot\s+it\b",
    r"\bperfect\b",
    r"\bresolved\b",
    r"\bfixed\s+it\b",
]
```

**Logic:** If thread has question-like message earlier AND ends with completion pattern → boost memory-worthiness.

#### 1.2 Multi-Turn / Refinement Bias

Detect non-trivial problem-solving:

```python
REFINEMENT_PATTERNS = [
    r"^actually\b",
    r"^instead\b",
    r"^what\s+about\b",
    r"^wait\b",
    r"^let\s+me\s+clarify\b",
]
```

**Logic:** Multiple user turns with refinement → boost score (indicates iterative problem-solving).

#### 1.3 De-duplicate Meta Interactions

Down-rank low-value turns:

```python
META_NOISE_PATTERNS = [
    r"^hi$",
    r"^hello$",
    r"^test$",
    r"^ok$",
    r"^you\s+there\??$",
    r"^thanks$",  # Standalone thanks without context
]
```

#### 1.4 Implementation Structure

```python
class HeuristicSignals(NamedTuple):
    """Structured heuristic evaluation for a thread/exchange."""
    is_qa_like: bool
    has_completion_token: bool
    turn_count: int
    has_refinements: bool
    is_meta_noise: bool

def compute_thread_heuristics(messages: List[dict]) -> HeuristicSignals:
    """Analyze full thread for memory-worthiness signals."""
    ...

def heuristic_score(signals: HeuristicSignals) -> float:
    """
    Compute memory-worthiness score from signals.

    base + qa_bonus + completion_bonus + turn_bonus - noise_penalty
    """
    score = 0.3  # base
    if signals.is_qa_like and signals.has_completion_token:
        score += 0.3  # Resolved Q&A is high value
    if signals.turn_count >= 3:
        score += 0.1 * min(signals.turn_count - 2, 3)  # Multi-turn bonus, capped
    if signals.has_refinements:
        score += 0.15  # Iterative refinement is valuable
    if signals.is_meta_noise:
        score -= 0.4  # Heavy penalty for noise
    return max(0.0, min(1.0, score))
```

**Key principle:** Deterministic and cheap—pure string methods + regex, no model calls.

---

### Phase 2: Hybrid Search with Cheap Lexical Fallback (Medium Effort, High Value)

**Goal:** Fix cases where vectors fail on exactness (identifiers, file names, table names).

**Files to modify/create:**
- Create [sem_mem/lexical_index.py](sem_mem/lexical_index.py) - Simple lexical search
- Modify [sem_mem/core.py](sem_mem/core.py) - Hybrid `recall()` method
- Modify [sem_mem/vector_index.py](sem_mem/vector_index.py) - Add text storage for lexical

#### 2.1 Identifier Detection

```python
def looks_like_identifier(query: str) -> bool:
    """
    Detect if query looks like an exact thing (file, table, id).

    Signals:
    - Contains underscores: tbl_orders_001
    - Contains hyphens with alphanumerics: my-config-file
    - Mixed digits+letters: user123, v2.1.0
    - Long contiguous token (>20 chars)
    - File extensions: .py, .json, .md
    """
    patterns = [
        r"\w+_\w+",           # underscore_separated
        r"\w+-\w+-\w+",       # hyphen-separated (3+ parts)
        r"[a-zA-Z]+\d+",      # letters+digits
        r"\d+[a-zA-Z]+",      # digits+letters
        r"\.\w{2,4}$",        # file extension
        r"\S{20,}",           # very long token
    ]
    return any(re.search(p, query) for p in patterns)
```

#### 2.2 Cheap Lexical Index (No Dependencies)

```python
class LexicalIndex:
    """Simple in-memory lexical search. No external dependencies."""

    def __init__(self):
        self.documents: Dict[str, str] = {}  # id -> raw text
        self.tokens: Dict[str, Set[str]] = {}  # token -> set of doc ids

    def add(self, doc_id: str, text: str):
        """Index a document."""
        self.documents[doc_id] = text
        for token in self._tokenize(text):
            self.tokens.setdefault(token, set()).add(doc_id)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search by token overlap.
        Returns: List of (doc_id, score) tuples.
        """
        query_tokens = set(self._tokenize(query))
        scores = {}
        for token in query_tokens:
            for doc_id in self.tokens.get(token, []):
                scores[doc_id] = scores.get(doc_id, 0) + 1

        # Normalize by query length
        results = [(doc_id, score / len(query_tokens))
                   for doc_id, score in scores.items()]
        return sorted(results, key=lambda x: -x[1])[:k]

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, split on whitespace/punctuation."""
        return re.findall(r'\w+', text.lower())
```

#### 2.3 Hybrid Retrieval Flow

```python
def recall(self, query: str, k: int = 5) -> List[dict]:
    """Hybrid retrieval: vector + optional lexical."""

    # Always get vector results
    results_vector = self.vector_index.search(query, k)

    # If query looks like identifier, also try lexical
    if looks_like_identifier(query):
        results_lexical = self.lexical_index.search(query, k)
        return self._merge_results(results_vector, results_lexical)

    return results_vector

def _merge_results(self, vector_results, lexical_results, alpha=0.7):
    """
    Merge vector and lexical results.
    alpha: weight for vector scores (1-alpha for lexical)
    """
    # Combine scores, deduplicate by doc_id
    ...
```

**Decision:** Start with cheap lexical fallback. Only add `rank_bm25` dependency if limitations surface in practice.

---

### Phase 3: Time-Aware Scoring (Medium Effort, Medium Value)

**Goal:** Prefer relevant recent memories without killing old high-signal ones.

**Files to modify:**
- [sem_mem/core.py](sem_mem/core.py) - Scoring wrapper
- [sem_mem/vector_index.py](sem_mem/vector_index.py) - Ensure timestamps stored
- [sem_mem/config.py](sem_mem/config.py) - Tunable parameters

#### 3.1 Ensure Timestamps

Every memory gets a timestamp on creation/update:

```python
def remember(self, text: str, metadata: Optional[dict] = None):
    metadata = metadata or {}
    if "timestamp" not in metadata:
        metadata["timestamp"] = datetime.now().isoformat()
    ...
```

#### 3.2 Time-Decay Scoring

```python
from math import exp, log

def time_adjusted_score(
    sim_score: float,
    timestamp: str,
    half_life_days: float = 30.0,
    alpha: float = 0.3,
) -> float:
    """
    Apply time decay to similarity score.

    Args:
        sim_score: Raw similarity score (0-1)
        timestamp: ISO format timestamp
        half_life_days: Days until decay reaches 50%
        alpha: Floor weight (prevents old memories from dying completely)

    Returns:
        Adjusted score: sim_score * (alpha + (1 - alpha) * decay)
    """
    age_days = (datetime.now() - datetime.fromisoformat(timestamp)).days
    decay_rate = log(2) / half_life_days
    time_weight = exp(-decay_rate * age_days)

    return sim_score * (alpha + (1 - alpha) * time_weight)
```

#### 3.3 Configuration

```python
# sem_mem/config.py

# Time-aware retrieval
TIME_DECAY_ENABLED = True
TIME_DECAY_HALF_LIFE_DAYS = 30  # Days until 50% decay
TIME_DECAY_ALPHA = 0.3  # Floor weight for old memories
```

**Rationale:** `alpha=0.3` ensures even very old memories retain 30% of their original score if highly relevant. Stable identity facts won't disappear.

---

### Phase 4: Thread Normalization (Low Priority, Partial Exists)

**Current state:** Already have:
- Structured thread storage (`threads.json`)
- `summarize_conversation_window()` in thread_utils.py
- Auto-memory extracts salient facts

**Future improvements (only if needed):**
- Force Markdown normalization for external ingests
- Split very long transcripts before embedding
- Deduplicate near-identical thread summaries

---

## Decision: What NOT to Implement

1. **Full knowledge graph (Neo4j/etc)** - Overkill for personal memory use case
2. **Client-side embeddings** - Our memory isn't static like a function dictionary
3. **Multi-tenant isolation** - Explicitly out of scope per README
4. **Heavy BM25 dependency** - Start with cheap lexical; add `rank_bm25` only if needed

---

## Final Roadmap (Laura's Refinement)

| Order | Phase | Effort | Impact | Key Change |
|-------|-------|--------|--------|------------|
| **1** | Phase 1: Heuristics++ | Low | High | Add Q&A completion, refinement, noise patterns |
| **2** | Phase 2: Hybrid v1 | Medium | High | Cheap lexical index (no deps), identifier detection |
| **3** | Phase 3: Time-decay | Medium | Medium | Half-life scoring with alpha floor |
| **4** | Phase 4: Normalization | Low | Low | Only if ingesting complex external data |

---

## Implementation Checklist

### Phase 1 (COMPLETED - Dec 2024)
- [x] Add `HeuristicSignals` NamedTuple to `auto_memory.py`
- [x] Implement `QUESTION_PATTERNS` and `COMPLETION_PATTERNS`
- [x] Implement `REFINEMENT_PATTERNS` for multi-turn detection
- [x] Implement `META_NOISE_PATTERNS` for down-ranking
- [x] Add `compute_thread_heuristics()` function
- [x] Add `compute_thread_heuristic_score()` function
- [x] Add thread-level completion detection to `thread_utils.py`:
  - `is_thread_completed()` - detect completed Q&A threads
  - `get_thread_memory_score()` - compute memory-worthiness score
  - `analyze_thread()` - full analysis with recommendation

### Phase 2 (COMPLETED - Dec 2024)
- [x] Create `sem_mem/lexical_index.py` with `LexicalIndex` class
- [x] Add `looks_like_identifier()` helper (detects `tbl_orders_001`, `my-config.json`, `getUserById`)
- [x] Wire lexical index into `SemanticMemory.__init__()` with auto-rebuild from HNSW
- [x] Update `recall()` for hybrid retrieval (vector + lexical merge when query is identifier-like)
- [x] Add lexical index persistence (`lexical_index.json` alongside HNSW files)
- [x] Test with identifier-heavy queries

### Phase 3 (COMPLETED - Dec 2024)
- [x] Ensure all memories have timestamps in metadata (`created_at` field added in `HNSWIndex.add()`)
- [x] Add `time_adjusted_score()` function to `core.py`
- [x] Add config params: `TIME_DECAY_ENABLED`, `TIME_DECAY_HALF_LIFE_DAYS`, `TIME_DECAY_ALPHA`
- [x] Wire into retrieval scoring (applied after outcome scoring in `recall()`)
- [x] Test with mix of old/new memories (verified decay curve and alpha floor)

---

## Next Steps

**Phases 1, 2, and 3 are complete.** Phase 4 (Thread Normalization) is documented but marked as low priority/optional - only needed if ingesting complex external data.

### New Exports Available

```python
from sem_mem import (
    # Auto-memory heuristics (Phase 1)
    HeuristicSignals,
    compute_thread_heuristics,
    compute_thread_heuristic_score,

    # Thread analysis helpers (Phase 1)
    is_thread_completed,
    get_thread_memory_score,
    analyze_thread,

    # Lexical index for hybrid search (Phase 2)
    LexicalIndex,
    looks_like_identifier,
    merge_search_results,
)

# Time-decay config (Phase 3) - from sem_mem.config
from sem_mem.config import (
    TIME_DECAY_ENABLED,        # Master switch (default: True)
    TIME_DECAY_HALF_LIFE_DAYS, # Days until 50% decay (default: 30)
    TIME_DECAY_ALPHA,          # Floor weight for old memories (default: 0.3)
)

# Phase 1: Thread analysis
messages = [
    {"role": "user", "content": "How do I configure SSL?"},
    {"role": "assistant", "content": "Here's how..."},
    {"role": "user", "content": "Thanks, that worked!"},
]

if is_thread_completed(messages):
    print("Thread is a completed Q&A")

score = get_thread_memory_score(messages)  # 0.0-1.0
analysis = analyze_thread(messages)
# {"signals": {...}, "score": 0.6, "is_completed": True, "recommendation": "save"}

# Phase 2: Identifier detection (used automatically in recall())
looks_like_identifier("tbl_orders_001")  # True
looks_like_identifier("my-config.json")  # True
looks_like_identifier("getUserById")     # True (CamelCase)
looks_like_identifier("what is X?")      # False

# Hybrid search is automatic when query looks like identifier:
# memory.recall("tbl_orders_001")  # Uses vector + lexical merge
```

### How Hybrid Search Works (Phase 2)

When `recall()` is called:
1. Vector search (HNSW) always runs
2. If `looks_like_identifier(query)` is True, lexical search also runs
3. Results are merged: `0.7 * vector_score + 0.3 * lexical_score`
4. Logs indicate when hybrid search was used: `"🔤 Query looks like identifier, adding lexical search..."`

### How Time-Decay Scoring Works (Phase 3)

When `recall()` returns results, time decay is applied:
1. Each memory's `created_at` timestamp is checked
2. A decay factor is computed: `decay = exp(-ln(2) * age_days / half_life)`
3. Time weight: `weight = alpha + (1 - alpha) * decay`
4. Final score: `adjusted_score = sim_score * weight`

**Example decay curve** (half_life=30, alpha=0.3):
| Age | Decay Factor | Time Weight | Score Multiplier |
|-----|--------------|-------------|------------------|
| 0 days | 1.0 | 1.0 | 100% |
| 7 days | 0.85 | 0.90 | 90% |
| 30 days | 0.50 | 0.65 | 65% |
| 60 days | 0.25 | 0.48 | 48% |
| 90 days | 0.13 | 0.39 | 39% |
| 1 year | ~0 | 0.30 | 30% (floor) |

The alpha floor ensures important old memories (identity facts, stable preferences) remain retrievable even after long periods.
