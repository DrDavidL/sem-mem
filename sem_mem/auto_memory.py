"""
Automatic memory consolidation based on salience detection.

Evaluates exchanges and decides what's worth remembering for future conversations.
Uses a hybrid approach: cheap heuristics first, LLM judge for ambiguous cases.

Memory Policy:
- Prioritize durable, high-value memories (identity, preferences, decisions)
- Treat explicit corrections as overriding older facts
- Tag memories with 'kind' for better retrieval handling
"""

import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Any, TYPE_CHECKING, NamedTuple

from .config import QUERY_EXPANSION_MODEL

if TYPE_CHECKING:
    from openai import OpenAI, AsyncOpenAI
    from .providers import BaseChatProvider


# Model for salience evaluation (same cheap model as query expansion)
AUTO_MEMORY_MODEL = QUERY_EXPANSION_MODEL

# Memory kinds for structured storage
MEMORY_KIND_IDENTITY = "identity"      # Who the user is
MEMORY_KIND_PREFERENCE = "preference"  # What they prefer
MEMORY_KIND_DECISION = "decision"      # Decisions made together
MEMORY_KIND_CORRECTION = "correction"  # Updates/corrections to prior info
MEMORY_KIND_FACT = "fact"              # General factual information


@dataclass
class MemorySignal:
    """Result of salience evaluation."""
    should_remember: bool
    memory_text: str  # Distilled/summarized fact to store
    salience: float  # 0-1 importance score
    reason: str  # Why it's worth remembering (for debugging)
    kind: str = MEMORY_KIND_FACT  # Type of memory for retrieval handling


# Heuristic patterns that suggest high salience
EXPLICIT_MARKERS = [
    r"\bremember\s+(this|that)\b",
    r"\bimportant\b",
    r"\balways\b",
    r"\bnever\b",
    r"\bdon'?t\s+forget\b",
    r"\bkeep\s+in\s+mind\b",
    r"\bfor\s+future\s+reference\b",
    r"\bmy\s+(name|email|phone|address|preference)\b",
    r"\bi\s+(am|work|live|prefer|like|hate|need|want)\b",
    r"\bi'?m\s+(a|an|the)\b",
]

CONFIRMATION_PATTERNS = [
    r"\byes,?\s+(exactly|that'?s?\s+right|correct|perfect)\b",
    r"\bthat'?s?\s+(it|correct|right|perfect)\b",
    r"\bexactly\b",
    r"\bprecisely\b",
    r"\bspot\s+on\b",
]

# Patterns suggesting factual information
FACTUAL_PATTERNS = [
    r"\bis\s+called\b",
    r"\bis\s+located\b",
    r"\bwas\s+born\b",
    r"\bworks?\s+at\b",
    r"\blives?\s+in\b",
    r"\bmy\s+\w+\s+is\b",
]

# Patterns for explicit preferences (high priority for memory)
PREFERENCE_PATTERNS = [
    r"\bi\s+prefer\b",
    r"\bi\s+like\s+when\b",
    r"\bi\s+don'?t\s+like\s+when\b",
    r"\bgoing\s+forward,?\s+please\b",
    r"\bfrom\s+now\s+on\b",
    r"\bplease\s+always\b",
    r"\bplease\s+never\b",
    r"\bi\s+would\s+prefer\b",
    r"\bi'?d\s+rather\b",
    r"\bmy\s+preference\s+is\b",
]

# Patterns for identity/role statements (high priority for memory)
IDENTITY_PATTERNS = [
    r"\bi'?m\s+(a|an)\s+\w+",  # "I'm a physician", "I'm an engineer"
    r"\bi\s+am\s+(a|an)\s+\w+",
    r"\bi\s+work\s+(as|in|at|for)\b",
    r"\bmy\s+(job|role|title|position)\s+(is|as)\b",
    r"\bi'?ve\s+been\s+(a|an|working)\b",
    r"\bi\s+specialize\s+in\b",
    r"\bmy\s+background\s+is\b",
    r"\bi\s+have\s+\d+\s+(years?|kids?|children)\b",
    r"\bi'?m\s+married\b",
    r"\bi\s+live\s+(in|with)\b",
]

# Patterns for corrections/updates (override older facts)
CORRECTION_PATTERNS = [
    r"\bactually,?\b",
    r"\bi\s+changed\s+my\s+mind\b",
    r"\bno,?\s+(that'?s?\s+not|earlier\s+i\s+said)\b",
    r"\blet\s+me\s+correct\b",
    r"\bi\s+was\s+wrong\b",
    r"\bthat'?s?\s+not\s+(right|correct|accurate)\b",
    r"\bi\s+meant\s+to\s+say\b",
    r"\bforget\s+what\s+i\s+said\b",
    r"\bignore\s+(my\s+)?previous\b",
    r"\bupdated?\s+(my\s+)?thinking\b",
    r"\bon\s+second\s+thought\b",
    r"\bscratch\s+that\b",
]

# =============================================================================
# Thread-Level Heuristics (Phase 1 Enhancement)
# =============================================================================

# Question-like patterns (detect Q&A threads)
QUESTION_PATTERNS = [
    r"\?$",                              # Ends with question mark
    r"^how\s+(do|can|should|would)\s+i\b",
    r"^why\s+(is|does|did|are|was|were)\b",
    r"^can\s+you\s+(help|explain|show)\b",
    r"^what\s+(is|are|does|should|would)\b",
    r"^where\s+(is|are|do|can)\b",
    r"^when\s+(should|do|did|will)\b",
    r"^is\s+(there|it|this)\b",
    r"^could\s+you\b",
    r"^would\s+you\b",
]

# Completion tokens (indicate resolved problem-solving)
COMPLETION_PATTERNS = [
    r"\bthanks?\b",
    r"\bthank\s+you\b",
    r"\bthat\s+worked\b",
    r"\bgot\s+it\b",
    r"\bperfect\b",
    r"\bresolved\b",
    r"\bfixed\s+it\b",
    r"\bproblem\s+solved\b",
    r"\bexcellent\b",
    r"\bawesome\b",
    r"\bgreat,?\s+that\b",
    r"\bmakes\s+sense\b",
]

# Refinement patterns (indicate iterative problem-solving)
REFINEMENT_PATTERNS = [
    r"^actually\b",
    r"^instead\b",
    r"^what\s+about\b",
    r"^wait\b",
    r"^let\s+me\s+clarify\b",
    r"^sorry,?\s+i\s+meant\b",
    r"^no,?\s+i\s+mean\b",
    r"^to\s+clarify\b",
    r"^one\s+more\s+thing\b",
    r"^also\b",
    r"^but\s+what\s+if\b",
    r"^follow.?up\b",
]

# Meta-noise patterns (low-value interactions to down-rank)
META_NOISE_PATTERNS = [
    r"^hi$",
    r"^hello$",
    r"^hey$",
    r"^test$",
    r"^testing$",
    r"^ok$",
    r"^okay$",
    r"^k$",
    r"^yes$",
    r"^no$",
    r"^you\s+there\??$",
    r"^thanks$",              # Standalone thanks without context
    r"^thank\s+you$",         # Standalone thank you without context
    r"^cool$",
    r"^nice$",
    r"^good$",
    r"^\.\.\.$",
    r"^nevermind$",
    r"^nvm$",
]


class HeuristicSignals(NamedTuple):
    """
    Structured heuristic evaluation for a thread/exchange.

    Used to compute memory-worthiness score from cheap signals
    without any LLM calls.
    """
    is_qa_like: bool           # Thread contains question-like patterns
    has_completion_token: bool  # Thread ends with completion signal
    turn_count: int            # Number of user turns in thread
    has_refinements: bool      # User refined/clarified during thread
    is_meta_noise: bool        # Exchange is low-value meta interaction
    has_identity: bool         # Contains identity statements
    has_preference: bool       # Contains preference statements
    has_correction: bool       # Contains correction statements


def _matches_any_pattern(text: str, patterns: List[str]) -> bool:
    """Check if text matches any pattern in the list."""
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower, re.IGNORECASE) for p in patterns)


def _is_question_like(text: str) -> bool:
    """Check if text looks like a question."""
    return _matches_any_pattern(text, QUESTION_PATTERNS)


def _has_completion_token(text: str) -> bool:
    """Check if text contains completion/resolution signals."""
    return _matches_any_pattern(text, COMPLETION_PATTERNS)


def _has_refinement(text: str) -> bool:
    """Check if text indicates refinement/clarification."""
    return _matches_any_pattern(text, REFINEMENT_PATTERNS)


def _is_meta_noise(text: str) -> bool:
    """Check if text is low-value meta interaction."""
    # Must match the ENTIRE message (short, standalone)
    text_clean = text.lower().strip()
    # Only consider short messages as potential noise
    if len(text_clean) > 50:
        return False
    return _matches_any_pattern(text_clean, META_NOISE_PATTERNS)


def compute_thread_heuristics(messages: List[dict]) -> HeuristicSignals:
    """
    Analyze a full thread for memory-worthiness signals.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        HeuristicSignals with detected patterns
    """
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    turn_count = len(user_messages)

    if not user_messages:
        return HeuristicSignals(
            is_qa_like=False,
            has_completion_token=False,
            turn_count=0,
            has_refinements=False,
            is_meta_noise=True,
            has_identity=False,
            has_preference=False,
            has_correction=False,
        )

    # Check if any early message is question-like
    is_qa_like = any(_is_question_like(msg) for msg in user_messages[:-1]) if len(user_messages) > 1 else _is_question_like(user_messages[0])

    # Check if final user message has completion token
    last_user_msg = user_messages[-1]
    has_completion = _has_completion_token(last_user_msg)

    # Check for refinements in any user message
    has_refinements = any(_has_refinement(msg) for msg in user_messages)

    # Check if the entire exchange is meta-noise
    # Only true if ALL user messages are noise AND total content is very short
    all_messages_noise = all(_is_meta_noise(msg) for msg in user_messages)
    total_content = " ".join(user_messages)
    is_meta_noise = all_messages_noise and len(total_content) < 100

    # Check for high-value content types
    combined_user_text = " ".join(user_messages)
    has_identity = looks_like_identity(combined_user_text)
    has_preference = looks_like_preference(combined_user_text)
    has_correction = looks_like_correction(combined_user_text)

    return HeuristicSignals(
        is_qa_like=is_qa_like,
        has_completion_token=has_completion,
        turn_count=turn_count,
        has_refinements=has_refinements,
        is_meta_noise=is_meta_noise,
        has_identity=has_identity,
        has_preference=has_preference,
        has_correction=has_correction,
    )


def compute_thread_heuristic_score(signals: HeuristicSignals) -> float:
    """
    Compute memory-worthiness score from heuristic signals.

    Formula: base + qa_bonus + completion_bonus + turn_bonus + content_bonus - noise_penalty

    Args:
        signals: HeuristicSignals from compute_thread_heuristics()

    Returns:
        Float 0.0-1.0 representing memory-worthiness
    """
    score = 0.3  # Base score

    # Resolved Q&A is high value
    if signals.is_qa_like and signals.has_completion_token:
        score += 0.3

    # Multi-turn bonus (capped at +0.3)
    if signals.turn_count >= 3:
        score += 0.1 * min(signals.turn_count - 2, 3)

    # Iterative refinement is valuable
    if signals.has_refinements:
        score += 0.15

    # High-value content types
    if signals.has_identity:
        score += 0.25
    if signals.has_preference:
        score += 0.2
    if signals.has_correction:
        score += 0.3  # Corrections are very important to remember

    # Heavy penalty for noise
    if signals.is_meta_noise:
        score -= 0.5

    return max(0.0, min(1.0, score))


def looks_like_preference(text: str) -> bool:
    """Check if text contains explicit preference statements."""
    text_lower = text.lower()
    return any(re.search(p, text_lower, re.IGNORECASE) for p in PREFERENCE_PATTERNS)


def looks_like_identity(text: str) -> bool:
    """Check if text contains identity/role statements."""
    text_lower = text.lower()
    return any(re.search(p, text_lower, re.IGNORECASE) for p in IDENTITY_PATTERNS)


def looks_like_correction(text: str) -> bool:
    """Check if text contains correction/update statements."""
    text_lower = text.lower()
    return any(re.search(p, text_lower, re.IGNORECASE) for p in CORRECTION_PATTERNS)


def detect_memory_kind(text: str) -> Tuple[str, float]:
    """
    Detect the kind of memory and a priority boost for salience.

    Returns:
        Tuple of (memory_kind, salience_boost)
    """
    if looks_like_correction(text):
        return MEMORY_KIND_CORRECTION, 0.4  # Corrections are high priority
    if looks_like_identity(text):
        return MEMORY_KIND_IDENTITY, 0.3
    if looks_like_preference(text):
        return MEMORY_KIND_PREFERENCE, 0.3
    return MEMORY_KIND_FACT, 0.0


def format_structured_memory(
    text: str,
    kind: str = MEMORY_KIND_FACT,
    meta: Optional[dict] = None
) -> str:
    """
    Format a memory with structured metadata for better retrieval handling.

    Args:
        text: The memory text to store
        kind: Type of memory (identity, preference, decision, correction, fact)
        meta: Optional additional metadata

    Returns:
        JSON-encoded string with structure for later parsing
    """
    payload = {
        "kind": kind,
        "text": text,
        "timestamp": datetime.now().isoformat(),
    }
    if meta:
        payload["meta"] = meta
    return json.dumps(payload)


def parse_structured_memory(memory_str: str) -> Tuple[str, dict]:
    """
    Parse a potentially structured memory string.

    Returns:
        Tuple of (text, metadata) where metadata may include kind, timestamp, etc.
        If memory is not structured JSON, returns (original_string, {})
    """
    try:
        data = json.loads(memory_str)
        if isinstance(data, dict) and "text" in data:
            text = data.pop("text")
            return text, data
    except (json.JSONDecodeError, TypeError):
        pass
    return memory_str, {}


def _count_pattern_matches(text: str, patterns: List[str]) -> int:
    """Count how many patterns match in text."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower, re.IGNORECASE))


def _has_named_entities(text: str) -> bool:
    """Simple check for capitalized words that might be named entities."""
    # Look for sequences of capitalized words (excluding sentence starts)
    words = text.split()
    for i, word in enumerate(words[1:], 1):  # Skip first word
        if word and word[0].isupper() and not words[i-1].endswith(('.', '!', '?')):
            return True
    return False


def compute_quick_salience(user_msg: str, assistant_msg: str) -> Tuple[float, str]:
    """
    Compute salience score using cheap heuristics.

    Returns:
        Tuple of (salience_score, memory_kind) where:
        - salience_score: Float 0-1, higher means more likely worth remembering
        - memory_kind: Type of memory detected (identity, preference, correction, fact)
    """
    combined = f"{user_msg} {assistant_msg}"
    score = 0.0

    # First, check for high-priority memory types (preferences, identity, corrections)
    # These get a significant salience boost and are tagged appropriately
    memory_kind, kind_boost = detect_memory_kind(user_msg)
    score += kind_boost

    # Explicit markers in user message are strong signals
    explicit_count = _count_pattern_matches(user_msg, EXPLICIT_MARKERS)
    score += min(explicit_count * 0.25, 0.5)

    # Confirmation patterns suggest resolved understanding
    confirmation_count = _count_pattern_matches(assistant_msg, CONFIRMATION_PATTERNS)
    score += min(confirmation_count * 0.15, 0.3)

    # Factual patterns suggest memorable information
    factual_count = _count_pattern_matches(combined, FACTUAL_PATTERNS)
    score += min(factual_count * 0.1, 0.2)

    # Named entities suggest specific, memorable content
    if _has_named_entities(user_msg):
        score += 0.1

    # Personal pronouns in statements (not questions) suggest self-disclosure
    if re.search(r"\bi\s+(am|was|have|had|work|live|prefer|like|need)\b", user_msg.lower()):
        if not user_msg.strip().endswith('?'):
            score += 0.2

    # Very short exchanges are usually not worth remembering
    if len(user_msg) < 20 and len(assistant_msg) < 50:
        score *= 0.5

    # Questions without answers are not worth remembering
    if user_msg.strip().endswith('?') and len(assistant_msg) < 100:
        score *= 0.7

    return min(score, 1.0), memory_kind


SALIENCE_EVAL_PROMPT = """Evaluate if this exchange contains information worth saving to long-term memory for future conversations.

SAVE if it contains:
- Personal facts about the user (name, job, preferences, constraints, background)
- Decisions or conclusions reached after discussion
- Corrections to previous assumptions or misunderstandings
- Important context that would help in future conversations
- Specific instructions or preferences for how to interact

DO NOT SAVE:
- Generic greetings or small talk
- Questions without meaningful answers
- Technical troubleshooting that won't recur
- Temporary or time-sensitive information
- Information that's already common knowledge

If worth saving, extract a concise fact (1-2 sentences) that captures the essential information.

Respond with JSON:
{
    "should_remember": true/false,
    "memory_text": "concise fact to remember (or empty string if not remembering)",
    "salience": 0.0-1.0,
    "reason": "brief explanation"
}"""


class AutoMemory:
    """
    Automatic memory consolidation based on salience detection.

    Uses a hybrid approach:
    1. Cheap heuristics first (no API cost)
    2. LLM judge for ambiguous cases (0.3-0.7 salience)
    3. Auto-remember high-confidence signals (>0.8)
    """

    def __init__(
        self,
        chat_provider: Optional["BaseChatProvider"] = None,
        client: Optional["OpenAI"] = None,
        salience_threshold: float = 0.5,
        use_llm_judge: bool = True,
        model: str = AUTO_MEMORY_MODEL,
    ):
        """
        Initialize AutoMemory.

        Args:
            chat_provider: Chat provider for LLM evaluation (preferred)
            client: Legacy OpenAI client (for backward compat)
            salience_threshold: Minimum salience to remember (default 0.5)
            use_llm_judge: If True, use LLM for ambiguous cases
            model: Model to use for salience evaluation
        """
        self._chat_provider = chat_provider
        self._legacy_client = client
        self.salience_threshold = salience_threshold
        self.use_llm_judge = use_llm_judge
        self.model = model

    @property
    def client(self) -> "OpenAI":
        """Legacy client access for backward compatibility."""
        if self._legacy_client is not None:
            return self._legacy_client
        if self._chat_provider is not None and hasattr(self._chat_provider, 'client'):
            return self._chat_provider.client
        raise AttributeError("No client available. Provide chat_provider or client.")

    def evaluate(self, user_msg: str, assistant_msg: str) -> MemorySignal:
        """
        Evaluate whether an exchange should be saved to memory.

        Args:
            user_msg: The user's message
            assistant_msg: The assistant's response

        Returns:
            MemorySignal with evaluation results including memory kind
        """
        # Quick heuristic check - now returns (score, kind)
        quick_score, memory_kind = compute_quick_salience(user_msg, assistant_msg)

        # High confidence - remember without LLM
        if quick_score > 0.8:
            return MemorySignal(
                should_remember=True,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason=f"High-confidence heuristic match ({memory_kind})",
                kind=memory_kind
            )

        # Low confidence - skip
        if quick_score < 0.3 or not self.use_llm_judge:
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text="" if quick_score < self.salience_threshold else self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason="Low heuristic score" if quick_score < 0.3 else "Heuristic evaluation only",
                kind=memory_kind
            )

        # Ambiguous - use LLM judge
        return self._llm_evaluate(user_msg, assistant_msg, quick_score, memory_kind)

    def _extract_fact_heuristic(self, user_msg: str, assistant_msg: str) -> str:
        """Extract a fact using simple heuristics."""
        # If user is stating something about themselves, use that
        if re.search(r"^i\s+(am|'m|was|have|had|work|live|prefer|like|need|want)\b", user_msg.lower()):
            # Clean up and return user's statement
            return user_msg.strip()

        # Otherwise, combine user query with key part of response
        if len(assistant_msg) > 200:
            # Take first sentence of response
            first_sentence = assistant_msg.split('.')[0] + '.'
            return f"Q: {user_msg}\nA: {first_sentence}"

        return f"Q: {user_msg}\nA: {assistant_msg}"

    def _llm_evaluate(self, user_msg: str, assistant_msg: str, quick_score: float, memory_kind: str) -> MemorySignal:
        """Use LLM to evaluate salience."""
        try:
            exchange = f"USER: {user_msg}\n\nASSISTANT: {assistant_msg}"

            # Use provider if available, otherwise fall back to legacy client
            if self._chat_provider is not None:
                response = self._chat_provider.chat(
                    messages=[{"role": "user", "content": exchange}],
                    model=self.model,
                    instructions=SALIENCE_EVAL_PROMPT,
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )
                content = response.text or "{}"
            else:
                # Legacy path using OpenAI client directly
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SALIENCE_EVAL_PROMPT},
                        {"role": "user", "content": exchange}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=200,
                )
                content = response.choices[0].message.content or "{}"

            result = json.loads(content)

            return MemorySignal(
                should_remember=result.get("should_remember", False),
                memory_text=result.get("memory_text", ""),
                salience=result.get("salience", quick_score),
                reason=result.get("reason", "LLM evaluation"),
                kind=memory_kind  # Preserve heuristic-detected kind
            )
        except Exception as e:
            # Fallback to heuristic on error
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg) if quick_score >= self.salience_threshold else "",
                salience=quick_score,
                reason=f"LLM error, using heuristic: {e}",
                kind=memory_kind
            )


class AsyncAutoMemory:
    """Async version of AutoMemory for use with FastAPI."""

    def __init__(
        self,
        client: "AsyncOpenAI",
        salience_threshold: float = 0.5,
        use_llm_judge: bool = True,
        model: str = AUTO_MEMORY_MODEL,
    ):
        self.client = client
        self.salience_threshold = salience_threshold
        self.use_llm_judge = use_llm_judge
        self.model = model

    async def evaluate(self, user_msg: str, assistant_msg: str) -> MemorySignal:
        """Evaluate whether an exchange should be saved to memory."""
        quick_score, memory_kind = compute_quick_salience(user_msg, assistant_msg)

        if quick_score > 0.8:
            return MemorySignal(
                should_remember=True,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason=f"High-confidence heuristic match ({memory_kind})",
                kind=memory_kind
            )

        if quick_score < 0.3 or not self.use_llm_judge:
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text="" if quick_score < self.salience_threshold else self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason="Low heuristic score" if quick_score < 0.3 else "Heuristic evaluation only",
                kind=memory_kind
            )

        return await self._llm_evaluate(user_msg, assistant_msg, quick_score, memory_kind)

    def _extract_fact_heuristic(self, user_msg: str, assistant_msg: str) -> str:
        """Extract a fact using simple heuristics."""
        if re.search(r"^i\s+(am|'m|was|have|had|work|live|prefer|like|need|want)\b", user_msg.lower()):
            return user_msg.strip()

        if len(assistant_msg) > 200:
            first_sentence = assistant_msg.split('.')[0] + '.'
            return f"Q: {user_msg}\nA: {first_sentence}"

        return f"Q: {user_msg}\nA: {assistant_msg}"

    async def _llm_evaluate(self, user_msg: str, assistant_msg: str, quick_score: float, memory_kind: str) -> MemorySignal:
        """Use LLM to evaluate salience."""
        try:
            exchange = f"USER: {user_msg}\n\nASSISTANT: {assistant_msg}"

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SALIENCE_EVAL_PROMPT},
                    {"role": "user", "content": exchange}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200,
            )

            content = response.choices[0].message.content or "{}"
            result = json.loads(content)

            return MemorySignal(
                should_remember=result.get("should_remember", False),
                memory_text=result.get("memory_text", ""),
                salience=result.get("salience", quick_score),
                reason=result.get("reason", "LLM evaluation"),
                kind=memory_kind
            )
        except Exception as e:
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg) if quick_score >= self.salience_threshold else "",
                salience=quick_score,
                reason=f"LLM error, using heuristic: {e}",
                kind=memory_kind
            )
