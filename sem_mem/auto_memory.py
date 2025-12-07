"""
Automatic memory consolidation based on salience detection.

Evaluates exchanges and decides what's worth remembering for future conversations.
Uses a hybrid approach: cheap heuristics first, LLM judge for ambiguous cases.
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple
from openai import OpenAI, AsyncOpenAI

from .config import QUERY_EXPANSION_MODEL


# Model for salience evaluation (same cheap model as query expansion)
AUTO_MEMORY_MODEL = QUERY_EXPANSION_MODEL


@dataclass
class MemorySignal:
    """Result of salience evaluation."""
    should_remember: bool
    memory_text: str  # Distilled/summarized fact to store
    salience: float  # 0-1 importance score
    reason: str  # Why it's worth remembering (for debugging)


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


def compute_quick_salience(user_msg: str, assistant_msg: str) -> float:
    """
    Compute salience score using cheap heuristics.

    Returns:
        Float 0-1 where higher means more likely worth remembering
    """
    combined = f"{user_msg} {assistant_msg}"
    score = 0.0

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

    return min(score, 1.0)


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
        client: OpenAI,
        salience_threshold: float = 0.5,
        use_llm_judge: bool = True,
        model: str = AUTO_MEMORY_MODEL,
    ):
        """
        Initialize AutoMemory.

        Args:
            client: OpenAI client for LLM evaluation
            salience_threshold: Minimum salience to remember (default 0.5)
            use_llm_judge: If True, use LLM for ambiguous cases
            model: Model to use for salience evaluation
        """
        self.client = client
        self.salience_threshold = salience_threshold
        self.use_llm_judge = use_llm_judge
        self.model = model

    def evaluate(self, user_msg: str, assistant_msg: str) -> MemorySignal:
        """
        Evaluate whether an exchange should be saved to memory.

        Args:
            user_msg: The user's message
            assistant_msg: The assistant's response

        Returns:
            MemorySignal with evaluation results
        """
        # Quick heuristic check
        quick_score = compute_quick_salience(user_msg, assistant_msg)

        # High confidence - remember without LLM
        if quick_score > 0.8:
            return MemorySignal(
                should_remember=True,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason="High-confidence heuristic match"
            )

        # Low confidence - skip
        if quick_score < 0.3 or not self.use_llm_judge:
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text="" if quick_score < self.salience_threshold else self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason="Low heuristic score" if quick_score < 0.3 else "Heuristic evaluation only"
            )

        # Ambiguous - use LLM judge
        return self._llm_evaluate(user_msg, assistant_msg, quick_score)

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

    def _llm_evaluate(self, user_msg: str, assistant_msg: str, quick_score: float) -> MemorySignal:
        """Use LLM to evaluate salience."""
        try:
            exchange = f"USER: {user_msg}\n\nASSISTANT: {assistant_msg}"

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
                reason=result.get("reason", "LLM evaluation")
            )
        except Exception as e:
            # Fallback to heuristic on error
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg) if quick_score >= self.salience_threshold else "",
                salience=quick_score,
                reason=f"LLM error, using heuristic: {e}"
            )


class AsyncAutoMemory:
    """Async version of AutoMemory for use with FastAPI."""

    def __init__(
        self,
        client: AsyncOpenAI,
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
        quick_score = compute_quick_salience(user_msg, assistant_msg)

        if quick_score > 0.8:
            return MemorySignal(
                should_remember=True,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason="High-confidence heuristic match"
            )

        if quick_score < 0.3 or not self.use_llm_judge:
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text="" if quick_score < self.salience_threshold else self._extract_fact_heuristic(user_msg, assistant_msg),
                salience=quick_score,
                reason="Low heuristic score" if quick_score < 0.3 else "Heuristic evaluation only"
            )

        return await self._llm_evaluate(user_msg, assistant_msg, quick_score)

    def _extract_fact_heuristic(self, user_msg: str, assistant_msg: str) -> str:
        """Extract a fact using simple heuristics."""
        if re.search(r"^i\s+(am|'m|was|have|had|work|live|prefer|like|need|want)\b", user_msg.lower()):
            return user_msg.strip()

        if len(assistant_msg) > 200:
            first_sentence = assistant_msg.split('.')[0] + '.'
            return f"Q: {user_msg}\nA: {first_sentence}"

        return f"Q: {user_msg}\nA: {assistant_msg}"

    async def _llm_evaluate(self, user_msg: str, assistant_msg: str, quick_score: float) -> MemorySignal:
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
                reason=result.get("reason", "LLM evaluation")
            )
        except Exception as e:
            return MemorySignal(
                should_remember=quick_score >= self.salience_threshold,
                memory_text=self._extract_fact_heuristic(user_msg, assistant_msg) if quick_score >= self.salience_threshold else "",
                salience=quick_score,
                reason=f"LLM error, using heuristic: {e}"
            )
