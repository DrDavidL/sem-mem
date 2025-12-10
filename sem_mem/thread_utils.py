"""
Pure utility functions for thread/conversation management.

These functions are intentionally stateless and UI-agnostic, so they can be
reused across different surfaces (Streamlit, FastAPI, CLI, etc.).

The actual "thread" data model lives in the UI layer (e.g., app.py's session_state).
This module just provides helpers for:
- Title generation
- Token estimation
- Conversation summarization (for windowing and deletion)
- Thread completion detection (heuristics)
"""

import json
from typing import List, Dict, Optional, TYPE_CHECKING, Union

from .auto_memory import (
    HeuristicSignals,
    compute_thread_heuristics,
    compute_thread_heuristic_score,
)

from .config import (
    AUTO_THREAD_RENAME_MODEL,
    AUTO_THREAD_RENAME_MAX_WORDS,
    CONVERSATION_SUMMARY_MODEL,
    CONVERSATION_SUMMARY_MAX_CHARS,
    CONVERSATION_SUMMARY_MIN_MESSAGES,
    CONVERSATION_SUMMARY_LEAVE_RECENT,
    FAREWELL_SUMMARY_MODEL,
    FAREWELL_SUMMARY_MAX_CHARS,
    get_api_key,
)

if TYPE_CHECKING:
    from openai import OpenAI
    from .providers import BaseChatProvider


def generate_thread_title(
    messages: List[Dict],
    chat_provider: Optional["BaseChatProvider"] = None,
    client: Optional["OpenAI"] = None,
    model: str = AUTO_THREAD_RENAME_MODEL,
    max_words: int = AUTO_THREAD_RENAME_MAX_WORDS,
) -> str:
    """
    Generate a short, descriptive title for a conversation thread.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        chat_provider: Chat provider to use (preferred)
        client: Legacy OpenAI client (for backward compat)
        model: Model to use for title generation
        max_words: Maximum words in the generated title

    Returns:
        A short title string (empty string on failure)

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "How do I configure nginx?"},
        ...     {"role": "assistant", "content": "Here's how to configure nginx..."},
        ...     {"role": "user", "content": "What about SSL certificates?"},
        ... ]
        >>> title = generate_thread_title(messages)  # doctest: +SKIP
        >>> len(title.split()) <= 8  # doctest: +SKIP
        True
    """
    if not messages:
        return ""

    # Build conversation summary for the model
    conversation_text = _format_messages_for_title(messages)

    system_prompt = f"""You are naming this conversation so the user can find it later.

Rules:
- Use at most {max_words} words
- Be specific and descriptive
- Avoid generic titles like "Conversation", "Chat", "Discussion", or "General chat"
- Focus on the main topic or goal
- Use title case

Respond with ONLY the title, nothing else."""

    try:
        # Use provider if available
        if chat_provider is not None:
            response = chat_provider.chat(
                messages=[{"role": "user", "content": f"Generate a title for this conversation:\n\n{conversation_text}"}],
                model=model,
                instructions=system_prompt,
                temperature=0.3,
                max_tokens=50,
            )
            title = response.text or ""
        else:
            # Legacy path: create client if needed
            if client is None:
                from openai import OpenAI
                api_key = get_api_key()
                if not api_key:
                    return ""
                client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate a title for this conversation:\n\n{conversation_text}"}
                ],
                temperature=0.3,
                max_tokens=50,
            )
            title = response.choices[0].message.content or ""

        # Clean up: remove quotes, extra whitespace
        title = title.strip().strip('"\'')
        return title
    except Exception:
        return ""


def _format_messages_for_title(messages: List[Dict], max_chars: int = 2000) -> str:
    """
    Format messages into a condensed string for title generation.

    Truncates to max_chars to avoid sending too much context.
    """
    parts = []
    total_chars = 0

    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        # Strip system logs from assistant messages
        if role == "ASSISTANT":
            if "**System Logs:**" in content:
                content = content.split("**System Logs:**")[0].strip()
            if "**Retrieved Context:**" in content:
                content = content.split("**Retrieved Context:**")[0].strip()

        # Truncate individual messages if needed
        if len(content) > 500:
            content = content[:500] + "..."

        line = f"{role}: {content}"
        if total_chars + len(line) > max_chars:
            break
        parts.append(line)
        total_chars += len(line) + 1

    return "\n".join(parts)


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_message_tokens(messages: List[Dict]) -> int:
    """
    Rough token estimate for a list of {role, content} messages.

    Uses a simple heuristic: ~4 characters per token on average.
    This is approximate but sufficient for deciding when to summarize.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Estimated token count (integer)

    Example:
        >>> msgs = [{"role": "user", "content": "Hello world"}]
        >>> estimate_message_tokens(msgs)
        3
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        # Include role in estimate (adds ~2-4 tokens per message for role + formatting)
        total_chars += len(content) + len(role) + 4  # 4 for formatting overhead
    return total_chars // 4


# =============================================================================
# Window Selection
# =============================================================================

def select_summary_window(
    messages: List[Dict],
    existing_windows: List[Dict],
    leave_recent: int = CONVERSATION_SUMMARY_LEAVE_RECENT,
    min_messages: int = CONVERSATION_SUMMARY_MIN_MESSAGES,
) -> Optional[tuple]:
    """
    Choose the next [start_index, end_index) span to summarize.

    This yields coarse, forward-moving windows ("chapters") rather than
    overlapping micro-windows. Each window covers a contiguous segment
    of messages that hasn't been summarized yet.

    Args:
        messages: Full list of messages in the thread
        existing_windows: List of existing summary window dicts with 'end_index'
        leave_recent: Number of recent messages to leave unsummarized
        min_messages: Minimum messages required before any summarization

    Returns:
        Tuple (start_index, end_index) or None if no window should be created

    Behavior:
        - If len(messages) < min_messages: return None
        - If no existing windows:
            start = 0
            end = max(0, len(messages) - leave_recent)
        - If existing windows:
            start = max(w["end_index"] for w in existing_windows)
            end = max(start, len(messages) - leave_recent)
        - If end <= start: return None

    Example:
        >>> msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        >>> select_summary_window(msgs, [], leave_recent=6, min_messages=10)
        (0, 14)
        >>> select_summary_window(msgs, [{"end_index": 14}], leave_recent=6, min_messages=10)
        None
    """
    if len(messages) < min_messages:
        return None

    if not existing_windows:
        start = 0
        end = max(0, len(messages) - leave_recent)
    else:
        # Find where the last summary ended
        last_end = max(w.get("end_index", 0) for w in existing_windows)
        start = last_end
        end = max(start, len(messages) - leave_recent)

    # No new content to summarize
    if end <= start:
        return None

    return (start, end)


# =============================================================================
# Conversation Summarization
# =============================================================================

CONVERSATION_SUMMARY_PROMPT = """You are summarizing a segment of a longer, ongoing conversation between a user and an AI assistant.

Focus on durable information the assistant should remember:
- Key decisions or conclusions reached
- User preferences and constraints
- Corrections or clarifications made
- Important background context and plans
- Action items or follow-ups mentioned

Exclude:
- Small talk and generic pleasantries
- Raw system logs and debug output
- Token-by-token narration of the conversation
- Information that's purely transient or time-sensitive

Write a concise summary using this structure (skip sections if not applicable):

**Key Points:**
- ...

**Decisions:**
- ...

**User Preferences:**
- ...

**Open Questions / TODOs:**
- ...

Be brief but complete. Focus on what would be useful context for future conversations."""


def summarize_conversation_window(
    messages: List[Dict],
    chat_provider: Optional["BaseChatProvider"] = None,
    client: Optional["OpenAI"] = None,
    model: str = CONVERSATION_SUMMARY_MODEL,
    max_chars: int = CONVERSATION_SUMMARY_MAX_CHARS,
) -> str:
    """
    Summarize a slice of messages into a durable semantic summary.

    Args:
        messages: List of message dicts to summarize (a window/slice)
        chat_provider: Chat provider to use (preferred)
        client: Legacy OpenAI client (for backward compat)
        model: Model to use for summarization
        max_chars: Maximum characters of conversation to include in prompt

    Returns:
        Summary text string (empty string on failure)

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "I prefer morning meetings"},
        ...     {"role": "assistant", "content": "Noted! I'll remember that."},
        ... ]
        >>> summary = summarize_conversation_window(messages)  # doctest: +SKIP
        >>> "morning" in summary.lower()  # doctest: +SKIP
        True
    """
    if not messages:
        return ""

    # Format messages for summarization (with cleanup)
    conversation_text = _format_messages_for_summary(messages, max_chars)

    if not conversation_text.strip():
        return ""

    try:
        # Use provider if available
        if chat_provider is not None:
            response = chat_provider.chat(
                messages=[{"role": "user", "content": f"Summarize this conversation segment:\n\n{conversation_text}"}],
                model=model,
                instructions=CONVERSATION_SUMMARY_PROMPT,
                temperature=0.3,
                max_tokens=1000,
            )
            summary = response.text or ""
        else:
            # Legacy path: create client if needed
            if client is None:
                from openai import OpenAI
                api_key = get_api_key()
                if not api_key:
                    return ""
                client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CONVERSATION_SUMMARY_PROMPT},
                    {"role": "user", "content": f"Summarize this conversation segment:\n\n{conversation_text}"}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            summary = response.choices[0].message.content or ""

        return summary.strip()
    except Exception:
        return ""


def _format_messages_for_summary(messages: List[Dict], max_chars: int = 8000) -> str:
    """
    Format messages into a string for summarization.

    Similar to _format_messages_for_title but with higher limits and
    focused on preserving content for meaningful summarization.

    Args:
        messages: List of message dicts
        max_chars: Maximum total characters to include

    Returns:
        Formatted conversation string
    """
    parts = []
    total_chars = 0

    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")

        # Strip system logs and debug noise from assistant messages
        content = _strip_system_noise(content)

        # Skip empty messages after stripping
        if not content.strip():
            continue

        line = f"{role}: {content}"

        # Check if adding this would exceed limit
        if total_chars + len(line) > max_chars:
            # Add truncated version if we have room
            remaining = max_chars - total_chars - 50  # Reserve for "..."
            if remaining > 100:
                parts.append(f"{role}: {content[:remaining]}...")
            break

        parts.append(line)
        total_chars += len(line) + 2  # +2 for newlines

    return "\n\n".join(parts)


def _strip_system_noise(content: str) -> str:
    """
    Remove system logs, debug output, and other noise from message content.

    Args:
        content: Raw message content

    Returns:
        Cleaned content string
    """
    # Strip everything after common noise markers
    noise_markers = [
        "**System Logs:**",
        "**Retrieved Context:**",
        "**Debug:**",
        "---\n`",  # Common log format
    ]

    for marker in noise_markers:
        if marker in content:
            content = content.split(marker)[0]

    return content.strip()


# =============================================================================
# Farewell Summary (Thread Deletion)
# =============================================================================

FAREWELL_SUMMARY_PROMPT = """You are summarizing an entire conversation that is about to be deleted from the UI.

Produce a short summary focusing on durable information that will still matter later:
- Key decisions or conclusions reached
- User preferences and constraints
- Important background context
- Persistent instructions or standing plans
- Any open questions or follow-ups that should be remembered

Exclude:
- Small talk and generic pleasantries
- Raw system logs and debug output
- Transient information that won't matter later

Write a concise summary using this structure (skip sections if not applicable):

**Key Points:**
- ...

**Decisions:**
- ...

**User Preferences:**
- ...

**Open Questions / TODOs:**
- ...

Be brief but complete. This is the final record of this conversation."""


def summarize_deleted_thread(
    messages: List[Dict],
    chat_provider: Optional["BaseChatProvider"] = None,
    client: Optional["OpenAI"] = None,
    model: str = FAREWELL_SUMMARY_MODEL,
    max_chars: int = FAREWELL_SUMMARY_MAX_CHARS,
) -> str:
    """
    Produce a 'farewell' summary of an entire thread for long-term memory.

    This is called when a thread is about to be deleted, to preserve
    durable takeaways in L2 memory before the thread is removed from the UI.

    Args:
        messages: Complete list of message dicts from the thread
        chat_provider: Chat provider to use (preferred)
        client: Legacy OpenAI client (for backward compat)
        model: Model to use for summarization
        max_chars: Maximum characters of conversation to include in prompt

    Returns:
        Summary text string (empty string on failure or if thread is trivial)

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "I prefer dark mode for all apps"},
        ...     {"role": "assistant", "content": "Noted! I'll remember that preference."},
        ... ]
        >>> summary = summarize_deleted_thread(messages)  # doctest: +SKIP
        >>> "dark mode" in summary.lower()  # doctest: +SKIP
        True
    """
    # Don't bother summarizing empty or trivial threads
    if not messages or len(messages) < 2:
        return ""

    # Format messages for summarization (reuse existing helper)
    conversation_text = _format_messages_for_summary(messages, max_chars)

    if not conversation_text.strip():
        return ""

    try:
        # Use provider if available
        if chat_provider is not None:
            response = chat_provider.chat(
                messages=[{"role": "user", "content": f"Summarize this conversation before deletion:\n\n{conversation_text}"}],
                model=model,
                instructions=FAREWELL_SUMMARY_PROMPT,
                temperature=0.3,
                max_tokens=1000,
            )
            summary = response.text or ""
        else:
            # Legacy path: create client if needed
            if client is None:
                from openai import OpenAI
                api_key = get_api_key()
                if not api_key:
                    return ""
                client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": FAREWELL_SUMMARY_PROMPT},
                    {"role": "user", "content": f"Summarize this conversation before deletion:\n\n{conversation_text}"}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            summary = response.choices[0].message.content or ""

        return summary.strip()
    except Exception:
        return ""


# =============================================================================
# Thread Completion Detection (Phase 1 Enhancement)
# =============================================================================

def is_thread_completed(messages: List[Dict]) -> bool:
    """
    Detect if a thread appears to be a completed Q&A interaction.

    A thread is considered "completed" if:
    - It started with a question-like message
    - The final user message contains a completion token (thanks, got it, etc.)

    This is useful for deciding when a thread is worth saving to long-term memory.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        True if the thread appears to be a completed Q&A interaction

    Example:
        >>> msgs = [
        ...     {"role": "user", "content": "How do I configure SSL?"},
        ...     {"role": "assistant", "content": "Here's how..."},
        ...     {"role": "user", "content": "Thanks, that worked!"},
        ... ]
        >>> is_thread_completed(msgs)
        True
    """
    signals = compute_thread_heuristics(messages)
    return signals.is_qa_like and signals.has_completion_token


def get_thread_memory_score(messages: List[Dict]) -> float:
    """
    Compute a memory-worthiness score for a thread.

    Higher scores indicate the thread contains valuable information
    that should be persisted to long-term memory.

    Score factors:
    - Completed Q&A interactions (+0.3)
    - Multi-turn conversations (+0.1 per turn, capped)
    - Refinement/clarification patterns (+0.15)
    - Identity statements (+0.25)
    - Preference statements (+0.2)
    - Correction statements (+0.3)
    - Meta-noise penalty (-0.5)

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Float 0.0-1.0 representing memory-worthiness

    Example:
        >>> msgs = [{"role": "user", "content": "I'm a physician in Boston"}]
        >>> score = get_thread_memory_score(msgs)
        >>> score > 0.5  # Identity statement should score well
        True
    """
    signals = compute_thread_heuristics(messages)
    return compute_thread_heuristic_score(signals)


def analyze_thread(messages: List[Dict]) -> Dict:
    """
    Full analysis of a thread for memory-worthiness.

    Returns detailed signals plus computed score, useful for debugging
    or displaying to users why a thread is/isn't worth saving.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Dict with:
            - signals: HeuristicSignals as dict
            - score: Float 0.0-1.0
            - is_completed: Bool
            - recommendation: String ("save", "maybe", "skip")

    Example:
        >>> msgs = [
        ...     {"role": "user", "content": "I prefer dark mode"},
        ...     {"role": "assistant", "content": "Noted!"},
        ... ]
        >>> analysis = analyze_thread(msgs)
        >>> analysis["signals"]["has_preference"]
        True
    """
    signals = compute_thread_heuristics(messages)
    score = compute_thread_heuristic_score(signals)
    is_completed = signals.is_qa_like and signals.has_completion_token

    # Generate recommendation
    if score >= 0.6:
        recommendation = "save"
    elif score >= 0.4:
        recommendation = "maybe"
    else:
        recommendation = "skip"

    return {
        "signals": signals._asdict(),
        "score": score,
        "is_completed": is_completed,
        "recommendation": recommendation,
    }
