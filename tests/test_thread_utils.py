"""
Tests for sem_mem.thread_utils module.

Run with: pytest tests/test_thread_utils.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestGenerateThreadTitle:
    """Tests for generate_thread_title function."""

    def test_empty_messages_returns_empty_string(self):
        """Empty message list should return empty string without calling API."""
        from sem_mem.thread_utils import generate_thread_title

        result = generate_thread_title([])
        assert result == ""

    def test_respects_max_words_in_prompt(self):
        """Verify that max_words is included in the prompt to the model."""
        from sem_mem.thread_utils import generate_thread_title

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test Title"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "How do I configure nginx?"},
            {"role": "assistant", "content": "Here's how to configure nginx..."},
            {"role": "user", "content": "What about SSL?"},
        ]

        result = generate_thread_title(messages, client=mock_client, max_words=5)

        # Verify API was called
        mock_client.chat.completions.create.assert_called_once()

        # Check that max_words appears in the system prompt
        call_args = mock_client.chat.completions.create.call_args
        system_message = call_args.kwargs["messages"][0]["content"]
        assert "5 words" in system_message

        assert result == "Test Title"

    def test_uses_correct_model(self):
        """Verify the specified model is used."""
        from sem_mem.thread_utils import generate_thread_title

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "My Title"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]

        generate_thread_title(messages, client=mock_client, model="gpt-4.1-mini")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4.1-mini"

    def test_strips_quotes_from_title(self):
        """Model sometimes wraps title in quotes - we should strip them."""
        from sem_mem.thread_utils import generate_thread_title

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '"Nginx SSL Configuration"'
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "SSL stuff"}]

        result = generate_thread_title(messages, client=mock_client)
        assert result == "Nginx SSL Configuration"

    def test_handles_api_error_gracefully(self):
        """API errors should return empty string, not raise."""
        from sem_mem.thread_utils import generate_thread_title

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        messages = [{"role": "user", "content": "Hello"}]

        result = generate_thread_title(messages, client=mock_client)
        assert result == ""

    def test_prompt_includes_conversation_content(self):
        """Verify conversation content is passed to the model."""
        from sem_mem.thread_utils import generate_thread_title

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Title"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "Help me with Python decorators"},
            {"role": "assistant", "content": "Decorators are functions that wrap..."},
        ]

        generate_thread_title(messages, client=mock_client)

        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args.kwargs["messages"][1]["content"]

        # Verify conversation content is included
        assert "Python decorators" in user_message
        assert "USER:" in user_message
        assert "ASSISTANT:" in user_message

    def test_strips_system_logs_from_assistant_messages(self):
        """System logs in assistant messages should not be sent to title generator."""
        from sem_mem.thread_utils import _format_messages_for_title

        messages = [
            {"role": "user", "content": "What is X?"},
            {
                "role": "assistant",
                "content": "X is a thing.\n\n**System Logs:**\n`L1 HIT`\n\n**Retrieved Context:**\n> some context"
            },
        ]

        result = _format_messages_for_title(messages)

        assert "X is a thing" in result
        assert "System Logs" not in result
        assert "Retrieved Context" not in result
        assert "L1 HIT" not in result


class TestFormatMessagesForTitle:
    """Tests for _format_messages_for_title helper."""

    def test_respects_max_chars(self):
        """Should truncate to max_chars."""
        from sem_mem.thread_utils import _format_messages_for_title

        messages = [
            {"role": "user", "content": "A" * 1000},
            {"role": "assistant", "content": "B" * 1000},
        ]

        result = _format_messages_for_title(messages, max_chars=500)
        assert len(result) <= 600  # Some buffer for truncation markers

    def test_truncates_long_individual_messages(self):
        """Individual messages over 500 chars should be truncated."""
        from sem_mem.thread_utils import _format_messages_for_title

        messages = [
            {"role": "user", "content": "X" * 600},
        ]

        result = _format_messages_for_title(messages)
        assert "..." in result
        assert len(result) < 600


# =============================================================================
# Tests for Token Estimation
# =============================================================================

class TestEstimateMessageTokens:
    """Tests for estimate_message_tokens function."""

    def test_empty_messages_returns_zero(self):
        """Empty message list should return 0 tokens."""
        from sem_mem.thread_utils import estimate_message_tokens

        result = estimate_message_tokens([])
        assert result == 0

    def test_simple_message_estimate(self):
        """Simple message should have reasonable token estimate."""
        from sem_mem.thread_utils import estimate_message_tokens

        messages = [{"role": "user", "content": "Hello world"}]
        result = estimate_message_tokens(messages)

        # "Hello world" is 11 chars + "user" (4) + 4 overhead = 19 chars
        # 19 / 4 = ~4 tokens
        assert 2 <= result <= 10  # Reasonable range

    def test_longer_conversation_estimate(self):
        """Longer conversation should have proportionally more tokens."""
        from sem_mem.thread_utils import estimate_message_tokens

        short_messages = [{"role": "user", "content": "Hi"}]
        long_messages = [
            {"role": "user", "content": "Hello, I have a question about Python programming"},
            {"role": "assistant", "content": "Of course! I'd be happy to help with Python."},
            {"role": "user", "content": "How do I use list comprehensions effectively?"},
        ]

        short_tokens = estimate_message_tokens(short_messages)
        long_tokens = estimate_message_tokens(long_messages)

        assert long_tokens > short_tokens * 3  # Should be significantly more

    def test_includes_role_in_estimate(self):
        """Token estimate should include role overhead."""
        from sem_mem.thread_utils import estimate_message_tokens

        # Same content, different roles
        msg1 = [{"role": "user", "content": "test"}]
        msg2 = [{"role": "assistant", "content": "test"}]

        # "assistant" is longer than "user", so should have slightly more tokens
        tokens1 = estimate_message_tokens(msg1)
        tokens2 = estimate_message_tokens(msg2)

        # Both should be small but msg2 slightly larger due to role
        assert tokens1 >= 1
        assert tokens2 >= tokens1


# =============================================================================
# Tests for Window Selection
# =============================================================================

class TestSelectSummaryWindow:
    """Tests for select_summary_window function."""

    def test_no_messages_returns_none(self):
        """No messages should return None."""
        from sem_mem.thread_utils import select_summary_window

        result = select_summary_window([], [], leave_recent=6, min_messages=10)
        assert result is None

    def test_few_messages_returns_none(self):
        """Fewer than min_messages should return None."""
        from sem_mem.thread_utils import select_summary_window

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        result = select_summary_window(messages, [], leave_recent=6, min_messages=10)
        assert result is None

    def test_first_window_selection(self):
        """First window should span from 0 to (len - leave_recent)."""
        from sem_mem.thread_utils import select_summary_window

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result = select_summary_window(messages, [], leave_recent=6, min_messages=10)

        assert result == (0, 14)  # 20 - 6 = 14

    def test_second_window_starts_at_last_end(self):
        """Subsequent windows should start where the last one ended."""
        from sem_mem.thread_utils import select_summary_window

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        existing_windows = [{"end_index": 14}]

        result = select_summary_window(messages, existing_windows, leave_recent=6, min_messages=10)

        # Should start at 14 (where last ended) and go to 30 - 6 = 24
        assert result == (14, 24)

    def test_no_new_content_returns_none(self):
        """If existing windows cover all but recent messages, return None."""
        from sem_mem.thread_utils import select_summary_window

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        existing_windows = [{"end_index": 14}]  # Only 6 recent left

        result = select_summary_window(messages, existing_windows, leave_recent=6, min_messages=10)
        assert result is None

    def test_multiple_existing_windows(self):
        """Should find max end_index from multiple windows."""
        from sem_mem.thread_utils import select_summary_window

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
        existing_windows = [
            {"end_index": 10},
            {"end_index": 25},  # This is the latest
            {"end_index": 20},
        ]

        result = select_summary_window(messages, existing_windows, leave_recent=6, min_messages=10)

        # Should start at 25 (max end_index) and go to 50 - 6 = 44
        assert result == (25, 44)

    def test_respects_leave_recent_parameter(self):
        """Different leave_recent values should affect window end."""
        from sem_mem.thread_utils import select_summary_window

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]

        result_6 = select_summary_window(messages, [], leave_recent=6, min_messages=10)
        result_10 = select_summary_window(messages, [], leave_recent=10, min_messages=10)

        assert result_6 == (0, 14)  # 20 - 6 = 14
        assert result_10 == (0, 10)  # 20 - 10 = 10

    def test_respects_min_messages_parameter(self):
        """Different min_messages values should gate window creation."""
        from sem_mem.thread_utils import select_summary_window

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(15)]

        result_10 = select_summary_window(messages, [], leave_recent=6, min_messages=10)
        result_20 = select_summary_window(messages, [], leave_recent=6, min_messages=20)

        assert result_10 == (0, 9)  # Passes min_messages=10
        assert result_20 is None  # Fails min_messages=20


# =============================================================================
# Tests for Conversation Summarization
# =============================================================================

class TestSummarizeConversationWindow:
    """Tests for summarize_conversation_window function."""

    def test_empty_messages_returns_empty_string(self):
        """Empty message list should return empty string."""
        from sem_mem.thread_utils import summarize_conversation_window

        result = summarize_conversation_window([])
        assert result == ""

    def test_calls_model_with_correct_structure(self):
        """Verify the model is called with expected prompt structure."""
        from sem_mem.thread_utils import summarize_conversation_window

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "**Key Points:**\n- User prefers morning meetings"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "I prefer morning meetings"},
            {"role": "assistant", "content": "Noted! I'll remember that preference."},
        ]

        result = summarize_conversation_window(messages, client=mock_client)

        # Verify API was called
        mock_client.chat.completions.create.assert_called_once()

        # Check call structure
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4.1-mini"
        assert len(call_args.kwargs["messages"]) == 2  # system + user

        # System message should contain summarization instructions
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "durable" in system_msg.lower() or "remember" in system_msg.lower()

        # User message should contain conversation
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "morning meetings" in user_msg

        assert result == "**Key Points:**\n- User prefers morning meetings"

    def test_uses_specified_model(self):
        """Verify custom model is passed through."""
        from sem_mem.thread_utils import summarize_conversation_window

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        summarize_conversation_window(messages, client=mock_client, model="custom-model")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "custom-model"

    def test_handles_api_error_gracefully(self):
        """API errors should return empty string, not raise."""
        from sem_mem.thread_utils import summarize_conversation_window

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        messages = [{"role": "user", "content": "Hello"}]

        result = summarize_conversation_window(messages, client=mock_client)
        assert result == ""

    def test_strips_system_noise_from_messages(self):
        """System logs should be stripped before summarization."""
        from sem_mem.thread_utils import _format_messages_for_summary

        messages = [
            {"role": "user", "content": "What is X?"},
            {
                "role": "assistant",
                "content": "X is important.\n\n**System Logs:**\n`L1 HIT`\n\n**Retrieved Context:**\n> old stuff"
            },
        ]

        result = _format_messages_for_summary(messages)

        assert "X is important" in result
        assert "System Logs" not in result
        assert "L1 HIT" not in result
        assert "Retrieved Context" not in result

    def test_respects_max_chars_limit(self):
        """Should truncate content to max_chars."""
        from sem_mem.thread_utils import _format_messages_for_summary

        messages = [
            {"role": "user", "content": "A" * 5000},
            {"role": "assistant", "content": "B" * 5000},
        ]

        result = _format_messages_for_summary(messages, max_chars=1000)

        # Should be truncated to around max_chars
        assert len(result) <= 1200  # Some buffer for truncation markers


class TestStripSystemNoise:
    """Tests for _strip_system_noise helper."""

    def test_strips_system_logs(self):
        """Should remove everything after **System Logs:**"""
        from sem_mem.thread_utils import _strip_system_noise

        content = "Important response.\n\n**System Logs:**\n`debug info`"
        result = _strip_system_noise(content)
        assert result == "Important response."

    def test_strips_retrieved_context(self):
        """Should remove everything after **Retrieved Context:**"""
        from sem_mem.thread_utils import _strip_system_noise

        content = "Main content here.\n\n**Retrieved Context:**\n> some quote"
        result = _strip_system_noise(content)
        assert result == "Main content here."

    def test_preserves_clean_content(self):
        """Clean content without noise markers should be unchanged."""
        from sem_mem.thread_utils import _strip_system_noise

        content = "This is a normal response with no noise."
        result = _strip_system_noise(content)
        assert result == content

    def test_handles_multiple_noise_markers(self):
        """Should handle content with multiple noise types."""
        from sem_mem.thread_utils import _strip_system_noise

        content = "Response.\n\n**System Logs:**\nlogs\n\n**Retrieved Context:**\ncontext"
        result = _strip_system_noise(content)
        # Should stop at first marker
        assert result == "Response."


# =============================================================================
# Tests for Farewell Summary (Thread Deletion)
# =============================================================================

class TestSummarizeDeletedThread:
    """Tests for summarize_deleted_thread function."""

    def test_empty_messages_returns_empty_string(self):
        """Empty message list should return empty string."""
        from sem_mem.thread_utils import summarize_deleted_thread

        result = summarize_deleted_thread([])
        assert result == ""

    def test_single_message_returns_empty_string(self):
        """Single message (trivial thread) should return empty string."""
        from sem_mem.thread_utils import summarize_deleted_thread

        messages = [{"role": "user", "content": "Hello"}]
        result = summarize_deleted_thread(messages)
        assert result == ""

    def test_calls_model_with_correct_structure(self):
        """Verify the model is called with expected prompt structure."""
        from sem_mem.thread_utils import summarize_deleted_thread

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "**Key Points:**\n- User prefers dark mode"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "I prefer dark mode for all apps"},
            {"role": "assistant", "content": "Noted! I'll remember that preference."},
        ]

        result = summarize_deleted_thread(messages, client=mock_client)

        # Verify API was called
        mock_client.chat.completions.create.assert_called_once()

        # Check call structure
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4.1-mini"
        assert len(call_args.kwargs["messages"]) == 2  # system + user

        # System message should contain farewell/deletion context
        system_msg = call_args.kwargs["messages"][0]["content"]
        assert "deleted" in system_msg.lower() or "final" in system_msg.lower()

        # User message should contain conversation
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "dark mode" in user_msg

        assert result == "**Key Points:**\n- User prefers dark mode"

    def test_uses_specified_model(self):
        """Verify custom model is passed through."""
        from sem_mem.thread_utils import summarize_deleted_thread

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]

        summarize_deleted_thread(messages, client=mock_client, model="custom-model")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "custom-model"

    def test_handles_api_error_gracefully(self):
        """API errors should return empty string, not raise."""
        from sem_mem.thread_utils import summarize_deleted_thread

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = summarize_deleted_thread(messages, client=mock_client)
        assert result == ""

    def test_strips_system_noise_from_messages(self):
        """System logs should be stripped before summarization."""
        from sem_mem.thread_utils import summarize_deleted_thread

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary without logs"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "What is X?"},
            {
                "role": "assistant",
                "content": "X is important.\n\n**System Logs:**\n`L1 HIT`\n\n**Retrieved Context:**\n> old stuff"
            },
        ]

        summarize_deleted_thread(messages, client=mock_client)

        # Check that the conversation text passed to the model is clean
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "X is important" in user_msg
        assert "System Logs" not in user_msg
        assert "L1 HIT" not in user_msg

    def test_respects_max_chars_parameter(self):
        """Should truncate content to max_chars."""
        from sem_mem.thread_utils import summarize_deleted_thread

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_client.chat.completions.create.return_value = mock_response

        # Create a very long conversation
        messages = [
            {"role": "user", "content": "A" * 5000},
            {"role": "assistant", "content": "B" * 5000},
        ]

        summarize_deleted_thread(messages, client=mock_client, max_chars=500)

        # Verify the content was truncated
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        # The formatted conversation should be significantly shorter than original
        assert len(user_msg) < 1000  # Should be truncated

    def test_handles_longer_conversation(self):
        """Should handle multi-turn conversations."""
        from sem_mem.thread_utils import summarize_deleted_thread

        mock_client = Mock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "**Key Points:**\n- Multiple topics discussed"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            {"role": "user", "content": "I'm a software engineer"},
            {"role": "assistant", "content": "Great! How can I help?"},
            {"role": "user", "content": "I prefer Python over JavaScript"},
            {"role": "assistant", "content": "Noted! Python is indeed powerful."},
            {"role": "user", "content": "My project deadline is next Friday"},
            {"role": "assistant", "content": "I'll keep that in mind."},
        ]

        result = summarize_deleted_thread(messages, client=mock_client)

        # Should succeed and return the summary
        assert "Key Points" in result

        # Check all messages were included
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "software engineer" in user_msg
        assert "Python" in user_msg
        assert "Friday" in user_msg
