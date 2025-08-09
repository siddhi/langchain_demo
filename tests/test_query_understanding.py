"""Tests for QueryUnderstandingChat class."""

import pytest
from perplexia_ai.week1.part1 import QueryUnderstandingChat


class TestQueryUnderstandingChat:
    """Test cases for QueryUnderstandingChat."""
    
    def test_process_message_returns_hello(self):
        """Test that process_message returns 'hello' for any input."""
        chat = QueryUnderstandingChat()
        result = chat.process_message("test input")
        assert result == "hello"