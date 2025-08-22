"""Tests for WebSearchChat class."""

import pytest
from perplexia_ai.week2.part1 import WebSearchChat


@pytest.fixture
def web_search_chat():
    """Create and initialize WebSearchChat instance for testing."""
    chat = WebSearchChat()
    chat.initialize()
    return chat


def test_web_search_capital_of_france(web_search_chat):
    """Test that WebSearchChat can answer what is the capital of France."""
    response = web_search_chat.process_message("What is the capital of France?")
    assert "Paris" in response
    assert "SOURCES" in response