"""Tests for QueryUnderstandingChat class."""

import pytest
from perplexia_ai.week1.part1 import QueryUnderstandingChat


@pytest.fixture
def chat():
    chat = QueryUnderstandingChat()
    chat.initialize()
    return chat

def test_routing(chat):
    assert chat.routing_chain.invoke({"question": "How many hydrogen atoms are in a molucule of water?"}) == "factual"
    assert chat.routing_chain.invoke({"question": "How does a car engine work?"}) == "analytical"
    assert chat.routing_chain.invoke({"question": "What's the difference between Java and Python?"}) == "comparison"
    assert chat.routing_chain.invoke({"question": "Define artificial intelligence"}) == "definition"
    assert chat.routing_chain.invoke({"question": "What is the best library for AI?"}) == "general"

def test_routing_tricky(chat):
    """Ask a question that is borderline between factual and definition"""
    assert chat.routing_chain.invoke({"question": "What is photosynthesis?"}) == "definition"

def test_math_question(chat):
    assert "4" in chat.process_message("What is 2 + 2?")
