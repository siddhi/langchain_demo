"""Tests for QueryUnderstandingChat class."""

import pytest
from perplexia_ai.week1.part1 import QueryUnderstandingChat


@pytest.fixture
def chat():
    chat = QueryUnderstandingChat()
    chat.initialize()
    return chat

@pytest.mark.parametrize("question,category", [
    ("How many hydrogen atoms are in a molucule of water?", "factual"),
    ("How does a car engine work?", "analytical"),
    ("What's the difference between Java and Python?", "comparison"),
    ("Define artificial intelligence", "definition"),
    ("What is the best library for AI?", "general")
])
def test_routing(chat, question, category):
    assert chat.routing_chain.invoke({"question": question}) == category

def test_routing_tricky(chat):
    """Ask a question that is borderline between factual and definition"""
    assert chat.routing_chain.invoke({"question": "What is photosynthesis?"}) == "definition"

def test_math_question(chat):
    assert "4" in chat.process_message("What is 2 + 2?")
