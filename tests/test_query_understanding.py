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
    ("What is 2 + 2?", "maths"),
    ("What is the best library for AI?", "general")
])
def test_routing(chat, question, category):
    assert chat.routing_chain.invoke({"question": question}) == category

def test_routing_tricky(chat):
    """Ask a question that is borderline between factual and definition"""
    assert chat.routing_chain.invoke({"question": "What is photosynthesis?"}) == "definition"

@pytest.mark.parametrize("question,answer", [
    ("What is 2 + 2?", "4"),
    ("What is 15% of 85?", "12.75"),
    ("If I have three dozen eggs, how many eggs do I have?", "36"),
    ("If I split a $200 bill amount 4 people with a 20% tip, how much does each pay?", "60")
])
def test_math_question(chat, question, answer):
    assert answer in chat.process_message(question)

def test_math_tricky(chat):
    """This test will fail without the tool call"""
    assert "179385280681.2326" in chat.process_message("How much is 145323.122 multiplied by 1234389.12?")
