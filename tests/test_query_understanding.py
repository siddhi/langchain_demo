"""Tests for QueryUnderstandingChat class."""

import pytest
from perplexia_ai.week1.part1 import get_question_type


def test_routing():
    assert get_question_type("How many hydrogen atoms are in a molucule of water?") == "factual"
    assert get_question_type("How does a car engine work?") == "analytical"
    assert get_question_type("What's the difference between Java and Python?") == "comparison"
    assert get_question_type("Define artificial intelligence") == "definition"
    assert get_question_type("What is the best library for AI?") == "general"

def test_routing_tricky():
    assert get_question_type("What is photosynthesis?") == "definition"
