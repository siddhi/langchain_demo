"""Tests for DocumentRAGChat class."""

import pytest
from pathlib import Path
from perplexia_ai.week2.part2 import DocumentRAGChat


def test_document_loading_and_chunking():
    """Test that documents are properly loaded and chunked."""
    # Create DocumentRAGChat instance
    chat = DocumentRAGChat()
    
    # Use test documents - process whatever PDFs we find
    test_docs_dir = Path("test_docs/")
    document_paths = list(test_docs_dir.glob("*.pdf"))
    
    # Process the documents (could be 0, 1, or more)
    chunks = chat._load_and_process_documents(document_paths)
    
    # Check chunk properties for any chunks we got
    for chunk in chunks[:5]:  # Check first 5 chunks
        assert len(chunk.page_content) <= 1000, f"Chunk too large: {len(chunk.page_content)} chars"
        assert len(chunk.page_content) > 0, "Empty chunk found"
        assert 'source' in chunk.metadata, "Chunk missing source metadata"
        assert 'test_docs' in chunk.metadata['source'], "Source should reference test_docs folder"
    
    print(f"Successfully processed {len(document_paths)} documents into {len(chunks)} chunks")


@pytest.fixture
def chat():
    """Create and initialize DocumentRAGChat instance for testing."""
    chat = DocumentRAGChat()
    chat.initialize("test_docs/")  # Use test documents instead of real ones
    return chat


def test_document_rag_omp_headquarters(chat):
    """Test end-to-end RAG with OMP headquarters question."""
    # Ask about OMP headquarters
    response = chat.process_message("Where is the headquarters of OPM?")
    
    # Should contain the specific building name
    assert "Theodore Roosevelt Federal Office Building" in response, \
           f"Response should mention Theodore Roosevelt Federal Office Building, got: {response}"
    
    # Should have sources section
    assert "SOURCES" in response, "Response should include sources section"


def test_document_rag_out_of_scope_question(chat):
    """Test that RAG system handles nonsensical questions appropriately."""
    # Ask a nonsensical question that neither documents nor web search can answer
    response = chat.process_message("How many dreams does a triangle have?")
    
    # Should respond with "I don't know the answer" as per the prompt
    assert "I don't know the answer" in response, \
           f"Response should contain 'I don't know the answer', got: {response}"
    
    # Should still have sources section
    assert "SOURCES" in response, "Response should include sources section"


def test_corrective_rag_pycon_workshop(chat):
    """Test corrective RAG with PyCon India 2025 workshop question requiring web search."""
    # Ask about specific PyCon India 2025 workshop - not in OPM documents or LLM training
    response = chat.process_message("Who is doing the workshop on 'Programming is playtime' at Pycon India 2025?")
    
    # Should contain the presenter's name
    assert "Siddharta" in response, \
           f"Response should mention Siddharta, got: {response}"
    
    # Should not say "I don't know" - should use web search instead
    assert not ("I don't know the answer" in response), \
           "Should fallback to web search instead of saying 'I don't know'"
    
    # Should still have sources section
    assert "SOURCES" in response, "Response should include sources section"
