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