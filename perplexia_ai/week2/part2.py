"""Part 2 - Document RAG implementation using LangGraph.

This implementation focuses on:
- Setting up document loading and processing
- Creating vector embeddings and storage
- Implementing retrieval-augmented generation
- Formatting responses with citations from OPM documents
"""

from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from perplexia_ai.core.chat_interface import ChatInterface
from pathlib import Path


# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class DocumentRAGChat(ChatInterface):
    """Week 2 Part 2 implementation for document RAG."""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.document_paths = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for document RAG.
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing
        - Create vector embeddings
        - Build retrieval system
        - Create LangGraph for RAG workflow
        """
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.embeddings = OpenAIEmbeddings()
        
        data_dir = Path("docs/")
        self.document_paths = list(data_dir.glob("*.pdf"))
        
        docs = self._load_and_process_documents(self.document_paths)
        self.vector_store = InMemoryVectorStore.from_documents(docs, self.embeddings)
        
        # TODO: Create the graph
        # Define nodes:
        # 1. Retrieval node: Finds relevant document sections
        # 2. Generation node: Creates response using retrieved context
        
        # Define the edges and the graph structure
        
        # Compile the graph
        pass
    
    def _load_and_process_documents(self, document_paths: list) -> list:
        """Load and process documents from given paths."""
        # 1. Load the documents from provided paths
        documents = []
        for doc_path in document_paths:
            loader = PyPDFLoader(str(doc_path))
            docs = loader.load()
            documents.extend(docs)
        
        # 2. Split into chunks suitable for text-embedding-3-small (8191 token limit)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Conservative chunk size for embeddings
            chunk_overlap=200,  # Overlap to maintain context
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        
        # 3. Return processed document chunks
        return chunks
    
    def _create_retrieval_node(self):
        """Create a node that retrieves relevant document sections."""
        # TODO: Implement retrieval node
        pass
    
    def _create_generation_node(self):
        """Create a node that generates responses using retrieved context."""
        # TODO: Implement generation node
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using document RAG.
        
        Should reject queries that are not answerable from the OPM documents.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on document knowledge
        """
        # TODO: Implement document RAG processing
        # 1. Format the input message
        # 2. Run the graph
        # 3. Extract the response
        
        # This is just a placeholder
        return f"Document RAG result for: {message}" 
