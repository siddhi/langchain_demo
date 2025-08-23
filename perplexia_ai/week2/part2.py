"""Part 2 - Document RAG implementation using LangGraph.

This implementation focuses on:
- Setting up document loading and processing
- Creating vector embeddings and storage
- Implementing retrieval-augmented generation
- Formatting responses with citations from OPM documents
"""

from typing import Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, END, StateGraph
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_community.tools import TavilySearchResults
from pathlib import Path


class RagState(TypedDict):
    question: str
    docs: list[Document]
    context: str
    is_context_good: bool
    answer: str

# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class DocumentRAGChat(ChatInterface):
    """Week 2 Part 2 implementation for document RAG."""
    
    def initialize(self, docs_path: str = "docs/") -> None:
        """Initialize components for document RAG.
        
        Args:
            docs_path: Path to directory containing PDF documents
        
        Students should:
        - Initialize the LLM
        - Set up document loading and processing
        - Create vector embeddings
        - Build retrieval system
        - Create LangGraph for RAG workflow
        """
        
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.embeddings = OpenAIEmbeddings()
        self.search_tool = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=True,
            include_images=False,
            search_depth="advanced"
        )

        data_dir = Path(docs_path)
        self.document_paths = list(data_dir.glob("*.pdf"))
        
        docs = self._load_and_process_documents(self.document_paths)
        self.vector_store = InMemoryVectorStore.from_documents(docs, self.embeddings)
        
        graph = StateGraph(RagState)
        graph.add_node("retrieval", self._create_retrieval_node())
        graph.add_node("create_context", self._create_context_node())
        graph.add_node("evaluation", self._create_evaluation_node())
        graph.add_node("web_search", self._create_web_search_node())
        graph.add_node("create_search_context", self._create_context_node())
        graph.add_node("generation", self._create_generation_node())
        graph.add_edge(START, "retrieval")
        graph.add_edge("retrieval", "create_context")
        graph.add_edge("create_context", "evaluation")
        graph.add_conditional_edges("evaluation", self._create_check_node(), {"generation": "generation", "web_search": "web_search"})
        graph.add_edge("web_search", "create_search_context")
        graph.add_edge("create_search_context", "generation")
        graph.add_edge("generation", END)
        self.graph = graph.compile()
    
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
        def retrieval_node(state: RagState) -> dict:
            question = state["question"]
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            docs = retriever.invoke(question)
            return {"docs": docs}
        return retrieval_node

    def _create_context_node(self):
        def context_node(state: RagState) -> dict:
            docs = state["docs"]
            context = "\n".join(f'{doc.metadata}:\n{doc.page_content}\n' for doc in docs)
            return {"context": context}
        return context_node

    def _create_evaluation_node(self):
        def evaluation_node(state: RagState) -> dict:
            prompt_template = """
Evaluate if the following context is relevant for answering the given question.

    - Output a single word: GOOD or BAD. Do not output anything else
    - Output GOOD if the context contains information that can be used to answer the question
    - For anything else, output BAD

Here are some examples

Context:

New Delhi is the capital of India.

India is in Asia

Asia is the biggest continent by land mass and population

Question: Which continent does New Delhi belong to?
Answer: GOOD

Context:

Populous is a video game release by Bullfrog in 1989

It was released for the Amiga

Question: Which is the most populous city in the World?
Answer: BAD

Now evaluate the context below

Context:
{context}

Question: {question}
Answer: """
            prompt = PromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": state["context"], "question": state["question"]})
            return {"is_context_good": "GOOD" in response}
        return evaluation_node

    def _create_check_node(self):
        def check_node(state: RagState) -> str:
            return "generation" if state["is_context_good"] else "web_search"
        return check_node

    def _create_web_search_node(self):
        def web_search(state: RagState) -> dict:
            query = state["question"]
            search_results = self.search_tool.invoke(query)
            documents = [Document(page_content=result['content'], metadata={'source': result['url']}) for result in search_results]
            return {"docs": documents}
        return web_search

    def _create_generation_node(self):
        """Create a node that generates responses using retrieved context."""
        def generation_node(state: RagState) -> dict:
            prompt_template = """
You are a helpful question answering bot. Use the context below to answer the question. Follow these rules:

    - Only use the context and nothing beyond the context
    - If the question is not answered in the context, say "I don't know the answer"

Context:
{context}

Question: {question}
Answer: """
            prompt = PromptTemplate.from_template(prompt_template)
            context = state["context"]
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context, "question": state["question"]})
            return {"answer": response}
        return generation_node
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using document RAG.
        
        Should reject queries that are not answerable from the OPM documents.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response based on document knowledge
        """

        state = self.graph.invoke({"question": message})
        sources = "\n".join(f"{i}. {doc.metadata['source']}" for i, doc in enumerate(state["docs"], start=1))
        full_answer = f"{state['answer']}\n\nSOURCES:\n\n{sources}"
        return full_answer
