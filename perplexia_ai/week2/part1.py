"""Part 1 - Web Search implementation using LangGraph.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from typing import Dict, List, Optional, TypedDict

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from perplexia_ai.core.chat_interface import ChatInterface


class WebSearchState(TypedDict):
    """State for the web search workflow."""
    query: str  # User's search query
    search_results: List[Dict]  # Raw search results from Tavily
    formatted_response: str  # Final response with citations

# NOTE: The TODOs are only a direction for you to start with.
# You are free to change the structure of the code as you see fit.
class WebSearchChat(ChatInterface):
    """Week 2 Part 1 implementation for web search using LangGraph."""
    
    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for web search.
        
        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph for web search workflow
        """
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.search_tool = TavilySearchResults(
            max_results=5,
            include_answer=True,
            include_raw_content=True,
            include_images=False,
            search_depth="advanced"
        )
        
        # TODO: Create the graph
        # Define nodes:
        
        # Define the edges and the graph structure
        
        # Compile the graph
        pass
    
    def _create_search_node(self):
        """Create a node that performs web search."""
        # TODO: Implement search node
        pass
    
    def _create_process_results_node(self):
        """Create a node that processes and formats search results."""
        # TODO: Implement process results node
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response with search results
        """
        # TODO: Implement web search processing
        # 1. Format the input message
        # 2. Run the graph
        # 3. Extract the response
        
        # This is just a placeholder
        return f"Web search result for: {message}" 
