"""Part 1 - Web Search implementation using LangGraph.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from typing import Dict, List, Optional, TypedDict

from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
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
        
        # Create the graph
        graph = StateGraph(WebSearchState)
        
        # Define nodes
        graph.add_node("search", self._create_search_node())
        graph.add_node("process_results", self._create_process_results_node())
        
        # Define the edges and the graph structure
        graph.add_edge(START, "search")
        graph.add_edge("search", "process_results")
        graph.add_edge("process_results", END)
        
        # Compile the graph
        self.graph = graph.compile()
    
    def _create_search_node(self):
        """Create a node that performs web search."""
        def search_node(state: WebSearchState) -> WebSearchState:
            query = state["query"]
            search_results = self.search_tool.invoke(query)
            return {"search_results": search_results}
        return search_node
    
    def _create_process_results_node(self):
        """Create a node that processes and formats search results."""
        def process_results_node(state: WebSearchState) -> WebSearchState:
            query = state["query"]
            search_results = state["search_results"]
            
            # Format search results for LLM
            context = "Based on the following search results, provide a comprehensive answer:\n\n"
            sources = []
            
            for i, result in enumerate(search_results, 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                
                context += f"[{i}] {title}\n{content}\n\n"
                sources.append(f"[{i}] {url}")
            
            # Create prompt for LLM
            prompt = f"""Question: {query}

{context}

Please provide a clear, accurate answer based on the search results above. Reference the sources using [1], [2], etc. when appropriate."""
            
            # Get LLM response
            llm_response = self.llm.invoke(prompt).content
            
            # Format final response with sources
            sources_section = "\n\nSOURCES:\n" + "\n".join(sources)
            formatted_response = llm_response + sources_section
            
            return {"formatted_response": formatted_response}
        
        return process_results_node
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using web search.
        
        Args:
            message: The user's input message
            chat_history: Previous conversation history
            
        Returns:
            str: The assistant's response with search results
        """
        # 1. Format the input message - create initial state
        initial_state = {
            "query": message,
            "search_results": [],
            "formatted_response": ""
        }
        
        # 2. Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # 3. Extract the response
        return final_state["formatted_response"] 
