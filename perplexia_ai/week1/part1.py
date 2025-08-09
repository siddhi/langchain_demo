"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

ROUTING_PROMPT = """
You are a query classifier. Given a question, you should classify it into one of five categories:
- factual: These are direct questions asking about a fact. Example: What is the biggest country by area?
- analytical: These questions require an in-depth answer with logical reasoning. Example: Explain Rayleigh scattering
- comparison: These questions ask about two or more things to be compared. Example: What is the difference between an asteroid and a meteor?
- definition: These questions ask for a definition of a term. Example: Define a strait
- general: Anything else

You should output only a single word containing the category of the question. Do not output anything else.

Question: {question}
Category: 
"""

model = init_chat_model("gpt-4o-mini", model_provider="openai")

def get_question_type(question: str) -> str:
    template = PromptTemplate.from_template(ROUTING_PROMPT)
    chain = template | model | StrOutputParser()
    return chain.invoke({"question": question})

class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        self.llm = model
        self.query_classifier_prompt = None
        self.response_prompts = {}
    
    def initialize(self) -> None:
        """Initialize components for query understanding.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """
        # TODO: Students implement initialization
        pass
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding.
        
        Students should:
        - Classify the query type
        - Generate appropriate response
        - Format based on query type
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # TODO: Students implement query understanding
        return "hello"
