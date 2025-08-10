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
from langchain_core.runnables import RunnableLambda

ROUTING_PROMPT = PromptTemplate.from_template("""
You are a query classifier. Given a question, you should classify it into one of five categories:
- factual: These are direct questions asking about a fact. Example: What is the biggest country by area?
- analytical: These questions require an in-depth answer with logical reasoning. Example: Explain Rayleigh scattering
- comparison: These questions ask about two or more things to be compared. Example: What is the difference between an asteroid and a meteor?
- definition: These questions ask for a definition of a term. Example: Define a strait
- general: Anything else

You should output only a single word containing the category of the question. Do not output anything else.

Question: {question}
Category: 
""")

FACTUAL_PROMPT = PromptTemplate.from_template("""
  You are a knowledgeable assistant that provides accurate, concise answers to factual questions.

  Guidelines:
  - Provide direct, factual answers based on well-established knowledge
  - Be precise and specific in your responses
  - If the question asks for a specific fact (like a formula, date, number), provide it clearly
  - Keep responses focused and avoid unnecessary elaboration
  - If you're uncertain about a fact, acknowledge the uncertainty

  Question: {question}

  Answer:""")

ANALYTICAL_PROMPT = PromptTemplate.from_template("""
  You are an expert analyst who provides thorough, well-reasoned explanations for complex questions.

  Guidelines:
  - Break down complex topics into logical components
  - Explain the underlying mechanisms, processes, or principles
  - Use clear reasoning and logical flow in your explanations
  - Provide context and background when helpful for understanding
  - Connect different concepts to show relationships
  - Use examples or analogies when they aid comprehension
  - Structure your response with clear organization (causes, effects, steps, etc.)

  Question: {question}

  Analysis:""")

COMPARISION_PROMPT = PromptTemplate.from_template("""
  You are a knowledgeable assistant who provides clear, structured comparisons between different concepts, objects,
  or ideas.

  Guidelines:
  - Identify the key dimensions or aspects for comparison
  - Present similarities and differences in an organized manner
  - Use parallel structure when describing each item being compared
  - Highlight the most significant distinctions first
  - Provide specific examples or characteristics for each item
  - Use clear contrast words (whereas, while, however, in contrast, etc.)
  - Structure your response to make the comparison easy to follow
  - Conclude with a brief summary of the main differences if helpful

  Question: {question}

  Comparison:""")

DEFINITION_PROMPT = PromptTemplate.from_template("""
You are a precise and knowledgeable assistant who provides clear, accurate definitions of terms and concepts.

Keep the definition to a single sentence, keep it concise and technically accurate

Question: {question}

Definition:""")

GENERAL_PROMPT = PromptTemplate.from_template("""
  You are a helpful and knowledgeable assistant who provides thoughtful, well-structured responses to a wide variety
  of questions.

  Guidelines:
  - Read the question carefully and determine what type of response is most appropriate
  - For factual questions: Provide direct, accurate answers with specific details
  - For analytical questions: Break down complex topics and explain underlying mechanisms
  - For comparative questions: Structure clear comparisons highlighting key similarities and differences
  - For definition questions: Start with a core definition and explain essential characteristics
  - For general questions: Use your best judgment to provide the most helpful response
  - Always be accurate and acknowledge when you're uncertain
  - Use clear, accessible language while maintaining appropriate depth
  - Structure your response logically for easy understanding
  - Provide examples when they would be helpful

  Question: {question}

  Response:""")


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""

    def initialize(self) -> None:
        """Initialize components for query understanding.

        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.routing_chain = ROUTING_PROMPT | self.llm | StrOutputParser()
        self.response_prompts = {
            "factual": FACTUAL_PROMPT,
            "analytical": ANALYTICAL_PROMPT,
            "comparison": COMPARISION_PROMPT,
            "definition": DEFINITION_PROMPT,
        }

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Evaluates the query intent and routes the query to a specific prompt based on the intent category"""

        def route_question(info: dict[str, str]):
            return self.response_prompts.get(info["category"], GENERAL_PROMPT) | self.llm | StrOutputParser()

        full_chain = {"category": self.routing_chain, "question": lambda x: x["question"] } | RunnableLambda(route_question)
        return full_chain.invoke({"question": message})
