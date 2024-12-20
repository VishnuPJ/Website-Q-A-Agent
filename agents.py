from crewai import Agent, LLM
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentCreator:
    """
    A class for creating different types of AI agents using the CrewAI framework.
    This class provides methods to create specialized agents for different tasks
    such as routing queries, retrieving information, and grading responses.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the AgentCreator with a base URL for the LLM service.
        Args:
            base_url (str): The base URL for the LLM service. Defaults to localhost:11434.
        """
        self.base_url = base_url

    def llm_provider(self) -> Optional[LLM]:
        """
        Create and configure the Language Learning Model provider.
        Returns:
            LLM: Configured LLM instance for agent use.
        Raises:
            ConnectionError: If unable to connect to the LLM service.
        """
        try:
            llm_general = LLM(
                model="ollama/qwen2.5",
                base_url=self.base_url
            )
            return llm_general
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {str(e)}")
            raise ConnectionError(f"Unable to connect to LLM service at {self.base_url}: {str(e)}")

    def router_agent(self) -> Optional[Agent]:
        """
        Create a router agent that evaluates and routes user queries.
        Returns:
            Agent: Configured router agent.
        Raises:
            Exception: If agent creation fails.
        """
        try:
            return Agent(
                role='Router',
                goal='Evaluate the user query and route it accordingly or flag it.',
                backstory=(
                    "You are an expert at processing user queries to ensure they are appropriate and meaningful. "
                    "First, check if the query contains any foul language or make no sense. "
                    "If the query contains foul language, respond with a polite warning. "
                    "If the query is nonsensical, ask the user to clarify. "
                    "Otherwise return 'VECTORSEARCH'"
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.llm_provider()
            )
        except Exception as e:
            logger.error(f"Failed to create router agent: {str(e)}")
            raise

    def retriever_agent(self) -> Optional[Agent]:
        """
        Create a retriever agent that handles information retrieval tasks.
        Returns:
            Agent: Configured retriever agent.
        Raises:
            Exception: If agent creation fails.
        """
        try:
            return Agent(
                role="Retriever",
                goal="Use the information retrieved from the vectorstore to answer the user's query.",
                backstory=(
                    "You are an assistant for question-answering tasks. "
                    "Use the information present in the retrieved context to answer the question. "
                    "You have to provide a clear concise answer."
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.llm_provider()
            )
        except Exception as e:
            logger.error(f"Failed to create retriever agent: {str(e)}")
            raise

    def grader_agent(self) -> Optional[Agent]:
        """
        Create a grader agent that assesses relevance of retrieved documents.
        Returns:
            Agent: Configured grader agent.
        Raises:
            Exception: If agent creation fails.
        """
        try:
            return Agent(
                role='Answer Grader',
                goal='Filter out erroneous retrievals',
                backstory=(
                    "You are a grader assessing relevance of a retrieved document to a user question. "
                    "If the document contains keywords related to the user question, grade it as relevant. "
                    "It does not need to be a stringent test. You have to make sure that the answer is relevant to the question."
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.llm_provider()
            )
        except Exception as e:
            logger.error(f"Failed to create grader agent: {str(e)}")
            raise

    def hallucination_grader_agent(self) -> Optional[Agent]:
        """
        Create a hallucination grader agent that detects potential hallucinations in responses.
        Returns:
            Agent: Configured hallucination grader agent.
        Raises:
            Exception: If agent creation fails.
        """
        try:
            return Agent(
                role="Hallucination Grader",
                goal="Filter out hallucination",
                backstory=(
                    "You are a hallucination grader assessing whether an answer is grounded in / supported by a set of facts. "
                    "Make sure you meticulously review the answer and check if the response provided is in alignment with the question asked"
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.llm_provider()
            )
        except Exception as e:
            logger.error(f"Failed to create hallucination grader agent: {str(e)}")
            raise

    def answer_grader_agent(self) -> Optional[Agent]:
        """
        Create an answer grader agent that assesses the quality and relevance of answers.
        Returns:
            Agent: Configured answer grader agent.
        Raises:
            Exception: If agent creation fails.
        """
        try:
            return Agent(
                role="Answer Grader",
                goal="Filter out hallucination from the answer.",
                backstory=(
                    "You are a grader assessing whether an answer is useful to resolve a question. "
                    "Make sure you meticulously review the answer and check if it makes sense for the question asked. "
                    "If the answer is relevant generate a clear and concise response. "
                    "If the answer generated is not relevant or you cannot identify the answer clearly mention that "
                    "the information is not available in the documentation."
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.llm_provider()
            )
        except Exception as e:
            logger.error(f"Failed to create answer grader agent: {str(e)}")
            raise