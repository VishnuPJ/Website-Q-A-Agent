from crewai import Task, Crew
from agents import AgentCreator
from tools import ToolCreator
from typing import Optional, Union
from logging import getLogger

logger = getLogger(__name__)

class TaskCreator(AgentCreator, ToolCreator):
    """
    A class that creates and manages various AI tasks for processing user queries.
    Inherits from AgentCreator and ToolCreator to access agent and tool creation capabilities.
    
    This class implements a pipeline of tasks for:
    - Query routing and validation
    - Information retrieval
    - Response grading
    - Hallucination detection
    - Final answer generation
    """

    def router_task(self) -> Task:
        """
        Creates a task for initial query evaluation and routing.
        Returns:
            Task: A CrewAI Task object configured for query routing
        The task evaluates user queries for:
        - Inappropriate language
        - Clarity and coherence
        - Routing decisions for further processing
        """
        try:
            Router_Task = Task(
                description=(
                    "Evaluate the user query: **{question}**.\n"
                    "Process the query to ensure it is appropriate and meaningful by following these steps:\n"
                    "- **Check for foul language**: If any offensive language is detected, respond with a polite warning.\n"
                    "- **Check for clarity**: If the query is nonsensical or lacks clarity, ask the user to clarify their question.\n"
                    "- **If the query is appropriate and clear**: Route it for further processing and analysis.\n\n"
                    "Return one of the following responses, without any preamble or explanation:\n"
                    "- 'WARNING: <your warning>', if foul language is detected.\n"
                    "- 'CLARIFICATION: <question you want to ask for clarification>', if the query is nonsensical or unclear.\n"
                    "- 'VECTORSEARCH', if the query is appropriate and meaningful."
                ),
                expected_output=(
                    "Return one of the following outputs:\n"
                    "- 'WARNING: <your warning>'\n"
                    "- 'CLARIFICATION: <question you want to ask for clarification>'\n"
                    "- 'VECTORSEARCH'.\n\n"
                    "Ensure that the output matches the evaluation of the query, maintaining a clear and concise response."
                ),
                agent=self.router_agent(),
            )
            return Router_Task
        except Exception as e:
            logger.error(f"Failed to create router task: {str(e)}")
            raise RuntimeError(f"Router task creation failed: {str(e)}")

    def retriever_task(self) -> Task:
        """
        Creates a task for retrieving information based on the validated query.
        Returns:
            Task: A CrewAI Task object configured for information retrieval
        Raises:
            RuntimeError: If tool creation or task setup fails
        """
        try:
            Retriever_task = Task(
                description=(
                    "Extract information for the question **{question}** with the help of the respective tool. "
                    "Use the markdown_rag_tool to retrieve information from the vectorstore in case the router task output is 'VECTORSEARCH'."
                ),
                expected_output=(
                    "You should analyse the output of the 'Router_Task' "
                    "If the response is 'VECTORSEARCH' then use the markdown_rag_tool to retrieve information from the vectorstore. "
                    "Return a clear and concise text as response. "
                    "Include source references (URLs) for answers when possible"
                ),
                agent=self.retriever_agent(),
                context=[self.router_task()],
                tools=[self.markdown_rag_tool("crawl_results.md")],
            )
            return Retriever_task
        except Exception as e:
            logger.error(f"Failed to create retriever task: {str(e)}")
            raise RuntimeError(f"Retriever task creation failed: {str(e)}")

    def grader_task(self) -> Task:
        """
        Creates a task for evaluating the relevance of retrieved information.
        Returns:
            Task: A CrewAI Task object configured for grading responses
        Raises:
            RuntimeError: If task creation or context setup fails
        """
        try:
            Grader_task = Task(
                description=(
                    "Based on the response from the Retriever_task **{response}** for the question **{question}** "
                    "evaluate whether the retrieved content is relevant to the question."
                ),
                expected_output=(
                    "Binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. "
                    "You must answer 'yes' if the response from the 'retriever_task' is in alignment with the question asked. "
                    "If your answer is 'yes' , you must return the response of Retriever_task also. "
                    "You must answer 'no' if the response from the 'Retriever_task' is not in alignment with the question asked."
                ),
                agent=self.grader_agent(),
                context=[self.retriever_task()],
            )
            return Grader_task
        except Exception as e:
            logger.error(f"Failed to create grader task: {str(e)}")
            raise RuntimeError(f"Grader task creation failed: {str(e)}")

    def hallucination_grader_task(self) -> Task:
        """
        Creates a task for detecting potential hallucinations in the response.
        Returns:
            Task: A CrewAI Task object configured for hallucination detection
        Raises:
            RuntimeError: If task creation or context setup fails
        """
        try:
            Hallucination_task = Task(
                description=(
                    "Based on the response from the grader_task for the question **{question}** "
                    "evaluate whether the answer **{response}** is grounded in / supported by a set of facts."
                ),
                expected_output=(
                    "Binary score 'yes' or 'no' score to indicate whether the answer is sync with the question asked. "
                    "Respond 'yes' if the answer is useful and contains fact about the question asked. "
                    "If your answer is 'yes' , you must return the response of grader_task also. "
                    "Respond 'no' if the answer is not useful and does not contains fact about the question asked. "
                    "Do not provide any preamble or explanations except for 'yes' or 'no'."
                ),
                agent=self.hallucination_grader_agent(),
                context=[self.grader_task()],
            )
            return Hallucination_task
        except Exception as e:
            logger.error(f"Failed to create hallucination grader task: {str(e)}")
            raise RuntimeError(f"Hallucination grader task creation failed: {str(e)}")

    def answer_grader_task(self) -> Task:
        """
        Creates a task for generating the final response based on all previous evaluations.
        Returns:
            Task: A CrewAI Task object configured for final answer generation
        Raises:
            RuntimeError: If task creation or context setup fails
        """
        try:
            Answer_task = Task(
                description=(
                    "Based on the response from the hallucination_grader_task for the question **{question}** "
                    "evaluate whether the answer **{response}** is useful to resolve the question. "
                    "If the answer is 'yes' return a clear and concise answer by returning {response}. "
                    "If the answer is 'no' then return 'information is not available in the document'"
                ),
                expected_output=(
                    "Return a clear and concise response if the response from 'hallucination_grader_task' is 'yes'. "
                    "Otherwise respond as 'Sorry! unable to find a valid response'."
                ),
                agent=self.answer_grader_agent(),
                context=[self.hallucination_grader_task()],
            )
            return Answer_task
        except Exception as e:
            logger.error(f"Failed to create answer grader task: {str(e)}")
            raise RuntimeError(f"Answer grader task creation failed: {str(e)}")