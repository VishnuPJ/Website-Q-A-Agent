import streamlit as st
import logging
from crewai import Crew, Flow, LLM
from tasks import TaskCreator
from pydantic import BaseModel, ValidationError
from crewai.flow.flow import listen, start, and_, or_, router
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseState(BaseModel):
    """State model for managing conversation flow."""
    user_input: str = ""
    agent_response: str = ""

class CreateCrew(Flow[ResponseState]):
    def __init__(self):
        """Initialize the flow controller and task creator."""
        try:
            super().__init__()
            self.task_creator = TaskCreator()
        except Exception as e:
            logger.error(f"Failed to initialize CreateCrew: {str(e)}")
            raise

    def router_crew(self) -> Crew:
        try:
            return Crew(
                agents=[self.task_creator.router_agent()],
                tasks=[self.task_creator.router_task()],
                verbose=True,
            )
        except Exception as e:
            logger.error(f"Failed to create router crew: {str(e)}")
            raise ValueError(f"Router crew creation failed: {str(e)}")

    def retriever_crew(self) -> Crew:
        try:
            return Crew(
                agents=[self.task_creator.retriever_agent()],
                tasks=[self.task_creator.retriever_task()],
                verbose=True,
            )
        except Exception as e:
            logger.error(f"Failed to create retriever crew: {str(e)}")
            raise ValueError(f"Retriever crew creation failed: {str(e)}")

    def checking_crew(self) -> Crew:
        try:
            return Crew(
                agents=[
                    self.task_creator.grader_agent(),
                    self.task_creator.hallucination_grader_agent(),
                    self.task_creator.answer_grader_agent()
                ],
                tasks=[
                    self.task_creator.grader_task(),
                    self.task_creator.hallucination_grader_task(),
                    self.task_creator.answer_grader_task()
                ],
                verbose=True,
            )
        except Exception as e:
            logger.error(f"Failed to create checking crew: {str(e)}")
            raise ValueError(f"Checking crew creation failed: {str(e)}")

    @start()
    def input_query(self) -> str:
        try:
            query = self.state.user_input.strip()
            if not query:
                raise ValueError("Query cannot be empty")
            logger.info(f"Processing query: {query}")
            return query
        except Exception as e:
            logger.error(f"Error processing input query: {str(e)}")
            raise

    @router(input_query)
    def route_query(self) -> str:
        try:
            inputs_route = {"question": self.state.user_input}
            response_route = self.router_crew().kickoff(inputs=inputs_route)
            return response_route.raw
        except Exception as e:
            logger.error(f"Error in query routing: {str(e)}")
            raise ValueError(f"Query routing failed: {str(e)}")

    @listen("VECTORSEARCH")
    def retrieve_info(self) -> str:
        try:
            inputs_ret = {"question": self.state.user_input}
            response_ret = self.retriever_crew().kickoff(inputs=inputs_ret)
            response_ret = response_ret.raw
            self.state.agent_response = response_ret
            return response_ret
        except Exception as e:
            logger.error(f"Error in information retrieval: {str(e)}")
            raise ValueError(f"Information retrieval failed: {str(e)}")

    @router(retrieve_info)
    def post_processing(self) -> str:
        try:
            inputs_process = {
                "question": self.state.user_input,
                "response": self.state.agent_response
            }
            response_process = self.checking_crew().kickoff(inputs=inputs_process)
            return response_process.raw
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            raise ValueError(f"Response post-processing failed: {str(e)}")

def main():
    st.set_page_config(
        page_title="CrewAI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    st.title("CrewAI Assistant")
    st.write("Ask any question and let our AI crew find the answer for you!")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create input field for user question
    user_input = st.text_input("Ask your question:", key="user_input")

    # Process button
    if st.button("Get Answer"):
        if user_input:
            try:
                # Show loading spinner while processing
                with st.spinner('Processing your question...'):
                    # Initialize CrewAI flow
                    crew_flow = CreateCrew()
                    crew_flow.state.user_input = user_input
                    
                    # Get response
                    result = crew_flow.kickoff()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_input,
                        "answer": result
                    })
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.divider()

    # Add sidebar with information
    with st.sidebar:
        st.header("About")
        st.write("""This application uses an AI-powered question-answering system to process documentation from help websites and accurately answer user queries about:
                    - Product Features: Provides detailed insights into product capabilities
                    - Integrations: Explains compatibility and setup with other tools
                    - Functionality: Clarifies how features work and resolve user questions effectively""")
        
        # # Add clear history button
        # if st.button("Clear Chat History"):
        #     st.session_state.chat_history = []
        #     st.experimental_rerun()

if __name__ == '__main__':
    main()

