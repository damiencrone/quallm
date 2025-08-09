"""
Test for the feedback() method - baseline test before implementing enhancements
"""
from pydantic import BaseModel, Field, conint
from quallm import LLMClient
from quallm.tasks import Task, TaskConfig


def test_feedback_method_works():
    """Verify the current feedback() method works (baseline test before any changes)"""
    
    # Define the Pydantic model for the structured response (from README)
    class TextAnalysisResponse(BaseModel):
        main_idea: str = Field(description="A concise summary of the main idea of the text.")
        sentiment_score: conint(ge=-10, le=10) = Field(
            description="An integer score from -10 (most negative) to 10 (most positive)."
        )
    
    # Create a TaskConfig for the task (from README)
    analysis_task_config = TaskConfig(
        response_model=TextAnalysisResponse,
        system_template="You are an AI assistant that analyzes text. Your goal is to identify the main idea and assign a sentiment score.",
        user_template="Please analyze the following text: {input_document}",
        output_attribute="main_idea"
    )
    
    # Create the Task instance from the configuration
    analysis_task = Task.from_config(analysis_task_config)
    
    # Initialize an LLMClient to provide the feedback
    feedback_llm = LLMClient(language_model="llama3.1") # For fast testing; not quality feedback
    
    # Get feedback on the task definition
    feedback_text = analysis_task.feedback(
        raters=feedback_llm,
        context="This sentiment analysis task will be used to classify responses to a survey question."
    )
    
    # Verify the feedback method returns a string with expected content
    assert isinstance(feedback_text, str)
    assert "TASK DEFINITION FEEDBACK" in feedback_text
    assert len(feedback_text) > 100  # Should return substantive feedback