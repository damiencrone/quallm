"""
Test for the feedback() method - baseline test before implementing enhancements
"""
from pydantic import BaseModel, Field, conint
from quallm import LLMClient
from quallm.tasks import Task, TaskConfig
from quallm.dataset import Dataset
from quallm.task_examples import DATA_ANALYSIS_INSTRUCTIONS, EXECUTION_ANALYSIS_INSTRUCTIONS


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


def test_prepare_feedback_task():
    """Test that _prepare_feedback_task builds prompts and data correctly for all modes"""
    
    # Create a simple task
    class SimpleResponse(BaseModel):
        result: str = Field(description="Result")
    
    task = Task.from_config(TaskConfig(
        response_model=SimpleResponse,
        system_template="Analyze",
        user_template="Analyze: {text}",
        data_args=["text"],  # Added missing data_args
        output_attribute="result"
    ))
    
    # Import FeedbackConfig for tests
    from quallm.feedback_config import FeedbackConfig
    config = FeedbackConfig()
    
    # Mode 0: No example data
    feedback_task, data_dict = task._prepare_feedback_task(
        context="Test context",
        example_data=None,
        task_raters=None,
        config=config
    )
    assert DATA_ANALYSIS_INSTRUCTIONS not in feedback_task.prompt.system_template
    assert EXECUTION_ANALYSIS_INSTRUCTIONS not in feedback_task.prompt.system_template
    assert data_dict["data_summary"] == "None provided"
    assert data_dict["output_summary"] == "None available"
    assert data_dict["observations"] == "None available"
    
    # Mode 1: Example data only
    example_data = Dataset([{"text": "sample"}], data_args=["text"])
    feedback_task, data_dict = task._prepare_feedback_task(
        context="Test context",
        example_data=example_data,
        task_raters=None,
        config=config
    )
    assert DATA_ANALYSIS_INSTRUCTIONS in feedback_task.prompt.system_template
    assert EXECUTION_ANALYSIS_INSTRUCTIONS not in feedback_task.prompt.system_template
    assert "Data schema (1 examples)" in data_dict["data_summary"]
    assert "type=str" in data_dict["data_summary"]  # Check for actual data type analysis
    assert data_dict["output_summary"] == "None available"
    assert "<observation>" in data_dict["observations"]  # Check for formatted observations
    assert "text: 'sample'" in data_dict["observations"]
    
    # Mode 2: Example data + task raters
    # Skip this test for now since Mode 2 (task execution) will be implemented in Phase 5
    # When implemented, this would actually run the task on the LLM
    # TODO: Add proper test with mocking or test LLM in Phase 5
    
    # Test validation: task_raters without example_data
    task_rater = LLMClient(language_model="llama3.1")
    try:
        task._prepare_feedback_task(
            context="Test", 
            example_data=None,
            task_raters=task_rater,
            config=config
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "example_data must be provided" in str(e)