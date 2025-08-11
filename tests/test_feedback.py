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


def test_feedback_mode_2_integration():
    """Test Mode 2: Example data + task execution with actual LLMs"""
    from quallm import LLMClient, Dataset
    from quallm.tasks import Task, TaskConfig
    from quallm.feedback_config import FeedbackConfig
    from pydantic import BaseModel, Field
    from enum import Enum
    import os
    
    # Skip test if no Ollama is available
    try:
        test_llm = LLMClient(language_model="olmo2", temperature=0.1)
        test_llm.test()
    except Exception:
        import pytest
        pytest.skip("Ollama not available for integration test")
    
    # Create a sentiment analysis task
    class Sentiment(str, Enum):
        positive = "positive"
        negative = "negative"
        neutral = "neutral"
    
    class SentimentResponse(BaseModel):
        sentiment: Sentiment = Field(description="The sentiment of the text")
        confidence: int = Field(description="Confidence score from 1-10", ge=1, le=10)
    
    task_config = TaskConfig(
        response_model=SentimentResponse,
        system_template="Analyze the sentiment of the provided text.",
        user_template="Text: {text}",
        data_args=["text"],
        output_attribute="sentiment"
    )
    task = Task.from_config(task_config)
    
    # Create example data
    example_data = Dataset([
        {"text": "I absolutely love this product! It's amazing!"},
        {"text": "This is terrible. I hate it."},
        {"text": "It's okay, nothing special."},
        {"text": "Best purchase ever! Highly recommend!"},
        {"text": "Worst experience of my life."}
    ], data_args=["text"])
    
    # Configure for fast testing
    config = FeedbackConfig(
        max_examples_to_process=3,  # Only process 3 examples
        max_examples_to_show=2,     # Only show 2 in output
        test_task_raters=False      # Skip rater testing for speed
    )
    
    # Test Mode 2: Get feedback with task execution
    feedback_llm = LLMClient(language_model="olmo2", temperature=0.1)
    task_llm = LLMClient(language_model="olmo2", temperature=0.1)
    
    feedback = task.feedback(
        raters=feedback_llm,
        context="Testing sentiment analysis on customer reviews",
        example_data=example_data,
        task_raters=task_llm,
        config=config
    )
    
    # Verify the feedback contains expected components
    assert "TASK DEFINITION FEEDBACK" in feedback
    assert "sentiment" in feedback.lower()  # Should analyze the sentiment field
    
    # Test the internal _prepare_feedback_task for Mode 2
    feedback_task, data_dict = task._prepare_feedback_task(
        context="Test Mode 2",
        example_data=example_data,
        task_raters=task_llm,
        config=config
    )
    
    # Verify Mode 2 data_dict has actual content (not placeholders)
    assert data_dict["data_summary"] != "None provided"
    assert data_dict["output_summary"] != "None available"
    assert data_dict["observations"] != "None available"
    
    # Check output summary has expected structure
    assert "Task raters:" in data_dict["output_summary"]
    assert "Success rate:" in data_dict["output_summary"]
    assert "Output distributions:" in data_dict["output_summary"]
    
    # Check observations have both inputs and outputs
    assert "<observation>" in data_dict["observations"]
    assert "<input>" in data_dict["observations"]
    assert "<output>" in data_dict["observations"]
    assert "sentiment:" in data_dict["observations"]
    assert "confidence:" in data_dict["observations"]