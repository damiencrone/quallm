import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from quallm.utils.instructor_mode_tester import InstructorModeTester, ModeTestResult, ModeEvaluationResults
from quallm.client import LLMClient
from quallm.tasks import Task
from quallm.prompt import Prompt
from quallm.dataset import Dataset
from pydantic import BaseModel
import instructor


class TestResponse(BaseModel):
    answer: str
    confidence: int


def test_instructor_mode_tester_initialization():
    """Test basic initialization of InstructorModeTester"""
    tester = InstructorModeTester("test-model")
    assert tester.model == "test-model"
    assert tester.base_url == "http://localhost:11434/v1"
    assert tester.api_key == "ollama"


def test_warm_up_test_with_working_client():
    """Test warm_up_test with a working client"""
    tester = InstructorModeTester("test-model")
    
    # Mock client that returns a string from test()
    mock_client = MagicMock()
    mock_client.test.return_value = "This is a test response"
    
    result = tester.warm_up_test(mock_client)
    assert result is True
    mock_client.test.assert_called_once()


def test_warm_up_test_with_failing_client():
    """Test warm_up_test with a failing client"""
    tester = InstructorModeTester("test-model")
    
    # Mock client that raises an exception
    mock_client = MagicMock()
    mock_client.test.side_effect = Exception("Connection failed")
    
    result = tester.warm_up_test(mock_client)
    assert result is False


def test_warm_up_test_with_non_string_response():
    """Test warm_up_test with client that returns non-string"""
    tester = InstructorModeTester("test-model")
    
    # Mock client that returns something other than string
    mock_client = MagicMock()
    mock_client.test.return_value = 42
    
    result = tester.warm_up_test(mock_client)
    assert result is False


def test_get_default_datasets():
    """Test that default datasets are created correctly"""
    tester = InstructorModeTester("test-model")
    datasets = tester._get_default_datasets()
    
    assert len(datasets) == 3
    assert all(isinstance(ds, Dataset) for ds in datasets)
    
    # Check that each dataset has the expected data_args
    assert datasets[0].data_args == ["question"]  # QA dataset
    assert datasets[1].data_args == ["text"]      # Classification dataset  
    assert datasets[2].data_args == ["content"]   # Nested analysis dataset


def test_get_default_tasks():
    """Test that default tasks are created correctly"""
    tester = InstructorModeTester("test-model")
    tasks = tester._get_default_tasks()
    
    assert len(tasks) == 3
    assert all(isinstance(task, Task) for task in tasks)
    
    # Check response model names
    assert tasks[0].response_model.__name__ == "BasicResponse"
    assert tasks[1].response_model.__name__ == "ClassificationResponse"
    assert tasks[2].response_model.__name__ == "NestedResponse"


def test_generate_recommendations():
    """Test recommendation generation based on validity scores"""
    tester = InstructorModeTester("test-model")
    
    # Perfect validity
    recommendations = tester._generate_recommendations(1.0, instructor.Mode.JSON)
    assert "perfectly" in recommendations[0]
    
    # High validity (now considered minor issues)
    recommendations = tester._generate_recommendations(0.98, instructor.Mode.JSON)
    assert "minor issues" in recommendations[0]
    
    # Good validity (now considered unreliable)
    recommendations = tester._generate_recommendations(0.85, instructor.Mode.JSON)
    assert "unreliable" in recommendations[0]
    
    # Partial validity (now considered mostly fails)
    recommendations = tester._generate_recommendations(0.50, instructor.Mode.JSON)
    assert "mostly fails" in recommendations[0]
    
    # No validity
    recommendations = tester._generate_recommendations(0.0, instructor.Mode.JSON)
    assert "does not work" in recommendations[0]


def test_mode_test_result_to_df():
    """Test ModeTestResult to_df conversion"""
    from quallm.utils.instructor_mode_tester import TestObservation
    
    observations = [
        TestObservation(
            task_index=0,
            task_display_name="BasicResponse",
            observation_num=0,
            success=True,
            response_time=0.12,
            error_type=None,
            error_message=None
        ),
        TestObservation(
            task_index=0,
            task_display_name="BasicResponse", 
            observation_num=1,
            success=False,
            response_time=0.25,
            error_type="ValidationError",
            error_message="Field required"
        )
    ]
    
    result = ModeTestResult(
        mode_name="JSON",
        works=True,
        results_by_task={},
        overall_validity=0.5,
        overall_accuracy=0.0,
        overall_avg_response_time=0.18,
        validity_median=0.5,
        response_time_median=0.18,
        response_time_min=0.12,
        response_time_max=0.25,
        error_types=[],
        recommendations=[],
        raw_observations=observations
    )
    
    df = result.to_df()
    assert len(df) == 2
    assert list(df.columns) == [
        'mode', 'task_index', 'task_display_name', 'observation_num', 'success', 
        'response_time', 'error_type', 'error_message'
    ]
    assert df.iloc[0]['success'] == True
    assert df.iloc[1]['success'] == False
    assert df.iloc[1]['error_type'] == "ValidationError"


def test_mode_evaluation_results():
    """Test ModeEvaluationResults container functionality"""
    # Create mock results
    mode_results = {
        "JSON": ModeTestResult(
            mode_name="JSON",
            works=True,
            results_by_task={},
            overall_validity=0.95,
            overall_accuracy=0.0,
            overall_avg_response_time=0.20,
            validity_median=1.0,
            response_time_median=0.19,
            response_time_min=0.12,
            response_time_max=0.35,
            error_types=[],
            recommendations=[],
            raw_observations=[]
        ),
        "MD_JSON": ModeTestResult(
            mode_name="MD_JSON", 
            works=True,
            results_by_task={},
            overall_validity=0.85,
            overall_accuracy=0.0,
            overall_avg_response_time=0.25,
            validity_median=1.0,
            response_time_median=0.24,
            response_time_min=0.15,
            response_time_max=0.40,
            error_types=[],
            recommendations=[],
            raw_observations=[]
        ),
        "TOOLS": ModeTestResult(
            mode_name="TOOLS",
            works=False,
            results_by_task={},
            overall_validity=0.0,
            overall_accuracy=0.0,
            overall_avg_response_time=0.0,
            validity_median=0.0,
            response_time_median=0.0,
            response_time_min=0.0,
            response_time_max=0.0,
            error_types=["Not supported"],
            recommendations=[],
            raw_observations=[]
        )
    }
    
    results = ModeEvaluationResults(mode_results, "test-model")
    
    # Test working modes
    working = results.get_working_modes()
    assert len(working) == 2
    assert "JSON" in working
    assert "MD_JSON" in working
    assert "TOOLS" not in working
    
    # Test recommended mode (should be JSON with higher validity)
    recommended = results.get_recommended_mode()
    assert recommended == "JSON"
    
    # Test summary
    summary = results.summary()
    assert "test-model" in summary
    assert "JSON" in summary
    assert "MD_JSON" in summary


@patch('quallm.utils.instructor_mode_tester.instructor.from_openai')
def test_create_llm_client_success(mock_instructor):
    """Test successful LLM client creation"""
    tester = InstructorModeTester("test-model")
    
    # Mock instructor client
    mock_client = MagicMock()
    mock_instructor.return_value = mock_client
    
    result = tester._create_llm_client(instructor.Mode.JSON)
    
    assert result is not None
    assert isinstance(result, LLMClient)
    mock_instructor.assert_called_once()


@patch('quallm.utils.instructor_mode_tester.instructor.from_openai')
def test_create_llm_client_failure(mock_instructor):
    """Test LLM client creation failure"""
    tester = InstructorModeTester("test-model")
    
    # Mock instructor to raise exception
    mock_instructor.side_effect = Exception("Connection failed")
    
    result = tester._create_llm_client(instructor.Mode.JSON)
    
    assert result is None


def test_find_recommended_mode_calls_evaluate_modes():
    """Test that find_recommended_mode calls evaluate_modes class method"""
    tester = InstructorModeTester("test-model")
    
    # Mock the class method
    with patch.object(InstructorModeTester, 'evaluate_modes') as mock_evaluate:
        mock_results = MagicMock()
        mock_results.get_recommended_mode.return_value = "JSON"
        mock_evaluate.return_value = mock_results
        
        result = tester.find_recommended_mode(echo=False)
        
        assert result == "JSON"
        mock_evaluate.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            include_default_tasks=True,
            custom_tasks=None,
            observations_per_task=20,
            echo=False,
            max_workers=1
        )


def test_end_to_end_mode_detection_with_mock():
    """End-to-end test of mode detection with mocked LLM responses"""
    
    # Mock the entire LLM interaction chain
    with patch('quallm.instructor_mode_tester.instructor.from_openai') as mock_instructor:
        # Mock instructor client
        mock_instructor_client = MagicMock()
        mock_instructor.return_value = mock_instructor_client
        
        # Mock LLMClient.test() method
        with patch.object(LLMClient, 'test') as mock_test:
            mock_test.return_value = "Test response"
            
            # Mock LLMClient.request() method  
            with patch.object(LLMClient, 'request') as mock_request:
                mock_request.return_value = TestResponse(answer="Test answer", confidence=8)
                
                # Run evaluation with minimal observations
                results = InstructorModeTester.evaluate_modes(
                    model="test-model",
                    observations_per_task=2,  # Minimal for testing
                    echo=False
                )
                
                # Verify results structure
                assert isinstance(results, ModeEvaluationResults)
                assert len(results.mode_results) == 4  # All four modes tested
                
                # Check that all modes were attempted (even if some failed)
                mode_names = set(results.mode_results.keys())
                expected_modes = {"JSON", "MD_JSON", "JSON_SCHEMA", "TOOLS"}
                assert mode_names == expected_modes
                
                # Get recommended mode
                recommended = results.get_recommended_mode()
                assert recommended is not None or len(results.get_working_modes()) == 0
                
                print(f"Test completed successfully. Recommended mode: {recommended}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])