import pytest
from unittest.mock import MagicMock, call
import numpy as np
import threading
import time

from quallm.predictor import Predictor
from quallm.prediction import Prediction
from quallm.client import LLMClient
from quallm.tasks import Task, TaskConfig # Using existing TaskConfig for simplicity
from quallm.dataset import Dataset
from pydantic import BaseModel

# Test Setup

class SimpleResponse(BaseModel):
    text: str

# Define a simple TaskConfig
SIMPLE_TASK_CONFIG = TaskConfig(
    response_model=SimpleResponse,
    system_template="You are a test assistant.",
    user_template="Data: {data_item}",
    data_args="data_item", # Ensure this matches the prompt
    output_attribute="text" # Needs to be a valid attribute of SimpleResponse
)

@pytest.fixture
def mock_llm_client():
    client = LLMClient(language_model="mock-model")
    # We will mock the request method on the instance
    client.request = MagicMock()
    return client

@pytest.fixture
def simple_task():
    return Task.from_config(SIMPLE_TASK_CONFIG)

@pytest.fixture
def simple_dataset():
    # Corresponds to 3 observations
    return Dataset(data=['item1', 'item2', 'item3'], data_args='data_item')

@pytest.fixture
def predictor_instance(simple_task, mock_llm_client):
    # Using a list for raters, even if it's one
    return Predictor(task=simple_task, raters=[mock_llm_client])

# Test for Prediction Resumption

def test_predict_resumes_partially_filled_predictions(
    predictor_instance, mock_llm_client, simple_task, simple_dataset
):
    n_obs = len(simple_dataset)
    n_raters = predictor_instance.n_raters # Should be 1 based on fixture

    # 1. Create an initial Prediction object with some entries filled
    #    and some as None.
    #    Let's say obs 0 is done, obs 1 is missing, obs 2 is done.
    existing_predictions = Prediction.__new__(
        Prediction, task=simple_task, n_obs=n_obs, n_raters=n_raters
    )

    # Pre-fill some responses
    # For rater 0 (index 0)
    pre_filled_response_obs0 = SimpleResponse(text="response_for_item1_rater0")
    pre_filled_response_obs2 = SimpleResponse(text="response_for_item3_rater0")

    existing_predictions[0, 0] = {'response': [pre_filled_response_obs0]}
    existing_predictions[1, 0] = None # This one needs to be predicted
    existing_predictions[2, 0] = {'response': [pre_filled_response_obs2]}

    # Configure the mock LLMClient.request to return a specific response
    # when called for the missing item (item2, which is simple_dataset[1])
    mocked_response_item2 = SimpleResponse(text="mocked_response_for_item2_rater0")
    
    # The request method will be called with system_prompt, user_prompt, response_model
    # We only care that it's called for the right item, and we'll make it return our mocked response
    mock_llm_client.request.return_value = mocked_response_item2

    # 2. Call Predictor.predict() with the existing Prediction object
    updated_predictions = predictor_instance.predict(
        data=simple_dataset,
        predictions=existing_predictions
    )

    # 3. Assert Correct Behavior

    #   a) Verify LLMClient.request was called only for the initially None entry
    #      The prompt for simple_dataset[1] (which is {'data_item': 'item2'})
    #      will be formatted by task.prompt.insert_role_and_data.
    #      We need to know what that formatted prompt looks like to assert the call.

    #      Let's get the expected formatted prompt for the item that was None
    formatted_prompt_item2 = simple_task.prompt.insert_role_and_data(
        **simple_dataset[1] # simple_dataset[1] is {'data_item': 'item2'}
    )
    
    expected_calls = [
        call(
            system_prompt=formatted_prompt_item2.system_prompt,
            user_prompt=formatted_prompt_item2.user_prompt,
            response_model=simple_task.response_model
        )
    ]
    mock_llm_client.request.assert_has_calls(expected_calls, any_order=False) # Assuming order if sequential for single rater
    assert mock_llm_client.request.call_count == 1 # Only called for the missing one

    #   b) Verify that the pre-filled entries remain unchanged
    assert updated_predictions[0, 0]['response'][0] == pre_filled_response_obs0
    assert updated_predictions[2, 0]['response'][0] == pre_filled_response_obs2

    #   c) Verify that the initially None entry is now filled
    assert updated_predictions[1, 0] is not None
    assert updated_predictions[1, 0]['response'][0] == mocked_response_item2
    
    #   d) Verify that all entries are now filled
    assert np.all(updated_predictions != None)

    #   e) Verify logs
    #      Check for a log message indicating resumption.
    #      The Predictor logs "Resuming predictions for X missing observation(s)..."
    resumption_log_found = False
    for log_entry in predictor_instance.logs:
        if "Resuming predictions" in log_entry.get("message", "") and "1 missing observation(s)" in log_entry.get("message", ""):
            resumption_log_found = True
            break
    assert resumption_log_found, "Resumption log message not found or incorrect."


def test_logs_property_returns_copy(predictor_instance):
    """Test that the logs property returns a copy, not the original list"""
    logs1 = predictor_instance.logs
    logs2 = predictor_instance.logs
    
    # They should be equal in content
    assert logs1 == logs2
    
    # But different objects
    assert logs1 is not logs2
    
    # Modifying the returned list shouldn't affect the internal state
    original_length = len(logs1)
    logs1.append({"test": "entry"})
    logs3 = predictor_instance.logs
    assert len(logs3) == original_length


def test_logs_thread_safety(predictor_instance, mock_llm_client, simple_dataset):
    """Test that concurrent access to logs doesn't cause errors"""
    mock_llm_client.request.return_value = SimpleResponse(text="test")
    
    errors = []
    iterations_completed = []
    
    def read_logs_repeatedly():
        try:
            for i in range(100):
                logs = predictor_instance.logs
                # Simulate processing each log
                for log in logs:
                    _ = log.get("message", "")
                time.sleep(0.001)  # Small delay to increase chance of race conditions
            iterations_completed.append(True)
        except Exception as e:
            errors.append(e)
    
    # Start prediction in background with multiple workers
    prediction_thread = threading.Thread(
        target=lambda: predictor_instance.predict(data=simple_dataset, max_workers=4)
    )
    
    # Start multiple threads reading logs concurrently
    reader_threads = [threading.Thread(target=read_logs_repeatedly) for _ in range(5)]
    
    # Start all threads
    prediction_thread.start()
    for t in reader_threads:
        t.start()
    
    # Wait for all threads to complete
    prediction_thread.join()
    for t in reader_threads:
        t.join()
    
    # Check no errors occurred
    assert len(errors) == 0, f"Thread safety errors occurred: {errors}"
    assert len(iterations_completed) == 5, "Not all reader threads completed"


def test_get_rater_info():
    """Test get_rater_info method"""
    from pydantic import BaseModel
    
    # Create a simple task
    class SimpleResponse(BaseModel):
        answer: str
    
    config = TaskConfig(
        response_model=SimpleResponse,
        system_template="Test",
        user_template="{text}",
        data_args=["text"],
        output_attribute="answer"
    )
    task = Task.from_config(config)
    
    # Create predictor with mock clients
    rater = LLMClient(language_model="test-model", temperature=0.5)
    predictor = Predictor(task=task, raters=[rater])
    
    rater_info = predictor.get_rater_info()
    assert len(rater_info) == 1
    assert "test-model (temp=0.5, mode=" in rater_info[0]


def test_get_error_summary_empty():
    """Test get_error_summary with no errors"""
    from pydantic import BaseModel
    
    # Create a simple task
    class SimpleResponse(BaseModel):
        answer: str
    
    config = TaskConfig(
        response_model=SimpleResponse,
        system_template="Test",
        user_template="{text}",
        data_args=["text"],
        output_attribute="answer"
    )
    task = Task.from_config(config)
    
    predictor = Predictor(task=task, raters=[LLMClient()])
    summary = predictor.get_error_summary()
    
    assert summary["total_errors"] == 0
    assert summary["error_categories"] == {}


# Core predict_single Tests

def test_predict_single_basic_success(predictor_instance, mock_llm_client, simple_task):
    """Test basic successful prediction flow in predict_single()"""
    # Setup mock response
    mock_response = SimpleResponse(text="mocked_response")
    mock_llm_client.request.return_value = mock_response
    
    # Setup formatted prompt mock
    mock_prompt = MagicMock()
    mock_prompt.system_prompt = "test system prompt"
    mock_prompt.user_prompt = "test user prompt"
    
    # Call predict_single
    result = predictor_instance.predict_single((0, mock_llm_client, mock_prompt))
    
    # Verify return format
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 0  # index
    assert isinstance(result[1], dict)
    assert 'response' in result[1]
    assert result[1]['response'] == [mock_response]
    
    # Verify LLMClient was called correctly
    mock_llm_client.request.assert_called_once_with(
        system_prompt="test system prompt",
        user_prompt="test user prompt", 
        response_model=simple_task.response_model
    )
    
    # Verify timing is captured in debug logs
    debug_logs = [log for log in predictor_instance.logs if log['level'] == 'DEBUG']
    assert len(debug_logs) >= 2  # start and end logs
    
    # Check for start log
    start_log = next((log for log in debug_logs if 'Beginning prediction' in log['message']), None)
    assert start_log is not None
    assert 'Index: 0' in start_log['message']
    
    # Check for completion log with duration
    completion_log = next((log for log in debug_logs if 'Returning prediction' in log['message'] and 'Duration:' in log['message']), None)
    assert completion_log is not None
    assert 'Index: 0' in completion_log['message']
    assert 'Duration:' in completion_log['message']
    
    # Verify duration format (X.XXXs)
    import re
    duration_match = re.search(r'Duration: (\d+\.\d{3})s', completion_log['message'])
    assert duration_match is not None
    duration = float(duration_match.group(1))
    assert duration >= 0


def test_predict_single_exception_handling(predictor_instance, mock_llm_client, simple_task):
    """Test exception handling in predict_single()"""
    
    # Setup mock to raise a generic exception 
    mock_llm_client.request.side_effect = Exception("Test exception")
    
    # Setup formatted prompt mock
    mock_prompt = MagicMock()
    mock_prompt.system_prompt = "test system prompt"
    mock_prompt.user_prompt = "test user prompt"
    
    # Call predict_single
    result = predictor_instance.predict_single((0, mock_llm_client, mock_prompt))
    
    # Verify return format for failure
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 0  # index
    assert result[1] is None  # None for failure
    
    # Verify error logging occurred
    error_logs = [log for log in predictor_instance.logs if log['level'] == 'ERROR']
    assert len(error_logs) >= 1
    
    error_log = error_logs[0]
    assert 'Index: 0' in error_log['message']
    assert 'Returning None' in error_log['message']
    assert 'Duration:' in error_log['message']
    assert 'Error:' in error_log['message']
    
    # Verify timing is still captured despite failure
    import re
    duration_match = re.search(r'Duration: (\d+\.\d{3})s', error_log['message'])
    assert duration_match is not None
    duration = float(duration_match.group(1))
    assert duration >= 0


def test_predict_single_timing_accuracy(predictor_instance, mock_llm_client, simple_task):
    """Test timing accuracy in predict_single()"""
    import time
    
    # Mock LLMClient with controlled delay
    def delayed_response(*args, **kwargs):
        time.sleep(0.1)  # 100ms delay
        return SimpleResponse(text="delayed_response")
    
    mock_llm_client.request.side_effect = delayed_response
    
    # Setup formatted prompt mock
    mock_prompt = MagicMock()
    mock_prompt.system_prompt = "test system prompt"
    mock_prompt.user_prompt = "test user prompt"
    
    # Call predict_single
    result = predictor_instance.predict_single((0, mock_llm_client, mock_prompt))
    
    # Extract duration from logs
    debug_logs = [log for log in predictor_instance.logs if log['level'] == 'DEBUG' and 'Duration:' in log['message']]
    assert len(debug_logs) >= 1
    
    import re
    duration_match = re.search(r'Duration: (\d+\.\d{3})s', debug_logs[0]['message'])
    assert duration_match is not None
    duration = float(duration_match.group(1))
    
    # Assert duration is approximately 0.1 seconds (±0.01s tolerance)
    assert 0.09 <= duration <= 0.11, f"Expected duration ~0.1s, got {duration}s"
    
    # Verify timing precision to 3 decimal places
    duration_str = duration_match.group(1)
    decimal_part = duration_str.split('.')[1]
    assert len(decimal_part) == 3, f"Expected 3 decimal places, got {len(decimal_part)}"


def test_predict_single_parameter_handling(predictor_instance, simple_task):
    """Test parameter handling with different values in predict_single()"""
    # Test different index values and formatted prompts
    test_cases = [
        {"index": 0, "system": "system0", "user": "user0"},
        {"index": 1, "system": "system1", "user": "user1"}, 
        {"index": 100, "system": "system100", "user": "user100"},
    ]
    
    for i, case in enumerate(test_cases):
        # Create fresh mock client for each test case
        mock_client = MagicMock()
        mock_client.request.return_value = SimpleResponse(text=f"response_{case['index']}")
        
        # Setup formatted prompt mock
        mock_prompt = MagicMock()
        mock_prompt.system_prompt = case["system"]
        mock_prompt.user_prompt = case["user"]
        
        # Clear logs before each test
        predictor_instance.clear_logs()
        
        # Call predict_single
        result = predictor_instance.predict_single((case["index"], mock_client, mock_prompt))
        
        # Verify parameters were passed correctly to LLMClient.request()
        mock_client.request.assert_called_once_with(
            system_prompt=case["system"],
            user_prompt=case["user"],
            response_model=simple_task.response_model
        )
        
        # Verify correct index in return value
        assert result[0] == case["index"]
        
        # Verify correct index in logging
        debug_logs = [log for log in predictor_instance.logs if log['level'] == 'DEBUG']
        for log in debug_logs:
            if 'Index:' in log['message']:
                assert f"Index: {case['index']}" in log['message']


def test_predict_single_thread_safety_basic(simple_task):
    """Test basic thread safety of predict_single()"""
    import threading
    import time
    import random
    
    # Create predictor instance
    predictor = Predictor(task=simple_task, raters=[LLMClient()])
    
    results = []
    errors = []
    
    def worker_function(worker_id):
        try:
            # Create mock client with unique response time per thread
            mock_client = MagicMock()
            delay = random.uniform(0.01, 0.05)  # Random delay 10-50ms
            
            def delayed_response(*args, **kwargs):
                time.sleep(delay)
                return SimpleResponse(text=f"worker_{worker_id}_response")
            
            mock_client.request.side_effect = delayed_response
            
            # Setup formatted prompt mock
            mock_prompt = MagicMock()
            mock_prompt.system_prompt = f"system_{worker_id}"
            mock_prompt.user_prompt = f"user_{worker_id}"
            
            # Call predict_single
            result = predictor.predict_single((worker_id, mock_client, mock_prompt))
            results.append((worker_id, result))
            
        except Exception as e:
            errors.append((worker_id, e))
    
    # Create and start multiple threads
    threads = []
    num_threads = 5
    
    for i in range(num_threads):
        thread = threading.Thread(target=worker_function, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify no errors occurred
    assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    # Verify all threads completed
    assert len(results) == num_threads
    
    # Verify each thread got correct results
    for worker_id, result in results:
        assert result[0] == worker_id  # correct index
        assert result[1] is not None  # successful prediction
        assert result[1]['response'][0].text == f"worker_{worker_id}_response"
    
    # Verify no log message corruption by checking each thread's logs contain correct index
    all_logs = predictor.logs
    
    for worker_id in range(num_threads):
        # Find logs for this worker
        worker_logs = [log for log in all_logs if f"Index: {worker_id}" in log.get('message', '')]
        assert len(worker_logs) >= 2, f"Worker {worker_id} missing logs"
        
        # Verify timing logs don't cross over between threads
        for log in worker_logs:
            # Each log should only contain this worker's index
            assert f"Index: {worker_id}" in log['message']
            # Should not contain other worker indices
            for other_id in range(num_threads):
                if other_id != worker_id:
                    assert f"Index: {other_id}" not in log['message']