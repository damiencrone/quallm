import pytest
from unittest.mock import MagicMock, call
import numpy as np
import threading
import time
import logging
import tempfile
import os

from quallm.predictor import Predictor
from quallm.prediction import Prediction
from quallm.client import LLMClient
from quallm.tasks import Task, TaskConfig # Using existing TaskConfig for simplicity
from quallm.dataset import Dataset
from pydantic import BaseModel
import pandas as pd

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

@pytest.fixture
def mock_llm_factory():
    """Factory for creating configured mock LLM clients"""
    def _create(model="mock", temp=0.5, delay=0.0, fail_indices=None):
        client = MagicMock()
        client.language_model = model
        client.temperature = temp
        client.mode = "JSON"
        client.role_args = {}
        
        call_count = [0]  # Use list for mutability in closure
        
        def mock_request(system_prompt, user_prompt, response_model):
            idx = call_count[0]
            call_count[0] += 1
            
            if delay:
                time.sleep(delay)
            
            if fail_indices and idx in fail_indices:
                raise Exception(f"Simulated failure at index {idx}")
            
            return SimpleResponse(text=f"{model}_response_{idx}")
        
        client.request = MagicMock(side_effect=mock_request)
        return client
    
    return _create

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


# Phase 0b Tests: Expanded Predictor Tests

def test_predict_parallel_execution_success(simple_task, mock_llm_factory):
    """Test parallel execution with variable response times and verify parallelism occurs"""
    # Create dataset with 10+ items
    data_items = [f"item{i}" for i in range(12)]
    dataset = Dataset(data=data_items, data_args='data_item')
    
    # Create mock client with variable response times (50-200ms)
    delays = [0.05, 0.1, 0.15, 0.2] * 3  # Cycle delays for variety
    mock_client = mock_llm_factory(model="test-parallel", temp=0.0, delay=0.0)
    
    # Override the request method to use variable delays
    original_request = mock_client.request.side_effect
    call_count = [0]
    
    def variable_delay_request(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        delay = delays[idx] if idx < len(delays) else 0.1
        time.sleep(delay)
        return original_request(*args, **kwargs)
    
    mock_client.request.side_effect = variable_delay_request
    
    # Create predictor and measure timing
    predictor = Predictor(task=simple_task, raters=[mock_client])
    start_time = time.time()
    
    # Execute with parallel workers
    predictions = predictor.predict(data=dataset, max_workers=4)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Assertions
    # All predictions should complete successfully (no None values)
    assert np.all(predictions != None), "Some predictions failed unexpectedly"
    
    # Results should maintain correct (observation, rater) ordering
    assert predictions.shape == (12, 1)
    # Note: In parallel execution, the response counter may not match observation order
    # Instead, verify all responses are valid and from the correct model
    for i in range(12):
        response = predictions[i, 0]['response'][0]
        assert response.text.startswith("test-parallel_response_"), f"Unexpected response format: {response.text}"
    
    # Total execution time should be less than sum of individual times (proves parallelism)
    expected_sequential_time = sum(delays)
    # Use a more generous threshold since system timing can vary
    assert total_time < expected_sequential_time * 0.9, f"Expected parallel execution to be faster than {expected_sequential_time * 0.9:.2f}s, got {total_time:.2f}s"
    
    # Debug logs should show all predictions started
    debug_logs = [log for log in predictor.logs if log['level'] == 'DEBUG' and 'Beginning prediction' in log['message']]
    assert len(debug_logs) == 12, f"Expected 12 start logs, got {len(debug_logs)}"
    
    # Verify all predictions completed - this is the main success criteria
    completion_logs = [log for log in predictor.logs if log['level'] == 'DEBUG' and 'Returning prediction' in log['message']]
    assert len(completion_logs) == 12, f"Expected 12 completion logs, got {len(completion_logs)}"


def test_predict_parallel_with_failures(simple_task, mock_llm_factory):
    """Test parallel execution with specific failure indices and verify partial success handling"""
    # Dataset of 10 items
    data_items = [f"item{i}" for i in range(10)]
    dataset = Dataset(data=data_items, data_args='data_item')
    
    # Mock LLMClient to raise Exception on indices [3, 7]
    mock_client = mock_llm_factory(
        model="test-failures", 
        temp=0.5, 
        delay=0.05,  # Small delay to make parallel execution apparent
        fail_indices=[3, 7]
    )
    
    # Create predictor and execute with parallel workers
    predictor = Predictor(task=simple_task, raters=[mock_client])
    predictions = predictor.predict(data=dataset, max_workers=4)
    
    # Assertions
    # predictions[3,0] and predictions[7,0] should be None
    assert predictions[3, 0] is None, "Expected failure at index 3"
    assert predictions[7, 0] is None, "Expected failure at index 7"
    
    # Other indices should contain valid predictions
    for i in range(10):
        if i not in [3, 7]:
            assert predictions[i, 0] is not None, f"Expected success at index {i}"
            response = predictions[i, 0]['response'][0]
            # In parallel execution, response counter might not match observation order
            assert response.text.startswith("test-failures_response_"), f"Unexpected response format: {response.text}"
    
    # ERROR logs should contain error messages (indices may vary due to parallel execution)
    error_logs = [log for log in predictor.logs if log['level'] == 'ERROR']
    
    # Should have exactly 2 error logs (for the two failures)
    assert len(error_logs) == 2, f"Expected 2 error logs, got {len(error_logs)}"
    
    # Each error should mention "Simulated failure"
    for log in error_logs:
        assert "Simulated failure" in log['message'], f"Expected simulated failure in error: {log['message']}"
    
    # Summary log should show correct success/failure counts
    summary_logs = [log for log in predictor.logs if 'predict() returned' in log.get('message', '')]
    assert len(summary_logs) >= 1, "Expected summary log"
    
    summary_msg = summary_logs[0]['message']
    # Should show 8 successful and 2 missing out of 10 total
    assert "8 successful predictions" in summary_msg
    assert "2 missing predictions" in summary_msg
    assert "10 total predictions" in summary_msg


@pytest.mark.parametrize("fill_pattern", [
    [None, "filled", None, "filled"],  # Alternating
    [None, None, None],                  # All empty
    ["filled", "filled", None],        # One missing
])
def test_predict_resumable_patterns(simple_task, mock_llm_factory, fill_pattern):
    """Test various resumption patterns (alternating, all empty, one missing)"""
    # Build Prediction with specified pattern
    n_obs = len(fill_pattern)
    n_raters = 1
    
    existing_predictions = Prediction.__new__(
        Prediction, task=simple_task, n_obs=n_obs, n_raters=n_raters
    )
    
    # Fill prediction with pattern
    filled_responses = {}
    none_count = 0
    for i, val in enumerate(fill_pattern):
        if val == "filled":
            response = SimpleResponse(text=f"pre_filled_{i}")
            existing_predictions[i, 0] = {'response': [response]}
            filled_responses[i] = response
        else:  # val is None
            existing_predictions[i, 0] = None
            none_count += 1
    
    # Create dataset matching the pattern size
    data_items = [f"item{i}" for i in range(n_obs)]
    dataset = Dataset(data=data_items, data_args='data_item')
    
    # Create mock client
    mock_client = mock_llm_factory(model="test-resume", temp=0.0)
    predictor = Predictor(task=simple_task, raters=[mock_client])
    
    # Execute prediction with existing predictions
    updated_predictions = predictor.predict(data=dataset, predictions=existing_predictions)
    
    # Assertions
    # Verify "Resuming predictions for N missing" log
    resumption_logs = [log for log in predictor.logs if 'Resuming predictions' in log.get('message', '')]
    assert len(resumption_logs) >= 1, "Expected resumption log message"
    
    resumption_msg = resumption_logs[0]['message']
    assert f"{none_count} missing observation(s)" in resumption_msg
    
    # Verify pre-filled values unchanged
    for i, val in enumerate(fill_pattern):
        if val == "filled":
            assert updated_predictions[i, 0]['response'][0] == filled_responses[i], f"Pre-filled value at index {i} was changed"
    
    # Verify LLMClient.request call count matches None count
    assert mock_client.request.call_count == none_count, f"Expected {none_count} calls, got {mock_client.request.call_count}"
    
    # Verify all predictions are now filled
    assert np.all(updated_predictions != None), "Some predictions still missing after resumption"


def test_predict_multiple_raters(simple_task, mock_llm_factory):
    """Test multiple LLMClient instances with different configurations"""
    # Create 3 LLMClient mocks with different configurations
    rater_configs = [
        ("gpt-4", 0.1),
        ("claude", 0.5), 
        ("llama", 0.9)
    ]
    
    raters = []
    for model, temp in rater_configs:
        client = mock_llm_factory(model=model, temp=temp)
        raters.append(client)
    
    # Dataset with 5 items
    data_items = [f"item{i}" for i in range(5)]
    dataset = Dataset(data=data_items, data_args='data_item')
    
    # Create predictor with multiple raters
    predictor = Predictor(task=simple_task, raters=raters)
    predictions = predictor.predict(data=dataset)
    
    # Assertions
    # predictions.shape should be (5, 3)
    assert predictions.shape == (5, 3), f"Expected shape (5, 3), got {predictions.shape}"
    
    # get_rater_info() should return list of 3 formatted strings
    rater_info = predictor.get_rater_info()
    assert len(rater_info) == 3, f"Expected 3 rater info strings, got {len(rater_info)}"
    
    # Each should contain model name, temperature, and mode
    expected_info = [
        ("gpt-4", "temp=0.1", "mode=JSON"),
        ("claude", "temp=0.5", "mode=JSON"),
        ("llama", "temp=0.9", "mode=JSON")
    ]
    
    for i, (model, temp_str, mode_str) in enumerate(expected_info):
        info = rater_info[i]
        assert model in info, f"Expected {model} in rater info: {info}"
        assert temp_str in info, f"Expected {temp_str} in rater info: {info}"
        assert mode_str in info, f"Expected {mode_str} in rater info: {info}"
    
    # Logs should show "Rater 0: gpt-4", "Rater 1: claude", etc.
    rater_logs = [log for log in predictor.logs if log['message'].startswith('Rater')]
    assert len(rater_logs) >= 3, f"Expected at least 3 rater logs, got {len(rater_logs)}"
    
    expected_rater_logs = [
        "Rater 0: gpt-4",
        "Rater 1: claude", 
        "Rater 2: llama"
    ]
    
    for expected_log in expected_rater_logs:
        found = any(expected_log in log['message'] for log in rater_logs)
        assert found, f"Expected log containing '{expected_log}' not found"
    
    # Verify all predictions are filled correctly by each rater
    for obs in range(5):
        for rater in range(3):
            assert predictions[obs, rater] is not None, f"Missing prediction at [{obs}, {rater}]"
            response = predictions[obs, rater]['response'][0]
            expected_model = rater_configs[rater][0]
            # In parallel execution, response counter might not be predictable
            assert response.text.startswith(f"{expected_model}_response_"), f"Wrong response model at [{obs}, {rater}]: {response.text}"


def test_predict_resumable_multi_rater_partial(simple_task, mock_llm_factory):
    """Test complex resumption patterns with multiple raters"""
    # Shape (4, 3) with pattern:
    # Rater 0: all filled
    # Rater 1: [filled, None, filled, None]  
    # Rater 2: all None
    n_obs, n_raters = 4, 3
    
    existing_predictions = Prediction.__new__(
        Prediction, task=simple_task, n_obs=n_obs, n_raters=n_raters
    )
    
    # Create raters
    rater_configs = [("rater0", 0.1), ("rater1", 0.5), ("rater2", 0.9)]
    raters = []
    for model, temp in rater_configs:
        client = mock_llm_factory(model=model, temp=temp)
        raters.append(client)
    
    # Fill pattern as described
    # Rater 0: all filled
    for obs in range(4):
        response = SimpleResponse(text=f"rater0_prefilled_{obs}")
        existing_predictions[obs, 0] = {'response': [response]}
    
    # Rater 1: [filled, None, filled, None]
    for obs in [0, 2]:  # Fill obs 0 and 2
        response = SimpleResponse(text=f"rater1_prefilled_{obs}")
        existing_predictions[obs, 1] = {'response': [response]}
    existing_predictions[1, 1] = None
    existing_predictions[3, 1] = None
    
    # Rater 2: all None
    for obs in range(4):
        existing_predictions[obs, 2] = None
    
    # Create dataset and predictor
    data_items = [f"item{i}" for i in range(4)]
    dataset = Dataset(data=data_items, data_args='data_item')
    
    predictor = Predictor(task=simple_task, raters=raters)
    updated_predictions = predictor.predict(data=dataset, predictions=existing_predictions)
    
    # Assertions
    # Verify LLMClient[0] never called (all were filled)
    assert raters[0].request.call_count == 0, "Rater 0 should not have been called"
    
    # Verify LLMClient[1] called 2 times (indices 1 and 3 were None)
    assert raters[1].request.call_count == 2, f"Expected rater 1 to be called 2 times, got {raters[1].request.call_count}"
    
    # Verify LLMClient[2] called 4 times (all were None)
    assert raters[2].request.call_count == 4, f"Expected rater 2 to be called 4 times, got {raters[2].request.call_count}"
    
    # Verify pre-filled values are unchanged
    # Rater 0 should be unchanged
    for obs in range(4):
        response = updated_predictions[obs, 0]['response'][0]
        assert response.text == f"rater0_prefilled_{obs}", f"Rater 0, obs {obs} was changed"
    
    # Rater 1 obs 0 and 2 should be unchanged
    for obs in [0, 2]:
        response = updated_predictions[obs, 1]['response'][0]
        assert response.text == f"rater1_prefilled_{obs}", f"Rater 1, obs {obs} was changed"
    
    # All predictions should now be filled
    assert np.all(updated_predictions != None), "Some predictions still missing after resumption"
    
    # Check resumption log mentions correct count (2 + 4 = 6 missing)
    resumption_logs = [log for log in predictor.logs if 'Resuming predictions' in log.get('message', '')]
    assert len(resumption_logs) >= 1, "Expected resumption log"
    assert "6 missing observation(s)" in resumption_logs[0]['message']


def test_echo_and_log_levels(simple_task, mock_llm_factory, capsys):
    """Test echo functionality and log level changes"""
    # Create simple dataset and mock client
    dataset = Dataset(data=['test1', 'test2'], data_args='data_item')
    mock_client = mock_llm_factory(model="echo-test", temp=0.0)
    
    # Test with echo enabled and DEBUG level
    predictor = Predictor(task=simple_task, raters=[mock_client], echo=True)
    predictor.set_echo_level(logging.DEBUG)
    
    predictor.predict(data=dataset)
    
    # Capture console output
    captured = capsys.readouterr()
    
    # Should see DEBUG logs including "Beginning prediction"
    assert "Beginning prediction" in captured.out, "Expected DEBUG logs in console output"
    assert "Returning prediction" in captured.out, "Expected completion logs in console output"
    
    # Clear console capture
    capsys.readouterr()
    
    # Change to ERROR level only
    predictor.set_echo_level(logging.ERROR)
    
    # Create a new dataset to generate more logs
    dataset2 = Dataset(data=['test3'], data_args='data_item')
    predictor.predict(data=dataset2)
    
    captured = capsys.readouterr()
    
    # Should NOT see DEBUG logs, only ERROR logs (if any)
    assert "Beginning prediction" not in captured.out, "DEBUG logs should not appear at ERROR level"
    
    # Test with echo disabled entirely
    predictor = Predictor(task=simple_task, raters=[mock_client], echo=False)
    predictor.predict(data=dataset)
    
    captured = capsys.readouterr()
    
    # Should see no DEBUG/INFO logs, only progress bars from tqdm are allowed
    lines = captured.out.split('\n')
    # Filter out progress bar lines (they contain "Predicting:")
    log_lines = [line for line in lines if line.strip() and "Predicting:" not in line]
    
    # Should have no actual log lines
    assert len(log_lines) == 0, f"Expected no log output with echo=False, got: {log_lines}"


def test_dataset_input_formats(simple_task, mock_llm_factory):
    """Test various input data formats (string, list, dict list, DataFrame, numpy)"""
    mock_client = mock_llm_factory(model="format-test", temp=0.0)
    predictor = Predictor(task=simple_task, raters=[mock_client])
    
    # Test inputs and expected shapes
    test_cases = [
        # Single string
        ("single item", (1, 1), "Single string input"),
        
        # List of strings
        (["a", "b", "c"], (3, 1), "List of strings input"),
        
        # List of dicts (matching data_args structure)
        ([{"data_item": "x"}, {"data_item": "y"}], (2, 1), "Dict list input"),
        
        # DataFrame (should extract based on data_args)
        (pd.DataFrame({"data_item": ["1", "2"]}), (2, 1), "DataFrame input"),
        
        # Numpy array
        (np.array(["item1", "item2"]), (2, 1), "Numpy array input")
    ]
    
    for data_input, expected_shape, description in test_cases:
        # Clear mock call count
        mock_client.request.reset_mock()
        
        # Make prediction
        predictions = predictor.predict(data=data_input)
        
        # Assertions
        assert predictions.shape == expected_shape, f"{description}: Expected shape {expected_shape}, got {predictions.shape}"
        
        # All predictions should be successful
        assert np.all(predictions != None), f"{description}: Some predictions failed"
        
        # Verify correct number of LLM calls were made
        expected_calls = expected_shape[0] * expected_shape[1]
        assert mock_client.request.call_count == expected_calls, f"{description}: Expected {expected_calls} calls, got {mock_client.request.call_count}"
        
        # Verify responses are valid SimpleResponse objects
        for i in range(expected_shape[0]):
            response = predictions[i, 0]['response'][0]
            assert isinstance(response, SimpleResponse), f"{description}: Expected SimpleResponse object"
            assert response.text.startswith("format-test_response_"), f"{description}: Unexpected response text: {response.text}"


def test_log_management_methods(simple_task, mock_llm_factory, capsys):
    """Test log utility methods (print_logs, clear_logs, logs_df, dump_logs)"""
    # Generate mix of success/failure predictions
    success_client = mock_llm_factory(model="success", temp=0.0)
    failure_client = mock_llm_factory(model="failure", temp=0.0, fail_indices=[0, 2])
    
    predictor = Predictor(task=simple_task, raters=[success_client])
    
    # Generate some logs with mixed success/failure
    dataset = Dataset(data=['test1', 'test2', 'test3'], data_args='data_item')
    
    # First do successful predictions
    predictor.predict(data=dataset)
    
    # Now simulate some failures by using failure client
    predictor.raters = [failure_client]
    try:
        predictor.predict(data=dataset)
    except:
        pass  # We expect some failures
    
    # Test print_logs(raw=True) - should output timestamp|level|message format
    predictor.print_logs(raw=True)
    captured = capsys.readouterr()
    
    # Should contain formatted log lines with timestamps and levels
    assert any("|INFO|" in line for line in captured.out.split('\n')), "Expected INFO logs with pipe formatting"
    assert any("|DEBUG|" in line for line in captured.out.split('\n')), "Expected DEBUG logs with pipe formatting"
    
    # Test print_logs(raw=False) - should output just messages
    predictor.print_logs(raw=False)
    captured = capsys.readouterr()
    
    # Should NOT contain pipe formatting, just messages
    assert "|INFO|" not in captured.out, "Raw=False should not contain pipe formatting"
    assert "|DEBUG|" not in captured.out, "Raw=False should not contain pipe formatting"
    assert len(captured.out.strip()) > 0, "Should have some output"
    
    # Test logs_df() - should have expected columns
    df = predictor.logs_df()
    
    expected_columns = ['timestamp', 'level', 'message', 'logger', 'func', 'line']
    for col in expected_columns:
        assert col in df.columns, f"Expected column '{col}' in logs DataFrame"
    
    # Should have rows
    assert len(df) > 0, "logs_df should contain log entries"
    
    # Should have various log levels
    log_levels = set(df['level'].unique())
    assert 'INFO' in log_levels, "Expected INFO level logs"
    
    # Test clear_logs() - should result in empty logs list
    initial_log_count = len(predictor.logs)
    assert initial_log_count > 0, "Should have logs before clearing"
    
    predictor.clear_logs()
    after_clear_count = len(predictor.logs)
    assert after_clear_count == 0, "Should have no logs after clearing"
    
    # Test dump_logs("test.json") - should create valid JSON file
    # First generate some logs again
    predictor.predict(data=['test'], max_workers=1)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_file_path = f.name
    
    try:
        predictor.dump_logs(test_file_path)
        
        # Verify file was created and contains valid JSON
        assert os.path.exists(test_file_path), "dump_logs should create file"
        
        import json
        with open(test_file_path, 'r') as f:
            loaded_logs = json.load(f)
        
        assert isinstance(loaded_logs, list), "Dumped logs should be a list"
        assert len(loaded_logs) > 0, "Dumped logs should contain entries"
        
        # Check structure of first log entry
        if loaded_logs:
            first_log = loaded_logs[0]
            assert isinstance(first_log, dict), "Log entries should be dictionaries"
            assert 'message' in first_log, "Log entries should have message field"
            assert 'level' in first_log, "Log entries should have level field"
            
    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)


def test_get_error_summary_with_various_errors(simple_task):
    """Test error categorization with various error types"""
    # Create mock client that will generate different types of errors
    mock_client = MagicMock()
    mock_client.language_model = "error-test"
    mock_client.temperature = 0.0
    mock_client.mode = "JSON"
    mock_client.role_args = {}
    
    # Define error types and which call indices should trigger them
    error_schedule = [
        None,  # Success (index 0)
        ValueError("JSON decode error occurred"),  # parsing error (index 1)
        ConnectionError("Connection refused by server"),  # connection error (index 2)
        None,  # Success (index 3) 
        ValueError("Validation failed for response"),  # validation error (index 4)
        TimeoutError("Request timed out after 30 seconds"),  # timeout error (index 5)
        None,  # Success (index 6)
        Exception("Some other generic error"),  # other error (index 7)
        ValueError("Another JSON parsing issue"),  # parsing error (index 8)
        ConnectionError("Network connection lost"),  # connection error (index 9)
    ]
    
    call_count = [0]
    
    def mock_request(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        
        if idx < len(error_schedule) and error_schedule[idx] is not None:
            raise error_schedule[idx]
        
        return SimpleResponse(text=f"success_response_{idx}")
    
    mock_client.request = MagicMock(side_effect=mock_request)
    
    # Create predictor and run predictions
    predictor = Predictor(task=simple_task, raters=[mock_client])
    dataset = Dataset(data=[f"item{i}" for i in range(10)], data_args='data_item')
    
    predictions = predictor.predict(data=dataset)
    
    # Get error summary
    summary = predictor.get_error_summary()
    
    # Expected: 2 ValidationErrors, 2 ConnectionErrors, 1 timeout, 1 other
    # Success: 4 out of 10
    
    # Print actual summary for debugging
    print(f"\nActual summary: {summary}")
    
    # Verify that we have some errors
    assert summary["total_errors"] >= 4, f"Expected at least 4 errors, got {summary['total_errors']}"
    
    # Check specific error categories exist
    error_categories = summary["error_categories"]
    assert "validation" in error_categories or "parsing" in error_categories, "Expected validation/parsing errors"
    assert "connection" in error_categories, "Expected connection errors"
    
    # Verify basic structure
    assert "success_rate" in summary, "Expected success_rate in summary"
    assert "tasks_started" in summary, "Expected tasks_started in summary"  
    assert "tasks_completed" in summary, "Expected tasks_completed in summary"
    
    # Verify predictions match expected success/failure pattern
    for i, expected_error in enumerate(error_schedule):
        if expected_error is None:
            assert predictions[i, 0] is not None, f"Expected success at index {i}"
        else:
            assert predictions[i, 0] is None, f"Expected failure at index {i}"


def test_validate_existing_predictions_errors(simple_task, mock_llm_factory):
    """Test validation errors for malformed prediction inputs"""
    mock_client = mock_llm_factory(model="validation-test", temp=0.0)
    predictor = Predictor(task=simple_task, raters=[mock_client])
    
    # Create test dataset
    dataset = Dataset(data=['item1', 'item2'], data_args='data_item')
    
    # Test 1: Wrong shape prediction
    wrong_shape_pred = Prediction.__new__(
        Prediction, task=simple_task, n_obs=3, n_raters=2  # Wrong dimensions
    )
    
    with pytest.raises(AssertionError, match="Shape.*does not match"):
        predictor.predict(data=dataset, predictions=wrong_shape_pred)
    
    # Test 2: Not a Prediction instance
    fake_prediction = [[None, None], [None, None]]  # Regular list, not Prediction
    
    with pytest.raises(AssertionError, match="must be an instance of Prediction"):
        predictor.predict(data=dataset, predictions=fake_prediction)
    
    # Test 3: All predictions already filled (no work to do)
    fully_filled_pred = Prediction.__new__(
        Prediction, task=simple_task, n_obs=2, n_raters=1
    )
    
    # Fill all predictions
    for i in range(2):
        response = SimpleResponse(text=f"filled_{i}")
        fully_filled_pred[i, 0] = {'response': [response]}
    
    with pytest.raises(AssertionError, match="All predictions have already been made"):
        predictor.predict(data=dataset, predictions=fully_filled_pred)
    
    # Test 4: Verify that valid partially filled predictions work (positive test)
    valid_partial_pred = Prediction.__new__(
        Prediction, task=simple_task, n_obs=2, n_raters=1  
    )
    
    # Fill only first prediction, leave second as None
    response = SimpleResponse(text="pre_filled")
    valid_partial_pred[0, 0] = {'response': [response]}
    valid_partial_pred[1, 0] = None  # This will need to be predicted
    
    # This should work without raising an exception
    result = predictor.predict(data=dataset, predictions=valid_partial_pred)
    
    # Verify it completed successfully
    assert result.shape == (2, 1)
    assert np.sum(result != None) == 2  # Both should now be filled
    assert result[0, 0]['response'][0] == response  # First should be unchanged
    assert result[1, 0] is not None  # Second should now be filled