import pytest
from unittest.mock import MagicMock, call
import numpy as np

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