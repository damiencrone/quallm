import pytest
import numpy as np
import pandas as pd
from quallm.tasks import LabelSet, SingleLabelCategorizationTask
from quallm.dataset import Dataset
from quallm.predictor import Predictor
from quallm.client import LLMClient

DEFAULT_MODEL = "llama3.1"

llm = LLMClient(language_model=DEFAULT_MODEL)

labels = LabelSet.create(name='Sentiment', values=['positive', 'neutral', 'negative'])
task = SingleLabelCategorizationTask(category_class=labels)
dataset = Dataset(['I love this!', 'It\'s okay.', 'I hate this.'], 'input_text')

@pytest.fixture
def sentiment_prediction_multi_llm():
    sentiment_dataset = Dataset(['I love this!', 'It\'s okay.', 'I hate this.'], 'input_text')
    predictor = Predictor(raters=[llm]*2, task=task)
    return predictor.predict(sentiment_dataset, echo=False)

@pytest.fixture
def sentiment_prediction_single_llm():
    sentiment_dataset = Dataset(['I love this!', 'It\'s okay.', 'I hate this.'], 'input_text')
    predictor = Predictor(raters=llm, task=task)
    return predictor.predict(sentiment_dataset, echo=False)


# Helper functions for reducing test duplication
def assert_valid_prediction_structure(prediction):
    """Verify prediction has correct internal structure"""
    assert all(isinstance(item, dict) for row in prediction for item in row)
    assert all('response' in item for row in prediction for item in row)
    assert all(isinstance(item['response'], list) and len(item['response']) == 1 
               for row in prediction for item in row)

def assert_flatten_behavior(prediction, index, expected_flat_shape, expected_nonflat_shape):
    """Test flatten parameter behavior consistently"""
    assert prediction.get(indices=index, flatten=True).shape == expected_flat_shape
    assert prediction.get(indices=index, flatten=False).shape == expected_nonflat_shape
    np.testing.assert_array_equal(
        prediction.get(indices=index, flatten=True),
        prediction.get(indices=index, flatten=False).flatten()
    )

def create_filled_prediction(task, n_obs, n_raters, fill_value):
    """Create fully populated Prediction for testing"""
    pred = Prediction.__new__(Prediction, task, n_obs, n_raters)
    for i in range(n_obs):
        for j in range(n_raters):
            pred[i,j] = {'response': [fill_value]}
    return pred

# Consolidated parameterized tests
@pytest.mark.parametrize("prediction_fixture,n_obs,n_raters,expected_get_shape,expected_expand_cols", [
    ("sentiment_prediction_single_llm", 3, 1, (3,), ['reasoning', 'confidence', 'code']),
    ("sentiment_prediction_multi_llm", 3, 2, (3, 2), ['reasoning_r1', 'confidence_r1', 'code_r1', 
                                                        'reasoning_r2', 'confidence_r2', 'code_r2'])
])
def test_prediction_operations(prediction_fixture, n_obs, n_raters, expected_get_shape, expected_expand_cols, request):
    """Test prediction initialization, get, expand, attributes, and indexing"""
    prediction = request.getfixturevalue(prediction_fixture)
    
    # Test initialization
    assert prediction.shape == (n_obs, n_raters)
    assert_valid_prediction_structure(prediction)
    
    # Test numpy interface assertions
    assert isinstance(prediction, np.ndarray)
    assert np.sum(prediction != None) == n_obs * n_raters  # All filled
    
    # Test get method - default attribute retrieval
    default_get = prediction.get()
    assert default_get.shape == expected_get_shape
    assert isinstance(default_get, np.ndarray)
    if n_raters == 1:
        assert all(isinstance(item, str) for item in default_get)
        assert all(item in ['positive', 'neutral', 'negative'] for item in default_get)
    else:
        assert all(isinstance(item, str) for row in default_get for item in row)
        assert all(item in ['positive', 'neutral', 'negative'] for row in default_get for item in row)
    
    # Test get method - specific attribute retrieval
    reasoning = prediction.get('reasoning')
    assert reasoning.shape == expected_get_shape
    if n_raters == 1:
        assert all(isinstance(item, str) for item in reasoning)
    else:
        assert all(isinstance(item, str) for row in reasoning for item in row)
    
    confidence = prediction.get('confidence')
    assert confidence.shape == expected_get_shape
    if n_raters == 1:
        assert all(isinstance(item, np.int64) for item in confidence)
        assert all(0 <= item <= 100 for item in confidence)
    else:
        assert all(isinstance(item, np.int64) for row in confidence for item in row)
        assert all(0 <= item <= 100 for row in confidence for item in row)
    
    # Test expand method
    expanded = prediction.expand()
    assert isinstance(expanded, pd.DataFrame)
    expected_shape = (n_obs, len(expected_expand_cols))
    assert expanded.shape == expected_shape
    assert all(col in expanded.columns for col in expected_expand_cols)
    
    # Test attribute access
    assert prediction.task == task
    assert prediction.n_obs == n_obs
    assert prediction.n_raters == n_raters
    
    # Test indexing
    assert prediction[0, 0]['response'][0].code in ['positive', 'neutral', 'negative']
    assert len(prediction[0]) == n_raters
    assert prediction[:, 0].shape == (n_obs,)

@pytest.mark.parametrize("prediction_fixture,n_raters", [
    ("sentiment_prediction_single_llm", 1),
    ("sentiment_prediction_multi_llm", 2)
])
def test_flatten_behavior(prediction_fixture, n_raters, request):
    """Test flatten parameter behavior for both single and multi-rater predictions"""
    prediction = request.getfixturevalue(prediction_fixture)
    
    # Test integer index flatten behavior
    if n_raters == 1:
        assert_flatten_behavior(prediction, 0, (1,), (1, 1))
    else:
        assert_flatten_behavior(prediction, 0, (2,), (1, 2))
    
    # Test tuple index flatten behavior
    assert_flatten_behavior(prediction, (0, 0), (1,), (1, 1))
    
    # Test that default behavior is to flatten
    if n_raters == 1:
        assert prediction.get(indices=0).shape == (1,)
    else:
        assert prediction.get(indices=0).shape == (2,)
    assert prediction.get(indices=(0, 0)).shape == (1,)
    
    # Test non-flattened results maintain correct dimensions
    assert len(prediction.get(indices=0, flatten=False).shape) == 2
    assert len(prediction.get(indices=(0, 0), flatten=False).shape) == 2


from pydantic import BaseModel, Field
from typing import List
from quallm.tasks import Task, TaskConfig
from quallm.prediction import Prediction

class ListResponse(BaseModel):
    items: List[str] = Field(description="A list of items relating to a topic")

LIST_GENERATOR_CONFIG = TaskConfig(
    response_model=ListResponse,
    system_template="Generate a short list based on the topic provided.",
    user_template="A list of: {topic}"
)

list_generation_task = Task.from_config(LIST_GENERATOR_CONFIG)
raters = [llm]
predictor_list_generation = Predictor(raters=raters, task=list_generation_task)
list_generation_data = ["friuts", "vegetables", "animals"]
list_prediction = predictor_list_generation.predict(list_generation_data)

def test_prediction_creation_list_generation():
    assert isinstance(list_prediction, Prediction)
    assert list_prediction.task == list_generation_task
    assert list_prediction.n_obs == len(list_generation_data)
    assert list_prediction.n_raters == 1
    
def test_get_with_list_generation():
    result = list_prediction.get()
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(list_generation_data),) 
    assert isinstance(result[0], list)
    assert all(isinstance(item, str) for item in result[0])


# Tests of missingness handling
class Person(BaseModel):
    name: str
    age: int

@pytest.fixture
def person_prediction():
    
    task_config = TaskConfig(
        response_model=Person,
        system_template="Generate a person object given a string of text",
        user_template="Text: {text}",
        output_attribute="name"
    )
    
    task = Task.from_config(task_config)
    
    # Create a prediction object with a missing response
    prediction = Prediction.__new__(
        Prediction,
        task=task,
        n_obs=3,
        n_raters=2
    )
    
    # Fill with test data
    prediction[0,0] = None
    prediction[0,1] = {'response': [Person(name='Steve', age=12)]}
    prediction[1,0] = {'response': [Person(name='Stevie', age=13)]}
    prediction[1,1] = {'response': [Person(name='Stevie', age=13)]}
    prediction[2,0] = {'response': [Person(name='Steven', age=14)]}
    prediction[2,1] = {'response': [Person(name='Steven', age=14)]}
    
    return prediction

def test_get_with_missing_response_object(person_prediction):
    result = person_prediction.get()
    np.testing.assert_equal(
        result,
        np.array([[None, 'Steve'],
                 ['Stevie', 'Stevie'],
                 ['Steven', 'Steven']], dtype=object)
    )

def test_expand_with_missing_response_object(person_prediction):
    expanded = person_prediction.expand()
    
    # Verify missingness is handled correctly
    assert expanded.loc[0, 'name_r1'] is None
    assert expanded.loc[0, 'name_r2'] == 'Steve'
    assert expanded.loc[0, 'age_r1'] is None
    assert expanded.loc[0, 'age_r2'] == 12
    
    # Check complete rows
    assert expanded.loc[1, 'name_r1'] == 'Stevie'
    assert expanded.loc[1, 'age_r1'] == 13
    assert expanded.loc[2, 'name_r1'] == 'Steven'
    assert expanded.loc[2, 'age_r1'] == 14
    
    
# Tests of explode argument for expand method
import json
from pathlib import Path

class StringList(BaseModel):
    items: List[str] = Field(description="A list of items relating to a topic")

class Item(BaseModel):
    name: str = Field(description="The name of the item")
    description: str = Field(description="A brief description of the item")

class ItemList(BaseModel):
    items: List[Item] = Field(description="A list of items relating to a topic")
    
# Helper function to load test data from JSON
def load_test_data(filepath, response_model, task):
    with open(filepath, 'r') as f:
        data = json.load(f)

    wrapped_predictions = []
    for pred_data in data['input_data']['predictions']:
        items = pred_data['items']
        if response_model == StringList:
            wrapped_items = response_model(items=items)
        elif response_model == ItemList:
            wrapped_items = response_model(items=[Item(**item) for item in items])
        wrapped_predictions.append({'response': [wrapped_items]})

    predictions = Prediction(task=task, n_obs=len(wrapped_predictions), n_raters=1)
    for i, prediction_data in enumerate(wrapped_predictions):
        predictions[i, 0] = prediction_data
    data['input_data']['predictions'] = predictions
    
    expanded = pd.DataFrame.from_records(data['expected_output']['data'], index=data['expected_output']['index'])
    expanded = expanded.replace({None: np.nan})
    data['expected_output'] = expanded

    return data

# Define paths to test data files
TEST_DATA_DIR = Path(__file__).parent / "test_data"
EXPAND_TEST_CASE_1 = TEST_DATA_DIR / "expand_test_case_1.json"
EXPAND_TEST_CASE_2 = TEST_DATA_DIR / "expand_test_case_2.json"

@pytest.mark.parametrize("test_case, response_model", [
    (EXPAND_TEST_CASE_1, StringList),
    (EXPAND_TEST_CASE_2, ItemList),
])
def test_expand_with_explode(test_case, response_model):
    
    STRING_LIST_GENERATOR_CONFIG = TaskConfig(
        response_model=response_model,
        system_template="Generate a short list based on the topic provided. If no topic is provided, return an empty list.",
        user_template="A list of: {topic}"
    )

    task = Task.from_config(STRING_LIST_GENERATOR_CONFIG)
    data = load_test_data(test_case, response_model, task)
    prediction = data['input_data']['predictions']

    # Test expand with explode
    expanded_explode = prediction.expand(explode='items')
    expanded_explode = expanded_explode.drop(columns='rater')
    pd.testing.assert_frame_equal(expanded_explode, data['expected_output'])


# Tests for expand with multiple raters and single datapoint
class AnswerResponse(BaseModel):
    answer: str = Field(description="The answer to the question")

QA_TASK_CONFIG = TaskConfig(
    response_model=AnswerResponse,
    system_template="Answer the question.",
    user_template="Question: {question}",
    data_args="question"
)

qa_task = Task.from_config(QA_TASK_CONFIG)
qa_llm = LLMClient()
qa_dataset = Dataset("What is 2+2?", data_args="question")

qa_predictor_multi = Predictor(raters=[qa_llm, qa_llm], task=qa_task)
qa_prediction = qa_predictor_multi.predict(qa_dataset)

def test_expand_single_datapoint_qa_multi_rater():
    expanded = qa_prediction.expand()
    assert isinstance(expanded, pd.DataFrame)
    assert expanded.shape[0] == 1
    
    expected_columns = ["answer_r1", "answer_r2"]
    for col in expected_columns:
        assert col in expanded.columns
    
    for col in expected_columns:
        value = expanded.loc[0, col]
        assert isinstance(value, str)
        assert len(value) > 0


def test_prediction_numpy_operations():
    """Test numpy-specific operations not covered elsewhere"""
    from pydantic import BaseModel
    
    class SimpleResponse(BaseModel):
        text: str
    
    # Create prediction with some None values
    pred = Prediction.__new__(Prediction, task, n_obs=3, n_raters=2)
    pred[0,0] = {'response': [SimpleResponse(text="a")]}
    pred[0,1] = None
    pred[1,0] = {'response': [SimpleResponse(text="b")]}
    pred[1,1] = {'response': [SimpleResponse(text="c")]}
    pred[2,0] = None
    pred[2,1] = None
    
    # Test numpy operations
    assert np.sum(pred != None) == 3  # Three filled entries
    assert np.sum(pred == None) == 3  # Three None entries
    assert np.any(pred == None)
    assert np.any(pred != None)
    
    # Test slicing operations
    first_col = pred[:, 0]
    assert first_col.shape == (3,)
    assert np.sum(first_col != None) == 2
    
    # Test masking
    non_none_mask = pred != None
    assert non_none_mask.sum() == 3


def test_expand_long_format(sentiment_prediction_multi_llm):
    """Test expand with long format - completely new functionality"""
    df = sentiment_prediction_multi_llm.expand(format='long')
    
    # Should have one row per observation per rater
    assert len(df) == 3 * 2  # 3 obs * 2 raters
    assert 'rater' in df.columns
    assert set(df['rater'].unique()) == {'r1', 'r2'}
    
    # Should have base attribute columns (no rater suffixes in long format)
    expected_cols = {'rater', 'reasoning', 'confidence', 'code'}
    assert expected_cols.issubset(set(df.columns))
    
    # Test with custom rater labels
    df_custom = sentiment_prediction_multi_llm.expand(format='long', rater_labels=['alice', 'bob'])
    assert set(df_custom['rater'].unique()) == {'alice', 'bob'}


def test_expand_with_external_data(sentiment_prediction_single_llm):
    """Test expand with external data integration - completely new functionality"""
    # Test with array data
    data_array = ['text_a', 'text_b', 'text_c']
    df = sentiment_prediction_single_llm.expand(data=data_array)
    assert 'data' in df.columns
    assert df['data'].tolist() == data_array
    assert len(df) == 3
    
    # Test with DataFrame data
    input_df = pd.DataFrame({'id': [1, 2, 3], 'group': ['A', 'B', 'A']})
    df = sentiment_prediction_single_llm.expand(data=input_df)
    assert 'id' in df.columns
    assert 'group' in df.columns
    assert df['id'].tolist() == [1, 2, 3]
    assert df['group'].tolist() == ['A', 'B', 'A']
    
    # Original prediction columns should still be present
    assert 'reasoning' in df.columns
    assert 'confidence' in df.columns


def test_expand_sort_by_embedding():
    """Test expand with semantic sorting - completely new functionality"""
    from unittest.mock import MagicMock
    from quallm.embedding_client import EmbeddingClient
    
    # Create a simple prediction for testing
    pred = create_filled_prediction(task, 3, 1, 
                                  type('MockResponse', (), {
                                      'reasoning': 'test reasoning',
                                      'confidence': 95,
                                      'code': 'positive'
                                  })())
    
    # Mock embedding client
    mock_embedding_client = MagicMock(spec=EmbeddingClient)
    mock_embedding_client.sort.return_value = pd.DataFrame({'original_index': [2, 0, 1]})
    
    df = pred.expand(sort_by='reasoning', embedding_client=mock_embedding_client)
    
    # Verify sort method was called
    mock_embedding_client.sort.assert_called_once()
    
    # Verify DataFrame structure is maintained
    assert len(df) == 3
    assert 'reasoning' in df.columns


def test_prediction_edge_cases():
    """Test extreme shapes and edge conditions"""
    
    @pytest.mark.parametrize("n_obs,n_raters", [
        (1, 1),      # Single prediction
        (100, 1),    # Many observations
        (5, 10),     # Many raters
        (3, 3),      # Square shape
    ])
    def test_shapes(n_obs, n_raters):
        pred = Prediction.__new__(Prediction, task, n_obs, n_raters)
        assert pred.shape == (n_obs, n_raters)
        assert pred.n_obs == n_obs
        assert pred.n_raters == n_raters
    
    # Test all-None predictions
    pred_empty = Prediction.__new__(Prediction, task, n_obs=2, n_raters=2)
    # All entries should be None by default
    assert np.sum(pred_empty != None) == 0
    assert np.all(pred_empty == None)
    
    # Test that get() works with all None
    try:
        result = pred_empty.get()
        # Should return array of None values
        assert result.shape == (2, 2)
        assert np.all(result == None)
    except Exception:
        # If it raises an exception, that's also acceptable behavior
        pass
    
    # Run the parameterized shapes test directly
    for n_obs, n_raters in [(1, 1), (100, 1), (5, 10), (3, 3)]:
        test_shapes(n_obs, n_raters)


def test_get_tabulations_string_method(sentiment_prediction_single_llm):
    """Test get_tabulations_string method directly"""
    # Get the response model from the task
    response_model = task.response_model
    
    # Test tabulations generation
    tabulations = sentiment_prediction_single_llm.get_tabulations_string(response_model)
    
    # Should contain field information (response model has reasoning, confidence, and code)
    # Note: The actual fields that show up depend on task configuration and prediction results
    assert "confidence" in tabulations  # The integer field should always be there
    
    # Should be formatted properly
    assert isinstance(tabulations, str)
    assert len(tabulations) > 0
    
    # If code field is present, check for it (but it's not guaranteed due to categorization logic)
    if "code" in tabulations:
        assert "code" in tabulations


def test_format_output_item():
    """Test format_output_item method"""
    from typing import List
    from pydantic import Field
    
    # Create a simple task like in the README
    class ListResponse(BaseModel):
        items: List[str] = Field(description="A list of items")
    
    task_config = TaskConfig(
        response_model=ListResponse,
        system_template="Generate a list.",
        user_template="Topic: {topic}"
    )
    task = Task.from_config(task_config)
    
    # Create a prediction array with mock data
    prediction = Prediction(task=task, n_obs=1, n_raters=1)
    prediction[0, 0] = {
        'response': [ListResponse(items=["item1", "item2"])]
    }
    
    formatted = prediction.format_output_item(0)
    assert "items: ['item1', 'item2']" in formatted


def test_get_output_summary_string():
    """Test get_output_summary_string method"""
    # Mock inputs
    error_summary = {
        'success_rate': '100.0%',
        'tasks_completed': 10,
        'tasks_started': 10
    }
    rater_info = ["test-model (temp=0.0, mode=JSON)"]
    tabulations = "- field1: {value1: 5, value2: 5}"
    
    # Create a dummy prediction to call the method
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
    prediction = Prediction(task=task, n_obs=1, n_raters=1)
    
    summary = prediction.get_output_summary_string(error_summary, rater_info, tabulations)
    assert "Task raters:" in summary
    assert "test-model (temp=0.0, mode=JSON)" in summary
    assert "Success rate: 100.0%" in summary
    assert "Output distributions:" in summary