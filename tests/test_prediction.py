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

predictor_multi_llm = Predictor(raters=[llm]*2, task=task)
sentiment_prediction_multi_llm = predictor_multi_llm.predict(dataset)

predictor_single_llm = Predictor(raters=llm, task=task)
sentiment_prediction_single_llm = predictor_single_llm.predict(dataset)


def test_multi_llm_prediction_initialization():
    assert sentiment_prediction_multi_llm.shape == (3, 2)
    assert all(isinstance(item, dict) for row in sentiment_prediction_multi_llm for item in row)
    assert all('response' in item for row in sentiment_prediction_multi_llm for item in row)
    assert all(isinstance(item['response'], list) and len(item['response']) == 1 
               for row in sentiment_prediction_multi_llm for item in row)

def test_multi_llm_get_method():
    # Default attribute retrieval
    default_get = sentiment_prediction_multi_llm.get()
    assert default_get.shape == (3, 2)
    assert isinstance(default_get, np.ndarray)
    assert all(isinstance(item, str) for row in default_get for item in row)
    assert all(item in ['positive', 'neutral', 'negative'] for row in default_get for item in row)

    # Specific attribute retrieval
    reasoning = sentiment_prediction_multi_llm.get('reasoning')
    assert reasoning.shape == (3, 2)
    assert all(isinstance(item, str) for row in reasoning for item in row)

    confidence = sentiment_prediction_multi_llm.get('confidence')
    assert confidence.shape == (3, 2)
    assert all(isinstance(item, np.int64) for row in confidence for item in row)
    assert all(0 <= item <= 100 for row in confidence for item in row)

    # Index-based retrieval
    assert sentiment_prediction_multi_llm.get(indices=0).shape == (2,)
    assert sentiment_prediction_multi_llm.get(indices=(0, 0)).shape == (1,)
    assert isinstance(sentiment_prediction_multi_llm.get(indices=(0, 0))[0], str)
    assert sentiment_prediction_multi_llm.get(indices=slice(0, 2)).shape == (2, 2)
    
    # Test integer index with flatten=True and flatten=False
    assert sentiment_prediction_multi_llm.get(indices=0, flatten=True).shape == (2,)
    assert sentiment_prediction_multi_llm.get(indices=0, flatten=False).shape == (1, 2)

    # Test tuple index with flatten=True and flatten=False
    assert sentiment_prediction_multi_llm.get(indices=(0, 0), flatten=True).shape == (1,)
    assert sentiment_prediction_multi_llm.get(indices=(0, 0), flatten=False).shape == (1, 1)

    # Test that the default behavior is to flatten
    assert sentiment_prediction_multi_llm.get(indices=0).shape == (2,)
    assert sentiment_prediction_multi_llm.get(indices=(0, 0)).shape == (1,)

    # Test that non-flattened results maintain the correct number of dimensions
    assert len(sentiment_prediction_multi_llm.get(indices=0, flatten=False).shape) == 2
    assert len(sentiment_prediction_multi_llm.get(indices=(0, 0), flatten=False).shape) == 2

    # Test that the values are the same regardless of flattening
    np.testing.assert_array_equal(
        sentiment_prediction_multi_llm.get(indices=0, flatten=True),
        sentiment_prediction_multi_llm.get(indices=0, flatten=False).flatten()
    )

def test_multi_llm_expand_method():
    expanded = sentiment_prediction_multi_llm.expand()
    assert isinstance(expanded, pd.DataFrame)
    assert expanded.shape == (3, 6)
    expected_columns = ['reasoning_r1', 'confidence_r1', 'code_r1', 
                        'reasoning_r2', 'confidence_r2', 'code_r2']
    assert all(col in expanded.columns for col in expected_columns)
    assert expanded['reasoning_r1'].dtype == object
    assert expanded['confidence_r1'].dtype == np.int64
    assert expanded['code_r1'].dtype == object
    assert expanded['reasoning_r2'].dtype == object
    assert expanded['confidence_r2'].dtype == np.int64
    assert expanded['code_r2'].dtype == object

def test_multi_llm_attribute_access():
    assert sentiment_prediction_multi_llm.task == task
    assert sentiment_prediction_multi_llm.n_obs == 3
    assert sentiment_prediction_multi_llm.n_raters == 2

def test_multi_llm_indexing():
    assert sentiment_prediction_multi_llm[0, 0]['response'][0].code in ['positive', 'neutral', 'negative']
    assert len(sentiment_prediction_multi_llm[0]) == 2
    assert sentiment_prediction_multi_llm[:, 0].shape == (3,)
    assert sentiment_prediction_multi_llm[0:2, 0:2].shape == (2, 2)
    


def test_single_llm_prediction_initialization():
    assert sentiment_prediction_single_llm.shape == (3, 1)
    assert all(isinstance(item, dict) for row in sentiment_prediction_single_llm for item in row)
    assert all('response' in item for row in sentiment_prediction_single_llm for item in row)
    assert all(isinstance(item['response'], list) and len(item['response']) == 1 
               for row in sentiment_prediction_single_llm for item in row)

def test_single_llm_get_method():
    # Default attribute retrieval
    default_get = sentiment_prediction_single_llm.get()
    assert default_get.shape == (3,)
    assert isinstance(default_get, np.ndarray)
    assert all(isinstance(item, str) for item in default_get)
    assert all(item in ['positive', 'neutral', 'negative'] for item in default_get)

    # Specific attribute retrieval
    reasoning = sentiment_prediction_single_llm.get('reasoning')
    assert reasoning.shape == (3,)
    assert all(isinstance(item, str) for item in reasoning)

    confidence = sentiment_prediction_single_llm.get('confidence')
    assert confidence.shape == (3,)
    assert all(isinstance(item, np.int64) for item in confidence)
    assert all(0 <= item <= 100 for item in confidence)

    # Index-based retrieval
    assert sentiment_prediction_single_llm.get(indices=0).shape == (1,)
    assert sentiment_prediction_single_llm.get(indices=(0, 0)).shape == (1,)
    assert isinstance(sentiment_prediction_single_llm.get(indices=(0, 0))[0], str)
    
    # Test integer index with flatten=True and flatten=False
    assert sentiment_prediction_single_llm.get(indices=0, flatten=True).shape == (1,)
    assert sentiment_prediction_single_llm.get(indices=0, flatten=False).shape == (1, 1)

    # Test tuple index with flatten=True and flatten=False
    assert sentiment_prediction_single_llm.get(indices=(0, 0), flatten=True).shape == (1,)
    assert sentiment_prediction_single_llm.get(indices=(0, 0), flatten=False).shape == (1, 1)

    # Test that the default behavior is to flatten
    assert sentiment_prediction_single_llm.get(indices=0).shape == (1,)
    assert sentiment_prediction_single_llm.get(indices=(0, 0)).shape == (1,)

    # Test that non-flattened results maintain the correct number of dimensions
    assert len(sentiment_prediction_single_llm.get(indices=0, flatten=False).shape) == 2
    assert len(sentiment_prediction_single_llm.get(indices=(0, 0), flatten=False).shape) == 2

    # Test that the values are the same regardless of flattening
    np.testing.assert_array_equal(
        sentiment_prediction_single_llm.get(indices=0, flatten=True),
        sentiment_prediction_single_llm.get(indices=0, flatten=False).flatten()
    )

def test_single_llm_expand_method():
    expanded = sentiment_prediction_single_llm.expand()
    assert isinstance(expanded, pd.DataFrame)
    assert expanded.shape == (3, 3)
    expected_columns = ['reasoning', 'confidence', 'code']
    assert all(col in expanded.columns for col in expected_columns)
    assert expanded['reasoning'].dtype == object
    assert expanded['confidence'].dtype == np.int64
    assert expanded['code'].dtype == object

def test_single_llm_attribute_access():
    assert sentiment_prediction_single_llm.task == task
    assert sentiment_prediction_single_llm.n_obs == 3
    assert sentiment_prediction_single_llm.n_raters == 1

def test_single_llm_indexing():
    assert sentiment_prediction_single_llm[0, 0]['response'][0].code in ['positive', 'neutral', 'negative']
    assert len(sentiment_prediction_single_llm[0]) == 1
    assert sentiment_prediction_single_llm[:, 0].shape == (3,)
    assert sentiment_prediction_single_llm[0:2, 0].shape == (2,)
    assert isinstance(sentiment_prediction_single_llm[0:2, 0], np.ndarray)
    assert sentiment_prediction_single_llm[0:2].shape == (2, 1)
    assert isinstance(sentiment_prediction_single_llm[0:2], np.ndarray)


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