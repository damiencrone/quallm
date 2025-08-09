import pytest
import pandas as pd
import numpy as np
from quallm.dataset import Dataset

@pytest.fixture
def test_df() -> pd.DataFrame:
    df = pd.DataFrame({
        'single_string': ["It's raining outside.", "Elephant", "Ant", "Dog", "Car", "Train"],
        'single_dict': [
            {'input_text': "It's raining outside."},
            {'input_text': "Elephant"},
            {'input_text': "Ant"},
            {'input_text': "Dog"},
            {'input_text': "Car"},
            {'input_text': "Train"}
        ],
        'multi_dict': [
            {'text': "It's raining outside.", 'location': "New York"},
            {'text': "Elephant", 'location': "Zoo"},
            {'text': "Ant", 'location': "Backyard"},
            {'text': "Dog", 'location': "London"},
            {'text': "Car", 'location': "Detroit"},
            {'text': "Train", 'location': "Tokyo"}
        ],
        'with_nans': ["Text", np.nan, "More text", np.nan, "Last text", np.nan],
        'with_nones': ["Text", None, "More text", None, "Last text", None],
        'with_empty_strings': ["Text", "", "More text", "", "Last text", ""],
        'all_nans': [np.nan] * 6,
        'all_nones': [None] * 6,
        'all_empty_strings': [""] * 6,
        'mixed_nulls': ["Text", np.nan, None, "", np.nan, None],
        'dict_with_nones': [
            {'text': "Text", 'value': 1},
            {'text': None, 'value': 2},
            {'text': "More text", 'value': None},
            {'text': None, 'value': None},
            {'text': "Last text", 'value': 5},
            {'text': "", 'value': None}
        ],
        'numeric': [1, 2, 3, 4, 5, 6],
        'mixed_types': [1, "two", 3.0, True, [5], {'six': 6}],
        'long_string': ['a' * 1000, 'b' * 2000, 'c' * 3000, 'd' * 4000, 'e' * 5000, 'f' * 6000],
        'special_chars': ["áéíóú", "ñç", "大家好", "🌟🌈", "\\n\\t", "\"quoted\""],
        'nested_dict': [
            {'a': {'b': 1}},
            {'a': {'b': 2}},
            {'a': {'b': 3}},
            {'a': {'b': 4}},
            {'a': {'b': 5}},
            {'a': {'b': 6}}
        ]
    })
    return df


# Type input type fixtures
@pytest.fixture
def single_string(test_df):
    return test_df['single_string'][0]

@pytest.fixture
def list_of_strings(test_df):
    return test_df['single_string'].tolist()

@pytest.fixture
def single_dict(test_df):
    return test_df['single_dict'][0]

@pytest.fixture
def list_of_dicts(test_df):
    return test_df['single_dict'].tolist()

@pytest.fixture
def series_of_strings(test_df):
    return test_df['single_string']

@pytest.fixture
def series_of_dicts(test_df):
    return test_df['single_dict']

@pytest.fixture
def single_column_df(test_df):
    return test_df[['single_string']]


@pytest.mark.parametrize("input_fixture, expected_length", [
    ("single_string", 1),
    ("list_of_strings", 6),
    ("single_dict", 1),
    ("list_of_dicts", 6),
    ("series_of_strings", 6),
    ("series_of_dicts", 6),
    ("single_column_df", 6),
])
def test_input_types(request, input_fixture, expected_length):
    input_data = request.getfixturevalue(input_fixture)
    dataset = Dataset(input_data, "input_text")
    assert len(dataset) == expected_length
    assert all(isinstance(item, dict) for item in dataset)
    assert all("input_text" in item for item in dataset)


# Missingness fixtures
@pytest.fixture
def with_nans(test_df):
    return test_df['with_nans']

@pytest.fixture
def with_nones(test_df):
    return test_df['with_nones']

@pytest.fixture
def with_empty_strings(test_df):
    return test_df['with_empty_strings']

@pytest.fixture
def all_nans(test_df):
    return test_df['all_nans']

@pytest.fixture
def all_nones(test_df):
    return test_df['all_nones']

@pytest.fixture
def all_empty_strings(test_df):
    return test_df['all_empty_strings']

@pytest.fixture
def mixed_nulls(test_df):
    return test_df['mixed_nulls']

@pytest.fixture
def dict_with_nones(test_df):
    return test_df['dict_with_nones']


@pytest.mark.parametrize("input_fixture, expected_behavior", [
    ("with_nans", pytest.raises(ValueError)),
    ("with_nones", pytest.raises(ValueError)),
    ("with_empty_strings", 6),  # Expected length of dataset, not error
    ("all_nans", pytest.raises(ValueError)),
    ("all_nones", pytest.raises(ValueError)),
    ("all_empty_strings", 6),  # Expected length of dataset, not error
    ("mixed_nulls", pytest.raises(ValueError)),
    ("dict_with_nones", pytest.raises(ValueError)),
])
def test_missing_data_handling(request, input_fixture, expected_behavior):
    input_data = request.getfixturevalue(input_fixture)
    
    if isinstance(expected_behavior, int):
        # We expect the Dataset to be created successfully
        dataset = Dataset(input_data, "input_text")
        assert len(dataset) == expected_behavior
        assert all(isinstance(item, dict) for item in dataset)
        assert all("input_text" in item for item in dataset)
    else:
        # We expect a ValueError to be raised
        with expected_behavior:
            Dataset(input_data, "input_text")


# Data arg handling
@pytest.fixture
def multi_dict(test_df):
    return test_df['multi_dict'].tolist()

@pytest.fixture
def series_of_multi_dicts(test_df):
    return test_df['multi_dict']

@pytest.mark.parametrize("input_fixture, data_args, expected_length", [
    ("single_dict", "input_text", 1),
    ("list_of_dicts", "input_text", 6),
    ("series_of_dicts", "input_text", 6),
    ("multi_dict", ["text", "location"], 6),
    ("series_of_multi_dicts", ["text", "location"], 6),
])
def test_data_args_handling(request, input_fixture, data_args, expected_length):
    input_data = request.getfixturevalue(input_fixture)
    dataset = Dataset(input_data, data_args)
    assert len(dataset) == expected_length
    assert all(isinstance(item, dict) for item in dataset)
    if isinstance(data_args, str):
        assert all(data_args in item for item in dataset)
    else:
        assert all(all(arg in item for arg in data_args) for item in dataset)

@pytest.mark.parametrize("input_fixture, data_args, expected_error", [
    ("multi_dict", "text", "Mismatch in data arguments"),
    ("single_dict", ["input_text", "extra_arg"], "Mismatch in data arguments"),
    ("multi_dict", ["wrong_arg1", "wrong_arg2"], "Mismatch in data arguments"),
    ("multi_dict", ["text", "missing_arg"], "Mismatch in data arguments"),
])
def test_data_args_errors(request, input_fixture, data_args, expected_error):
    input_data = request.getfixturevalue(input_fixture)
    with pytest.raises(ValueError, match=expected_error):
        Dataset(input_data, data_args)


def test_get_data_summary_string():
    """Test get_data_summary_string method"""
    data = [
        {"text": "hello", "number": 1},
        {"text": "world", "number": 2}
    ]
    dataset = Dataset(data, data_args=["text", "number"])
    summary = dataset.get_data_summary_string()
    
    assert "Data schema (2 examples)" in summary
    assert "text: type=str" in summary
    assert "number: type=int64" in summary


def test_format_data_item():
    """Test format_data_item method"""
    data = [{"text": "hello world", "number": 42}]
    dataset = Dataset(data, data_args=["text", "number"])
    formatted = dataset.format_data_item(0)
    
    assert "text: 'hello world'" in formatted
    assert "number: 42" in formatted


def test_format_observations_without_predictions():
    """Test format_observations without predictions"""
    data = [{"text": "test"}, {"text": "data"}]
    dataset = Dataset(data, data_args=["text"])
    observations = dataset.format_observations()
    
    assert "<observation>" in observations
    assert "<input>" in observations
    assert "text: 'test'" in observations or "text: 'data'" in observations
    assert "<output>" not in observations