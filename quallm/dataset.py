
from typing import Union, List, Dict, Any
import pandas as pd
import numpy as np

class Dataset(List[Dict[str, str]]):
    """
    A custom list-like class for managing data to be inserted into prompts.

    This class extends the built-in list type to store and manage a collection of 
    observations, where each observation is represented as a dictionary. The keys 
    in these dictionaries correspond to data argument names, and the values are the 
    corresponding values (data) to be inserted into a prompt at inference.

    Attributes:
        data_args (List[str]): The expected argument names for each data point.

    Args:
        data: The input data. Can be a single dictionary, a string (for single-argument datasets),
              or a list/array/series of dictionaries or values.
        data_args (str or List[str]): The name(s) of the data argument(s) expected in each observation.

    Raises:
        ValueError: If the input data structure doesn't match the expected format based on data_args,
                    or if there's a mismatch between the keys in the data and the specified data_args.

    Example:
        >>> dataset = Dataset(["text1", "text2", "text3"], "input_text")
        >>> print(dataset[0])
        {'input_text': 'text1'}
    """
    def __init__(self,
                 data: Union[str, Dict[str, Any], List[Any], pd.Series, np.ndarray, List[Dict[str, Any]]],
                 data_args: Union[str, List[str]]):
        if isinstance(data_args, str):
            data_args = [data_args]
        self.data_args = data_args
        super().__init__(self.standardize_input(data))
        
    def standardize_input(self, data) -> List[Dict[str, str]]:
        """Convert input into data list of dicts with one dict per observation"""
        data_args = self.data_args
        n_expected_data_args = len(data_args)
        
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                data = data.iloc[:, 0]  # Convert single-column DataFrame to Series
            else:
                # TODO: Add functionality to handle multiple dataframe columns as separate data args
                raise ValueError(f"DataFrame input must have only one column for {n_expected_data_args} data arg(s)")

        is_list_like = isinstance(data, (list, np.ndarray, pd.Series))
        is_list_of_dicts = is_list_like and all(isinstance(item, dict) for item in data)
    
        std_data = None
        if n_expected_data_args == 1:
            arg_name = self.data_args[0]
            if isinstance(data, str):
                std_data = [{arg_name: data}]
            elif is_list_like and is_list_of_dicts:
                std_data = data
            else:
                std_data = [{arg_name: item} for item in data]
        elif isinstance(data, dict):
            std_data = [data]
        elif is_list_like:
            if is_list_of_dicts:
                std_data = data
            else:
                raise ValueError(f"For multiple data args, list-like input must contain only dictionaries")
        
        if std_data is None:
            raise ValueError(f"Unsupported data type or structure for {n_expected_data_args} data args: {type(data)}")
        
        return [self.validate_datum(item, data_args) for item in std_data]

    @staticmethod
    def validate_datum(item, data_args):
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict, got {type(item)}")
        if set(item.keys()) != set(data_args):
            raise ValueError(f"Mismatch in data arguments. Expected: {data_args}, Received: {list(item.keys())}")
        for key, value in item.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                raise ValueError(f"None or NaN value found for key '{key}'")
        return item