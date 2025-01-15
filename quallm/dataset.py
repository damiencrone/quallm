
from typing import Optional, Union, List, Dict, Any
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
              a list/array/series of dictionaries or values, or a DataFrame.
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
                 data: Union[str, Dict[str, Any], List[Any], pd.Series, pd.DataFrame, np.ndarray, List[Dict[str, Any]]],
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
                data = data.to_dict(orient="records")

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

    @classmethod
    def from_samples(cls, 
                    data: pd.DataFrame,
                    n_samples: int,
                    sample_size: Union[int, List[int]],
                    random_state: Optional[int] = None,
                    labels: Optional[Dict[str, str]] = None,
                    separator: str = "-----") -> 'Dataset':
        """
        Create a Dataset instance from samples of a pandas DataFrame.

        This method generates multiple samples from the input DataFrame, where each sample
        is a combination of randomly selected rows. The resulting Dataset contains one
        observation per sample, where a "sample" is the unit of analysis to be passed to the
        LLM at inference time, with all columns from the original DataFrame combined into
        a single string representation.

        Args:
            data (pd.DataFrame): The input DataFrame to sample from.
            n_samples (int): The number of samples to generate.
            sample_size (Union[int, List[int]]): The number of rows to include in each sample.
                Can be a single integer for uniform sample sizes, or a list of integers for
                variable sample sizes. When a list is provided sample sizes are drawn randomly.
            random_state (Optional[int], default=None): Seed for the random number generator.
                If provided, ensures reproducibility of samples. Each sample uses a different
                seed derived from this base value.
            labels (Optional[Dict[str, str]], default=None): A dictionary to rename columns
                in the input DataFrame (i.e., to label variables differently for when they get
                passed to the LLM). Keys are original column names, values are new names that
                the LLM will see.
            separator (str, default="-----"): The string used to separate individual rows
                within a sample.

        Returns:
            Dataset: A new Dataset instance where each observation is a sample containing
            a sample of rows from the input DataFrame. The number of rows in each sample may
            vary if a list of sample sizes is provided.

        Raises:
            ValueError: If the input is not a pandas DataFrame, or if sample_size is neither a
            positive integer nor a list of positive integers.

        Example:
            >>> df = pd.DataFrame({
            ...     'id': range(1, 101),
            ...     'response': [f"Response {i}" for i in range(1, 101)],
            ...     'sentiment': ['positive', 'negative', 'neutral'] * 33 + ['positive']
            ... })
            >>> dataset = Dataset.from_samples(
            ...     data=df,
            ...     n_samples=5,
            ...     sample_size=3,
            ...     random_state=42,
            ...     labels={'id': 'ID', 'response': 'Response', 'sentiment': 'Sentiment'}
            ... )
            >>> print(dataset[0]['sample'])
            ID: 52
            Response: Response 52
            Sentiment: positive
            -----
            ID: 93
            Response: Response 93
            Sentiment: neutral
            -----
            ID: 15
            Response: Response 15
            Sentiment: neutral
        """
        
        # Data validation
        if isinstance(data, pd.DataFrame):
            if labels:
                data = data.rename(columns=labels)
        else:
            raise ValueError("Input data must be a pandas DataFrame")

        # Sample size validation
        if isinstance(sample_size, int):
            sample_size = [sample_size]
        elif not isinstance(sample_size, list) or not all(isinstance(s, int) and s > 0 for s in sample_size):
            raise ValueError("sample_size must be a positive integer or a list of positive integers")
        
        # Create an array of sampled sample sizes
        if random_state is not None:
            np.random.seed(random_state)
        sampled_sizes = np.random.choice(sample_size, size=n_samples)

        # Generate samples
        samples = []
        for n in range(n_samples):
            if random_state is not None:
                r = random_state + n
            else:
                r = None
            
            size = sampled_sizes[n]
            sample = data.sample(n=size, replace=True, random_state=r)
            sample_str = cls._combine_columns(sample, separator)
            samples.append({"sample": sample_str})

        # Create and return a new Dataset instance
        return cls(samples, data_args=["sample"])

    @staticmethod
    def _combine_columns(sample: pd.DataFrame, separator: str) -> str:
        combined = []
        for _, row in sample.iterrows():
            row_str = "\n".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            combined.append(row_str)
        return f"\n{separator}\n".join(combined)