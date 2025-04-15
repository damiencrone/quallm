import pandas as pd
import numpy as np
from typing import Union, List, Optional
from .embedding_client import EmbeddingClient
import warnings


class Prediction(np.ndarray):
    """
    A custom numpy array class for storing and managing predictions from LLM-assisted content analysis tasks.

    This class extends numpy.ndarray to provide additional functionality for handling
    predictions, including methods for data extraction, expansion into a DataFrame,
    and reliability analysis.

    Attributes:
        task: The content analysis task associated with this prediction.
        n_obs (int): The number of observations (data points) in the prediction.
        n_raters (int): The number of raters (LLMs) used for each prediction.

    Methods:
        get: Extract specific attributes from the prediction array.
        expand: Convert the prediction array into a pandas DataFrame.
        code_reliability: Compute reliability metrics for the predictions.

    Note:
        This class is designed to work with the Predictor class and various Task subclasses
        in the LLM-assisted content analysis framework.
    """
    def __new__(cls, task, n_obs, n_raters, dtype=object):
        obj = super(Prediction, cls).__new__(cls, (n_obs, n_raters), dtype=dtype)
        obj.task = task
        obj.n_obs = n_obs
        obj.n_raters = n_raters
        return obj
    
    @staticmethod
    def _extract_value(item, attribute):
        if item is None:
            return None
        val = getattr(item['response'][0], attribute)
        if hasattr(val, 'value'):
            return val.value
        else:
            return val

    def get(self, attribute=None, indices=None, flatten=True):
        """
        Extract specific attributes from the prediction array.

        Args:
            attribute (str, optional): The name of the attribute to extract. If None, 
                                    uses the task's output_attribute. Defaults to None.
            indices (int, tuple, or optional): Indices of the predictions to extract.
                                               If None, extracts from all predictions.
                                               Defaults to None.
            flatten (bool, optional): Whether to flatten the output for single observations.
                                    Defaults to True.

        Returns:
            numpy.ndarray: An array of extracted values.

        Note:
            The shape of the returned array depends on the provided indices and the flatten parameter:
            - If indices is None: shape is always (n_obs, n_raters)
            - If indices is an int and flatten is True: shape is (n_raters,)
            - If indices is an int and flatten is False: shape is (1, n_raters)
            - If indices is a tuple and flatten is True: shape is (1,)
            - If indices is a tuple and flatten is False: shape is (1, 1)

        Example:
            >>> prediction.get('confidence', indices=0, flatten=True)
            array([95, 87])
            >>> prediction.get('confidence', indices=0, flatten=False)
            array([[95, 87]])
        """
        if attribute is None:
            attribute = self.task.output_attribute
            
        if indices is None:
            output_data = self
        else:
            output_data = self[indices]
    
        # Determine if the attribute is a list-type or scalar
        is_list_type = self.task.is_attribute_list(attribute)
        
        if isinstance(output_data, dict):
            # Single item case
            result = np.array([[self._extract_value(output_data, attribute)]])
        elif isinstance(output_data, np.ndarray):
            if is_list_type:
                # For list-type attributes, preserve the lists
                result = np.empty(output_data.shape, dtype=object)
                for idx, item in np.ndenumerate(output_data):
                    result[idx] = self._extract_value(item, attribute)
            else:
                # For scalar attributes
                result = np.array([self._extract_value(item, attribute) for item in output_data.ravel()]).reshape(output_data.shape)
        else:
            raise TypeError(f"Unexpected type for output data: {type(output_data)}")
        
        # Only reshape if a specific index has been provided.
        if indices is not None:
            if not flatten:
                # If a single item was selected, reshape accordingly
                if isinstance(indices, int):
                    result = result.reshape(1, -1)
                elif isinstance(indices, tuple):
                    # Depending on tuple length we assume a single item is selected.
                    result = result.reshape(1, 1)
        else:
            # If indices is None, leave the result's shape intact.
            pass

        if flatten:
            # Only flatten if the result is effectively one-dimensional.
            is_1d = (result.ndim == 1) or (min(result.shape) == 1)
            if is_1d:
                result = result.flatten()
        
        return result
    
    def expand(self,
               rater_labels: list = None,
               data=None,
               format: str = None,
               explode: str = None,
               sort_by: Optional[Union[str, List[str]]] = None,
               embedding_client: Optional[EmbeddingClient] = None) -> pd.DataFrame:
        """
        Convert the prediction array into a pandas DataFrame.

        This method expands the prediction array into a detailed DataFrame, including
        all attributes of the response model for each rater.

        Args:
            rater_labels (list, optional): List of labels to use for each rater's columns.
                                        If None, uses '_r1', '_r2', etc. Defaults to None.
            data (array-like or pandas.DataFrame, optional): Additional data to include in the DataFrame.
                                        If an array-like object is provided, it's added as a single 'data' column.
                                        If a pandas.DataFrame is provided, its columns are directly merged into the 
                                        resulting DataFrame. Defaults to None.
            format (str, optional): Output format, either 'wide' or 'long'. Defaults to 'wide' unless explode is specified.
            explode (str, optional): Name of list-like response model attribute to explode into multiple
                                    rows. If the attribute is a list of Pydantic models, additional 
                                    columns will be created for each field in the models, and the original
                                    'explode' column will be dropped. Defaults to None.

        Returns:
            pandas.DataFrame: A DataFrame containing expanded prediction data. The structure depends on the format:
                - In 'wide' format: Each rater has a column for each attribute in the response model.
                If there's only one rater, no rater labels are added to the column names.
                - In 'long' format: Contains one row per observation per rater, with an additional 'rater' column.
                If an 'explode' argument is provided, the DataFrame will contain one row per item in each 
                exploded list *per observation* *per rater*.
                - If the optional data arugment is provided, the DataFrame will contain the additional data columns
                prepended to the result.

        Raises:
            ValueError: If input arguments are invalid (e.g., `rater_labels` is not a list,
                    number of rater labels does not match number of raters, `data` length does not
                    match the number of observations, `explode` is not a response model attribute
                    or is not a list-like attribute, exploding causes name collisions, or
                    'wide' format is specified with explode).
        """
        attributes = self.task.response_model.model_fields.keys()
        data_is_dataframe = False
        
        # Perform some input validation
        # Validate rater_labels
        if rater_labels is not None:
            if not isinstance(rater_labels, list):
                raise ValueError(f"rater_labels must be a list, not {type(rater_labels)}.")
            if len(rater_labels) != self.n_raters:
                raise ValueError(f"Number of rater labels ({len(rater_labels)}) does not match number of raters ({self.n_raters}).")
        else:
            rater_labels = [f"r{i+1}" for i in range(self.n_raters)]
        # Validate data
        if data is not None:
            data_is_dataframe = isinstance(data, pd.DataFrame)
            if not data_is_dataframe and 'data' in attributes:
                raise ValueError("Cannot prepend data because 'data' is already used in the response model.")
            if self.n_obs != len(data):
                raise ValueError(f"Data length ({len(data)}) does not match number of observations ({self.n_obs}).")
        # Validate format
        if format is not None and format not in ['wide', 'long']:
            raise ValueError(f"format must be either 'wide' or 'long'; got '{format}' instead.")
        if explode is not None:
            if format == 'wide':
                raise ValueError("Cannot use 'wide' format when explode is specified")
            format = 'long' if format is None else format
        format = 'wide' if format is None else format
        # Validate explode
        if explode is not None:
            if explode not in attributes:
                raise ValueError(f"Cannot explode on '{explode}' because it is not an attribute of the response model.")
            if data_is_dataframe and explode in data.columns:
                raise ValueError(f"Cannot explode on '{explode}' because it is already a column in the provided data.")
            if not self.task.is_attribute_list(explode):
                raise ValueError(f"Cannot explode on '{explode}' because it is not a list-like attribute.")
            
        rater_result_list = self._construct_rater_result_list(attributes, format, rater_labels)

        # Prepend data (if provided)
        if data is not None:
            rater_result_list = self._prepend_data_to_rater_responses(rater_result_list, data, format)

        # Construct DataFrame
        if format == 'wide':
            result = pd.DataFrame({k: v for d in rater_result_list for k, v in d.items()})
        elif format == 'long':
            result = pd.DataFrame([
                {**rater_result, **{k: v[i] for k, v in rater_result.items() if isinstance(v, (list, np.ndarray))}}
                for rater_result in rater_result_list
                for i in range(len(next(iter([v for v in rater_result.values() if isinstance(v, (list, np.ndarray))]))))
            ])
        
        # Post-process results
        if explode is not None:
            result = self._explode_results(result, explode)
        if sort_by is not None:
            result = self._sort_expanded_results(result, sort_by, embedding_client)
        
        return result
    
    def _construct_rater_result_list(self, attributes, format, rater_labels) -> List[dict]:
        """Constructs a list containing all LLM responses for each rater."""
        rater_result_list = []
        for rater in range(self.n_raters):
            rater_result = {}
            if format == 'long':
                rater_result['rater'] = rater_labels[rater]
            for attr in attributes:
                values = self.get(attr, flatten=False)
                if self.n_raters > 1:
                    if format == 'wide':
                        column_name = f"{attr}_{rater_labels[rater]}"
                    else:  # long format
                        column_name = attr
                    rater_result[column_name] = values[:, rater]
                else:
                    rater_result[attr] = values.flatten()
            rater_result_list.append(rater_result)
        return rater_result_list
    
    def _prepend_data_to_rater_responses(self, rater_result_list: List[dict], data, format: str):
        """
        Prepends data for each rater's result based on the specified format.
        
        Args:
            rater_result_list (list): List of dictionaries containing rater results.
            data: Data to be prepended.
            format (str): 'wide' or 'long' format.
        
        Returns:
            list: Updated list of dictionaries with additional data prepended.
        """
        indices = [0] if format == 'wide' else range(len(rater_result_list))
        for idx in indices:
            new_result = {}
            if isinstance(data, pd.DataFrame):
                for col in data.columns:
                    new_result[col] = data[col].tolist()
            else:
                new_result['data'] = data
            new_result.update(rater_result_list[idx])
            rater_result_list[idx] = new_result
        return rater_result_list
    
    def _explode_results(self, result: pd.DataFrame, explode: str) -> pd.DataFrame:
        result = result.explode(explode)
        explode_is_list_of_pydantic_models = self.task.is_attribute_list_of_pydantic_models(explode)
        if explode_is_list_of_pydantic_models:
            model_fields = self.task.response_model.model_fields[explode].annotation.__args__[0].model_fields.keys()
            for field in model_fields:
                result[field] = result[explode].apply(lambda x: getattr(x, field, np.nan) if not pd.isna(x) else np.nan)
            result = result.drop(columns=[explode])
        return result
    
    def _sort_expanded_results(self,
                               result: pd.DataFrame,
                               sort_by: Union[str, List[str]],
                               embedding_client: EmbeddingClient) -> pd.DataFrame:
        """Performs semantic sorting on the expanded results based on the specified column(s)."""
        idx_df = embedding_client.sort(result[[sort_by]])
        # TODO: use idx_df to sort results
        raise NotImplementedError("The _sort_expanded_results method has not been implemented yet.")
        return result

    def code_reliability(self):
        # TODO (later): Compute reliability metrics based on the results when n_raters > 1
        raise NotImplementedError("The code_reliability method has not been implemented yet.")