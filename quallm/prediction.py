
import pandas as pd
import numpy as np


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
        attr_type = self.task.response_model.model_fields[attribute].annotation
        is_list_type = attr_type is list
        
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
        
        if not flatten:
            result = result.reshape(1, -1)
        else:
            is_1d = len(result) == 1 or min(result.shape) == 1
            if flatten and is_1d:
                result = result.flatten()
        
        return result
    
    def expand(self, suffix: list = None, data=None):
        """
        Convert the prediction array into a pandas DataFrame.

        This method expands the prediction array into a detailed DataFrame, including
        all attributes of the response model for each rater.

        Args:
            suffix (list, optional): List of suffixes to use for each rater's columns. 
                                     If None, uses '_r1', '_r2', etc. Defaults to None.
            data (array-like, optional): Additional data to include in the DataFrame. 
                                         If provided, adds a 'data' column. Defaults to None.

        Returns:
            pandas.DataFrame: A DataFrame containing expanded prediction data with one row per
                              observation and one column per response model attribute (per rater).

        Note:
            The resulting DataFrame will have columns for each attribute in the response model,
            suffixed by the rater number or provided suffix. If there's only one rater,
            no suffix is added to the column names.
        """
        attributes = self.task.response_model.model_fields.keys()
        result_data = {}

        if data is not None:
            result_data['data'] = data

        for attr in attributes:
            values = self.get(attr)
            if self.n_raters > 1:
                for rater in range(self.n_raters):
                    if isinstance(suffix, list):
                        column_name = f"{attr}_{suffix[rater]}"
                    elif suffix is None:
                        column_name = f"{attr}_r{rater+1}"
                    else:
                        raise TypeError(f"Unexpected type for suffix: {type(suffix)}")
                    result_data[column_name] = values[:, rater]
            else:
                result_data[attr] = values.flatten()

        return pd.DataFrame(result_data)

    def code_reliability(self):
        # TODO (later): Compute reliability metrics based on the results when n_raters > 1
        raise NotImplementedError("The code_reliability method has not been implemented yet.")