from .client import LLMClient
from .dataset import Dataset
from .tasks import Task
from .prediction import Prediction

from typing import List, Dict, Union
import pandas as pd
import numpy as np
import concurrent.futures
    
    
class Predictor:
    """
    A pipeline for performing LLM-assisted content analysis tasks.

    This class manages the process of applying a specific content analysis task
    to a dataset using one or more language models (raters).

    Attributes:
        raters (List[LLMClient]): The language model clients to be used for predictions.
        n_raters (int): The number of raters (language models) being used (inferred from raters).
        task (Task): The content analysis task to be performed.

    Args:
        task (Task): The content analysis task to be performed.
        raters (Union[LLMClient, List[LLMClient]]): One or more language model clients.
            Defaults to a single LLMClient instance.

    Note:
        This class is designed to work with various content analysis tasks and
        can utilize multiple language models for each prediction task.
    """
    def __init__(self,
                 task: Task,
                 raters=[LLMClient()],
                 ):
        if isinstance(raters, LLMClient):
            raters = [raters]
        self.raters = raters
        self.n_raters = len(raters)
        self.task = task


    def predict_single(self, params):
        """Generate a prediction for a single observation"""
        index, language_model, formatted_prompt = params
        try:
            response = language_model.request(
                system_prompt=formatted_prompt.system_prompt,
                user_prompt=formatted_prompt.user_prompt,
                response_model=self.task.response_model
            )
            return (index, {'response': [response]})
        except Exception as e:
            return (index, None)


    def predict(self,
                data: Union[str, Dict[str, str], List[str], pd.Series, np.ndarray, List[Dict[str, str]]],
                predictions: Prediction = None,
                max_workers: int = 1
                ) -> Prediction:
        """
        Perform predictions on the given data using the configured task and raters.

        This method processes the input data, applies the content analysis task using
        the specified language models, and returns the predictions.
        If no existing predictions are provided, performs predictions for all observations. 
        If an existing Prediction object is provided via the predictions argument, only 
        performs predictions for missing (None) values in that object, preserving all 
        existing predictions.

        Args:
            data (Union[str, Dict[str, str], List[str], pd.Series, np.ndarray, List[Dict[str, str]]):
                The input data to be analyzed. Can be in various formats, which will be
                standardized internally.
            predictions (Prediction, optional): An existing Prediction object to update.
                If provided, only missing (None) predictions will be made, preserving
                existing ones. If None, a new Prediction object will be created and
                all predictions will be made. Defaults to None.
            max_workers (int, optional): The maximum number of worker threads to use
                for parallel processing. If 1, processing is done sequentially.
                Defaults to 1.

        Returns:
            Prediction: A Prediction object containing the results of the content analysis.
                If predictions argument was provided, this will be the same object with
                missing values filled in.

        Raises:
            ValueError: If the input data format is unsupported or if there's a mismatch
                between the data and the task requirements.
            AssertionError: If the provided predictions object is invalid or incompatible.

        Note:
            This method can handle both single-item and batch predictions. It also
            supports parallel processing for improved performance with larger datasets.
        """
        if not isinstance(data, Dataset):
            standardized_data = Dataset(data, data_args=self.task.prompt.data_args)
        else:
            standardized_data = data
        if predictions is None:
            predictions = Prediction.__new__(Prediction, task=self.task, n_obs=len(standardized_data), n_raters=self.n_raters)
        else:
            self.validate_existing_predictions(predictions, standardized_data)
            print(f"Resuming predictions for {np.sum(predictions == None)} missing observation(s) out of {predictions.size} total observation(s).")
        # Prepare prediction tasks
        tasks = []
        for i, data_point in enumerate(standardized_data):
            for j in range(self.n_raters):
                if predictions[i,j] is None:  # Only create tasks for unmade predictions
                    language_model = self.raters[j]
                    formatted_prompt = self.task.prompt.insert_data(**data_point)
                    tasks.append(((i, j), language_model, formatted_prompt))
        # Execute predictions
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.predict_single, tasks))
        else:
            results = [self.predict_single(task) for task in tasks]
        # Fill in predictions
        for index, result in results:
            if result is not None:
                predictions[index] = result
        if np.any(predictions == None):
            print(f"Returning successful predictions for {np.sum(predictions != None)} out of {len(predictions)}")
        return predictions
    
    
    def validate_existing_predictions(self, predictions: Prediction, standardized_data: Dataset):
        assert isinstance(predictions, Prediction), f"Received predictions object of type {type(predictions)}; predictions must be an instance of Prediction"
        assert np.any(predictions == None), "All predictions have already been made"
        assert predictions.shape == (len(standardized_data), self.n_raters), f"Shape of predictions {predictions.shape} does not match number of observations and raters"
