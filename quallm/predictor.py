from .client import LLMClient
from .dataset import Dataset
from .tasks import Task
from .prediction import Prediction

# Import version information
from quallm import __version__ as version

from typing import List, Dict, Union
import pandas as pd
import numpy as np
import concurrent.futures
import datetime as dt
import threading
import sys
import logging


class InMemoryLogHandler(logging.Handler):
    """
    A logging.Handler that keeps formatted log records in a simple list.
    Each record is a dict with timestamp, level, message, etc.
    """
    def __init__(self,
                 fmt: str = "%(asctime)s|%(levelname)s|%(message)s",
                 datefmt: str = "%Y-%m-%dT%H:%M:%S%z"):
        super().__init__(level=logging.DEBUG)
        self.formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        self.records: List[dict] = []
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            raw = self.formatter.format(record)
            entry = {
                "created_ts": record.created,
                "timestamp": self.formatter.formatTime(record, self.formatter.datefmt),
                "level": record.levelname,
                "logger": record.name,
                "func": record.funcName,
                "line": record.lineno,
                "message": record.getMessage(),
                "raw": raw
            }
            with self._lock:
                self.records.append(entry)
        except Exception:
            # If handler fails, avoid infinite recursion
            self.handleError(record)
    
    
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
                 echo: bool = False,
                 echo_level: int = logging.INFO,
                 ):
        if isinstance(raters, LLMClient):
            raters = [raters]
        self.raters = raters
        self.n_raters = len(raters)
        self.task = task
        self.run_timestamps = []
        self.echo_level = echo_level

        # Build a dedicated logger for this instance:
        logger_name = f"quallm.predictor.Predictor_{id(self):x}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        # Prevent double‐logging if root logging is configured
        self.logger.propagate = False

        # Attach our in‑memory handler
        self._log_handler = InMemoryLogHandler()
        self.logger.addHandler(self._log_handler)

        # Expose logs as a public attribute
        # Users can do: for entry in predictor.logs: ...
        self.logs: List[dict] = self._log_handler.records

        # Console logging
        self._console_handler = None
        self._set_echo(echo)
        self.logger.info(f"Initialized Predictor. quallm version: {version}")
    
    
    def _set_echo(self, value: bool = False):
        """Enable or disable echoing of logs to the console"""
        self.echo = value
        if value:
            # Add a console handler if it doesn't exist
            if self._console_handler is None:
                console = logging.StreamHandler(sys.stdout)
                console.setLevel(self.echo_level)
                console.setFormatter(self._log_handler.formatter)
                self.logger.addHandler(console)
                self._console_handler = console
        else:
            # Remove the console handler if it exists
            if self._console_handler is not None:
                self.logger.removeHandler(self._console_handler)
                self._console_handler.close()
                self._console_handler = None


    def set_echo_level(self, level: int):
        """
        Change the console echo level on the fly. If a console handler is already
        attached, updates its level immediately.
        """
        self.echo_level = level
        if self._console_handler is not None:
            self._console_handler.setLevel(level)


    def predict_single(self, params):
        """Generate a prediction for a single observation"""
        index, language_model, formatted_prompt = params
        start_time = dt.datetime.now()
        self.logger.debug(f"Index: {index}. Beginning prediction.")
        try:
            response = language_model.request(
                system_prompt=formatted_prompt.system_prompt,
                user_prompt=formatted_prompt.user_prompt,
                response_model=self.task.response_model
            )
            end_time = dt.datetime.now()
            pred_time = (end_time - start_time).total_seconds()
            length = len(getattr(response, self.task.output_attribute))
            self.logger.debug(f"Index: {index}. Returning prediction. Length: {length}. Duration: {pred_time:.3f}s.")
            return (index, {'response': [response]})
        except Exception as e:
            end_time = dt.datetime.now()
            pred_time = (end_time - start_time).total_seconds()
            self.logger.error(f"Index: {index}. Returning None. Duration: {pred_time:.3f}s. Error: {e}")
            return (index, None)


    def predict(self,
                data: Union[str, Dict[str, str], List[str], pd.Series, pd.DataFrame, np.ndarray, List[Dict[str, str]]],
                predictions: Prediction = None,
                max_workers: int = 1,
                echo: bool = False
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
            data (Union[str, Dict[str, str], List[str], pd.Series, pd.DataFrame, np.ndarray, List[Dict[str, str]]):
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
        self._set_echo(echo)
        run_num = len(self.run_timestamps)
        self.logger.info(f"predict() called. Run number: {run_num}. n_raters: {self.n_raters}. max_workers: {max_workers}.")
        start_time = dt.datetime.now()
        self.run_timestamps.append({"start": start_time.isoformat()})
        if not isinstance(data, Dataset):
            standardized_data = Dataset(data, data_args=self.task.prompt.data_args)
        else:
            standardized_data = data
        if predictions is None:
            predictions = Prediction.__new__(Prediction,
                                             task=self.task,
                                             n_obs=len(standardized_data),
                                             n_raters=self.n_raters)
        else:
            self.validate_existing_predictions(predictions, standardized_data)
            self.logger.info(f"Resuming predictions for {np.sum(predictions == None)} missing observation(s) out of {predictions.size} total observation(s).")
        # Prepare prediction tasks
        tasks = []
        for i, data_point in enumerate(standardized_data):
            for j in range(self.n_raters):
                if predictions[i,j] is None:  # Only create tasks for unmade predictions
                    language_model = self.raters[j]
                    role_and_data_args = language_model.role_args | data_point
                    formatted_prompt = self.task.prompt.insert_role_and_data(**role_and_data_args)
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
        self._log_completion(start_time, run_num, predictions)
        return predictions
    
    
    def _log_completion(self, start_time, run_num, predictions):
        end_time = dt.datetime.now()
        self.run_timestamps[run_num-1]["end"] = end_time.isoformat()
        run_duration = (end_time - start_time).total_seconds()
        self.run_timestamps[run_num-1]["run_duration_sec"] = run_duration
        self.logger.info(f"predict() finished in {run_duration:.3f}s")
        n_pred = predictions.size
        n_success = np.sum(predictions != None)
        n_missing = np.sum(predictions == None)
        self.logger.info(f"predict() returned {n_success} successful predictions and {n_missing} missing predictions out of {n_pred} total predictions")
    
    
    def validate_existing_predictions(self, predictions: Prediction, standardized_data: Dataset):
        assert isinstance(predictions, Prediction), f"Received predictions object of type {type(predictions)}; predictions must be an instance of Prediction"
        assert np.any(predictions == None), "All predictions have already been made; if passing a preexisting Prediction object, it must have at least one missing value"
        assert predictions.shape == (len(standardized_data), self.n_raters), f"Shape of predictions {predictions.shape} does not match number of observations {len(standardized_data)} and raters {self.n_raters}"
        
    
    def print_logs(self, raw: bool = True):
        """
        Print stored log records to stdout.
        If raw=True, prints the formatted line; otherwise prints the message only.
        """
        for rec in self.logs:
            line = rec["raw"] if raw else rec["message"]
            print(line)

    def clear_logs(self):
        """Empty the in‑memory log buffer."""
        with self._log_handler._lock:
            self._log_handler.records.clear()

    def logs_df(self) -> pd.DataFrame:
        """Return the log records as a pandas DataFrame for further analysis."""
        return pd.DataFrame(self.logs)

    def dump_logs(self, path: str):
        """Save the logs JSON‐style for later inspection."""
        import json
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2)