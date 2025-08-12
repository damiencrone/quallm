from typing import Dict, List, Optional, Any
import pandas as pd
import time
import statistics
import json
import os
from pydantic import BaseModel
import instructor
from ..client import LLMClient, DEFAULT_TEMPERATURE
from ..tasks import Task, TaskConfig
from ..dataset import Dataset
from ..predictor import Predictor
from openai import OpenAI
from enum import Enum


def load_diagnostic_data(filename: str) -> Dict[str, Any]:
    """
    Load diagnostic test data from JSON file.
    
    Args:
        filename: Name of the JSON file (without path) in the diagnostics data directory
        
    Returns:
        Dictionary containing the loaded data with 'data_args' and 'data' keys
        
    Raises:
        FileNotFoundError: If the diagnostic data file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    from importlib.resources import files
    
    # Load from package resources (works in both dev and installed mode)
    data_text = files('quallm.data.diagnostics').joinpath(filename).read_text()
    return json.loads(data_text)


# Default diagnostic task response models
class BasicResponse(BaseModel):
    """Response model for basic Q&A diagnostic task."""
    answer: str
    confidence: int  # 1-10


class CategoryEnum(str, Enum):
    """Categories for classification diagnostic task."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ClassificationResponse(BaseModel):
    """Response model for text classification diagnostic task."""
    category: CategoryEnum
    keywords: List[str]
    confidence: int
    reasoning: str


class SubCategory(BaseModel):
    """Subcategory for nested analysis diagnostic task."""
    name: str
    score: float


class NestedObject(BaseModel):
    """Nested object for complex analysis diagnostic task."""
    themes: List[str]
    sentiment_scores: Dict[str, float]


class NestedResponse(BaseModel):
    """Response model for nested analysis diagnostic task."""
    main_category: str
    subcategories: List[SubCategory]
    metadata: Dict[str, Any]
    nested_analysis: NestedObject


# Default diagnostic task configurations
DEFAULT_DIAGNOSTIC_TASK_CONFIGS = [
    TaskConfig(
        response_model=BasicResponse,
        system_template="You are a helpful assistant.",
        user_template="Answer this question: {question}",
        data_args=["question"],
        output_attribute="answer",
        output_type="text"
    ),
    TaskConfig(
        response_model=ClassificationResponse,
        system_template="You are a text analyzer.",
        user_template="Analyze this text: {text}",
        data_args=["text"],
        output_attribute=None,  # Multi-field output
        output_type="structured"
    ),
    TaskConfig(
        response_model=NestedResponse,
        system_template="You are an advanced analyzer.",
        user_template="Perform deep analysis on: {content}",
        data_args=["content"],
        output_attribute=None,
        output_type="complex_structured"
    )
]

# Create tasks from configs
DEFAULT_DIAGNOSTIC_TASKS = [Task.from_config(config) for config in DEFAULT_DIAGNOSTIC_TASK_CONFIGS]


class TestObservation(BaseModel):
    task_index: int  # Numeric task identifier (0, 1, 2...)
    task_display_name: str  # Response model class name for display
    observation_num: int
    success: bool
    response_time: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class TaskResult(BaseModel):
    task_index: int  # Numeric task identifier (0, 1, 2...)
    task_display_name: str  # Response model class name for display
    validity_rate: float  # % conforming to schema
    accuracy: Optional[float]  # % correct (not currently implemented, would require ground truth)
    avg_response_time: float
    errors: List[str]


class ResponseModeTestResult(BaseModel):
    mode_name: str
    works: bool
    # Metrics tracked per task
    results_by_task: Dict[int, TaskResult]  # task_index -> metrics
    # Aggregate metrics
    overall_validity: float  # % responses conforming to schema
    overall_accuracy: float  # % factually correct (not implemented, always 0.0)
    overall_avg_response_time: float
    validity_median: float  # Median success rate across observations
    response_time_median: float  # Median response time for successful requests
    response_time_min: float  # Minimum response time observed
    response_time_max: float  # Maximum response time observed
    error_types: List[str]
    recommendations: List[str]
    # Raw observation data for DataFrame conversion
    raw_observations: List[TestObservation] = []

    def to_df(self) -> pd.DataFrame:
        """
        Convert test results to a long-format pandas DataFrame.
        
        Each row represents a single test observation with mode, task,
        and observation-level details. This format is optimal for
        statistical analysis and visualization.
        
        Returns:
            pd.DataFrame with columns:
            - mode: The Instructor mode name
            - task_index: Numeric identifier for the diagnostic task (0, 1, 2...)
            - task_display_name: Response model class name for human-readable identification
            - observation_num: Sequential number of this observation within the task
            - success: Boolean indicating if the response was valid
            - response_time: Time taken for this specific request
            - error_type: Category of error if success=False, None otherwise
            - error_message: Detailed error message if available
        """
        rows = []
        for obs in self.raw_observations:
            rows.append({
                'mode': self.mode_name,
                'task_index': obs.task_index,
                'task_display_name': obs.task_display_name,
                'observation_num': obs.observation_num,
                'success': obs.success,
                'response_time': obs.response_time,
                'error_type': obs.error_type,
                'error_message': obs.error_message
            })
        
        return pd.DataFrame(rows)


class ResponseModeEvaluationResults:
    def __init__(self, mode_results: Dict[str, ResponseModeTestResult], model_name: str):
        """
        Container for complete mode evaluation results.
        
        Args:
            mode_results: Dictionary mapping mode names to ResponseModeTestResult objects
            model_name: Name of the model that was tested
        """
        self.mode_results = mode_results
        self.model_name = model_name
    
    def to_df(self) -> pd.DataFrame:
        """
        Combine all mode test results into a single long-format DataFrame.
        
        This method aggregates test observations from all modes into one DataFrame
        where each row represents a single test observation. A 'mode' column
        distinguishes which mode each observation came from.
        
        Returns:
            pd.DataFrame with columns:
            - mode: The Instructor mode name
            - task_index: Numeric identifier for the diagnostic task (0, 1, 2...)
            - task_display_name: Response model class name for human-readable identification
            - observation_num: Sequential number of this observation within the task
            - success: Boolean indicating if the response was valid
            - response_time: Time taken for this specific request
            - error_type: Category of error if success=False, None otherwise
            - error_message: Detailed error message if available
        """
        combined_rows = []
        
        for mode_name, mode_result in self.mode_results.items():
            # Get individual mode's DataFrame
            mode_df = mode_result.to_df()
            # All rows already have 'mode' column from individual to_df()
            combined_rows.append(mode_df)
        
        if combined_rows:
            return pd.concat(combined_rows, ignore_index=True)
        else:
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=[
                'mode', 'task_index', 'task_display_name', 'observation_num', 'success', 
                'response_time', 'error_type', 'error_message'
            ])
    
    def get_working_response_modes(self) -> Dict[str, ResponseModeTestResult]:
        """
        Get only the modes that successfully work with this model.
        
        Returns:
            Dictionary of mode names to ResponseModeTestResult objects for working modes only
        """
        return {name: result for name, result in self.mode_results.items() if result.works}
    
    def get_recommended_response_mode(self) -> str:
        """
        Get the recommended response mode name based on performance metrics.
        
        Sorts response modes by validity rate (primary) and response time (secondary).
        Higher validity rate is better; lower response time is better.
        
        Returns:
            Name of the recommended response mode, or None if no response modes work
        """
        working_modes = self.get_working_response_modes()
        if not working_modes:
            return None
            
        # Sort by validity rate (primary, descending) and response time (secondary, ascending)
        sorted_mode_names = sorted(working_modes.keys(), 
                                 key=lambda mode_name: (
                                     -working_modes[mode_name].overall_validity,  # Higher validity is better (negative for desc)
                                     working_modes[mode_name].overall_avg_response_time  # Lower time is better (positive for asc)
                                 ))
        
        return sorted_mode_names[0]  # Return the best mode
    
    def summary(self) -> str:
        """
        Generate a text summary of the evaluation results.
        
        Presents modes in recommendation order with detailed performance metrics.
        
        Returns:
            Formatted summary string showing working modes and detailed recommendations
        """
        working_modes = self.get_working_response_modes()
        lines = [f"Mode Evaluation Results for {self.model_name}"]
        lines.append("=" * 50)
        
        if working_modes:
            # Get modes sorted by recommendation criteria (validity first, then speed)
            sorted_modes = sorted(working_modes.keys(),
                                key=lambda mode_name: (
                                    -working_modes[mode_name].overall_validity,
                                    working_modes[mode_name].overall_avg_response_time
                                ))
            recommended = self.get_recommended_response_mode()
            
            # Count total observations and tasks from any working mode
            sample_result = next(iter(working_modes.values()))
            total_obs = len(sample_result.raw_observations) if sample_result.raw_observations else 0
            total_tasks = len(set(obs.task_index for obs in sample_result.raw_observations)) if sample_result.raw_observations else 0
            
            lines.append(f"Working modes: {', '.join(sorted_modes)}")
            lines.append(f"Recommended mode: {recommended}")
            if total_obs > 0:
                lines.append(f"Evaluated {total_obs} observations across {total_tasks} tasks")
            lines.append("")
            
            # Display modes in sorted order
            for mode_name in sorted_modes:
                result = working_modes[mode_name]
                lines.append(f"\n{mode_name}:")
                lines.append(f"  Valid responses: {result.overall_validity:.1%}")
                lines.append(f"  Time: {result.overall_avg_response_time:.3f}s avg, {result.response_time_median:.3f}s median ({result.response_time_min:.3f}s - {result.response_time_max:.3f}s)")
                # Always check for errors from raw observations
                error_counts = {}
                for obs in result.raw_observations:
                    if not obs.success and obs.error_type:
                        error_counts[obs.error_type] = error_counts.get(obs.error_type, 0) + 1
                
                if error_counts:
                    error_summary = []
                    for error_type in sorted(error_counts.keys()):
                        count = error_counts[error_type]
                        error_summary.append(f"{error_type} ({count})")
                    lines.append(f"  Errors: {', '.join(error_summary)}")
            
            # Also show failed modes
            for mode_name, result in self.mode_results.items():
                if not result.works:
                    lines.append(f"\n{mode_name}:")
                    lines.append(f"  Failed: {', '.join(result.error_types) if result.error_types else 'Unknown error'}")
                    
        else:
            lines.append("No working modes found.")
            # Show failed modes with their errors
            for mode_name, result in self.mode_results.items():
                if not result.works:
                    lines.append(f"\n{mode_name}:")
                    lines.append(f"  Failed: {', '.join(result.error_types) if result.error_types else 'Unknown error'}")
            
        return "\n".join(lines)


class InstructorResponseModeTester:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", 
                 api_key: str = "ollama", temperature: float = DEFAULT_TEMPERATURE):
        """Initialize tester for a specific model.
        
        Args:
            model: Model name (e.g., "llama3.1" for Ollama)
            base_url: API endpoint (Ollama default shown)
            api_key: API key for authentication ("ollama" for local)
            temperature: Temperature parameter for LLM inference (default DEFAULT_TEMPERATURE)
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        
    def find_recommended_response_mode(self, 
                         include_default_tasks: bool = True,
                         custom_tasks: List[Dict] = None,
                         observations_per_task: int = 20,
                         echo: bool = False,
                         max_workers: int = 1) -> str:
        """Find recommended Instructor response mode for structured outputs.
        
        This method is a convenience wrapper around evaluate_response_modes() that
        returns just the recommended response mode name.
        
        Args:
            include_default_tasks: Whether to run the three built-in diagnostic tasks (default True)
            custom_tasks: List of dictionaries with 'task' and 'dataset' keys. Each dict contains a Task object paired with its corresponding Dataset object (optional)
            observations_per_task: Number of test runs per task (default 20).
            echo: Whether to show detailed diagnostic output via console logging
            max_workers: Maximum number of worker threads for parallel processing (default 1)
        
        Returns:
            Name of the recommended Instructor response mode (e.g., "JSON", "MD_JSON")
        """
        # Use the class method for the core evaluation logic
        results = InstructorResponseModeTester.evaluate_response_modes(
            model=self.model,
            base_url=self.base_url, 
            api_key=self.api_key,
            temperature=self.temperature,
            include_default_tasks=include_default_tasks,
            custom_tasks=custom_tasks,
            observations_per_task=observations_per_task,
            echo=echo,
            max_workers=max_workers
        )
        
        return results.get_recommended_response_mode()
        
    def _create_llm_client(self, mode: instructor.Mode) -> LLMClient:
        """Create an LLMClient with the specified Instructor mode."""
        try:
            client = instructor.from_openai(
                OpenAI(base_url=self.base_url, api_key=self.api_key),
                mode=mode
            )
            return LLMClient(
                client=client,
                language_model=self.model,
                temperature=self.temperature
            )
        except Exception:
            return None
            
    def warm_up_test(self, llm_client: LLMClient) -> bool:
        """Run LLMClient.test() to warm up model and validate mode works.
        
        Returns:
            True if test passes, False otherwise
        """
        try:
            result = llm_client.test()
            return isinstance(result, str)
        except Exception:
            return False
    
    def _get_default_datasets(self) -> List['Dataset']:
        """Return the three default Dataset objects for diagnostic testing.
        
        These datasets are designed to work with the default diagnostic
        tasks: basic Q&A (uses "question"), classification (uses "text"), and
        nested analysis (uses "content").
        
        Returns:
            List of three Dataset objects with test data for the default diagnostic tasks
        """
        datasets = []
        
        # Load data from JSON files
        data_files = ['basic_qa.json', 'classification.json', 'nested_analysis.json']
        
        for filename in data_files:
            data_config = load_diagnostic_data(filename)
            dataset = Dataset(data_config['data'], data_args=data_config['data_args'])
            datasets.append(dataset)
            
        return datasets
        
    def _generate_recommendations(self, validity: float, mode: instructor.Mode, avg_time: float = 0.0, 
                                time_min: float = 0.0, time_max: float = 0.0, error_types: List[str] = None) -> List[str]:
        """
        Generate factual descriptions of mode performance.
        
        Args:
            validity: Success rate (0.0-1.0)
            mode: Instructor mode being evaluated
            avg_time: Average response time in seconds
            time_min: Minimum response time observed
            time_max: Maximum response time observed
            error_types: List of error types encountered
            
        Returns:
            List of factual descriptive strings
        """
        recommendations = []
        error_types = error_types or []
        
        # Success rate
        if validity == 0:
            recommendations.append(f"{mode.name} mode: No successful responses")
        else:
            recommendations.append(f"{mode.name} mode: {validity:.1%} success rate")
        
        # Timing
        if avg_time > 0:
            if time_min > 0 and time_max > 0:
                recommendations.append(f"Response time: {avg_time:.3f}s avg ({time_min:.3f}s - {time_max:.3f}s)")
            else:
                recommendations.append(f"Response time: {avg_time:.3f}s avg")
        
        # Errors
        if error_types:
            unique_errors = list(set(error_types))
            if len(unique_errors) <= 3:
                recommendations.append(f"Errors: {', '.join(unique_errors)}")
            else:
                recommendations.append(f"Errors: {len(unique_errors)} different types")
        
        return recommendations
    
    def _get_default_tasks(self) -> List[Task]:
        """Return the three default diagnostic tasks for mode testing.
        
        Returns the module-level DEFAULT_DIAGNOSTIC_TASKS which test
        fundamental structured output capabilities across increasing complexity.
        
        Returns:
            List of three Task objects configured for diagnostic testing
        """
        return DEFAULT_DIAGNOSTIC_TASKS
        
    def _build_task_dataset_pairs(self, include_default_tasks: bool, custom_tasks: List[Dict] = None) -> List[Dict]:
        """Build the list of task-dataset pairs for testing.
        
        Args:
            include_default_tasks: Whether to include default diagnostic tasks
            custom_tasks: Optional list of custom task-dataset pairs
            
        Returns:
            List of dictionaries with 'task' and 'dataset' keys
            
        Raises:
            ValueError: If custom_tasks have invalid structure or no tasks are configured
        """
        task_dataset_pairs = []
        
        if include_default_tasks:
            default_tasks = self._get_default_tasks()
            default_datasets = self._get_default_datasets()
            for task, dataset in zip(default_tasks, default_datasets):
                # Validate task-dataset compatibility
                try:
                    task.validate(dataset)
                except ValueError as e:
                    raise ValueError(f"Default task validation failed: {e}")
                task_dataset_pairs.append({"task": task, "dataset": dataset})
        
        if custom_tasks is not None:
            # Validate custom_tasks structure
            for i, task_dict in enumerate(custom_tasks):
                if not isinstance(task_dict, dict):
                    raise ValueError(f"custom_tasks[{i}] must be a dictionary")
                if set(task_dict.keys()) != {"task", "dataset"}:
                    raise ValueError(f"custom_tasks[{i}] must contain exactly 'task' and 'dataset' keys")
                # Validate types
                if not hasattr(task_dict["task"], "response_model"):
                    raise ValueError(f"custom_tasks[{i}]['task'] must be a Task object")
                
                # Validate task-dataset compatibility
                task, dataset = task_dict['task'], task_dict['dataset']
                try:
                    task.validate(dataset)
                except ValueError as e:
                    raise ValueError(f"Custom task {i} validation failed: {e}")
                    
            task_dataset_pairs.extend(custom_tasks)
        
        if not task_dataset_pairs:
            raise ValueError("No tasks to run - either include_default_tasks=True or provide custom_tasks")
            
        return task_dataset_pairs
    
    @classmethod
    def evaluate_response_modes(cls, 
                       model: str, 
                       base_url: str = "http://localhost:11434/v1",
                       api_key: str = "ollama",
                       temperature: float = DEFAULT_TEMPERATURE,
                       include_default_tasks: bool = True,
                       custom_tasks: List[Dict] = None,
                       observations_per_task: int = 20,
                       echo: bool = True,
                       max_workers: int = 1) -> 'ResponseModeEvaluationResults':
        """
        Evaluate all Instructor modes for a given model and return detailed results.
        
        This method runs comprehensive diagnostics on all four Instructor modes
        and returns complete test results for user inspection and analysis.
        
        Args:
            model: Model name (e.g., "llama3.1" for Ollama)
            base_url: API endpoint (Ollama default shown)
            api_key: API key for authentication ("ollama" for local)
            temperature: Temperature parameter for LLM inference (default DEFAULT_TEMPERATURE)
            include_default_tasks: Whether to run the three built-in diagnostic tasks (default True)
            custom_tasks: List of dictionaries with 'task' and 'dataset' keys. Each dict contains a Task object paired with its corresponding Dataset object (optional)
            observations_per_task: Number of test runs per task (default 20).
            echo: Whether to show detailed diagnostic output via console logging
            max_workers: Maximum number of worker threads for parallel processing (default 1)
            
        Returns:
            ResponseModeEvaluationResults object containing individual mode results and
            methods for accessing combined analysis across all modes.
        """
        tester = cls(model, base_url, api_key, temperature)
        
        # Build task-dataset pairs
        task_dataset_pairs = tester._build_task_dataset_pairs(include_default_tasks, custom_tasks)
        
        if echo:
            print(f"Evaluating Instructor modes for model: {model} (temperature={temperature}, max_retries=5)")
            print("=" * 50)
        
        # Test each mode
        mode_results = {}
        for mode in [instructor.Mode.JSON, instructor.Mode.MD_JSON, 
                     instructor.Mode.JSON_SCHEMA, instructor.Mode.TOOLS]:
            
            if echo:
                print(f"\nTesting mode: {mode.name}")
            
            # Test this mode
            result = tester.test_mode(mode, task_dataset_pairs, observations_per_task, echo, max_workers)
            mode_results[mode.name] = result
            
            if echo:
                tester._print_mode_result(result)
        
        # Create and return results
        results = ResponseModeEvaluationResults(mode_results=mode_results, model_name=model)
        
        if echo:
            tester._print_evaluation_summary(results)
        
        return results
    
    def _print_mode_result(self, result: ResponseModeTestResult):
        """Print the results for a single mode test."""
        if result.works:
            print(f"  ✓ Mode works")
        else:
            print(f"  ✗ Mode failed - {', '.join(result.error_types)}")
    
    def _print_evaluation_summary(self, results: ResponseModeEvaluationResults):
        """Print the summary of all mode evaluations."""
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETE")
        print("=" * 50)
        
        working_modes = results.get_working_response_modes()
        if working_modes:
            print(f"Working modes: {', '.join(working_modes.keys())}")
            recommended = results.get_recommended_response_mode()
            if recommended:
                print(f"Recommended mode: {recommended}")
        else:
            print("No working modes found for this model.")

    def test_mode(self, mode: instructor.Mode,
                  task_dataset_pairs: List[Dict],
                  observations_per_task: int = 20,
                  echo: bool = False,
                  max_workers: int = 1) -> ResponseModeTestResult:
        """Test a specific Instructor mode with diagnostic tasks."""
        # Create client for this specific mode
        llm_client = self._create_llm_client(mode)
        if llm_client is None:
            return ResponseModeTestResult(
                mode_name=mode.name,
                works=False,
                overall_validity=0.0,
                overall_accuracy=0.0,
                overall_avg_response_time=0.0,
                validity_median=0.0,
                response_time_median=0.0,
                response_time_min=0.0,
                response_time_max=0.0,
                results_by_task={},
                error_types=["Failed to create client"],
                recommendations=["Mode not supported by this model/configuration"],
                raw_observations=[]
            )
        
        # Warm-up AND validate in one step using LLMClient.test()
        if not self.warm_up_test(llm_client):
            return ResponseModeTestResult(
                mode_name=mode.name,
                works=False,
                overall_validity=0.0,
                overall_accuracy=0.0,
                overall_avg_response_time=0.0,
                validity_median=0.0,
                response_time_median=0.0,
                response_time_min=0.0,
                response_time_max=0.0,
                results_by_task={},
                error_types=["Failed warm-up test"],
                recommendations=[],
                raw_observations=[]
            )
        
        # Model is now warm, proceed with diagnostic tests using predictor-based implementation
        test_results = self._run_predictor_based_tests(llm_client, mode, task_dataset_pairs, observations_per_task, echo, max_workers)
        return test_results

    
    def _run_predictor_based_tests(self, llm_client: LLMClient, mode: instructor.Mode,
                                   task_dataset_pairs: List[Dict], observations_per_task: int, echo: bool = False, max_workers: int = 1) -> ResponseModeTestResult:
        """
        Run diagnostic tasks using Predictor-based architecture.
        
        This method creates a Predictor instance per (mode, task) combination and uses
        Dataset cycling to achieve the required number of observations per task.
        It extracts timing and error data from Prediction metadata.
        
        Args:
            llm_client: LLMClient configured with the mode to test
            mode: Instructor mode being tested
            task_dataset_pairs: List of task-dataset pairs to run
            observations_per_task: Number of observations required per task
            echo: Whether to display verbose diagnostic information
            max_workers: Maximum number of worker threads for parallel processing
            
        Returns:
            ResponseModeTestResult with same structure as _run_diagnostic_tests
        """
        raw_observations = []
        all_successes = []
        all_timings = []
        all_errors = []
        predictors = []  # Store predictors (not currently used but kept for future use)
        
        for task_idx, task_pair in enumerate(task_dataset_pairs):
            task = task_pair["task"]
            dataset = task_pair["dataset"]
            task_index = task_idx
            task_display_name = task.response_model.__name__
            
            if echo:
                print(f"  Task {task_index + 1}/{len(task_dataset_pairs)}: {task_display_name}")
            
            # Build dataset with cycling to achieve observations_per_task count
            cycled_data = []
            for i in range(observations_per_task):
                cycled_data.append(dataset[i % len(dataset)])
            cycled_dataset = Dataset(cycled_data, data_args=dataset.data_args)
            
            # Create Predictor instance for this mode-task combination
            # Set echo=False to avoid conflicting with tqdm progress bars
            predictor = Predictor(task=task, raters=[llm_client], echo=False)
            predictors.append(predictor)  # Store for future use
            
            task_successes = []
            task_timings = []
            task_errors = []
            
            # Run prediction with parallel processing support
            prediction_failed = False
            try:
                prediction = predictor.predict(cycled_dataset, max_workers=max_workers)
            except Exception as e:
                prediction_failed = True
                prediction_error = e
            
            if not prediction_failed:
                # Extract metadata for all predictions
                timing_data = prediction.get_timing()
                metadata_data = prediction.get_metadata()
                
                # Process each prediction and extract metadata
                for i in range(observations_per_task):
                    # Get actual result for this observation (single rater, so j=0)
                    result = prediction[i, 0] if i < prediction.n_obs else None
                    
                    # Extract timing from metadata
                    timing_info = timing_data[i, 0] if timing_data is not None else None
                    response_time = timing_info.get('duration', 0.0) if timing_info else 0.0
                    
                    # Extract error information from metadata  
                    metadata_info = metadata_data[i, 0] if metadata_data is not None else None
                    success = metadata_info.get('success', result is not None) if metadata_info else (result is not None)
                    error_type = metadata_info.get('error_type') if metadata_info else None
                    error_message = metadata_info.get('error_message') if metadata_info else None
                    
                    # Handle case where result is None but metadata doesn't indicate failure
                    if result is None and success:
                        success = False
                        error_type = error_type or "PredictionFailure"
                        error_message = error_message or "Prediction returned None"
                    
                    raw_observations.append(TestObservation(
                        task_index=task_index,
                        task_display_name=task_display_name,
                        observation_num=i,
                        success=success,
                        response_time=response_time,
                        error_type=error_type,
                        error_message=error_message[:200] if error_message else None
                    ))
                    
                    # Track per-task statistics
                    task_successes.append(success)
                    all_successes.append(success)
                    if success and response_time > 0:
                        task_timings.append(response_time)
                        all_timings.append(response_time)
                    if not success and error_message:
                        task_errors.append(error_message)
                        all_errors.append(error_message)
                        
                # Display per-task results
                if echo:
                    successful_count = sum(task_successes)
                    failed_count = len(task_successes) - successful_count
                    
                    success_rate = (successful_count / observations_per_task * 100) if observations_per_task > 0 else 0
                    print(f"    Valid responses: {successful_count}/{observations_per_task} ({success_rate:.0f}%)")
                    
                    if task_timings:
                        avg_task_time = sum(task_timings) / len(task_timings)
                        min_task_time = min(task_timings)
                        max_task_time = max(task_timings)
                        median_task_time = statistics.median(task_timings)
                        print(f"    Time: {avg_task_time:.2f}s avg, {median_task_time:.2f}s median ({min_task_time:.2f}s - {max_task_time:.2f}s)")
                    
                    if failed_count > 0:
                        error_types = set()
                        for obs in raw_observations:
                            if obs.task_index == task_index and not obs.success and obs.error_type:
                                error_types.add(obs.error_type)
                        if error_types:
                            print(f"    Errors: {', '.join(sorted(error_types))}")
            else:
                # Entire task failed - no metadata available
                error_type = prediction_error.__class__.__name__
                error_message = str(prediction_error)[:200]
                for i in range(observations_per_task):
                    raw_observations.append(TestObservation(
                        task_index=task_index,
                        task_display_name=task_display_name,
                        observation_num=i,
                        success=False,
                        response_time=0.0,
                        error_type=error_type,
                        error_message=error_message
                    ))
                    all_successes.append(False)
                    all_errors.append(error_message)
                
                if echo:
                    print(f"    Valid responses: 0/{observations_per_task} (0%)")
                    print(f"    Error: Task failed - {error_type}")
        
        # Calculate aggregate metrics from actual data
        overall_validity = sum(all_successes) / len(all_successes) if all_successes else 0.0
        avg_time = sum(all_timings) / len(all_timings) if all_timings else 0.0
        
        # Calculate distributional metrics
        
        numeric_successes = [1.0 if s else 0.0 for s in all_successes]
        validity_median = statistics.median(numeric_successes) if numeric_successes else 0.0
        
        if all_timings:
            response_time_median = statistics.median(all_timings)
            response_time_min = min(all_timings)
            response_time_max = max(all_timings)
        else:
            response_time_median = 0.0
            response_time_min = 0.0
            response_time_max = 0.0
        
        # Collect unique error types
        unique_errors = set()
        for obs in raw_observations:
            if not obs.success and obs.error_type:
                unique_errors.add(obs.error_type)
        
        # Display summary across all tasks
        if echo and len(task_dataset_pairs) > 1:
            print(f"  Summary across {len(task_dataset_pairs)} tasks:")
            print(f"    Overall valid responses: {overall_validity:.1%}")
            if all_timings:
                print(f"    Overall time: {avg_time:.2f}s avg, {response_time_median:.2f}s median")
            if unique_errors:
                # Count error occurrences
                error_counts = {}
                for obs in raw_observations:
                    if not obs.success and obs.error_type:
                        error_counts[obs.error_type] = error_counts.get(obs.error_type, 0) + 1
                error_summary = []
                for error_type in sorted(error_counts.keys()):
                    count = error_counts[error_type]
                    error_summary.append(f"{error_type} ({count})")
                print(f"    Error types: {', '.join(error_summary)}")
        
        return ResponseModeTestResult(
            mode_name=mode.name,
            works=overall_validity > 0,
            overall_validity=overall_validity,
            overall_accuracy=0.0,  # Accuracy evaluation not implemented (would require ground truth)
            overall_avg_response_time=avg_time,
            validity_median=validity_median,
            response_time_median=response_time_median,
            response_time_min=response_time_min,
            response_time_max=response_time_max,
            results_by_task={},  # Task-level breakdown not currently populated
            error_types=list(unique_errors) if unique_errors else [],
            recommendations=self._generate_recommendations(
                overall_validity, mode, avg_time, response_time_min, response_time_max, list(unique_errors) if unique_errors else []
            ),
            raw_observations=raw_observations
        )