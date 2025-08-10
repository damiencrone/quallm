from typing import Dict, List, Optional, Any
import pandas as pd
import time
from pydantic import BaseModel
import instructor
from ..client import LLMClient
from ..tasks import Task, TaskConfig
from ..dataset import Dataset
from ..predictor import Predictor
from openai import OpenAI
from enum import Enum
import numpy as np


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
    accuracy: Optional[float]  # % correct (only for default tasks, None for user tasks)
    avg_response_time: float
    errors: List[str]


class ModeTestResult(BaseModel):
    mode_name: str
    works: bool
    # Metrics tracked per task
    results_by_task: Dict[int, TaskResult]  # task_index -> metrics
    # Aggregate metrics
    overall_validity: float  # % responses conforming to schema
    overall_accuracy: float  # % factually correct (default tasks only, 0.0 for user tasks)
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


class ModeEvaluationResults:
    def __init__(self, mode_results: Dict[str, ModeTestResult], model_name: str):
        """
        Container for complete mode evaluation results.
        
        Args:
            mode_results: Dictionary mapping mode names to ModeTestResult objects
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
    
    def get_working_modes(self) -> Dict[str, ModeTestResult]:
        """
        Get only the modes that successfully work with this model.
        
        Returns:
            Dictionary of mode names to ModeTestResult objects for working modes only
        """
        return {name: result for name, result in self.mode_results.items() if result.works}
    
    def get_recommended_mode(self) -> str:
        """
        Get the recommended mode name based on performance metrics.
        
        Returns:
            Name of the recommended mode, or None if no modes work
        """
        working_modes = self.get_working_modes()
        if not working_modes:
            return None
            
        return max(working_modes.keys(), 
                  key=lambda x: (working_modes[x].overall_validity, 
                               -working_modes[x].overall_avg_response_time))
    
    def summary(self) -> str:
        """
        Generate a text summary of the evaluation results.
        
        Returns:
            Formatted summary string showing working modes and recommendations
        """
        working_modes = self.get_working_modes()
        lines = [f"Mode Evaluation Results for {self.model_name}"]
        lines.append("=" * 50)
        
        if working_modes:
            lines.append(f"Working modes: {', '.join(working_modes.keys())}")
            recommended = self.get_recommended_mode()
            lines.append(f"Recommended mode: {recommended}")
            lines.append("")
            
            for mode_name, result in working_modes.items():
                lines.append(f"{mode_name}: {result.overall_validity:.1%} success, "
                           f"{result.overall_avg_response_time:.3f}s avg")
        else:
            lines.append("No working modes found.")
            
        return "\n".join(lines)


class InstructorModeTester:
    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1", 
                 api_key: str = "ollama"):
        """Initialize tester for a specific model.
        
        Args:
            model: Model name (e.g., "llama3.1" for Ollama)
            base_url: API endpoint (Ollama default shown)
            api_key: API key for authentication ("ollama" for local)
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        
    def find_recommended_mode(self, 
                         include_default_tasks: bool = True,
                         custom_tasks: List[Dict] = None,
                         observations_per_task: int = 20,
                         echo: bool = False,
                         max_workers: int = 1) -> str:
        """Find recommended Instructor mode for structured outputs.
        
        This method is a convenience wrapper around evaluate_modes() that
        returns just the recommended mode name.
        
        Args:
            include_default_tasks: Whether to run the three built-in diagnostic tasks (default True)
            custom_tasks: List of dictionaries with 'task' and 'dataset' keys. Each dict contains a Task object paired with its corresponding Dataset object (optional)
            observations_per_task: Number of test runs per task (default 20).
            echo: Whether to show detailed diagnostic output via console logging
            max_workers: Maximum number of worker threads for parallel processing (default 1)
        
        Returns:
            Name of the recommended Instructor mode (e.g., "JSON", "MD_JSON")
        """
        # Use the class method for the core evaluation logic
        results = InstructorModeTester.evaluate_modes(
            model=self.model,
            base_url=self.base_url, 
            api_key=self.api_key,
            include_default_tasks=include_default_tasks,
            custom_tasks=custom_tasks,
            observations_per_task=observations_per_task,
            echo=echo,
            max_workers=max_workers
        )
        
        return results.get_recommended_mode()
        
    def _create_llm_client(self, mode: instructor.Mode) -> LLMClient:
        """Create an LLMClient with the specified Instructor mode."""
        try:
            client = instructor.from_openai(
                OpenAI(base_url=self.base_url, api_key=self.api_key),
                mode=mode
            )
            return LLMClient(
                client=client,
                language_model=self.model
                # Uses DEFAULT_TEMPERATURE from LLMClient
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
        # Basic Q&A dataset
        qa_data = [
            {"question": "What is the main theme of this content?"},
            {"question": "How would you summarize this information?"},
            {"question": "What are the key points discussed?"}
        ]
        qa_dataset = Dataset(qa_data, data_args=["question"])
        
        # Classification dataset
        classification_data = [
            {"text": "This is sample text for analysis."},
            {"text": "Another piece of text to classify."},
            {"text": "A third example for classification testing."}
        ]
        classification_dataset = Dataset(classification_data, data_args=["text"])
        
        # Nested analysis dataset
        nested_data = [
            {"content": "Sample content for deep analysis."},
            {"content": "Complex content requiring detailed examination."},
            {"content": "Rich textual material for comprehensive analysis."}
        ]
        nested_dataset = Dataset(nested_data, data_args=["content"])
        
        return [qa_dataset, classification_dataset, nested_dataset]
        
    def _generate_recommendations(self, validity: float, mode: instructor.Mode) -> List[str]:
        if validity >= 1.0:
            return [f"{mode.name} mode works perfectly with {validity:.1%} success rate"]
        elif validity >= 0.95:
            return [f"{mode.name} mode has minor issues with {validity:.1%} success rate", "May occasionally fail on complex structures"]
        elif validity >= 0.80:
            return [f"{mode.name} mode is unreliable with {validity:.1%} success rate", "Not recommended for production use"]
        elif validity > 0:
            return [f"{mode.name} mode mostly fails with {validity:.1%} success rate", "Serious reliability issues"]
        else:
            return [f"{mode.name} mode does not work with this model"]
    
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
            task_dataset_pairs.extend(custom_tasks)
        
        if not task_dataset_pairs:
            raise ValueError("No tasks to run - either include_default_tasks=True or provide custom_tasks")
            
        return task_dataset_pairs
    
    @classmethod
    def evaluate_modes(cls, 
                       model: str, 
                       base_url: str = "http://localhost:11434/v1",
                       api_key: str = "ollama",
                       include_default_tasks: bool = True,
                       custom_tasks: List[Dict] = None,
                       observations_per_task: int = 20,
                       echo: bool = True,
                       max_workers: int = 1) -> 'ModeEvaluationResults':
        """
        Evaluate all Instructor modes for a given model and return detailed results.
        
        This method runs comprehensive diagnostics on all four Instructor modes
        and returns complete test results for user inspection and analysis.
        
        Args:
            model: Model name (e.g., "llama3.1" for Ollama)
            base_url: API endpoint (Ollama default shown)
            api_key: API key for authentication ("ollama" for local)
            include_default_tasks: Whether to run the three built-in diagnostic tasks (default True)
            custom_tasks: List of dictionaries with 'task' and 'dataset' keys. Each dict contains a Task object paired with its corresponding Dataset object (optional)
            observations_per_task: Number of test runs per task (default 20).
            echo: Whether to show detailed diagnostic output via console logging
            max_workers: Maximum number of worker threads for parallel processing (default 1)
            
        Returns:
            ModeEvaluationResults object containing individual mode results and
            methods for accessing combined analysis across all modes.
        """
        tester = cls(model, base_url, api_key)
        
        # Build task-dataset pairs
        task_dataset_pairs = tester._build_task_dataset_pairs(include_default_tasks, custom_tasks)
        
        if echo:
            print(f"Evaluating Instructor modes for model: {model}")
            print("=" * 50)
        
        # Test each mode
        mode_results = {}
        for mode in [instructor.Mode.JSON, instructor.Mode.MD_JSON, 
                     instructor.Mode.JSON_SCHEMA, instructor.Mode.TOOLS]:
            
            if echo:
                print(f"\nTesting mode: {mode.name}")
            
            # Test this mode
            result = tester.test_mode(mode, task_dataset_pairs, observations_per_task)
            mode_results[mode.name] = result
            
            if echo:
                tester._print_mode_result(result)
        
        # Create and return results
        results = ModeEvaluationResults(mode_results=mode_results, model_name=model)
        
        if echo:
            tester._print_evaluation_summary(results)
        
        return results
    
    def _print_mode_result(self, result: ModeTestResult):
        """Print the results for a single mode test."""
        if result.works:
            print(f"  ✓ Success rate: {result.overall_validity:.1%}")
            print(f"  ✓ Avg response time: {result.overall_avg_response_time:.2f}s")
            if result.overall_accuracy > 0:
                print(f"  ✓ Accuracy: {result.overall_accuracy:.1%}")
        else:
            print(f"  ✗ Failed - {', '.join(result.error_types)}")
    
    def _print_evaluation_summary(self, results: ModeEvaluationResults):
        """Print the summary of all mode evaluations."""
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETE")
        print("=" * 50)
        
        working_modes = results.get_working_modes()
        if working_modes:
            print(f"Working modes: {', '.join(working_modes.keys())}")
            recommended = results.get_recommended_mode()
            if recommended:
                print(f"Recommended mode: {recommended}")
        else:
            print("No working modes found for this model.")

    def test_mode(self, mode: instructor.Mode,
                  task_dataset_pairs: List[Dict],
                  observations_per_task: int = 20) -> ModeTestResult:
        """Test a specific Instructor mode with diagnostic tasks."""
        # Create client for this specific mode
        llm_client = self._create_llm_client(mode)
        if llm_client is None:
            return ModeTestResult(
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
            return ModeTestResult(
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
        
        # Model is now warm, proceed with diagnostic tests
        test_results = self._run_diagnostic_tests(llm_client, mode, task_dataset_pairs, observations_per_task)
        return test_results

    def _run_diagnostic_tests(self, llm_client: LLMClient, mode: instructor.Mode,
                              task_dataset_pairs: List[Dict], observations_per_task: int) -> ModeTestResult:
        """Run diagnostic tasks and aggregate results."""
        successes = []
        timings = []
        errors = []
        raw_observations = []
        
        for task_idx, task_pair in enumerate(task_dataset_pairs):
            task = task_pair["task"]
            dataset = task_pair["dataset"]
            task_index = task_idx  # Use numeric index for task identity
            task_display_name = task.response_model.__name__  # Use response model name for display
            
            for i in range(observations_per_task):
                # Sample from the dataset for this observation
                data_item = dataset[i % len(dataset)]  # Cycle through dataset if fewer items than observations
                try:
                    start = time.time()
                    # Format the prompt with data arguments
                    formatted_prompt = task.prompt.insert_role_and_data(**data_item)
                    response = llm_client.request(
                        system_prompt=formatted_prompt.system_prompt,
                        user_prompt=formatted_prompt.user_prompt,
                        response_model=task.response_model
                    )
                    elapsed = time.time() - start
                    successes.append(True)
                    timings.append(elapsed)
                    # Record successful observation
                    raw_observations.append(TestObservation(
                        task_index=task_index,
                        task_display_name=task_display_name,
                        observation_num=i,
                        success=True,
                        response_time=elapsed,
                        error_type=None,
                        error_message=None
                    ))
                except Exception as e:
                    elapsed = time.time() - start
                    successes.append(False)
                    errors.append(str(e))
                    # Record the actual exception class name
                    error_type = e.__class__.__name__
                    # Record failed observation
                    raw_observations.append(TestObservation(
                        task_index=task_index,
                        task_display_name=task_display_name,
                        observation_num=i,
                        success=False,
                        response_time=elapsed,
                        error_type=error_type,
                        error_message=str(e)[:200]  # Truncate long error messages
                    ))
        
        overall_validity = sum(successes) / len(successes)
        avg_time = sum(timings) / len(timings) if timings else 0
        
        # Calculate distributional metrics
        import statistics
        
        # Convert boolean successes to numeric for median calculation (1.0 for True, 0.0 for False)
        numeric_successes = [1.0 if s else 0.0 for s in successes]
        validity_median = statistics.median(numeric_successes) if numeric_successes else 0.0
        
        # Calculate response time statistics (successful requests only)
        if timings:
            response_time_median = statistics.median(timings)
            response_time_min = min(timings)
            response_time_max = max(timings)
        else:
            response_time_median = 0.0
            response_time_min = 0.0
            response_time_max = 0.0
        
        return ModeTestResult(
            mode_name=mode.name,
            works=overall_validity > 0,
            overall_validity=overall_validity,
            overall_accuracy=0.0,  # Accuracy evaluation not implemented yet
            overall_avg_response_time=avg_time,
            validity_median=validity_median,
            response_time_median=response_time_median,
            response_time_min=response_time_min,
            response_time_max=response_time_max,
            results_by_task={},  # Task-level breakdown not implemented yet
            error_types=list(set(errors)),
            recommendations=self._generate_recommendations(overall_validity, mode),
            raw_observations=raw_observations
        )