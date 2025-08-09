
from pydantic import BaseModel, Field, conint
from enum import Enum
import typing
from typing import List, Dict, Union, TypeVar, Generic, Type, Optional
from datetime import datetime

from .client import LLMClient
from .prompt import Prompt


class TaskConfig:
    """
    A configuration class for initializing tasks in an LLM-assisted content analysis framework.

    This class encapsulates all the necessary components to define a specific task,
    including the response model, prompt templates, task-specific arguments,
    and output specifications.

    Attributes:
        response_model: The Pydantic model defining the structure of the LLM's output.
        system_template (str): Template string for the system prompt.
        user_template (str): Template string for the user prompt.
        prompt (Prompt): A Prompt object containing the system and user templates.
        task_arg_values (Dict): Dictionary of task-specific argument values.
        output_attribute (str, optional): The name of the attribute in the response model to be used as output. None for multi-field models where no default is specified.
        output_type (str): The type of output (e.g., 'single_label', 'taxonomy').
        kwargs (Dict): Optional extra arguments for specific tasks.

    Args:
        response_model: The Pydantic model for the task's output structure.
        system_template (str): Template string for the system prompt.
        user_template (str): Template string for the user prompt.
        task_arg_values (Dict, optional): Dictionary of task-specific argument values (required if prompt has task args).
        data_args (Union[str, List[str]], optional): Argument name(s) to be filled with data at inference time (required if prompt has > 1 argument).
        output_attribute (str, optional): Name of the output attribute in the response model. If not provided, automatically inferred for single-field models, None for multi-field models.
        output_type (str, optional): Type of the task output (if unspecified, inferred from response_model and output_attribute).
        kwargs (Dict, optional): Additional keyword arguments for specific tasks.

    Raises:
        AssertionError: If the keys in task_arg_values don't match the prompt's task arguments.
        ValueError: If the response_model is not a Pydantic model.
        ValueError: If the output_attribute is specified but is not a top-level attribute of the response model.
    """
    def __init__(self,
                 response_model,
                 system_template: str,
                 user_template: str,
                 task_arg_values: Dict = None,
                 data_args: Union[str, List[str]] = None,
                 role_args: Union[str, List[str]] = None,
                 output_attribute: str = None,
                 output_type: str = None,
                 kwargs: Dict = None # Optional extra arguments for specific tasks
                 ):
        self.response_model = response_model
        if role_args is None:
            role_args = []
        elif isinstance(role_args, str):
            role_args = [role_args]
        self.prompt = Prompt(system_template, user_template, data_args, role_args)
        self.task_arg_values = task_arg_values or {}
        self.output_attribute = self.infer_output_attribute(output_attribute)
        self.output_type = self.infer_output_type(output_type)
        self.kwargs = kwargs or {}
        self.validate()
        
    def infer_output_attribute(self, output_attribute):
        if output_attribute is not None:
            return output_attribute
        attributes = self.response_model.model_fields.keys()
        if len(attributes) == 1:
            return next(iter(attributes))
        else:
            # Multi-field models have no default output attribute
            return None
    
    def infer_output_type(self, output_type):
        if output_type is not None:
            return output_type
        attr_type = self.response_model.model_fields[self.output_attribute].annotation
        if hasattr(attr_type, '__origin__'):  # For generic types like List, Dict
            return attr_type.__origin__.__name__.lower()
        elif issubclass(attr_type, BaseModel):  # For Pydantic models
            return attr_type.__name__.lower()
        else:
            return attr_type.__name__.lower()

    def validate(self):
        assert set(self.task_arg_values.keys()) == set(self.prompt.task_args), \
            f"Keys in task_arg_values: {set(self.task_arg_values.keys())} do not match prompt task_args: {set(self.prompt.task_args)}"
        if not issubclass(self.response_model, BaseModel):
            raise ValueError("response_model must be a Pydantic model")
        if self.output_attribute is not None and self.output_attribute not in self.response_model.model_fields:
            raise ValueError(f"output_attribute '{self.output_attribute}' must be a top-level attribute of the response model")

    def get_prompt(self, custom_task_args: Dict=None) -> Prompt:
        task_args = self.task_arg_values.copy()
        prompt = Prompt(system_template=self.prompt.system_template,
                        user_template=self.prompt.user_template,
                        data_args=self.prompt.data_args,
                        role_args=self.prompt.role_args)
        if custom_task_args:
            task_args.update(custom_task_args)
        return prompt.define_task(**task_args)


class Task():
    """
    A base class for defining tasks in an LLM-assisted content analysis framework.

    This class serves as a foundation for specific task types, encapsulating
    common attributes and methods used across different content analysis tasks.

    Attributes:
        response_model: The Pydantic model defining the structure of the LLM's output.
        prompt (Prompt): A Prompt object containing the formatted system and user templates.
        output_attribute (str, optional): The name of the attribute in the response model to be used as output. None for multi-field models where no default is specified.
        output_type (str): The type of output (e.g., 'single_label', 'taxonomy').

    Args:
        response_model: The Pydantic model for the task's output structure.
        prompt (Prompt): A Prompt object with formatted templates.
        output_attribute (str): Name of the output attribute in the response model.
        output_type (str): Type of the task output.
        
    Raises:
        ValueError: If the response_model is not a Pydantic model.
        ValueError: If the output_attribute is specified but is not a top-level attribute of the response model.

    Note:
        This class serves as a base for specific task subclasses.
    """
    def __init__(self,
                 response_model,
                 prompt: Prompt,
                 output_attribute: str,
                 output_type: str
                 ):
        self.response_model = response_model # The pydantic model defining the structure of the output to be created by the LLM
        self.prompt = prompt
        self.output_attribute = output_attribute
        self.output_type = output_type
        self.validate_output_attribute()
        
    def validate_output_attribute(self):
        if not issubclass(self.response_model, BaseModel):
            raise ValueError("response_model must be a Pydantic model")
        
        if self.output_attribute is not None and self.output_attribute not in self.response_model.model_fields:
            raise ValueError(f"output_attribute '{self.output_attribute}' is not a top-level attribute of the response model")
        
    @classmethod
    def from_config(cls, config: TaskConfig, **kwargs):
        """
        Create a Task instance from a TaskConfig object.

        This class method provides an alternative way to initialize a Task or its subclass
        using a pre-defined TaskConfig object. It allows for easy creation of task instances
        with standardized configurations.

        Args:
            cls: The class on which this method is called (Task or a subclass).
            config (TaskConfig): A TaskConfig object containing all necessary configuration parameters.
            **kwargs: Additional keyword arguments that can override or supplement the config.
                These are merged with any default kwargs specified in the TaskConfig.

        Returns:
            An instance of the Task class (or subclass) initialized with the provided configuration.

        Note:
            This method is particularly useful for creating instances of Task subclasses
            that may require additional parameters beyond those in the base Task class.

        Example:
            >>> config = TaskConfig(...)
            >>> task = SingleLabelCategorizationTask.from_config(config, category_class=labels)
        """
        merged_kwargs = {**config.kwargs, **kwargs} # Merge default and user-supplied inputs (e.g., category_class)
        return cls(
            response_model=config.response_model,
            prompt=config.get_prompt(),
            output_attribute=config.output_attribute,
            output_type=config.output_type,
            **merged_kwargs
        )
        
    def feedback(self,
                 raters: Union[LLMClient, List[LLMClient]],
                 context: str = "None provided",
                 example_data: Optional[Union['Dataset', List, 'pd.DataFrame']] = None,
                 task_raters: Optional[Union[LLMClient, List[LLMClient]]] = None,
                 config: Optional[Union['FeedbackConfig', dict]] = None,
                 save_to_file: bool = False,
                 output_filename: Optional[str] = None) -> str:
        """
        Solicits feedback on the current task's definition using one or more LLMs.

        This method passes the current task's system prompt template, user prompt template,
        and Pydantic response model schema (as a JSON string) to one or more
        LLM clients. These clients, guided by the `TaskDefinitionFeedbackTask`,
        will then provide feedback on the clarity, robustness, and potential issues
        of the current task's design.

        Args:
            raters (Union[LLMClient, List[LLMClient]]): A single `LLMClient` or a list
                of `LLMClient` instances that will generate the feedback. More capable
                models are strongly advised for this role to ensure high-quality,
                actionable feedback.
            context (str, optional): An optional string containing any additional contextual
                information about the task being reviewed (e.g., its specific goals,
                the type of data it will handle, intended downstream applications, or known
                constraints). Defaults to "None provided". Providing clear context can
                significantly improve the relevance and depth of the feedback received.
            example_data: Optional example data to analyze (Dataset, List, or DataFrame).
            task_raters: LLM client(s) to run the task on examples (if provided, task will be run).
            config: Configuration object or dict for feedback options.
            save_to_file (bool, optional): If True, the generated feedback string will
                be saved to a text file. Defaults to False.
            output_filename (Optional[str], optional): The desired filename (including path
                if necessary) for the saved feedback. If `save_to_file` is True and
                this is None, a default filename `feedback_<timestamp>.txt` will be used.
                Defaults to None.

        Returns:
            A string containing the feedback on the current task's definition.
        """
        from .predictor import Predictor
        from .dataset import Dataset

        # Convert example_data to Dataset if needed
        if example_data is not None and not isinstance(example_data, Dataset):
            example_data = Dataset(example_data, data_args=self.prompt.data_args)
        
        # Use helper method to prepare feedback task and data
        feedback_task, feedback_data_dict = self._prepare_feedback_task(
            context=context,
            example_data=example_data,
            task_raters=task_raters,
            config=config
        )
        
        # Create dataset from the prepared data dictionary
        dataset = Dataset(data=[feedback_data_dict], data_args=list(feedback_data_dict.keys()))

        predictor = Predictor(task=feedback_task, raters=raters)
        prediction_result = predictor.predict(dataset)

        feedback_strings_array = prediction_result.get()
        list_of_feedback_strings = list(feedback_strings_array)
        linebreak = "\n\n----------------------------------------\n\n"
        feedback_string = """TASK DEFINITION FEEDBACK\n
Disclaimer: LLMs can be (confidently) wrong, and when prompted to give feedback, will always come up with something.
Use your judgment to decide if the feedback is useful or not.""" + linebreak + linebreak.join(list_of_feedback_strings)
        
        if output_filename is not None:
            save_to_file = True
        if save_to_file:
            filename_to_use = output_filename
            if filename_to_use is None:
                # Generate timestamped filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_to_use = f"feedback_{timestamp}.txt"
            with open(filename_to_use, "w", encoding="utf-8") as f:
                f.write(feedback_string)
            print(f"Feedback saved to {filename_to_use}")

        return feedback_string

    def _prepare_feedback_task(self,
                              context: str,
                              example_data: Optional['Dataset'],
                              task_raters: Optional[Union[LLMClient, List[LLMClient]]],
                              config: Optional[Union['FeedbackConfig', dict]]) -> tuple:
        """
        Prepare the feedback task and data dictionary for feedback generation.
        
        This method handles all the logic for constructing the feedback prompt,
        including conditional assembly based on whether example data and/or
        task execution is requested.
        
        Args:
            context: Context about the task's intended use
            example_data: Optional Dataset with example data (already converted)
            task_raters: Optional LLM client(s) to run the task on examples
            config: Configuration for feedback options (currently unused in Phase 2)
        
        Returns:
            Tuple of (feedback_task, feedback_data_dict) ready for prediction
        """
        from .task_examples import (
            TASK_DEFINITION_FEEDBACK_CONFIG,
            DATA_ANALYSIS_INSTRUCTIONS,
            EXECUTION_ANALYSIS_INSTRUCTIONS,
            TaskFeedbackResponse
        )
        
        # Validate combination of parameters
        if task_raters is not None and example_data is None:
            raise ValueError("example_data must be provided when task_raters is specified")
        
        # Initialize placeholder values for new fields
        data_summary = "None provided"
        output_summary = "None available" 
        observations = "None available"
        
        # Phase 2: Use placeholder strings when example data is provided
        if example_data is not None:
            data_summary = f"Example data provided: {len(example_data)} samples (detailed analysis pending implementation)"
            
            # If task_raters are also provided, indicate task execution would happen
            if task_raters is not None:
                output_summary = "Task execution on examples pending implementation"
                observations = "Interleaved input-output observations pending implementation"
            else:
                # Data only mode
                observations = "Input observations pending implementation"
        
        # Build conditional system prompt
        system_prompt = TASK_DEFINITION_FEEDBACK_CONFIG.prompt.system_template
        if example_data is not None:
            system_prompt += "\n\n" + DATA_ANALYSIS_INSTRUCTIONS
        if task_raters is not None:
            system_prompt += "\n\n" + EXECUTION_ANALYSIS_INSTRUCTIONS
        
        # Create enhanced feedback task configuration
        enhanced_feedback_config = TaskConfig(
            response_model=TaskFeedbackResponse,
            system_template=system_prompt,
            user_template=TASK_DEFINITION_FEEDBACK_CONFIG.prompt.user_template,
            data_args=TASK_DEFINITION_FEEDBACK_CONFIG.prompt.data_args,
            output_attribute=TASK_DEFINITION_FEEDBACK_CONFIG.output_attribute
        )
        
        feedback_task = Task.from_config(enhanced_feedback_config)
        
        # Prepare the data dictionary
        feedback_data_dict = {
            "user_provided_context": context,
            "original_system_prompt": self.prompt.system_prompt,
            "original_user_prompt": self.prompt.user_prompt,
            "original_response_model": self.response_model.model_json_schema(),
            "data_summary": data_summary,
            "output_summary": output_summary,
            "observations": observations
        }
        
        return feedback_task, feedback_data_dict

    def is_attribute_list(self, attribute: str) -> bool:
        """
        Checks if a given attribute in the response model is a list type.
        """
        if attribute not in self.response_model.model_fields:
            raise ValueError(f"Attribute '{attribute}' not found in the response model.")

        annotation = self.response_model.model_fields[attribute].annotation
        if hasattr(annotation, '__origin__'): # Check if it's a parameterized type
            return annotation.__origin__ is list # Use "is" for identity comparison
        elif hasattr(typing, 'List') and typing.List is list:
            return issubclass(annotation, list)
        else: # Fallback: treat as scalar if not a parameterized or typing.List
            return False

    def is_attribute_list_of_pydantic_models(self, attribute: str) -> bool:
        """
        Checks if a given attribute in the response model is a list of Pydantic models.
        """
        if attribute not in self.response_model.model_fields:
            raise ValueError(f"Attribute '{attribute}' not found in the response model.")
        if not self.is_attribute_list(attribute):
            return False
        annotation = self.response_model.model_fields[attribute].annotation
        list_type = typing.get_args(annotation)[0]
        if issubclass(list_type, BaseModel):
            return True
        return False

class LabelSet(str, Enum):
    """General class for category sets (e.g., labels in a sentiment analysis task)"""
    def create(name: str, values: List[str]) -> Type['LabelSet']:
        """Generate a label set given a set name and a list of labels"""
        return Enum(name, {value.upper(): value for value in values}, type=LabelSet)

# Generic type for the category
Label = TypeVar('Label', bound=LabelSet)


class SingleLabelCategorization(BaseModel, Generic[Label]):
    """A generic categorization decision (typically from an LLM)"""
    reasoning: str
    confidence: conint(ge=0, le=100) = Field(description="A 0-100 confidence rating")
    code: Label
    
SINGLE_LABEL_CATEGORIZATION_CONFIG = TaskConfig(
    response_model=SingleLabelCategorization,
    system_template="Role: {role_description}\nContext: {context_description}\nInstructions: {task_description}",
    user_template="{input_label}: {input_text}",
    data_args=["input_text"],
    task_arg_values={
        "role_description": "You are an expert in categorization.",
        "context_description": "You are analyzing text to determine its category.",
        "task_description": "Analyze the given text and determine its category. Provide reasoning for your decision and a confidence score.",
        "input_label": "Text"
    },
    output_attribute='code',
    output_type='single_label'
)
    
class SingleLabelCategorizationTask(Task):
    """A generic categorization task which outputs a categorization decision, given a prompt and response model (output schema)"""
    def __init__(self, category_class: Type[LabelSet], prompt=None):
        if prompt is None:
            prompt = SINGLE_LABEL_CATEGORIZATION_CONFIG.get_prompt()
        self.category_class = category_class
        super().__init__(
            response_model=SingleLabelCategorization[category_class],
            prompt=prompt,
            output_attribute=SINGLE_LABEL_CATEGORIZATION_CONFIG.output_attribute,
            output_type=SINGLE_LABEL_CATEGORIZATION_CONFIG.output_type
        )

    @classmethod
    def from_config(cls, config: TaskConfig, category_class: Type[LabelSet], **kwargs):
        return cls(
            category_class=category_class,
            prompt=config.get_prompt()
        )


# Import examples to maintain backwards compatibility
from .task_examples import *