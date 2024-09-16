
from pydantic import BaseModel, Field, conint
from enum import Enum
from typing import List, Dict, Union, TypeVar, Generic, Type

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
        output_attribute (str): The name of the attribute in the response model to be used as output.
        output_type (str): The type of output (e.g., 'single_label', 'taxonomy').
        kwargs (Dict): Optional extra arguments for specific tasks.

    Args:
        response_model: The Pydantic model for the task's output structure.
        system_template (str): Template string for the system prompt.
        user_template (str): Template string for the user prompt.
        task_arg_values (Dict, optional): Dictionary of task-specific argument values (required if prompt has task args).
        data_args (Union[str, List[str]], optional): Argument name(s) to be filled with data at inference time (required if prompt has > 1 argument).
        output_attribute (str, optional): Name of the output attribute in the response model (required if response model has > 1 attribute).
        output_type (str, optional): Type of the task output (if unspecified, inferred from response_model and output_attribute).
        kwargs (Dict, optional): Additional keyword arguments for specific tasks.

    Raises:
        AssertionError: If the keys in task_arg_values don't match the prompt's task arguments.
        ValueError: If the response_model is not a Pydantic model.
        ValueError: If the output_attribute is not a top-level attribute of the response model.
    """
    def __init__(self,
                 response_model,
                 system_template: str,
                 user_template: str,
                 task_arg_values: Dict = None,
                 data_args: Union[str, List[str]] = None,
                 output_attribute: str = None,
                 output_type: str = None,
                 kwargs: Dict = None # Optional extra arguments for specific tasks
                 ):
        self.response_model = response_model
        self.prompt = Prompt(system_template, user_template, data_args)
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
            raise ValueError(f"Cannot infer output_attribute from response model attributes: {attributes}. Please provide output_attribute explicitly.")
    
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
        if self.output_attribute not in self.response_model.model_fields:
            raise ValueError(f"output_attribute '{self.output_attribute}' must be a top-level attribute of the response model")

    def get_prompt(self, custom_task_args: Dict=None) -> Prompt:
        task_args = self.task_arg_values.copy()
        prompt = Prompt(self.prompt.system_template, self.prompt.user_template, self.prompt.data_args)
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
        output_attribute (str): The name of the attribute in the response model to be used as output.
        output_type (str): The type of output (e.g., 'single_label', 'taxonomy').

    Args:
        response_model: The Pydantic model for the task's output structure.
        prompt (Prompt): A Prompt object with formatted templates.
        output_attribute (str): Name of the output attribute in the response model.
        output_type (str): Type of the task output.
        
    Raises:
        ValueError: If the response_model is not a Pydantic model.
        ValueError: If the output_attribute is not a top-level attribute of the response model.

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
        
        if self.output_attribute not in self.response_model.model_fields:
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


class Sentiment(LabelSet):
    POSITIVE  = "positive"
    NEGATIVE  = "negative"
    NEUTRAL   = "neutral"
    MIXED     = "mixed"
    UNCERTAIN = "uncertain"

class SentimentAnalysis(SingleLabelCategorization[Sentiment]):
    """A single label categorization decision where the label set consists of sentiment categories"""
    pass
    
SENTIMENT_ANALYSIS_CONFIG = TaskConfig(
    response_model=SentimentAnalysis,
    system_template="Role: {role_description}\nContext: {context_description}\nInstructions: {task_description}",
    user_template="{input_label}: {input_text}",
    data_args=["input_text"],
    task_arg_values={
        "role_description": "You are an expert in sentiment analysis.",
        "context_description": "You are analyzing text to determine the sentiment expressed.",
        "task_description": "Analyze the given text and determine its sentiment. Provide reasoning for your decision and a confidence score.",
        "input_label": "Text"
    },
    output_attribute='code',
    output_type='single_label'
)

class SentimentAnalysisTask(SingleLabelCategorizationTask):
    def __init__(self, prompt=None):
        if prompt is None:
            prompt = SENTIMENT_ANALYSIS_CONFIG.get_prompt()
        super().__init__(category_class=Sentiment, prompt=prompt)


class Concept(BaseModel):
    name: str = Field(description="A human-readable name for the concept")
    id: str = Field(description="A descriptive id in all lowercase, with underscores instead of spaces")
    definition: str
    criteria: List[str] = Field(min_length=3, description="Criteria which a judge could use to unambiguously determine whether the concept is present or absent in the text")
    
class Taxonomy(BaseModel):
    """A generic concept description (typically from an LLM)"""
    concepts: list[Concept]
    
CONCEPT_EXTRACTION_CONFIG = TaskConfig(
    response_model=Taxonomy,
    system_template="Role: {role_description}\nContext: {context_description}\nInstructions: {task_description}",
    user_template="{input_label}:\n{input_text}",
    data_args=["input_text"],
    task_arg_values={
        "role_description": "You are an expert in conceptual analysis and taxonomy creation.",
        "context_description": "You are analyzing text(s) to extract key concepts and create a structured taxonomy.",
        "task_description": """
Analyze the given text(s) and extract key concepts to create a taxonomy. For each concept:
1. Provide a clear, concise name in tite case.
2. Generate a descriptive ID in lowercase, using underscores instead of spaces.
3. Write a comprehensive definition.
4. List at least 3 specific criteria that could be used to unambiguously determine the presence or absence of the concept in a text.

Ensure that the concepts are distinct, relevant, and cover the main ideas presented in the text. Aim for a balance between breadth and depth in your taxonomy.
""",
        "input_label": "Text for Analysis"
    },
    output_attribute='concepts',
    output_type='taxonomy'
)

CONCEPT_AGGREGATION_CONFIG = TaskConfig(
    response_model=Taxonomy,
    system_template="Role: {role_description}\nContext: {context_description}\nInstructions: {task_description}",
    user_template="{input_label}:\n{input_taxonomies}",
    data_args=["input_taxonomies"],
    task_arg_values={
        "role_description": "You are an expert in conceptual analysis and taxonomy aggregation.",
        "context_description": "You are combining multiple taxonomies created by different sources into a single, comprehensive taxonomy.",
        "task_description": """
Analyze the given taxonomies and combine them into a single, exhaustive, non-redundant taxonomy. For each concept in the final taxonomy:
1. Provide a clear, concise name in tite case.
2. Generate a descriptive ID in lowercase, using underscores instead of spaces.
3. Write a comprehensive definition that incorporates insights from all input taxonomies.
4. List at least 3 specific criteria that could be used to unambiguously determine the presence or absence of the concept in a text.

Ensure that the final taxonomy:
- Merges similar concepts appropriately
- Provides a comprehensive coverage of the domain represented by all input taxonomies
""",
        "input_label": "Input Taxonomies"
    },
    output_attribute='concepts',
    output_type='taxonomy'
)

    
class ConceptExtractionTask(Task):
    def __init__(self, prompt=None):
        if prompt is None:
            prompt = CONCEPT_EXTRACTION_CONFIG.get_prompt()
        super().__init__(
            response_model=Taxonomy,
            prompt=prompt,
            output_attribute=CONCEPT_EXTRACTION_CONFIG.output_attribute,
            output_type=CONCEPT_EXTRACTION_CONFIG.output_type)


class PresenceAbsence(LabelSet):
    PRESENT = "present"
    ABSENT = "absent"

class PresenceAbsenceCoding(SingleLabelCategorization[PresenceAbsence]):
    """A single label categorization decision where the label set consists of presence or absence categories"""
    pass

PRESENCE_ABSENCE_CODING_CONFIG = TaskConfig(
    response_model=PresenceAbsenceCoding,
    system_template="""Role: {role_description}\nContext: {context_description}\nInstructions: {task_description}\n\n{concept_label}: {concept}\n\nRemember to focus solely on whether this concept is unambiguously and explicitly present in the text.""",
    user_template="{input_label}: {input_text}",
    data_args=["input_text", "concept"],
    task_arg_values={
        "role_description": "You are an expert in content analysis and coding.",
        "context_description": "You are analyzing text to determine the presence or absence of a specific concept.",
        "task_description": "Analyze the given text and determine whether the specified concept is unambiguously and explicitly present. Provide reasoning for your decision and a confidence score.",
        "input_label": "Text",
        "concept_label": "Concept to code"
    },
    output_attribute='code',
    output_type='single_label'
)

class PresenceAbsenceCodingTask(SingleLabelCategorizationTask):
    def __init__(self, prompt=None):
        if prompt is None:
            prompt = PRESENCE_ABSENCE_CODING_CONFIG.get_prompt()
        super().__init__(category_class=PresenceAbsence, prompt=prompt)
