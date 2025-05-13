from pydantic import BaseModel, Field, conint
from enum import Enum
from typing import List, Type

# Import primitives from the main tasks.py
from .tasks import TaskConfig, Task, LabelSet, SingleLabelCategorization, SingleLabelCategorizationTask
from .prompt import Prompt # Needed if TaskConfig.get_prompt() is to be understood by type checkers or for direct Prompt use.


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
