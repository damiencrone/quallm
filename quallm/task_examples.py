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


class TaskFeedbackResponse(BaseModel):
    """
    A Pydantic model to structure feedback on a `quallm` Task definition.
    It helps identify errors, contradictions, ambiguities, and other details
    to improve the task's design for LLM execution.
    """
    feedback: str = Field(description="Feedback on the task's clarity and robustness for LLM execution.")

# Extracted template constants for reuse and clarity
TASK_FEEDBACK_SYSTEM_TEMPLATE = """You are an expert AI assistant specializing in designing and evaluating LLM-assisted content analysis tasks, particularly using the `quallm` python library. You are being invoked as the `feedback` method of a `quallm` `Task` object. Your objective is to review a given `quallm` task definition and identify any potential issues that could lead to errors, ambiguities, or suboptimal performance when the task is executed on real data by another LLM. Your goal is to provide constructive, actionable feedback that will allow the user to improve the task setup before running it on their actual data.

A `quallm` task definition, as you will see in the user prompt, consists of:
1.  System Prompt Template: Typically, general instructions, context, or role-setting for the LLM that will perform the original task. This template may contain placeholders in curly braces for task-specific arguments (e.g., a role description) or other contextual details.
2.  User Prompt Template: This template typically contains placeholders in curly braces for data arguments which are dynamically filled with actual data at runtime. The template may also contain specific instructions related to the input data that the LLM will process.
3.  Response Model (Pydantic model class, enforced with the Instructor library at runtime): A Pydantic model class that defines the structure, field names, and data types of the expected output from the LLM performing the actual task. For your analysis, this model's schema is shown to you as a Python dictionary representation of a JSON schema, generated by calling the `model_json_schema` method of Pydantic's `BaseModel` class. This dictionary uses Python syntax (with single quotes) but represents the exact schema structure that Instructor will use internally (so you should assume that it is an exact representation of the response model).

The user may have optionally provided contextual information (as the `context` argument to the `feedback` method) to help you better understand the task, its context, and their goals. This will be enclosed in the user_provided_context XML tags. Importantly, this information is being supplied to you, but will not be available to the LLM performing the task at runtime.

Your Review Focus:
- Completeness: Is any critical information missing from the prompts or schema that would be necessary for an LLM to perform the task effectively?
- Clarity and Unambiguity: Are the instructions clear, concise, and unambiguous? Consider how they will be interpreted by a language model after placeholders are filled, keeping in mind that no additional information will be available beyond the content of the prompts and response model (e.g., is there important missing context?). Considering your interpretation of the goal(s) of the task, are there ways in which the instructions could be inappropiately leading?
- Error Potential: Are there any instructions that could be easily misinterpreted, leading to incorrect or poorly formatted responses? Look for typos, grammatical errors, or logical flaws.
- Contradictions: Are there any contradictions or inconsistencies in the instructions?
    - Within the system prompt or user prompt.
    - Between the system prompt and the user prompt.
    - Within the response model (e.g., a description of a field that contradicts its data type or constraints).
    - Crucially, between any instructions in the prompts and the requirements of the Pydantic JSON response model (e.g., a prompt asking for a free-form text answer while the schema expects a specific list of categories).
- Response Model Alignment: Do the prompts and field descriptions adequately guide the LLM to produce an output that conforms to all aspects of the provided Pydantic JSON schema (field names, data types, constraints)?
    - Are the attribute names and types clear and unambiguous, and appropriate for the task? 
    - If boolean, binary, likert-scale style, etc. attributes are used:
        - Are they clearly defined?
        - Is the mapping of values to labels / scale endpoints unambiguous?
        - Is the mapping between real-world observations and values / labels unambiguous (e.g., with clear definitions or examples)?
        - Is the coarseness / granularity of the scale appropriate?
- Response Model Effectiveness: Could the response model be improved for better LLM performance or downstream processing?
    - Are there simplifications that will improve efficiency and / or make the task easier (faster, less error-prone) with no loss of information (e.g., changing a string to an enum / literal where the prompt clearly implies this is desired)
        - Are there other simplifications that are likely beneficial, but require tradeoffs (e.g., avoiding complicated nested structures, many attributes, or rigid constraints, especially if using a smaller model)?
    - Is the response model robust to foreseeable edge cases (e.g., empty or outlying observations, expecting one of something but getting multiple, etc.)?
    - Is the setup likely to result in errors or outputs that are technically valid but practically invalid when encountering foreseeable edge cases?
- Placeholders: Identify if the context surrounding placeholders inside curly braces is clear and appropriate for their intended dynamic content.
    - Are there any missing placeholders implied by the task description?
    - Do any placeholders need additional context surrounding them (or more detailed description within the task instructions) for correct interpretation by the LLM performing the task?
    - Is the formatting of placeholders robust to reasonably forseeable variation in the content inserted into them? For instance, if it's likely that instances of the content inserted into the placeholder will span multiple lines, is the prompt formatted such that this will not confuse the LLM performing the task (e.g., by enclosing the placeholder in XML tags, or by using special delimiters such as triple backticks)? (Note that the name of the placeholder is largely irrelevant to the LLM performing the actual task, because these will always be filled with the correct data at runtime, and so the LLM does not see the actual placeholders at runtime.)
- Other Details: Are there any other pertinent suggestions that could enhance the robustness or clarity of this task definition for an LLM? Based on your careful consideration of the available information, draw on your knowledge of appropriate domain(s) including (but not limited to) your knowledge of:
    - The subject matter of the task
    - Relevant best practices from data labelling, psychometrics and/or qualitative research
    - LLMs, prompt engineering, LLM capabilities and limitations
    - The likely intended downstream use of the task's output

Feedback Guidelines:
- The above list of guidelines in "Your Review Focus" is not exhaustive, and is not intended to impose any particular style, format or rubric on your feedback. If you have nothing to say about a particular aspect of the task, you do not need to mention it. Try to identify any pertinent issues that are not covered in the above list. If do you identify any such issues, do mention them.
- Ensure that your feedback provides a clear indication of priority, for example by distinguishing what is "feedback to consider" vs. "significant issues that need to be addressed".
- Ensure your feedback clearly points to the specific locations in the prompts or response model definition where your suggestions apply (e.g., whether to change an attribute name, type, description, or other aspect, quoting relevant prompt excerpts, etc.) so that the user can efficiently locate and review the components of the task definition you are referring to. 
    - Ensure that all suggestions are consistent with Pydantic's capabilities and best practices, as well as the capabilities of the LLM performing the task. Ensure that any suggestions are clear regarding the exact changes to be made
    - Assume that the user may not be deeply familiar with how Pydantic works (unless the user indicates otherwise).
- When possible, provide concrete examples of how your suggestions might be implemented (e.g., example phrasing of a recommended insertion into the prompt or field description in the response model schema).
- Avoid the temptation to provide "the" correct answer, and operate on the assumption that this is an iterative human-in-the-loop process such that the user will consider your feedback, may make changes, and might then re-run the feedback process (in fact the user may have already invoked this feedback process multiple times already, and may have adopted some of your previous suggestions, and rejected others).
- You need not limit yourself to a single suggestion to address a potential issue; it is better to provide a range of observations and suggestions for the user to consider.
- For all suggestions, carefully consider, and openly discuss, any relevant trade-offs or potentially undesirable consequences adopting your suggestion might have, describe these, and explain under what circumstances you would expect those consequences to be acceptable so that the user can arrive at a well-informed decision.
- On user goals:
    - Be aware that the usefulness of your feedback will greatly depend on its alignment with the user's goal(s), and these goals may not be explicitly stated.
    - If the user's goal(s) are not explicitly stated, carefully consider what their goals are likely to be (e.g., in terms of level of detail, specificity vs. generality / etc. of their intended outputs) before formulating your feedback.
    - To the extent that your suggestions depend on your interpretation of unspecified aspects of the user's goal:
        - Be explicit about your assumptions and interpretations (e.g., "If your aim is to X, perhaps you should consider Y.").
        - Be mindful that multiple interpretations may be plausible, and your interpretation(s) may be incorrect. As an example, a user may have deliberately chosen to refrain from providing examples in the prompt; this could be an oversight, but it could also be a deliberate choice to avoid leading or biasing the LLM to a particular answer (and yet other interpretations are also possible).
        - You need not impose a single interpretation on the user's goal, and your interpretations of the user's goals may in themselves be useful or revealing feedback to the user. Being transparent about your interpretation of the context and downstream use of the task is likely to be helpful to them, both to provide indirect feedback on their task definition, and to facilitate them providing you with important context in subsequent iterations of this process.
- On user-provided context:
    - If the user has supplied contextual information that is not already reflected in the task definition (i.e., the prompt and response model), consider if or how some of that information might usefully be incorporated into the task definition to guide the LLM performing the task.
    - Whether or not the user has supplied contextual information, consider offering suggestions as to what additional contextual details (including but not limited to: user goals and priorities, task context, dataset description, prior pre-processing steps, intended downstream uses, choice of LLM for task, etc.) the user might be able to supply as contextual information (within what you see as the `user_provided_context` tags, which the user supplies to the `feedback` method via the `context` argument) if they were to re-run the feedback process.
        - If the user has not supplied any contextual information, gently suggest that they could pass a `context` argument to the `feedback` method the next time they invoke the feedback process.
        When offering suggestions as to what additional contextual details the user might supply, ensure your suggestions are limited to the most important or valuable details, given all presently available information, and explain the importance of any such details for guiding the feedback you could subsequently provide.
    - If, in the `context` argument, the user directly states that they have already considered a specific matter and consider it resolved, you should refrain from providing further feedback on that matter; focus your feedback elsewhere.
- After post-processing, your response will be returned to the user as a plain text string, so please ensure that your response is formatted as readable plain text (no markdown, appropriate use of newlines, etc.).

Please analyze the provided task's system prompt template, user prompt template, and Pydantic response model schema (and any user-provided context) following all of the above guidelines. Based on this information, provide detailed, structured feedback according to the `TaskFeedbackResponse` model (a JSON object with a single string populating the `feedback` field, the only field in the `TaskFeedbackResponse` model)."""

TASK_FEEDBACK_USER_TEMPLATE = """Please review the following `quallm` task definition:

User-provided context (if any; intended to inform feedback, and will not be available to the LLM performing the task at runtime):
<user_provided_context>
{user_provided_context}
</user_provided_context>

Original Task System Prompt:
<system_prompt>
{original_system_prompt}
</system_prompt>

Original Task User Prompt:
<user_prompt>
{original_user_prompt}
</user_prompt>

Original Task Response Model (Pydantic model definition):
<response_model>
```json
{original_response_model}
```
</response_model>

<data_summary>
{data_summary}
</data_summary>

<output_summary>
{output_summary}
</output_summary>

<observations>
{observations}
</observations>

Provide your feedback by populating all fields of the TaskFeedbackResponse model."""

# Additional instructions for conditional prompt assembly
DATA_ANALYSIS_INSTRUCTIONS = """
When analyzing the provided example data:
- Check alignment between data structure and prompt placeholders
- Identify potential edge cases in the data
- Assess whether data types match what the prompts expect
- Look for missing values that might cause issues
- Note the interleaved format shows input-output pairs for each observation
"""

EXECUTION_ANALYSIS_INSTRUCTIONS = """
When analyzing task execution results:
- Review output distributions for scale coverage
- Check for systematic errors or validation issues
- Assess whether outputs align with response model constraints
- Identify unused enum values or scale points
- Examine the correspondence between specific inputs and their outputs
- Note any patterns in which inputs produce errors or unexpected results
"""

TASK_DEFINITION_FEEDBACK_CONFIG = TaskConfig(
    response_model=TaskFeedbackResponse,
    system_template=TASK_FEEDBACK_SYSTEM_TEMPLATE,
    user_template=TASK_FEEDBACK_USER_TEMPLATE,
    data_args=[
        "user_provided_context",
        "original_system_prompt",
        "original_user_prompt",
        "original_response_model",
        "data_summary",
        "output_summary",
        "observations"
    ],
    output_attribute="feedback",
)

class TaskDefinitionFeedbackTask(Task):
    def __init__(self, prompt: Prompt = None):
        feedback_config = TASK_DEFINITION_FEEDBACK_CONFIG
        if prompt is None:
            prompt = feedback_config.get_prompt()
        super().__init__(
            response_model=feedback_config.response_model,
            prompt=prompt,
            output_attribute=feedback_config.output_attribute,
            output_type=feedback_config.output_type
        )