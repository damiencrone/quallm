# quallm

quallm is a Python library designed to simplify and streamline LLM-assisted content analysis tasks. It provides a flexible framework for defining, executing, and analyzing various content analysis (and similar) tasks using language models. quallm is built to work seamlessly with both local (i.e., on-device) models through [Ollama](https://ollama.com) and cloud-hosted models (e.g., those provided by Together, Groq, OpenAI, and other services), providing the flexibility to choose the most suitable option for a given task.

Key features of quallm include:

- A modular design that separates tasks, data handling, inference, and output handling
- Output validation using [Instructor](https://python.useinstructor.com)
- Support for defining arbitrary structured outputs using [Pydantic](https://docs.pydantic.dev/latest/) models
- Built-in, customizable tasks for common analyses like single-labeled classification
- Support for multiple "raters" (different language model instances) for the same task
- Options for parallel processing to improve performance on larger datasets

## Installation

You can install quallm and all its dependencies using pip:

```bash
pip install git+https://github.com/damiencrone/quallm.git
```

This command will install quallm along with all required dependencies as specified in the setup.py file.

If you prefer using conda, you can set up a new environment and install quallm as follows:

```bash
conda create -n quallm_env python=3.10
conda activate quallm_env
pip install git+https://github.com/damiencrone/quallm.git
```

These instructions will create a new conda environment with Python 3.10 and install quallm with all its dependencies.

**Note**: to use local LLMs, you will need to install [Ollama](https://ollama.com), and will need to have already downloaded whatever model(s) you intend to use. After installing Ollama, this can simply be achieved with the following bash command (which would download the [Phi-3.5-mini](https://ollama.com/library/phi3.5) model used in the first demo below):

```bash
ollama pull phi3.5
```

If using cloud-hosted LLMs, you will likely need to create an account and set up an API key with your chosen provider. For further details on setting up different LLMs with different providers, refer to the [Instructor](https://python.useinstructor.com) documentation.

## Usage

quallm revolves around a few simple elements. The most important of these is the "task", which is essentially the combination of (1) an output schema (a Pydantic model) defining the structure of the response the LLM will generate, and (2) a prompt template providing instructions for the LLM. Once a task is defined, all one needs is one or more LLMs (or "raters") to perform the task, and a dataset, which contains each individual observation (e..g, a survey response, document, etc.) which the task will be performed on.

Here are some basic examples of how to use quallm:

### Using a pre-existing task

Simple tasks such as single-label classification or sentiment analysis can be performed with pre-existing tasks[^1]. In the example below, we use a relatively small local LLM (Phi-3.5), which will likely work on most consumer devices. In this task, we ask the LLM to classify three texts, using a pre-configured sentiment analysis task, which returns a sentiment classification (one of five categories), along with an explanation and confidence rating.

```python
from quallm import LLMClient, Dataset, Predictor
from quallm.tasks import SentimentAnalysisTask

llm = LLMClient(language_model="phi3.5") # Defaults to Ollama but will work with any instructor client
task = SentimentAnalysisTask()
dataset = Dataset(['I love this!', 'It\'s okay.', 'I hate this.'], 'input_text')

# Create predictor and get predictions
predictor = Predictor(raters=llm, task=task)
predictions = predictor.predict(dataset)

results = predictions.get()
print(results)
# Output:
# ['positive' 'neutral' 'negative']

expanded_results = predictions.expand()
print(expanded_results)
# Output:
#                                            reasoning  confidence      code
# 0  The text expresses a clear sense of enjoyment ...          90  positive
# 1  The text expresses acceptance or a lack of str...          85   neutral
# 2  The text expresses a strong negative emotion t...          95  negative
```

[^1]: Although pre-defined tasks *do* come with pre-written prompt templates, users are advised to tailor prompts to their specific use cases, as the default prompts are unlikely to be optimal for a given combination of task, LLM, and dataset.

### Defining a new task

Aribtrary tasks can also be defined using a `TaskConfig` with (at minimum) a Pydantic model, and system and user prompt template. In the example below, we define a trivial task: The response model (i.e., the thing the LLM is tasked with generating) is a list of strings on a given topic (the `ListResponse` Pydantic model). The prompt template (in the definition of `task_config`) is a barebones template with a placeholder for the topic (which is the datapoint or observation which is piped into the prompt template at inference time). In this case, the "data" is a single observation: "ethical precepts". As instructed, the LLM returns a list of ethical precepts.

```python
from pydantic import BaseModel, Field
from typing import List
from quallm.tasks import Task, TaskConfig
from quallm import LLMClient, Predictor

class ListResponse(BaseModel):
    items: List[str] = Field(description="A list of items relating to a topic")

task_config = TaskConfig(
    response_model=ListResponse,
    system_template="Generate a short list based on the topic provided.",
    user_template="Topic: {topic}"
)

llm_client = LLMClient(language_model="phi3.5")
list_generation_task = Task.from_config(task_config)
predictor = Predictor(task=list_generation_task, raters=llm_client)
prediction = predictor.predict("ethical precepts")

# Print the results
results = prediction.get()
results
# Output:
# array([list(['Do unto others as you would have them do unto you (Golden Rule)', 'Seek justice and fairness', 'Respect the dignity of every individual', 'Act with integrity in all situations'])],
#   dtype=object)
```

### Using multiple language models, parallelization and arbitrary label sets

Many use cases for quallm will likely entail labelling tasks in which labels are predicted for a large number of observations. As such, this example provides a barebones demonstration of how one might approach such a use case, using multiple raters, parallelizing predictions, and defining custom label sets (using the `LabelSet` class, which can be passed to a generic `SingleLabelCategorizationTask`, or if preferred, an entirely user-defined task).

In this toy example, we define a categorization task where two local LLMs (Phi-3.5 and Llama-3.1) are tasked with categorizing a list of four objects as either an animal, a vehicle, or other:

```python
from quallm import LLMClient, Predictor
from quallm.tasks import LabelSet, SingleLabelCategorizationTask

# Initialize two different LLM clients
llama_llm = LLMClient(language_model="llama3.1")
phi_llm = LLMClient(language_model="phi3.5")

# Create a custom label set
labels = LabelSet.create(name='Kind', values=['animal', 'vehicle', 'other'])

# Initialize the task with the custom label set
task = SingleLabelCategorizationTask(category_class=labels)

# Create a predictor with multiple raters
predictor = Predictor(raters=[llama_llm, phi_llm], task=task)

# Make predictions in parallel
pred = predictor.predict(data=['Elephant', 'Dog', 'Car', 'Train'], max_workers=5)

# Print expanded results
print(pred.expand(suffix=['llama', 'phi']))
# Output:
#    reasoning_llama                      reasoning_phi                       confidence_llama  confidence_phi  code_llama  code_phi
# 0  The text 'Elephant' is categoriz...  The given text 'Elephant' refer...                95              98     animal    animal
# 1  The text 'Dog' is classified as ...  The text 'Dog' refers to a comm...                95              98     animal    animal
# 2  The text 'Car' matches the categ...  The provided text 'Car' refers ...                80             100    vehicle   vehicle
# 3  The text 'Train' refers to a mod...  The text 'Train' refers to a mo...                80              95    vehicle   vehicle
```

## Features

- Flexible task creation for various content analysis and generation tasks using Pydantic models
- Support for multiple LLM raters using Instructor
- Parallelization of predictions
- Streamlined input and output management

## License

This project is licensed under the [MIT License](LICENSE.txt).