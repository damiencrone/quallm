# quallm

quallm is a Python library designed to simplify and streamline LLM-assisted content analysis tasks. It provides a flexible framework for defining, executing, and analyzing various content analysis (and similar) tasks using language models. quallm is built to work seamlessly with both local (i.e., on-device) models through [Ollama](https://ollama.com) and cloud-hosted models (e.g., those provided by Together, Groq, OpenAI, and other services), providing the flexibility to choose the most suitable option for a given task.

Key features of quallm include:

- A modular design that separates tasks, data handling, inference, and output handling
- Output validation using [Instructor](https://python.useinstructor.com)
- Allows streamlined routing of requests through [LiteLLM](https://docs.litellm.ai/docs/)
- Support for defining arbitrary structured outputs using [Pydantic](https://docs.pydantic.dev/latest/) models
- Built-in, customizable tasks for common analyses like single-labeled classification
- Support for multiple "raters" (different language model instances) and "roles" (different prompts) for the same task
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

If using cloud-hosted LLMs, you will likely need to create an account and set up an API key with your chosen provider. The easiest way to use a cloud provider will be to initialize a client with `LLMClient.from_litellm()`.

## Usage

quallm revolves around a few simple elements. The most important of these is the `Task`, which is essentially the combination of (1) an output schema (a Pydantic model) defining the structure of the response the LLM will generate, and (2) a prompt template providing instructions for the LLM. Once a task is defined, all one needs is one or more LLMs (or "raters") to perform the task, and a dataset, which contains each individual observation (e..g, a survey response, document, etc.) which the task will be performed on.

Here are some basic examples of how to use quallm:

### Setting up and testing an LLM

An LLM (or rater) is configured with the `LLMClient` class (which is built around Instructor's `client.Instructor` class). In the example below, we use a locally-hosted LLM (Phi-3.5, a 3.8B parameter model released by Microsoft in August 2024) but the same class can be used to configure cloud-hosted LLMs (with some additional inputs). Once your LLM is instantiated, you can perfrom a simple test to ensure it is working as expected. The test method takes an optional question argument; you can probably guess the default.

```python
from quallm import LLMClient

llm = LLMClient(language_model="phi3.5")
llm.test()
# Output:
# 'Hot dogs are technically considered sandwiches as they consist of meat between slices of bread.'
```

If using a cloud-hosted LLM, you can specify the model using `LLMClient.from_litellm()`. Assuming your API key is set in your environment, you can use the following code to configure a cloud-hosted LLM:

```python
from quallm import LLMClient

# With OpenAI
llm = LLMClient.from_litellm(language_model="openai/gpt-4o")
llm.test()
# Output:
# "A hot dog is a sandwich because it's meat in bread."

# With Gemini, at a higher temperature
llm = LLMClient.from_litellm(language_model="gemini/gemini-2.0-flash", temperature=2.0)
llm.test()
# Output:
# "A hot dog is not a sandwich because it is its own category!"
```

For more information on providers supported by LiteLLM, see the [LiteLLM documentation](https://docs.litellm.ai/docs/providers). Otherwise, if not routing requests through LiteLLM, for further details on setting up different LLMs with different providers, refer to the [Instructor](https://python.useinstructor.com/integrations/) documentation on integrations with specific LLM providers.

### Using a pre-existing task

Simple tasks such as single-label classification or sentiment analysis can be performed with (and easily adapted from) pre-existing tasks[^1]. In the example below, we use a relatively small local LLM (Phi-3.5), which will likely work on most consumer devices. In this task, we ask the LLM to classify three texts, using a pre-configured sentiment analysis task (an instance of `Task`), which returns a sentiment classification (one of five categories), along with an explanation and confidence rating.

```python
from quallm import LLMClient, Dataset, Predictor
from quallm.tasks import SentimentAnalysisTask

llm = LLMClient(language_model="phi3.5")
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

[^1]: Although pre-defined tasks *do* come with pre-written prompt templates, these are primarily for demonstration purposes. Users are advised to tailor prompts to their specific use cases, as the default prompts may be subject to change, and are unlikely to be optimal for your given combination of task, LLM, and dataset.

### Defining a new task

Aribtrary tasks can also be defined using a `TaskConfig` with (at minimum) a Pydantic model, and system and user prompt template. In the example below, we define a trivial task: The response model (i.e., the thing the LLM is tasked with generating) is a list of strings on a given topic (the `ListResponse` Pydantic model). The prompt template (in the definition of `task_config`) is a barebones template with a placeholder for the topic (which is the datapoint or observation which is piped into the prompt template at inference time). In this case, the "data" are two observations: "ethical precepts" and "moral transgressions". As instructed, the LLM returns a lists of both.

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

data = ["ethical precepts", "moral transgressions"]
llm_client = LLMClient(language_model="phi3.5")
list_generation_task = Task.from_config(task_config)
predictor = Predictor(task=list_generation_task, raters=llm_client)
prediction = predictor.predict(data)

# Print the results
results = prediction.expand(data=data, explode="items")
results
# Output:
#                   data                                              items
# 0     ethical precepts                       Do no harm (non-maleficence)
# 1     ethical precepts                   Be honest and act with integrity
# 2     ethical precepts  Respect others' rights and dignity (respect fo...
# 3     ethical precepts  Promote fairness and justice in actions, ensdi...
# 4 moral transgressions                              Cheating on a partner
# 5 moral transgressions               Stealing from others without remorse
# 6 moral transgressions   Lying to manipulate or deceive for personal gain
# 7 moral transgressions         Disrespecting someone's dignity and rights
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
#                        reasoning_llama                       reasoning_phi  confidence_llama  confidence_phi  code_llama  code_phi
# 0  The text 'Elephant' is categoriz...  The given text 'Elephant' refer...                95              98      animal    animal
# 1  The text 'Dog' is classified as ...  The text 'Dog' refers to a comm...                95              98      animal    animal
# 2  The text 'Car' matches the categ...  The provided text 'Car' refers ...                80             100     vehicle   vehicle
# 3  The text 'Train' refers to a mod...  The text 'Train' refers to a mo...                80              95     vehicle   vehicle
```

### Assigning different roles to different raters

In some cases, users may want to assign different roles to different raters. For example, one rater might take the perspective of a social psychologist, while another might take the perspective of a political scientist. This can be done by defining (1) defining role arguments in the task config, (2) defining a list of role dictionaries and passing these to the `assign_roles` method of the `LLMClient` object (which returns a list of raters, each with a different role).

Note that the content of the role is defined by placeholders in the system (and/or user) prompt template. In this specific example, the role is simply defined by a single placeholder `{role}`, which appears in one location in the system prompt template, but much more complex roles or combinations of instructions can be defined by adding more placeholders to the prompt template (along with corresponding key-value pairs in the role dictionaries for each rater).

```python
from pydantic import BaseModel, Field
from typing import List
from quallm.tasks import Task, TaskConfig
from quallm import LLMClient, Predictor

class ListResponse(BaseModel):
    items: List[str] = Field(description="A list of items relating to a topic")

task_config = TaskConfig(
    response_model=ListResponse,
    system_template="{role}. Generate a short list based on the topic provided.",
    user_template="Topic: {topic}",
    data_args="topic",
    role_args="role",
)

data = ["foods", "inanimate objects"]
llm = llm_a = LLMClient(language_model="olmo2")
roles = [{"role": "You are a helpful assistant who only responds with things beginning with the letter A"},
         {"role": "You are a helpful assistant who only responds with things beginning with the letter B"}]
rater_list = llm.assign_roles(roles)
list_generation_task = Task.from_config(task_config)
predictor = Predictor(task=list_generation_task, raters=rater_list)
prediction = predictor.predict(data)

# Print the results
results = prediction.expand(data=data, explode="items")
print(results)
# Output:
#                data   rater      items
# 0 foods               r1     Apples
# 0 foods               r1     Avocado
# 0 foods               r1     Almonds
# 0 foods               r1     Artichoke
# 1 inanimate objects   r1     apple
# 1 inanimate objects   r1     airplane
# 1 inanimate objects   r1     antenna
# 2 foods               r2     Banana
# 2 foods               r2     Blueberry
# 2 foods               r2     Bread
# 2 foods               r2     Beans
# 3 inanimate objects   r2     book
# 3 inanimate objects   r2     ball
# 3 inanimate objects   r2     bottle
```

## License

This project is licensed under the [MIT License](LICENSE.txt).