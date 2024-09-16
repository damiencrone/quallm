
from openai import OpenAI
import instructor

DEFAULT_LANGUAGE_MODEL = "gemma2"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama" # Used for ollama (which doesn't need an API key)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RETRIES = 5

default_client = instructor.from_openai(
    OpenAI(
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
    ),
    mode=instructor.Mode.JSON,
)


class LLMClient:
    """Class containing an Instructor LLM client and request parameters"""
    def __init__(self,
                 client: instructor.client.Instructor = default_client,
                 language_model: str = DEFAULT_LANGUAGE_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_retries: int = DEFAULT_RETRIES):
        self.client = client
        self.language_model = language_model
        self.temperature = temperature
        self.max_retries = max_retries

    def request(self, system_prompt, user_prompt, response_model):
        """Request a response from LLM given a prompt"""
        resp = self.client.chat.completions.create(
            model=self.language_model,
            messages=[{'role': 'system', 'content': system_prompt},
                      {'role': 'user', 'content': user_prompt}],
            response_model=response_model,
            temperature=self.temperature,
            max_retries=self.max_retries
        )
        return resp