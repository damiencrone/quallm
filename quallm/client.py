
from openai import OpenAI
from pydantic import BaseModel
from typing import List
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
                 max_retries: int = DEFAULT_RETRIES,
                 role_args: dict = {}):
        self.client = client
        self.language_model = language_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.role_args = role_args

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
    
    def set_role_args(self, role_args: dict):
        """Set the role arguments for the LLM client"""
        self.role_args = role_args
        
    def assign_roles(self, roles: List[dict]) -> list:
        """Generate a list of LLM clients with different roles defined by a list of role argument dictionaries"""
        rater_list = []
        for role in roles:
            llm = self.copy()
            llm.set_role_args(role)
            rater_list.append(llm)
        return rater_list
    
    def copy(self):
        return LLMClient(
            client=self.client,
            language_model=self.language_model,
            temperature=self.temperature,
            max_retries=self.max_retries,
            role_args=self.role_args
        )
    
    def test(self, question=None) -> str:
        """Test the LLM client with a simple request"""
        class TestResponse(BaseModel):
            response: str
        if question is None:
            question = "Is a hot dog a sandwich, and if so, why?"
        resp = self.request(
            system_prompt="You are a helpful but opinionated assistant.",
            user_prompt=f"Provide a definitive 10 word statement that answers the question: {question}",
            response_model=TestResponse
        )
        return resp.response