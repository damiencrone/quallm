from openai import OpenAI
from litellm import completion
import instructor
from pydantic import BaseModel
from typing import List

# Default configuration goes directly through Ollama
DEFAULT_LANGUAGE_MODEL = "olmo2:13b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"  # Ollama doesn't actually need an API key.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RETRIES = 5

# Create the default client that goes directly through Ollama.
DEFAULT_CLIENT = instructor.from_openai(
    OpenAI(
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
    ),
    mode=instructor.Mode.JSON,
)


class LLMClient:
    """Class containing an Instructor LLM client and request parameters."""
    def __init__(self,
                 client: instructor.client.Instructor = DEFAULT_CLIENT,
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
        """Request a response from LLM given a prompt."""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        resp = self.client.chat.completions.create(
            model=self.language_model,
            messages=messages,
            response_model=response_model,
            temperature=self.temperature,
            max_retries=self.max_retries
        )
        return resp

    @classmethod
    def from_litellm(cls,
                     language_model: str = "ollama/olmo2:13b",
                     temperature: float = DEFAULT_TEMPERATURE,
                     max_retries: int = DEFAULT_RETRIES,
                     role_args: dict = {}):
        """Create an LLMClient instance using a client that routes via litellm.
        This is an alternative constructor for users who want to use the litellm
        backend."""
        client = instructor.from_litellm(completion)
        return cls(client=client,
                   language_model=language_model,
                   temperature=temperature,
                   max_retries=max_retries,
                   role_args=role_args)
    
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
    
    @property
    def mode(self) -> str:
        """
        Get the current Instructor mode as a readable string.
        
        Returns:
            String representation of the current mode (e.g., "JSON", "MD_JSON", "TOOLS", "JSON_SCHEMA")
            Returns "UNKNOWN" if mode cannot be determined
        
        Example:
            >>> client = LLMClient()
            >>> client.mode
            "JSON"
        """
        try:
            # Access the mode from the instructor client
            if hasattr(self.client, 'mode') and hasattr(self.client.mode, 'name'):
                return self.client.mode.name
            elif hasattr(self.client, 'mode'):
                return str(self.client.mode)
            else:
                return "UNKNOWN"
        except AttributeError:
            return "UNKNOWN"

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