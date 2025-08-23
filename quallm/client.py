from openai import OpenAI
from litellm import completion
import instructor
from pydantic import BaseModel
from typing import List, Union

# Default configuration goes directly through Ollama
DEFAULT_LANGUAGE_MODEL = "olmo2:13b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"  # Ollama doesn't actually need an API key.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RETRIES = 5
DEFAULT_TIMEOUT = 600.0

# Create the default client that goes directly through Ollama.
DEFAULT_CLIENT = instructor.from_openai(
    OpenAI(
        base_url=DEFAULT_BASE_URL,
        api_key=DEFAULT_API_KEY,
        timeout=DEFAULT_TIMEOUT,
    ),
    mode=instructor.Mode.JSON_SCHEMA,
)


class LLMClient:
    """Class containing an Instructor LLM client and request parameters.
    
    Args:
        client: Instructor client instance to use
        language_model: Model name to use for requests
        temperature: Temperature value for LLM inference  
        max_retries: Maximum number of retries for failed requests
        role_args: Role-specific arguments for prompts
        mode: Instructor response mode to use
        base_url: API endpoint URL
        api_key: API authentication key
        evaluate_response_modes: Whether to run response mode evaluation
        timeout: Request timeout in seconds (default 600.0)
    """
    def __init__(self,
                 client: instructor.client.Instructor = DEFAULT_CLIENT,
                 language_model: str = DEFAULT_LANGUAGE_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_retries: int = DEFAULT_RETRIES,
                 role_args: dict = {},
                 mode: Union[instructor.Mode, str, None] = None,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key: str = DEFAULT_API_KEY,
                 evaluate_response_modes: bool = None,
                 timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize an LLMClient instance.
        
        Args:
            client: Instructor client instance
            language_model: Model name to use
            temperature: Temperature for generation
            max_retries: Maximum number of retries
            role_args: Role-specific arguments
            mode: Instructor mode (JSON, MD_JSON, etc.) or 'auto'
            base_url: API base URL
            api_key: API key
            evaluate_response_modes: Whether to evaluate response modes
            timeout: Request timeout in seconds (default 600.0)
        """
        
        # If using DEFAULT_CLIENT but timeout differs from default, create new client
        if client == DEFAULT_CLIENT and timeout != DEFAULT_TIMEOUT:
            client = instructor.from_openai(
                OpenAI(base_url=base_url, api_key=api_key, timeout=timeout),
                mode=instructor.Mode.JSON_SCHEMA
            )
        
        # Handle auto mode logic first
        if mode == "auto":
            if evaluate_response_modes is None:
                # Auto-enable evaluation when mode="auto" and evaluation not explicitly specified
                evaluate_response_modes = True
            elif evaluate_response_modes is False:
                # Only raise error if user explicitly set evaluate_response_modes=False with mode="auto"
                raise ValueError("mode='auto' cannot be used with evaluate_response_modes=False")
        
        # Validate other parameter combinations
        if evaluate_response_modes and mode is not None and mode != "auto":
            raise ValueError("Cannot specify both evaluate_response_modes=True and mode (other than 'auto')")
        
        # Handle automatic response mode evaluation
        if evaluate_response_modes:
            if client != DEFAULT_CLIENT:
                raise ValueError("Cannot specify both client and evaluate_response_modes=True")
            
            # Import locally to avoid circular dependencies
            from quallm.utils.instructor_response_mode_tester import InstructorResponseModeTester
            
            # Run evaluation to find best response mode
            results = InstructorResponseModeTester.evaluate_response_modes(
                model=language_model,
                temperature=temperature,
                base_url=base_url,
                api_key=api_key,
                echo=False,  # Use default diagnostic tasks only, no console output
                timeout=timeout
            )
            recommended_mode_name = results.get_recommended_response_mode()
            if recommended_mode_name is None:
                raise ValueError(f"No working Instructor response modes found for model {language_model}")
            
            # Map mode name to Instructor mode
            mode_map = {
                "JSON": instructor.Mode.JSON,
                "MD_JSON": instructor.Mode.MD_JSON,
                "JSON_SCHEMA": instructor.Mode.JSON_SCHEMA,
                "TOOLS": instructor.Mode.TOOLS
            }
            recommended_mode = mode_map[recommended_mode_name]
            
            # Create client with recommended mode
            client = instructor.from_openai(
                OpenAI(base_url=base_url, api_key=api_key, timeout=timeout),
                mode=recommended_mode
            )
        
        # Handle mode parameter (mixed-type approach)
        elif mode is not None:
            if isinstance(mode, str):
                # Convert string mode to instructor.Mode enum
                mode_map = {
                    "JSON": instructor.Mode.JSON,
                    "MD_JSON": instructor.Mode.MD_JSON,
                    "JSON_SCHEMA": instructor.Mode.JSON_SCHEMA,
                    "TOOLS": instructor.Mode.TOOLS
                }
                if mode not in mode_map:
                    raise ValueError(f"Invalid mode string: {mode}. Valid options: {list(mode_map.keys())} or 'auto'")
                instructor_mode = mode_map[mode]
                if client == DEFAULT_CLIENT:
                    client = instructor.from_openai(
                        OpenAI(base_url=base_url, api_key=api_key, timeout=timeout),
                        mode=instructor_mode
                    )
            elif isinstance(mode, instructor.Mode):
                # Direct instructor.Mode enum usage
                if client == DEFAULT_CLIENT:
                    client = instructor.from_openai(
                        OpenAI(base_url=base_url, api_key=api_key, timeout=timeout),
                        mode=mode
                    )
            else:
                raise TypeError(f"mode must be str, instructor.Mode, or None. Got {type(mode)}")
        
        self.client = client
        self.language_model = language_model
        self.temperature = temperature
        self.max_retries = max_retries
        self.role_args = role_args
        self.timeout = timeout

    def request(self, system_prompt, user_prompt, response_model):
        """Request a response from LLM given a prompt."""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        
        # Build parameters dict
        create_params = {
            'model': self.language_model,
            'messages': messages,
            'response_model': response_model,
            'temperature': self.temperature,
            'max_retries': self.max_retries
        }
        
        # LiteLLM requires per-request timeout
        if self.backend == 'litellm':
            create_params['timeout'] = self.timeout
        
        resp = self.client.chat.completions.create(**create_params)
        return resp

    @classmethod
    def from_litellm(cls,
                     language_model: str = "ollama/olmo2:13b",
                     temperature: float = DEFAULT_TEMPERATURE,
                     max_retries: int = DEFAULT_RETRIES,
                     role_args: dict = {},
                     timeout: float = DEFAULT_TIMEOUT):
        """Create an LLMClient instance using a client that routes via litellm.
        This is an alternative constructor for users who want to use the litellm
        backend.
        
        Args:
            language_model: Model name to use
            temperature: Temperature for generation
            max_retries: Maximum number of retries
            role_args: Role-specific arguments
            timeout: Request timeout in seconds (default 600.0)
        """
        client = instructor.from_litellm(completion)
        return cls(client=client,
                   language_model=language_model,
                   temperature=temperature,
                   max_retries=max_retries,
                   role_args=role_args,
                   timeout=timeout)
    
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
        """
        Create a copy of this LLMClient instance.
        
        Returns:
            LLMClient: A new instance with the same configuration.
        """
        return LLMClient(
            client=self.client,
            language_model=self.language_model,
            temperature=self.temperature,
            max_retries=self.max_retries,
            role_args=self.role_args,
            timeout=self.timeout
        )
    
    @property
    def backend(self) -> str:
        """Detect the backend provider for this client.
        
        Returns:
            str: Backend name ('litellm', 'openai', or 'unknown')
        """
        if hasattr(self.client, 'create_fn'):
            fn_module = getattr(self.client.create_fn, '__module__', '')
            if fn_module and '.' in fn_module:
                return fn_module.split('.')[0]
        return 'unknown'

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
    
    def evaluate_available_response_modes(self, **kwargs):
        """Evaluate which Instructor response modes work with this client's configuration."""
        from quallm.utils.instructor_response_mode_tester import InstructorResponseModeTester
        return InstructorResponseModeTester.evaluate_response_modes(
            model=self.language_model,
            temperature=self.temperature,
            base_url=DEFAULT_BASE_URL,  # Use default base_url as it's not stored in client
            api_key=DEFAULT_API_KEY,    # Use default api_key as it's not stored in client
            timeout=kwargs.pop('timeout', self.timeout),  # Use client timeout as default
            **kwargs
        )
    
    @classmethod
    def from_response_mode_evaluation(cls,
                                    model: str,
                                    temperature: float = DEFAULT_TEMPERATURE,
                                    base_url: str = DEFAULT_BASE_URL,
                                    api_key: str = DEFAULT_API_KEY,
                                    include_default_tasks: bool = True,
                                    custom_tasks: List = None,
                                    echo: bool = True,
                                    max_workers: int = 1,
                                    timeout: float = 30.0,
                                    **kwargs) -> 'LLMClient':
        """
        Create an LLMClient with automatically selected best response mode.
        
        This method runs diagnostic tests to find the recommended Instructor response mode
        for the specified model, then returns a pre-configured LLMClient.
        
        Args:
            model: Model name to optimize for
            temperature: Temperature value for LLM inference (default DEFAULT_TEMPERATURE)
            base_url: API endpoint URL
            api_key: API authentication key
            include_default_tasks: Whether to run the three built-in diagnostic tasks (default True)
            custom_tasks: List of dictionaries with 'task' and 'dataset' keys. Each dict contains a Task object paired with its corresponding Dataset object (optional)
            echo: Whether to show detailed diagnostic output via console logging
            max_workers: Maximum number of worker threads for parallel processing (default 1)
            timeout: Request timeout in seconds (default 30.0)
            **kwargs: Additional arguments passed to LLMClient.__init__()
            
        Returns:
            LLMClient instance configured with recommended Instructor response mode
            
        Raises:
            ValueError: If no working response modes are found for the model
        """
        # Import locally to avoid circular dependencies
        from quallm.utils.instructor_response_mode_tester import InstructorResponseModeTester
        
        # Use the standalone evaluation method for full results
        evaluation_results = InstructorResponseModeTester.evaluate_response_modes(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            include_default_tasks=include_default_tasks,
            custom_tasks=custom_tasks,
            echo=echo,
            max_workers=max_workers,
            timeout=timeout
        )
        
        # Select recommended mode from results
        recommended_mode_name = evaluation_results.get_recommended_response_mode()
        if recommended_mode_name is None:
            raise ValueError(f"No working Instructor response modes found for model {model}")
            
        mode_map = {
            "JSON": instructor.Mode.JSON,
            "MD_JSON": instructor.Mode.MD_JSON,
            "JSON_SCHEMA": instructor.Mode.JSON_SCHEMA,
            "TOOLS": instructor.Mode.TOOLS
        }
        
        selected_client = instructor.from_openai(
            OpenAI(base_url=base_url, api_key=api_key, timeout=timeout),
            mode=mode_map[recommended_mode_name]
        )
        
        if echo:
            print(f"Created LLMClient with recommended response mode: {recommended_mode_name}")
            
        return cls(client=selected_client, language_model=model, temperature=temperature, max_retries=kwargs.get('max_retries', DEFAULT_RETRIES), role_args=kwargs.get('role_args', {}), timeout=timeout)