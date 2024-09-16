
from typing import List, Union
import re

class Prompt:
    """
    A class to manage and format prompt templates for language model tasks.

    This class handles the creation and formatting of system and user prompts,
    separating task-specific arguments from data arguments. It allows for 
    step-wise formatting of prompts, first with task-specific information and 
    then with data for inference.

    Attributes:
        system_template (str): Template string for the system prompt.
        user_template (str): Template string for the user prompt.
        data_args (List[str]): List of argument names to be filled with data at inference time.
        task_args (List[str]): List of argument names to be filled with task-specific information.
        has_formattable_task_arguments (bool): Indicates if there are still unformatted task arguments.
        unformatted_task_arguments (set): Set of task argument names that haven't been formatted yet.
        formatted_task_arguments (dict): Dictionary of task arguments that have been formatted.

    Args:
        system_template (str): Template string for the system prompt.
        user_template (str): Template string for the user prompt.
        data_args (Union[str, List[str]]): Argument name(s) to be filled with data at inference time.
    """
    def __init__(self, system_template: str, user_template: str, data_args: Union[str, List[str]]):
        self.system_template = system_template
        self.user_template = user_template
        if isinstance(data_args, str):
            data_args = [data_args]
        all_arguments = self.extract_arguments(system_template, user_template)
        self.data_args = self.infer_data_args(all_arguments, data_args)
        self.task_args = [arg for arg in all_arguments if arg not in self.data_args]
        
        # Initialize formatting status
        self.has_formattable_task_arguments = bool(self.task_args)
        self.unformatted_task_arguments = set(self.task_args)  # All task arguments start as unformatted
        self.formatted_task_arguments = {}  # To hold the formatted values

    def extract_arguments(self, *templates: str) -> List[str]:
        """Extract all argument names from the provided templates."""
        placeholders = set()
        for template in templates:
            placeholders.update(re.findall(r'\{(.*?)\}', template))
        return list(placeholders)
    
    def infer_data_args(self, all_arguments, data_args):
        """If not specified, try to determine data arg from other inputs"""
        if len(all_arguments) == 1 and data_args is None:
            return all_arguments
        elif data_args is None:
            raise ValueError(f"Cannot infer data_args from prompt template arguments: {all_arguments}. Please provide data_args explicitly.")
        else:
            return data_args
    
    def format_template(self, template: str, **kwargs) -> str:
        """Formats the template using provided keyword arguments."""
        template_args = self.extract_arguments(template)
        formattable_args = {k: kwargs.get(k, '{' + k + '}') for k in template_args}
        return template.format(**formattable_args)
    
    def define_task(self, **kwargs):
        """Format the system and user prompts with provided task args (task-defining keyword arguments)."""
        if not self.task_args:
            self.has_formattable_task_arguments = False
            self.system_prompt = self.system_template
            self.user_prompt = self.user_template
            return self
        else:
            assert self.has_formattable_task_arguments, f"All task arguments [{', '.join(self.task_args)}] have already been formatted; No task arguments available for formatting."
            assert set(kwargs.keys()) == set(self.task_args), "Keys in kwargs do not match the expected task arguments."
            for arg in self.task_args:
                if arg in kwargs:
                    self.formatted_task_arguments[arg] = kwargs[arg]
                    self.unformatted_task_arguments.discard(arg)
            self.system_prompt = self.format_template(self.system_template, **self.formatted_task_arguments)
            self.user_prompt = self.format_template(self.user_template, **self.formatted_task_arguments)
            self.has_formattable_task_arguments = bool(self.unformatted_task_arguments)
        return self
    
    def insert_data(self, **kwargs):
        """Format the system and user prompts with provided data arguments, returning a new instance of the prompt."""
        assert set(kwargs.keys()) == set(self.data_args), "Keys in kwargs do not match the expected data arguments."
        formatted_prompt = Prompt(self.system_template, self.user_template, self.data_args)
        formatted_prompt.system_prompt = formatted_prompt.format_template(self.system_prompt, **kwargs)
        formatted_prompt.user_prompt = formatted_prompt.format_template(self.user_prompt, **kwargs)
        return formatted_prompt