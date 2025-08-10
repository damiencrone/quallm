__version__ = "0.8.2"

from .client import LLMClient
from .dataset import Dataset
from .prompt import Prompt
from .tasks import *
from .quallm import *
from .utils.instructor_mode_tester import InstructorModeTester, ModeTestResult, ModeEvaluationResults