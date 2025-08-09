"""
Configuration class for feedback functionality.
"""

from pydantic import BaseModel, Field


class FeedbackConfig(BaseModel):
    """
    Configuration for feedback with example data.
    
    Controls aspects like how many examples to show, text truncation limits,
    and thresholds for various analysis features.
    """
    
    max_examples_to_process: int = Field(default=20, ge=0, description="How many examples to run through task")
    max_examples_to_show: int = Field(default=5, ge=0, description="How many examples to show in feedback")
    max_text_length: int = Field(default=200, ge=1, description="Maximum text length before truncation")
    test_task_raters: bool = Field(default=True, description="Run test() method on each rater before examples")
    tabulation_threshold: int = Field(default=11, ge=1, description="Max unique values to tabulate")
    num_bins: int = Field(default=5, ge=2, description="Number of bins for high-cardinality integers")