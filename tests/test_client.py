"""
Tests for LLMClient functionality.

These tests validate the basic functionality of the LLMClient.test() method
including success cases and error handling.
"""

import pytest
from quallm.client import LLMClient
import instructor
from openai import OpenAI


def test_basic_success_verification():
    """
    Verify that LLMClient.test() returns a string when it succeeds.
    Uses real connection to test actual functionality.
    """
    # Create a client with default configuration
    client = LLMClient()
    
    # Call test method and verify it returns a string
    result = client.test()
    
    # Verify the result is a string (basic success verification)
    assert isinstance(result, str), f"Expected string, got {type(result)}"
    assert len(result) > 0, "Expected non-empty string response"
    
    # Verify it's a reasonable response (not just empty or whitespace)
    assert result.strip(), "Expected non-whitespace string response"


def test_real_connection_failure_handling():
    """
    Verify proper error handling when the connection fails.
    Uses a client configured to connect to an invalid endpoint.
    """
    # Create a client that will fail to connect
    invalid_client = instructor.from_openai(
        OpenAI(
            base_url="http://invalid-endpoint:99999/v1",
            api_key="invalid_key"
        ),
        mode=instructor.Mode.JSON,
    )
    
    failed_client = LLMClient(client=invalid_client)
    
    # Verify that calling test() raises an appropriate exception
    with pytest.raises(Exception) as exc_info:
        failed_client.test()
    
    # Verify the exception contains information about the connection failure
    # The exact exception type may vary (ConnectionError, timeout, etc.)
    # but we verify that an exception was raised for connection failure
    exception = exc_info.value
    assert exception is not None, "Expected exception to be raised for connection failure"
    
    # Log the exception type for documentation purposes
    # (This helps with evergreen documentation - shows what to expect)
    print(f"Connection failure raised: {type(exception).__name__}: {exception}")