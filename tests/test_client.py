"""
Tests for LLMClient functionality.

These tests validate the basic functionality of the LLMClient.test() method
including success cases and error handling.
"""

import pytest
from quallm.client import LLMClient
import instructor
from openai import OpenAI
from unittest.mock import Mock, patch
from pydantic import BaseModel
import httpx


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


def test_timeout_values_and_preservation():
    """Test timeout defaults, custom values, and preservation in copy."""
    from quallm.client import DEFAULT_TIMEOUT
    # Default timeout
    client = LLMClient()
    assert client.timeout == DEFAULT_TIMEOUT
    
    # Custom timeout
    client_custom = LLMClient(timeout=30.0)
    assert client_custom.timeout == 30.0
    
    # Copy preserves timeout
    copy = client_custom.copy()
    assert copy.timeout == 30.0


def test_timeout_actually_times_out():
    """Verify that a very low timeout actually causes a timeout error."""
    client = LLMClient(language_model="olmo2", timeout=0.001)
    
    with pytest.raises((httpx.TimeoutException, httpx.ConnectTimeout, Exception)) as exc_info:
        client.test()
    
    error_str = str(exc_info.value).lower()
    assert 'timeout' in error_str or 'timed out' in error_str or '0.001' in str(exc_info.value)


def test_backend_specific_timeout_handling():
    """Test that LiteLLM includes timeout in request but OpenAI doesn't."""
    
    # Mock for LiteLLM backend
    mock_litellm = Mock()
    mock_litellm.chat.completions.create.return_value = Mock()
    mock_litellm.create_fn = Mock()
    mock_litellm.create_fn.__module__ = 'litellm.main'
    
    client_litellm = LLMClient(client=mock_litellm, timeout=25.0)
    
    class TestModel(BaseModel):
        result: str
    
    # LiteLLM should include timeout in request
    client_litellm.request("system", "user", TestModel)
    call_kwargs = mock_litellm.chat.completions.create.call_args[1]
    assert 'timeout' in call_kwargs
    assert call_kwargs['timeout'] == 25.0
    
    # Mock for OpenAI backend  
    mock_openai = Mock()
    mock_openai.chat.completions.create.return_value = Mock()
    mock_openai.create_fn = Mock()
    mock_openai.create_fn.__module__ = 'openai.resources.chat'
    
    client_openai = LLMClient(client=mock_openai, timeout=25.0)
    
    # OpenAI should NOT include timeout in request
    client_openai.request("system", "user", TestModel)
    call_kwargs = mock_openai.chat.completions.create.call_args[1]
    assert 'timeout' not in call_kwargs


def test_backend_detection():
    """Test backend detection for routing timeout behavior."""
    # Standard client should detect as 'openai'
    client = LLMClient()
    assert client.backend == 'openai'
    
    # LiteLLM client should detect as 'litellm'
    client_lite = LLMClient.from_litellm()
    assert client_lite.backend == 'litellm'


@patch('quallm.utils.instructor_response_mode_tester.InstructorResponseModeTester.evaluate_response_modes')
def test_factory_methods_accept_timeout(mock_evaluate):
    """Test that factory methods properly handle timeout parameter."""
    # Mock evaluation for from_response_mode_evaluation
    mock_result = Mock()
    mock_result.get_recommended_response_mode.return_value = "JSON"
    mock_evaluate.return_value = mock_result
    
    # Test from_litellm
    client1 = LLMClient.from_litellm(timeout=20.0)
    assert client1.timeout == 20.0
    
    # Test from_response_mode_evaluation
    client2 = LLMClient.from_response_mode_evaluation(model="test", timeout=25.0)
    assert client2.timeout == 25.0
    
    # Verify timeout was passed to evaluator
    call_kwargs = mock_evaluate.call_args[1]
    assert call_kwargs['timeout'] == 25.0