"""Test configuration and fixtures for ollama-proxy tests.

This module provides common test fixtures and configuration to ensure
consistent test environment setup across all test modules.
"""

import os
import pytest
from unittest.mock import patch, AsyncMock

# Ensure providers are registered before any tests run
from src.providers.init_providers import register_all_providers

# Force provider registration
register_all_providers()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with required environment variables."""
    with patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test-api-key-1234567890",
        "ENVIRONMENT": "development",
    }):
        yield


@pytest.fixture
def mock_openrouter_response():
    """Create a mock OpenRouter response for testing."""
    from src.openrouter import OpenRouterResponse
    
    mock_response_data = {
        "data": [
            {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
            {"id": "openai/gpt-4", "name": "OpenAI: GPT-4"},
        ]
    }
    return OpenRouterResponse(
        data=mock_response_data, 
        status_code=200, 
        headers={}, 
        metrics=AsyncMock()
    )


@pytest.fixture
def mock_chat_response():
    """Create a mock chat completion response."""
    from src.openrouter import OpenRouterResponse

    mock_response_data = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": "Hello"}}],
    }
    return OpenRouterResponse(
        data=mock_response_data,
        status_code=200,
        headers={},
        metrics=AsyncMock()
    )


@pytest.fixture
def test_client():
    """Create a test client with proper multi-provider setup."""
    from fastapi.testclient import TestClient
    from src.app import create_app
    from src.multi_provider_config import MultiProviderSettings

    # Create settings with OpenRouter enabled
    settings = MultiProviderSettings(
        openrouter_enabled=True,
        openrouter_api_key="test-api-key-1234567890",
    )

    # Mock provider initialization to avoid actual API calls
    with patch("src.providers.openrouter_provider.OpenRouterProvider.initialize") as mock_init:
        mock_init.return_value = None

        with patch("src.providers.openrouter_provider.OpenRouterProvider.list_models") as mock_list:
            mock_list.return_value = [
                {"id": "google/gemini-pro", "name": "gemini-pro:latest"},
                {"id": "openai/gpt-4", "name": "gpt-4:latest"},
            ]

            app = create_app(settings)
            with TestClient(app) as client:
                yield client
