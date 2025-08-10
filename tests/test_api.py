from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import AppState, Settings
from src.openrouter import OpenRouterClient, OpenRouterResponse


@pytest.fixture
def client():
    """Create test client with proper multi-provider setup."""
    from src.multi_provider_config import MultiProviderSettings

    # Create settings with OpenRouter enabled
    settings = MultiProviderSettings(
        openrouter_enabled=True,
        openrouter_api_key="test-api-key-1234567890",
    )

    # Mock the OpenRouterClient.fetch_models method
    with patch("src.openrouter.OpenRouterClient.fetch_models") as mock_fetch:
        mock_response_data = {
            "data": [
                {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
                {"id": "openai/gpt-4", "name": "OpenAI: GPT-4"},
            ]
        }
        mock_response = OpenRouterResponse(
            data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
        )
        mock_fetch.return_value = mock_response

        app = create_app(settings)
        with TestClient(app) as c:
            yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.text == "Ollama is running"


def test_api_version(client):
    response = client.get("/api/version")
    assert response.status_code == 200
    assert response.json() == {"version": "0.2.0"}


def test_api_tags(client):
    response = client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 2
    assert data["models"][0]["name"] == "gemini-pro:latest"
    assert data["models"][1]["name"] == "gpt-4:latest"


@patch("src.openrouter.OpenRouterClient.chat_completion")
def test_api_chat_non_streaming(mock_chat_completion, client):
    mock_response_data = {
        "id": "chatcmpl-123",
        "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
    }
    mock_response = OpenRouterResponse(
        data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
    )
    mock_chat_completion.return_value = mock_response

    payload = {
        "model": "gemini-pro:latest",  # Use the model name that exists in our mock data
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"] == "Hello"
    assert data["done"] is True


@patch("src.openrouter.OpenRouterClient.chat_completion_stream")
def test_api_chat_streaming(mock_chat_completion_stream, client):
    async def mock_stream():
        yield b"""data: {"id": "1", "choices": [{"delta": {"content": "Hel"}}]}

"""
        yield b"""data: {"id": "2", "choices": [{"delta": {"content": "lo"}}]}

"""
        yield b"""data: [DONE]

"""

    mock_chat_completion_stream.return_value = mock_stream()

    payload = {
        "model": "gemini-pro:latest",  # Use the model name that exists in our mock data
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    assert "Hel" in response.text
    assert "lo" in response.text
