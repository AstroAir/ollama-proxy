"""Comprehensive tests for the API module to improve coverage."""

import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from src.app import create_app
from src.models import OllamaChatRequest, OllamaGenerateRequest, OllamaEmbeddingsRequest
from src.multi_provider_config import MultiProviderSettings
from src.openrouter import OpenRouterResponse


@pytest.fixture
def mock_openrouter_client():
    """Create a mock OpenRouter client."""
    client = AsyncMock()
    client.fetch_models.return_value = OpenRouterResponse(
        data={
            "data": [
                {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
                {"id": "openai/gpt-4", "name": "OpenAI: GPT-4"},
                {"id": "anthropic/claude-3-sonnet", "name": "Anthropic: Claude 3 Sonnet"},
            ]
        },
        status_code=200,
        headers={},
        metrics=AsyncMock()
    )
    return client


@pytest.fixture
def test_app(mock_openrouter_client):
    """Create test app with mocked dependencies."""
    settings = MultiProviderSettings(
        openrouter_enabled=True,
        openrouter_api_key="test-api-key-1234567890",
    )
    
    with patch("src.openrouter.OpenRouterClient") as mock_client_class:
        mock_client_class.return_value = mock_openrouter_client
        app = create_app(settings)
        return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    with TestClient(test_app) as c:
        yield c


class TestAPIEndpoints:
    """Test API endpoints comprehensively."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct message."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == "Ollama is running"

    def test_api_version(self, client):
        """Test API version endpoint."""
        response = client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.2.0"

    def test_api_tags_endpoint(self, client):
        """Test API tags endpoint returns model list."""
        with patch("src.api.get_app_state") as mock_get_state:
            mock_state = Mock()
            mock_state.all_models = [
                {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
                {"id": "openai/gpt-4", "name": "OpenAI: GPT-4"},
            ]
            mock_state.model_filter.models = set()
            mock_get_state.return_value = mock_state
            
            response = client.get("/api/tags")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) >= 0

    def test_api_show_endpoint(self, client):
        """Test API show endpoint for model details."""
        with patch("src.api.get_app_state") as mock_get_state:
            mock_state = Mock()
            mock_state.all_models = [
                {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
            ]
            mock_state.model_filter.models = set()
            mock_get_state.return_value = mock_state
            
            response = client.post("/api/show", json={"name": "gemini-pro"})
            assert response.status_code == 200
            data = response.json()
            assert "modelfile" in data

    def test_api_chat_endpoint(self, client, mock_openrouter_client):
        """Test API chat endpoint."""
        # Mock the chat completion response
        mock_openrouter_client.chat_completion.return_value = OpenRouterResponse(
            data={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "google/gemini-pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I help you today?"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "total_tokens": 18
                }
            },
            status_code=200,
            headers={},
            metrics=AsyncMock()
        )
        
        with patch("src.api.get_app_state") as mock_get_state, \
             patch("src.api.get_openrouter_client") as mock_get_client:
            
            mock_state = Mock()
            mock_state.all_models = [
                {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
            ]
            mock_state.model_filter.models = set()
            mock_state.ollama_to_openrouter_map = {"gemini-pro": "google/gemini-pro"}
            mock_get_state.return_value = mock_state
            mock_get_client.return_value = mock_openrouter_client
            
            chat_request = {
                "model": "gemini-pro",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "stream": False
            }
            
            response = client.post("/api/chat", json=chat_request)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert data["message"]["role"] == "assistant"

    def test_api_generate_endpoint(self, client):
        """Test API generate endpoint."""
        with patch("src.api.get_app_state") as mock_get_state, \
             patch("src.openrouter.OpenRouterClient") as mock_client_class:
            
            mock_client = AsyncMock()
            mock_client.chat_completion.return_value = OpenRouterResponse(
                data={
                    "choices": [{"message": {"content": "Generated response"}}]
                },
                status_code=200,
                headers={},
                metrics=AsyncMock()
            )
            mock_client_class.return_value = mock_client
            
            mock_state = Mock()
            mock_state.all_models = [{"id": "google/gemini-pro", "name": "Google: Gemini Pro"}]
            mock_state.model_filter.models = set()
            mock_state.ollama_to_openrouter_map = {"gemini-pro": "google/gemini-pro"}
            mock_get_state.return_value = mock_state
            
            generate_request = {
                "model": "gemini-pro",
                "prompt": "Tell me a joke",
                "stream": False
            }
            
            response = client.post("/api/generate", json=generate_request)
            assert response.status_code == 200
            data = response.json()
            assert "response" in data

    def test_api_embeddings_endpoint(self, client, mock_openrouter_client):
        """Test API embeddings endpoint."""
        mock_openrouter_client.embeddings.return_value = OpenRouterResponse(
            data={
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3],
                        "index": 0
                    }
                ],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            },
            status_code=200,
            headers={},
            metrics=AsyncMock()
        )
        
        with patch("src.api.get_app_state") as mock_get_state, \
             patch("src.api.get_openrouter_client") as mock_get_client:
            
            mock_state = Mock()
            mock_state.all_models = [
                {"id": "openai/text-embedding-ada-002", "name": "OpenAI: Ada 002"},
            ]
            mock_state.model_filter.models = set()
            mock_state.ollama_to_openrouter_map = {"ada-002": "openai/text-embedding-ada-002"}
            mock_get_state.return_value = mock_state
            mock_get_client.return_value = mock_openrouter_client
            
            embeddings_request = {
                "model": "ada-002",
                "prompt": "Hello world"
            }
            
            response = client.post("/api/embeddings", json=embeddings_request)
            assert response.status_code == 200
            data = response.json()
            assert "embedding" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        with patch("src.api.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_collector.get_health_status.return_value = {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": 100.0
            }
            mock_get_collector.return_value = mock_collector
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        with patch("src.api.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_collector.get_metrics.return_value = []
            mock_collector.get_endpoint_stats.return_value = {}
            mock_get_collector.return_value = mock_collector
            
            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert "statistics" in data
            assert "timestamp" in data

    def test_unsupported_endpoints(self, client):
        """Test unsupported endpoints return 501."""
        unsupported_endpoints = [
            ("/api/create", "post"),
            ("/api/copy", "post"),
            ("/api/delete", "delete"),
            ("/api/pull", "post"),
            ("/api/push", "post"),
        ]
        
        for endpoint, method in unsupported_endpoints:
            if method == "post":
                response = client.post(endpoint, json={})
            elif method == "delete":
                response = client.delete(endpoint)
            
            assert response.status_code == 501
            assert "not supported" in response.json()["detail"]


class TestAPIErrorHandling:
    """Test API error handling scenarios."""

    def test_invalid_model_name(self, client):
        """Test handling of invalid model names."""
        with patch("src.api.get_app_state") as mock_get_state:
            mock_state = Mock()
            mock_state.all_models = []
            mock_state.model_filter.models = set()
            mock_state.ollama_to_openrouter_map = {}
            mock_get_state.return_value = mock_state
            
            chat_request = {
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
            
            response = client.post("/api/chat", json=chat_request)
            assert response.status_code == 400

    def test_malformed_request(self, client):
        """Test handling of malformed requests."""
        response = client.post("/api/chat", json={"invalid": "request"})
        assert response.status_code == 422  # Validation error
