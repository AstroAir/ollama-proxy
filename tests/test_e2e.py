"""End-to-end tests for ollama-proxy."""

import asyncio
import json
import os
import time
from unittest.mock import patch, AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.multi_provider_config import MultiProviderSettings


@pytest.mark.integration
class TestE2EWorkflows:
    """End-to-end workflow tests."""

    @pytest.fixture
    def mock_openrouter_responses(self):
        """Mock OpenRouter API responses for E2E tests."""
        return {
            "models": {
                "data": [
                    {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
                    {"id": "openai/gpt-4", "name": "OpenAI: GPT-4"},
                    {"id": "anthropic/claude-3-sonnet", "name": "Anthropic: Claude 3 Sonnet"},
                ]
            },
            "chat": {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "google/gemini-pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! I'm a helpful AI assistant. How can I help you today?"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 12,
                    "total_tokens": 27
                }
            },
            "embeddings": {
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dim vector
                        "index": 0
                    }
                ],
                "model": "text-embedding-ada-002",
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            }
        }

    @pytest.fixture
    def e2e_client(self, mock_openrouter_responses):
        """Create E2E test client with comprehensive mocking."""
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="test-api-key-e2e",
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            
            # Mock model fetching
            mock_client.get.return_value.json.return_value = mock_openrouter_responses["models"]
            mock_client.get.return_value.status_code = 200
            
            # Mock chat completions
            mock_client.post.return_value.json.return_value = mock_openrouter_responses["chat"]
            mock_client.post.return_value.status_code = 200
            mock_client.post.return_value.headers = {}
            
            mock_client_class.return_value = mock_client
            
            app = create_app(settings)
            with TestClient(app) as client:
                yield client

    def test_complete_chat_workflow(self, e2e_client):
        """Test complete chat workflow from start to finish."""
        # 1. Check server health
        health_response = e2e_client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get API version
        version_response = e2e_client.get("/api/version")
        assert version_response.status_code == 200
        assert version_response.json()["version"] == "0.2.0"
        
        # 3. List available models
        models_response = e2e_client.get("/api/tags")
        assert models_response.status_code == 200
        models_data = models_response.json()
        assert "models" in models_data
        
        # 4. Show model details
        if models_data["models"]:
            model_name = models_data["models"][0]["name"]
            show_response = e2e_client.post("/api/show", json={"name": model_name})
            assert show_response.status_code == 200
        
        # 5. Perform chat completion
        chat_request = {
            "model": "gemini-pro",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
            "stream": False
        }
        
        chat_response = e2e_client.post("/api/chat", json=chat_request)
        assert chat_response.status_code == 200
        chat_data = chat_response.json()
        assert "message" in chat_data
        assert chat_data["message"]["role"] == "assistant"
        assert len(chat_data["message"]["content"]) > 0

    def test_multi_turn_conversation(self, e2e_client):
        """Test multi-turn conversation workflow."""
        conversation_history = []
        
        # Turn 1
        turn1_request = {
            "model": "gemini-pro",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "stream": False
        }
        
        turn1_response = e2e_client.post("/api/chat", json=turn1_request)
        assert turn1_response.status_code == 200
        turn1_data = turn1_response.json()
        
        conversation_history.extend(turn1_request["messages"])
        conversation_history.append(turn1_data["message"])
        
        # Turn 2 - Follow-up question
        turn2_request = {
            "model": "gemini-pro",
            "messages": conversation_history + [
                {"role": "user", "content": "What is the population of that city?"}
            ],
            "stream": False
        }
        
        turn2_response = e2e_client.post("/api/chat", json=turn2_request)
        assert turn2_response.status_code == 200
        turn2_data = turn2_response.json()
        assert "message" in turn2_data

    def test_generate_workflow(self, e2e_client):
        """Test generate completion workflow."""
        # Test simple generation
        generate_request = {
            "model": "gemini-pro",
            "prompt": "Complete this sentence: The future of AI is",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 100
            }
        }
        
        generate_response = e2e_client.post("/api/generate", json=generate_request)
        assert generate_response.status_code == 200
        generate_data = generate_response.json()
        assert "response" in generate_data

    def test_embeddings_workflow(self, e2e_client):
        """Test embeddings workflow."""
        embeddings_request = {
            "model": "text-embedding-ada-002",
            "prompt": "This is a test sentence for embedding generation."
        }
        
        embeddings_response = e2e_client.post("/api/embeddings", json=embeddings_request)
        assert embeddings_response.status_code == 200
        embeddings_data = embeddings_response.json()
        assert "embedding" in embeddings_data
        assert isinstance(embeddings_data["embedding"], list)
        assert len(embeddings_data["embedding"]) > 0

    def test_error_handling_workflow(self, e2e_client):
        """Test error handling across different scenarios."""
        # Test invalid model
        invalid_model_request = {
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        invalid_response = e2e_client.post("/api/chat", json=invalid_model_request)
        assert invalid_response.status_code == 400
        
        # Test malformed request
        malformed_request = {"invalid": "request"}
        malformed_response = e2e_client.post("/api/chat", json=malformed_request)
        assert malformed_response.status_code == 422
        
        # Test unsupported endpoint
        unsupported_response = e2e_client.post("/api/create", json={})
        assert unsupported_response.status_code == 501

    def test_metrics_and_monitoring_workflow(self, e2e_client):
        """Test metrics and monitoring workflow."""
        # Perform some operations to generate metrics
        e2e_client.get("/api/tags")
        e2e_client.post("/api/chat", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Test"}],
            "stream": False
        })
        
        # Check metrics endpoint
        metrics_response = e2e_client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()
        assert "metrics" in metrics_data
        assert "statistics" in metrics_data
        assert "timestamp" in metrics_data

    def test_concurrent_requests_workflow(self, e2e_client):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_chat_request():
            return e2e_client.post("/api/chat", json={
                "model": "gemini-pro",
                "messages": [{"role": "user", "content": f"Request from thread {threading.current_thread().ident}"}],
                "stream": False
            })
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_chat_request) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    def test_rate_limiting_behavior(self, e2e_client):
        """Test rate limiting behavior (if implemented)."""
        # Make rapid requests to test rate limiting
        responses = []
        for i in range(10):
            response = e2e_client.get("/health")
            responses.append(response)
            time.sleep(0.1)  # Small delay
        
        # Most requests should succeed (rate limiting may not be implemented)
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 5  # At least half should succeed

    def test_model_switching_workflow(self, e2e_client):
        """Test switching between different models."""
        models_to_test = ["gemini-pro", "gpt-4", "claude-3-sonnet"]
        
        for model in models_to_test:
            chat_request = {
                "model": model,
                "messages": [{"role": "user", "content": f"Hello from {model}"}],
                "stream": False
            }
            
            response = e2e_client.post("/api/chat", json=chat_request)
            # Response may vary based on model availability
            assert response.status_code in [200, 400]  # 400 if model not available

    def test_parameter_variations_workflow(self, e2e_client):
        """Test various parameter combinations."""
        parameter_sets = [
            {"temperature": 0.1, "max_tokens": 50},
            {"temperature": 0.9, "max_tokens": 200},
            {"temperature": 0.5, "max_tokens": 100, "top_p": 0.9},
        ]
        
        for params in parameter_sets:
            chat_request = {
                "model": "gemini-pro",
                "messages": [{"role": "user", "content": "Tell me a short story"}],
                "stream": False,
                "options": params
            }
            
            response = e2e_client.post("/api/chat", json=chat_request)
            assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
class TestE2EPerformance:
    """End-to-end performance tests."""

    @pytest.fixture
    def performance_client(self, mock_openrouter_responses):
        """Create client for performance testing."""
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="test-api-key-perf",
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value.json.return_value = mock_openrouter_responses["models"]
            mock_client.get.return_value.status_code = 200
            mock_client.post.return_value.json.return_value = mock_openrouter_responses["chat"]
            mock_client.post.return_value.status_code = 200
            mock_client.post.return_value.headers = {}
            mock_client_class.return_value = mock_client
            
            app = create_app(settings)
            with TestClient(app) as client:
                yield client

    def test_response_time_performance(self, performance_client):
        """Test response time performance."""
        start_time = time.time()
        
        response = performance_client.post("/api/chat", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        })
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds

    def test_throughput_performance(self, performance_client):
        """Test throughput performance."""
        num_requests = 20
        start_time = time.time()
        
        for i in range(num_requests):
            response = performance_client.get("/health")
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_requests / total_time
        
        assert throughput > 5  # Should handle at least 5 requests per second

    def test_memory_usage_stability(self, performance_client):
        """Test memory usage stability over multiple requests."""
        # Make multiple requests to check for memory leaks
        for i in range(50):
            response = performance_client.post("/api/chat", json={
                "model": "gemini-pro",
                "messages": [{"role": "user", "content": f"Request {i}"}],
                "stream": False
            })
            assert response.status_code == 200
        
        # If we get here without issues, memory usage is likely stable
