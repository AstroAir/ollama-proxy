"""Integration tests for the ollama-proxy application."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.openrouter import OpenRouterResponse


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            # Mock the fetch_models call during app startup
            with patch("src.openrouter.OpenRouterClient.fetch_models") as mock_fetch:
                mock_response_data = {
                    "data": [
                        {"id": "openai/gpt-4", "name": "OpenAI: GPT-4"},
                        {"id": "anthropic/claude-3-sonnet",
                            "name": "Anthropic: Claude 3 Sonnet"},
                        {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
                    ]
                }
                mock_response = OpenRouterResponse(
                    data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
                )
                mock_fetch.return_value = mock_response

                app = create_app()
                with TestClient(app) as c:
                    yield c

    def test_complete_chat_workflow_non_streaming(self, client):
        """Test complete chat workflow without streaming."""
        # Mock the OpenRouter chat completion
        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            mock_response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
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
                    "completion_tokens": 15,
                    "total_tokens": 25
                }
            }
            mock_response = OpenRouterResponse(
                data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
            )
            mock_chat.return_value = mock_response

            # Test the complete workflow
            payload = {
                "model": "gpt-4:latest",
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "stream": False
            }

            response = client.post("/api/chat", json=payload)

            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["message"]["role"] == "assistant"
            assert data["message"]["content"] == "Hello! How can I help you today?"
            assert data["done"] is True

            # Verify the OpenRouter client was called correctly
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args
            # Resolved model name
            assert call_args[0][0]["model"] == "openai/gpt-4"
            assert call_args[0][0]["messages"][0]["content"] == "Hello, how are you?"
            assert call_args[1]["stream"] is False

    def test_complete_chat_workflow_streaming(self, client):
        """Test complete chat workflow with streaming."""
        # Mock the OpenRouter streaming chat completion
        with patch("src.openrouter.OpenRouterClient.chat_completion_stream") as mock_chat:
            async def mock_stream():
                yield b'data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": "Hello"}}]}\n\n'
                yield b'data: {"id": "chatcmpl-123", "choices": [{"delta": {"content": " there!"}}]}\n\n'
                yield b'data: [DONE]\n\n'

            mock_chat.return_value = mock_stream()

            # Test the complete streaming workflow
            payload = {
                "model": "claude-3-sonnet:latest",
                "messages": [
                    {"role": "user", "content": "Say hello"}
                ],
                "stream": True
            }

            response = client.post("/api/chat", json=payload)

            # Verify response
            assert response.status_code == 200
            response_text = response.text
            assert "Hello" in response_text
            assert " there!" in response_text
            assert "[DONE]" in response_text

            # Verify the OpenRouter client was called correctly
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args
            # Resolved model name
            assert call_args[0][0]["model"] == "anthropic/claude-3-sonnet"

    def test_model_resolution_workflow(self, client):
        """Test model name resolution workflow."""
        # Test that various model name formats are resolved correctly
        response = client.get("/api/tags")
        assert response.status_code == 200

        data = response.json()
        models = data["models"]

        # Verify models are properly formatted for Ollama compatibility
        model_names = [model["name"] for model in models]
        assert "gpt-4:latest" in model_names
        assert "claude-3-sonnet:latest" in model_names
        assert "gemini-pro:latest" in model_names

        # Verify model details
        gpt4_model = next(m for m in models if m["name"] == "gpt-4:latest")
        assert "size" in gpt4_model
        assert "digest" in gpt4_model
        assert "modified_at" in gpt4_model

    def test_error_handling_workflow(self, client):
        """Test error handling throughout the application."""
        # Test model not found error
        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            from src.providers.base import ProviderError, ProviderType
            from src.exceptions import ErrorContext

            # Make the mock raise a model not found error
            mock_chat.side_effect = ProviderError(
                "Model not found: nonexistent-model",
                provider_type=ProviderType.OPENROUTER,
                context=ErrorContext(additional_data={"model": "nonexistent-model"})
            )

            payload = {
                "model": "nonexistent-model:latest",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }

            # Use TestClient that doesn't raise server exceptions
            with TestClient(client.app, raise_server_exceptions=False) as error_client:
                response = error_client.post("/api/chat", json=payload)
                assert response.status_code == 500  # Provider errors become 500 errors
                data = response.json()
                assert "error" in data

    def test_health_and_monitoring_workflow(self, client):
        """Test health check and monitoring endpoints."""
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == "Ollama is running"

        # Test version endpoint
        response = client.get("/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.2.0"

    def test_request_validation_workflow(self, client):
        """Test request validation and error responses."""
        # Use TestClient that doesn't raise server exceptions
        with TestClient(client.app, raise_server_exceptions=False) as error_client:
            # Test invalid JSON
            response = error_client.post(
                "/api/chat", content="invalid json", headers={"content-type": "application/json"})
            assert response.status_code == 400  # JSON decode error

            # Test missing required fields
            response = error_client.post("/api/chat", json={})
            assert response.status_code == 400  # Missing required fields

        # Test invalid model format
        response = client.post("/api/chat", json={
            "model": "",  # Empty model name
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert response.status_code == 400  # Empty model name validation error

    def test_cors_and_headers_workflow(self, client):
        """Test CORS and header handling."""
        # Test CORS preflight
        response = client.options("/api/chat", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        })
        assert response.status_code == 200

        # Test actual request with CORS
        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            mock_response = OpenRouterResponse(
                data={"choices": [{"message": {"content": "Hello"}}]},
                status_code=200, headers={}, metrics=AsyncMock()
            )
            mock_chat.return_value = mock_response

            response = client.post("/api/chat",
                                   json={
                                       "model": "gpt-4:latest",
                                       "messages": [{"role": "user", "content": "Hello"}],
                                       "stream": False
                                   },
                                   headers={"Origin": "http://localhost:3000"}
                                   )
            assert response.status_code == 200


class TestConcurrentRequests:
    """Test concurrent request handling."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("src.openrouter.OpenRouterClient.fetch_models") as mock_fetch:
                mock_response_data = {
                    "data": [{"id": "openai/gpt-4", "name": "OpenAI: GPT-4"}]
                }
                mock_response = OpenRouterResponse(
                    data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
                )
                mock_fetch.return_value = mock_response

                app = create_app()
                with TestClient(app) as c:
                    yield c

    def test_concurrent_chat_requests(self, client):
        """Test handling multiple concurrent chat requests."""
        import concurrent.futures
        import threading

        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            # Use a simple counter - the mock will track call count automatically
            call_counter = [0]  # Use list for mutable reference

            def mock_chat_response(*args, **kwargs):
                call_counter[0] += 1
                return OpenRouterResponse(
                    data={"choices": [
                        {"message": {"content": f"Response {call_counter[0]}"}}]},
                    status_code=200, headers={}, metrics=AsyncMock()
                )

            mock_chat.side_effect = mock_chat_response

            def make_request(i):
                payload = {
                    "model": "gpt-4:latest",
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "stream": False
                }
                return client.post("/api/chat", json=payload)

            # Make 5 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request, i) for i in range(5)]
                responses = [future.result()
                             for future in concurrent.futures.as_completed(futures)]

            # Verify all requests succeeded
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "Response" in data["message"]["content"]

            # Verify all requests were processed
            assert mock_chat.call_count == 5


class TestPerformanceAndLimits:
    """Test performance characteristics and system limits."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("src.openrouter.OpenRouterClient.fetch_models") as mock_fetch:
                mock_response_data = {
                    "data": [{"id": "openai/gpt-4", "name": "OpenAI: GPT-4"}]
                }
                mock_response = OpenRouterResponse(
                    data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
                )
                mock_fetch.return_value = mock_response

                app = create_app()
                with TestClient(app) as c:
                    yield c

    def test_large_request_handling(self, client):
        """Test handling of large requests."""
        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            mock_response = OpenRouterResponse(
                data={"choices": [
                    {"message": {"content": "Processed large request"}}]},
                status_code=200, headers={}, metrics=AsyncMock()
            )
            mock_chat.return_value = mock_response

            # Create a large message (but within reasonable limits)
            large_content = "This is a test message. " * 1000  # ~25KB

            payload = {
                "model": "gpt-4:latest",
                "messages": [{"role": "user", "content": large_content}],
                "stream": False
            }

            response = client.post("/api/chat", json=payload)
            assert response.status_code == 200

            # Verify the large content was passed through
            mock_chat.assert_called_once()
            call_args = mock_chat.call_args
            assert len(call_args[0][0]["messages"][0]["content"]) > 20000

    def test_request_timeout_handling(self, client):
        """Test request timeout handling."""
        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            # Simulate a timeout
            import asyncio

            async def slow_response(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate slow response
                return OpenRouterResponse(
                    data={"choices": [
                        {"message": {"content": "Slow response"}}]},
                    status_code=200, headers={}, metrics=AsyncMock()
                )

            mock_chat.side_effect = slow_response

            payload = {
                "model": "gpt-4:latest",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }

            # This should still work as our timeout is reasonable
            response = client.post("/api/chat", json=payload)
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
