"""Tests for enhanced OpenRouter client with modern features."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from src.config import Environment, Settings
from src.exceptions import NetworkError, OpenRouterError
from src.openrouter import (
    OpenRouterClient,
    OpenRouterEndpoint,
    OpenRouterResponse,
    RequestMetrics,
)


class TestOpenRouterEndpoint:
    """Test OpenRouterEndpoint enum."""

    def test_endpoint_values(self):
        """Test endpoint string values."""
        assert OpenRouterEndpoint.MODELS == "/models"
        assert OpenRouterEndpoint.CHAT_COMPLETIONS == "/chat/completions"
        assert OpenRouterEndpoint.COMPLETIONS == "/completions"
        assert OpenRouterEndpoint.EMBEDDINGS == "/embeddings"
        assert OpenRouterEndpoint.GENERATION == "/generation"
        assert OpenRouterEndpoint.PROVIDERS == "/providers"
        assert OpenRouterEndpoint.LIMITS == "/auth/key"
        assert OpenRouterEndpoint.COSTS == "/generation/{id}/cost"


class TestRequestMetrics:
    """Test RequestMetrics dataclass with enhanced features."""

    def test_default_creation(self):
        """Test default metrics creation."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        assert metrics.endpoint == OpenRouterEndpoint.MODELS
        assert metrics.end_time is None
        assert metrics.model is None
        assert metrics.stream is False
        assert metrics.request_size == 0
        assert metrics.response_size == 0
        assert metrics.status_code is None
        metrics.error is None
        assert metrics.request_id.startswith("req_")
        assert metrics.retry_count == 0
        assert metrics.cache_hit is False

    def test_mark_complete(self):
        """Test marking metrics as complete."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        metrics.mark_complete(status_code=200, error=None)

        assert metrics.end_time is not None
        assert metrics.status_code == 200
        assert metrics.error is None
        assert metrics.is_complete

    def test_increment_retry(self):
        """Test retry counter increment."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        assert metrics.retry_count == 0

        metrics.increment_retry()
        assert metrics.retry_count == 1

        metrics.increment_retry()
        assert metrics.retry_count == 2

    def test_mark_cache_hit(self):
        """Test cache hit marking."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        assert metrics.cache_hit is False

        metrics.mark_cache_hit()
        assert metrics.cache_hit is True

    def test_is_successful(self):
        """Test success detection."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)

        # No status code - not successful
        assert not metrics.is_successful

        # 200 status - successful
        metrics.mark_complete(status_code=200)
        assert metrics.is_successful

        # 404 status - not successful
        metrics.mark_complete(status_code=404)
        assert not metrics.is_successful

    def test_performance_category(self):
        """Test performance categorization."""
        import time

        # Test excellent performance (< 100ms)
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        metrics.start_time = time.time()
        metrics.end_time = metrics.start_time + 0.05  # 50ms
        assert metrics.performance_category == "excellent"

        # Test good performance (100-500ms)
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        metrics.start_time = time.time()
        metrics.end_time = metrics.start_time + 0.3  # 300ms
        assert metrics.performance_category == "good"

        # Test acceptable performance (500-1000ms)
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        metrics.start_time = time.time()
        metrics.end_time = metrics.start_time + 0.8  # 800ms
        assert metrics.performance_category == "acceptable"

        # Test slow performance (1000-5000ms)
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        metrics.start_time = time.time()
        metrics.end_time = metrics.start_time + 3.0  # 3000ms
        assert metrics.performance_category == "slow"

        # Test very slow performance (> 5000ms)
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        metrics.start_time = time.time()
        metrics.end_time = metrics.start_time + 10.0  # 10000ms
        assert metrics.performance_category == "very_slow"

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = RequestMetrics(
            endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS, model="gpt-4", stream=True
        )
        metrics.mark_complete(status_code=200)

        result = metrics.to_dict()

        assert result["endpoint"] == "/chat/completions"
        assert result["model"] == "gpt-4"
        assert result["stream"] is True
        assert result["status_code"] == 200
        assert result["is_complete"] is True
        assert result["is_successful"] is True
        assert "performance_category" in result
        assert "request_id" in result


class TestOpenRouterResponse:
    """Test OpenRouterResponse with enhanced features."""

    def test_basic_creation(self):
        """Test basic response creation."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        response = OpenRouterResponse(
            data={"test": "data"},
            status_code=200,
            headers={"content-type": "application/json"},
            metrics=metrics,
        )

        assert response.data["test"] == "data"
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.metrics == metrics

    def test_is_success(self):
        """Test success detection."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)

        success_response = OpenRouterResponse(
            data={},
            status_code=200,
            headers={},
            metrics=metrics,
        )
        assert success_response.is_success

        error_response = OpenRouterResponse(
            data={},
            status_code=404,
            headers={},
            metrics=metrics,
        )
        assert not error_response.is_success

    def test_usage_extraction(self):
        """Test usage information extraction."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS)

        # Response with usage
        response_with_usage = OpenRouterResponse(
            data={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
            status_code=200,
            headers={},
            metrics=metrics,
        )
        usage = response_with_usage.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20

        # Response without usage
        response_without_usage = OpenRouterResponse(
            data={"choices": []}, status_code=200, headers={}, metrics=metrics
        )
        assert response_without_usage.usage is None

    def test_model_used_extraction(self):
        """Test model used extraction."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS)

        response = OpenRouterResponse(
            data={"model": "gpt-4"}, status_code=200, headers={}, metrics=metrics
        )
        assert response.model_used == "gpt-4"

    def test_finish_reason_extraction(self):
        """Test finish reason extraction."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS)

        response = OpenRouterResponse(
            data={"choices": [{"finish_reason": "stop"}]},
            status_code=200,
            headers={},
            metrics=metrics,
        )
        assert response.finish_reason == "stop"

    def test_get_choices(self):
        """Test choices extraction."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS)

        choices_data = [
            {"message": {"content": "Hello"}},
            {"message": {"content": "World"}},
        ]
        response = OpenRouterResponse(
            data={"choices": choices_data}, status_code=200, headers={}, metrics=metrics
        )

        choices = response.get_choices()
        assert len(choices) == 2
        assert choices[0]["message"]["content"] == "Hello"
        assert choices[1]["message"]["content"] == "World"

    def test_get_content(self):
        """Test content extraction."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS)

        # Test message content
        response_message = OpenRouterResponse(
            data={"choices": [{"message": {"content": "Hello world"}}]},
            status_code=200,
            headers={},
            metrics=metrics,
        )
        assert response_message.get_content() == "Hello world"

        # Test delta content (streaming)
        response_delta = OpenRouterResponse(
            data={"choices": [{"delta": {"content": "Hello"}}]},
            status_code=200,
            headers={},
            metrics=metrics,
        )
        assert response_delta.get_content() == "Hello"

    def test_get_error_info(self):
        """Test error information extraction."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS)

        # Test dict error
        response_dict_error = OpenRouterResponse(
            data={"error": {"message": "API error", "code": "invalid_request"}},
            status_code=400,
            headers={},
            metrics=metrics,
        )
        error_info = response_dict_error.get_error_info()
        assert error_info["message"] == "API error"
        assert error_info["code"] == "invalid_request"

        # Test string error
        response_string_error = OpenRouterResponse(
            data={"error": "Simple error message"},
            status_code=400,
            headers={},
            metrics=metrics,
        )
        error_info = response_string_error.get_error_info()
        assert error_info["message"] == "Simple error message"

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS)
        response = OpenRouterResponse(
            data={"model": "gpt-4", "choices": [{"finish_reason": "stop"}]},
            status_code=200,
            headers={"content-type": "application/json"},
            metrics=metrics,
        )

        result = response.to_dict()

        assert result["data"]["model"] == "gpt-4"
        assert result["status_code"] == 200
        assert result["headers"]["content-type"] == "application/json"
        assert result["is_success"] is True
        assert result["model_used"] == "gpt-4"
        assert result["finish_reason"] == "stop"
        assert "metrics" in result


class TestOpenRouterClient:
    """Test OpenRouterClient with enhanced features."""

    def test_client_initialization(self):
        """Test client initialization."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            client = OpenRouterClient(settings)

        assert client.settings == settings
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert client.timeout == 300
        assert client._client is None
        assert client.request_count == 0
        assert client.error_count == 0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            client = OpenRouterClient(settings)

        # No requests - 0% error rate
        assert client.error_rate == 0.0

        # Simulate some requests and errors
        client._request_count = 10
        client._error_count = 2
        assert client.error_rate == 20.0

    def test_get_headers_basic(self):
        """Test basic header generation."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            client = OpenRouterClient(settings)

        headers = client._get_headers()

        assert headers["Authorization"] == "Bearer test-api-key-1234567890"
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers
        assert "X-Title" in headers
        assert "X-Request-ID" in headers
        assert "X-Client-Version" in headers
        assert "X-Client-Type" in headers

    def test_get_headers_with_site_info(self):
        """Test header generation with site information."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            client = OpenRouterClient(settings)

        headers = client._get_headers(
            site_url="https://example.com", site_name="Test Site"
        )

        assert headers["HTTP-Referer"] == "https://example.com"
        assert headers["X-Title"] == "Test Site"

    def test_get_headers_environment_specific(self):
        """Test environment-specific headers."""
        # Development environment
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890", "ENVIRONMENT": "development"}):
            dev_settings = Settings()  # type: ignore[call-arg]
            dev_client = OpenRouterClient(dev_settings)
            dev_headers = dev_client._get_headers()
            assert dev_headers["X-Debug-Mode"] == "true"

        # Production environment
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890", "ENVIRONMENT": "production"}):
            prod_settings = Settings()  # type: ignore[call-arg]
            prod_client = OpenRouterClient(prod_settings)
            prod_headers = prod_client._get_headers()
            assert prod_headers["X-Environment"] == "production"

    def test_enhance_chat_payload(self):
        """Test chat payload enhancement."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            client = OpenRouterClient(settings)

        basic_payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        enhanced = client._enhance_chat_payload(
            basic_payload, provider="openai", transforms=["middle-out"]
        )

        assert enhanced["model"] == "gpt-4"
        assert enhanced["provider"] == "openai"
        assert enhanced["transforms"] == ["middle-out"]
        assert enhanced["max_tokens"] == 4096  # Default added
        assert enhanced["temperature"] == 0.7  # Default added


if __name__ == "__main__":
    pytest.main([__file__])

class TestOpenRouterClientRequests:
    """Test OpenRouterClient request methods with mocking."""

    @pytest.fixture
    def client(self):
        """Fixture for OpenRouterClient."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            return OpenRouterClient(settings)

    @pytest.mark.asyncio
    async def test_get_models_success(self, client):
        """Test successful get_models call."""
        mock_response = httpx.Response(
            200, json={"data": [{"id": "gpt-4"}, {"id": "claude-3"}]}
        )
        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ) as mock_get:
            response = await client.get_models()

            mock_get.assert_called_once()
            assert response.is_success
            assert len(response.data["data"]) == 2
            assert response.data["data"][0]["id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_models_network_error(self, client):
        """Test get_models with network error."""
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection failed"),
        ):
            with pytest.raises(NetworkError, match="Failed to connect to OpenRouter"):
                await client.get_models()

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming_success(self, client):
        """Test successful non-streaming chat completion."""
        mock_response_data = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello"}}],
        }
        mock_response = httpx.Response(200, json=mock_response_data)
        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
            response = await client.chat_completion(payload, stream=False)

            mock_post.assert_called_once()
            assert response.is_success
            assert response.get_content() == "Hello"

    @pytest.mark.asyncio
    async def test_chat_completion_streaming_success(self, client):
        """Test successful streaming chat completion."""
        stream_chunks = [
            b"""data: {"id": "1", "choices": [{"delta": {"content": "Hel"}}]} 

""",
            b"""data: {"id": "2", "choices": [{"delta": {"content": "lo"}}]} 

""",
            b"""data: [DONE]\n\n""",
        ]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = stream_chunks
        mock_response = httpx.Response(200, content=mock_stream)
        mock_response.aiter_bytes = AsyncMock(return_value=stream_chunks)

        with patch(
            "httpx.AsyncClient.stream", new_callable=AsyncMock, return_value=mock_response
        ) as mock_stream_method:
            payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
            stream_iterator = await client.chat_completion(payload, stream=True)

            mock_stream_method.assert_called_once()
            results = [chunk async for chunk in stream_iterator]
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_chat_completion_api_error(self, client):
        """Test chat completion with API error."""
        mock_response = httpx.Response(
            400, json={"error": {"message": "Invalid request"}}
        )
        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(OpenRouterError, match="OpenRouter API error"):
                payload = {"model": "gpt-4", "messages": []}  # Invalid payload
                await client.chat_completion(payload, stream=False)

    @pytest.mark.asyncio
    async def test_client_shutdown(self, client):
        """Test client shutdown."""
        # First, make a request to initialize the client
        mock_response = httpx.Response(200, json={"data": []})
        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ):
            await client.get_models()

        # Now, test shutdown
        mock_aclose = AsyncMock()
        client._client.aclose = mock_aclose
        await client.shutdown()

        mock_aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_providers_success(self, client):
        """Test successful get_providers call."""
        mock_response = httpx.Response(200, json={"data": [{"id": "openai"}]})
        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ) as mock_get:
            response = await client.get_providers()
            mock_get.assert_called_once()
            assert response.is_success
            assert len(response.data["data"]) == 1

    @pytest.mark.asyncio
    async def test_get_limits_success(self, client):
        """Test successful get_limits call."""
        mock_response = httpx.Response(200, json={"limit": 100, "remaining": 50})
        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ) as mock_get:
            response = await client.get_limits()
            mock_get.assert_called_once()
            assert response.is_success
            assert response.data["limit"] == 100

    @pytest.mark.asyncio
    async def test_get_costs_success(self, client):
        """Test successful get_costs call."""
        mock_response = httpx.Response(200, json={"cost": 0.123})
        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            response = await client.get_costs("some-id")
            mock_post.assert_called_once()
            assert response.is_success
            assert response.data["cost"] == 0.123

    @pytest.mark.asyncio
    async def test_chat_completion_500_error(self, client):
        """Test chat completion with 500 API error."""
        mock_response = httpx.Response(500, json={"error": {"message": "Internal Server Error"}})
        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(OpenRouterError, match="OpenRouter API error"):
                payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}
                await client.chat_completion(payload, stream=False)
