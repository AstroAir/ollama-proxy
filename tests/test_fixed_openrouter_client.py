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


class TestOpenRouterClientRequests:
    """Test OpenRouterClient request methods with mocking."""

    @pytest.fixture
    def client(self):
        """Fixture for OpenRouterClient."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            return OpenRouterClient(settings)

    @pytest.mark.asyncio
    async def test_fetch_models_success(self, client):
        """Test successful fetch_models call."""
        mock_response = httpx.Response(
            200, json={"data": [{"id": "gpt-4"}, {"id": "claude-3"}]}
        )
        with patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response
        ) as mock_get:
            response = await client.fetch_models()

            mock_get.assert_called_once()
            assert response.is_success
            assert len(response.data["data"]) == 2
            assert response.data["data"][0]["id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_fetch_models_network_error(self, client):
        """Test fetch_models with network error."""
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection failed"),
        ):
            with pytest.raises(NetworkError, match="Network error fetching models"):
                await client.fetch_models()

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
            b"""data: [DONE]

""",
        ]
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = stream_chunks
        mock_response = httpx.Response(200)
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
        mock_response.request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response
        ):
            with pytest.raises(OpenRouterError, match="Bad request: Invalid request"):
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
            await client.fetch_models()

        # Now, test shutdown
        mock_aclose = AsyncMock()
        client._client.aclose = mock_aclose
        await client.close()

        mock_aclose.assert_called_once()
        assert client._client is None
