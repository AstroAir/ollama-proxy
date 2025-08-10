"""Anthropic Claude API provider implementation.

This module provides an implementation of the AIProvider interface for
Anthropic's Claude API, including the Messages API and proper format transformation.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Union

import httpx
import structlog

from .base import (
    AIProvider,
    ProviderCapability,
    ProviderConfig,
    ProviderError,
    ProviderMetrics,
    ProviderResponse,
    ProviderType,
)

logger = structlog.get_logger(__name__)


class AnthropicProvider(AIProvider):
    """Anthropic Claude API provider implementation."""

    def __init__(self, config: ProviderConfig):
        # Ensure the provider type is correct
        if config.provider_type != ProviderType.ANTHROPIC:
            raise ValueError(f"Expected Anthropic provider type, got {config.provider_type}")

        # Set default capabilities if not provided
        if not config.capabilities:
            config = ProviderConfig(
                provider_type=config.provider_type,
                api_key=config.api_key,
                base_url=config.base_url or "https://api.anthropic.com",
                timeout=config.timeout,
                max_retries=config.max_retries,
                max_concurrent_requests=config.max_concurrent_requests,
                capabilities=frozenset([
                    ProviderCapability.CHAT_COMPLETION,
                    ProviderCapability.STREAMING,
                    ProviderCapability.VISION,  # Claude 3 supports vision
                ]),
                custom_headers=config.custom_headers,
                model_mapping=config.model_mapping,
            )

        super().__init__(config)

    def get_headers(self, extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
        """Get headers for Anthropic API requests."""
        headers = {
            "x-api-key": self.config.api_key,  # Anthropic uses x-api-key instead of Authorization
            "Content-Type": "application/json",
            "User-Agent": f"ollama-proxy/0.1.0 (anthropic)",
            "anthropic-version": "2023-06-01",  # Required API version header
        }

        # Add custom headers from config
        headers.update(self.config.custom_headers)

        # Add extra headers if provided
        if extra_headers:
            headers.update(extra_headers)

        return headers

    async def list_models(self) -> ProviderResponse:
        """List available Anthropic models.

        Note: Anthropic doesn't have a public models endpoint, so we return
        a static list of known Claude models.
        """
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/models",
        )

        # Static list of known Claude models
        models_data = {
            "data": [
                {
                    "id": "claude-3-opus-20240229",
                    "object": "model",
                    "created": 1709251200,
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-3-sonnet-20240229",
                    "object": "model",
                    "created": 1709251200,
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-3-haiku-20240307",
                    "object": "model",
                    "created": 1709856000,
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-2.1",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-2.0",
                    "object": "model",
                    "created": 1690000000,
                    "owned_by": "anthropic",
                },
                {
                    "id": "claude-instant-1.2",
                    "object": "model",
                    "created": 1680000000,
                    "owned_by": "anthropic",
                },
            ]
        }

        metrics.mark_complete(status_code=200)

        logger.debug(
            "Listed Anthropic models (static)",
            model_count=len(models_data["data"]),
            **metrics.to_dict(),
        )

        return ProviderResponse(
            data=models_data,
            status_code=200,
            headers={},
            metrics=metrics,
            provider_type=self.provider_type,
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Create a chat completion using Anthropic Messages API."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/v1/messages",
            model=model,
        )
        self._request_count += 1

        # Transform messages to Anthropic format
        anthropic_messages, system_message = self._prepare_messages(messages)

        # Build request payload
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),  # Required for Anthropic
        }

        # Add system message if present
        if system_message:
            payload["system"] = system_message

        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "stop_sequences" in kwargs:
            payload["stop_sequences"] = kwargs["stop_sequences"]
        elif "stop" in kwargs:
            # Convert OpenAI-style stop to Anthropic stop_sequences
            stop = kwargs["stop"]
            if isinstance(stop, str):
                payload["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                payload["stop_sequences"] = stop

        # Handle streaming
        if stream:
            payload["stream"] = True

        metrics.request_size = len(json.dumps(payload).encode("utf-8"))

        if stream:
            return self._chat_completion_stream(payload, metrics)
        else:
            return await self._chat_completion_non_stream(payload, metrics)

    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str | None]:
        """Prepare messages for Anthropic API format.

        Anthropic requires:
        1. System messages to be separate from the messages array
        2. Messages to alternate between user and assistant
        3. No consecutive messages from the same role
        """
        system_message = None
        anthropic_messages = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                # Extract system message (Anthropic handles it separately)
                if system_message is None:
                    system_message = content
                else:
                    # Combine multiple system messages
                    system_message += "\n\n" + content
            elif role in ["user", "assistant"]:
                # Handle content that might be a list (for vision/multimodal)
                if isinstance(content, list):
                    # Keep as-is for multimodal content
                    anthropic_messages.append({
                        "role": role,
                        "content": content
                    })
                else:
                    anthropic_messages.append({
                        "role": role,
                        "content": content
                    })

        # Ensure messages alternate and start with user
        if anthropic_messages and anthropic_messages[0]["role"] != "user":
            # If first message is not from user, add a placeholder
            anthropic_messages.insert(0, {
                "role": "user",
                "content": "Hello"
            })

        return anthropic_messages, system_message

    async def _chat_completion_non_stream(
        self,
        payload: Dict[str, Any],
        metrics: ProviderMetrics,
    ) -> ProviderResponse:
        """Handle non-streaming chat completion."""
        client = await self.get_client()

        try:
            response = await client.post(
                f"{self.config.base_url}/v1/messages",
                headers=self.get_headers(),
                json=payload,
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.response_size = len(json.dumps(data).encode("utf-8"))
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "Anthropic chat completion successful",
                model=payload.get("model"),
                **metrics.to_dict(),
            )

            return ProviderResponse(
                data=data,
                status_code=response.status_code,
                headers=dict(response.headers),
                metrics=metrics,
                provider_type=self.provider_type,
            )

        except httpx.RequestError as e:
            self._error_count += 1
            metrics.mark_complete(error=str(e))
            logger.error(
                "Network error in Anthropic chat completion",
                error=str(e),
                **metrics.to_dict(),
            )
            raise ProviderError(
                message=f"Network error in chat completion: {e}",
                provider_type=self.provider_type,
                status_code=500,
            ) from e

    async def _chat_completion_stream(
        self,
        payload: Dict[str, Any],
        metrics: ProviderMetrics,
    ) -> AsyncIterator[bytes]:
        """Handle streaming chat completion."""
        client = await self.get_client()

        try:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/v1/messages",
                headers=self.get_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    await self.handle_error(response, metrics)

                async for chunk in response.aiter_bytes():
                    yield chunk

        except httpx.RequestError as e:
            self._error_count += 1
            metrics.mark_complete(error=str(e))
            logger.error(
                "Network error in Anthropic streaming chat completion",
                error=str(e),
                **metrics.to_dict(),
            )
            raise ProviderError(
                message=f"Network error in streaming chat completion: {e}",
                provider_type=self.provider_type,
                status_code=500,
            ) from e

    async def text_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Create a text completion using Anthropic API.

        Note: Anthropic doesn't have a separate text completion endpoint,
        so we convert this to a chat completion with the prompt as a user message.
        """
        # Convert text completion to chat completion
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, model, stream, **kwargs)

    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create embeddings using Anthropic API.

        Note: Anthropic doesn't provide embeddings, so this raises an error.
        """
        raise ProviderError(
            message="Anthropic does not provide embeddings API",
            provider_type=self.provider_type,
            status_code=501,
        )

    def transform_ollama_to_provider(
        self,
        ollama_request: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama request format to Anthropic format."""
        if endpoint == "chat":
            return self._transform_ollama_chat_to_anthropic(ollama_request)
        elif endpoint == "generate":
            return self._transform_ollama_generate_to_anthropic(ollama_request)
        else:
            # Default: pass through with minimal changes
            return ollama_request.copy()

    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Anthropic response format to Ollama format."""
        if endpoint in ["chat", "generate"]:
            return self._transform_anthropic_to_ollama(provider_response)
        else:
            # Default: pass through with minimal changes
            return provider_response.copy()

    def _transform_ollama_chat_to_anthropic(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama chat request to Anthropic format."""
        messages = ollama_request.get("messages", [])
        anthropic_messages, system_message = self._prepare_messages(messages)

        anthropic_request = {
            "model": ollama_request.get("model", "claude-3-sonnet-20240229"),
            "messages": anthropic_messages,
            "max_tokens": 4096,  # Required for Anthropic
        }

        # Add system message if present
        if system_message:
            anthropic_request["system"] = system_message

        # Map Ollama options to Anthropic parameters
        options = ollama_request.get("options", {})
        if "temperature" in options:
            anthropic_request["temperature"] = options["temperature"]
        if "max_tokens" in options:
            anthropic_request["max_tokens"] = options["max_tokens"]
        if "top_p" in options:
            anthropic_request["top_p"] = options["top_p"]
        if "top_k" in options:
            anthropic_request["top_k"] = options["top_k"]
        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                anthropic_request["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                anthropic_request["stop_sequences"] = stop

        # Handle streaming
        if ollama_request.get("stream", False):
            anthropic_request["stream"] = True

        return anthropic_request

    def _transform_ollama_generate_to_anthropic(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama generate request to Anthropic format."""
        prompt = ollama_request.get("prompt", "")
        system = ollama_request.get("system")

        # Create messages array
        messages = [{"role": "user", "content": prompt}]

        anthropic_request = {
            "model": ollama_request.get("model", "claude-3-sonnet-20240229"),
            "messages": messages,
            "max_tokens": 4096,  # Required for Anthropic
        }

        # Add system message if present
        if system:
            anthropic_request["system"] = system

        # Map Ollama options to Anthropic parameters
        options = ollama_request.get("options", {})
        if "temperature" in options:
            anthropic_request["temperature"] = options["temperature"]
        if "max_tokens" in options:
            anthropic_request["max_tokens"] = options["max_tokens"]
        if "top_p" in options:
            anthropic_request["top_p"] = options["top_p"]
        if "top_k" in options:
            anthropic_request["top_k"] = options["top_k"]
        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                anthropic_request["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                anthropic_request["stop_sequences"] = stop

        # Handle streaming
        if ollama_request.get("stream", False):
            anthropic_request["stream"] = True

        return anthropic_request

    def _transform_anthropic_to_ollama(self, anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Anthropic response to Ollama format."""
        import time

        if "content" not in anthropic_response or not anthropic_response["content"]:
            return {"error": "No content in Anthropic response"}

        # Extract content from Anthropic response
        content_blocks = anthropic_response["content"]
        if isinstance(content_blocks, list) and content_blocks:
            # Get text from first content block
            content = content_blocks[0].get("text", "")
        else:
            content = str(content_blocks)

        ollama_response = {
            "model": anthropic_response.get("model", "unknown"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": content,  # For generate endpoint
            "message": {  # For chat endpoint
                "role": "assistant",
                "content": content,
            },
            "done": anthropic_response.get("stop_reason") is not None,
        }

        # Add usage information if available
        if "usage" in anthropic_response:
            usage = anthropic_response["usage"]
            ollama_response.update({
                "prompt_eval_count": usage.get("input_tokens", 0),
                "eval_count": usage.get("output_tokens", 0),
                "total_duration": 0,  # Not available from Anthropic
                "load_duration": 0,   # Not available from Anthropic
                "prompt_eval_duration": 0,  # Not available from Anthropic
                "eval_duration": 0,   # Not available from Anthropic
            })

        return ollama_response