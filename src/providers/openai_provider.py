"""OpenAI API provider implementation.

This module provides an implementation of the AIProvider interface for
OpenAI's API, including chat completions, embeddings, and other endpoints.
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


class OpenAIProvider(AIProvider):
    """OpenAI API provider implementation."""

    def __init__(self, config: ProviderConfig):
        # Ensure the provider type is correct
        if config.provider_type != ProviderType.OPENAI:
            raise ValueError(f"Expected OpenAI provider type, got {config.provider_type}")

        # Set default capabilities if not provided
        if not config.capabilities:
            config = ProviderConfig(
                provider_type=config.provider_type,
                api_key=config.api_key,
                base_url=config.base_url or "https://api.openai.com/v1",
                timeout=config.timeout,
                max_retries=config.max_retries,
                max_concurrent_requests=config.max_concurrent_requests,
                capabilities=frozenset([
                    ProviderCapability.CHAT_COMPLETION,
                    ProviderCapability.TEXT_COMPLETION,
                    ProviderCapability.EMBEDDINGS,
                    ProviderCapability.STREAMING,
                    ProviderCapability.FUNCTION_CALLING,
                    ProviderCapability.VISION,
                ]),
                custom_headers=config.custom_headers,
                model_mapping=config.model_mapping,
            )

        super().__init__(config)

    async def list_models(self) -> ProviderResponse:
        """List available OpenAI models."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/models",
        )
        self._request_count += 1

        client = await self.get_client()

        try:
            response = await client.get(
                f"{self.config.base_url}/models",
                headers=self.get_headers(),
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "Listed OpenAI models",
                model_count=len(data.get("data", [])),
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
                "Network error listing OpenAI models",
                error=str(e),
                **metrics.to_dict(),
            )
            raise ProviderError(
                message=f"Network error listing models: {e}",
                provider_type=self.provider_type,
                status_code=500,
            ) from e

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Create a chat completion using OpenAI API."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/chat/completions",
            model=model,
        )
        self._request_count += 1

        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Add optional parameters
        for key, value in kwargs.items():
            if key in [
                "temperature", "max_tokens", "top_p", "frequency_penalty",
                "presence_penalty", "stop", "functions", "function_call",
                "tools", "tool_choice", "response_format", "seed", "user"
            ]:
                payload[key] = value

        metrics.request_size = len(json.dumps(payload).encode("utf-8"))

        if stream:
            return self._chat_completion_stream(payload, metrics)
        else:
            return await self._chat_completion_non_stream(payload, metrics)

    async def _chat_completion_non_stream(
        self,
        payload: Dict[str, Any],
        metrics: ProviderMetrics,
    ) -> ProviderResponse:
        """Handle non-streaming chat completion."""
        client = await self.get_client()

        try:
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.get_headers(),
                json=payload,
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.response_size = len(json.dumps(data).encode("utf-8"))
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "OpenAI chat completion successful",
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
                "Network error in OpenAI chat completion",
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
                f"{self.config.base_url}/chat/completions",
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
                "Network error in OpenAI streaming chat completion",
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
        """Create a text completion using OpenAI API."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/completions",
            model=model,
        )
        self._request_count += 1

        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
        }

        # Add optional parameters
        for key, value in kwargs.items():
            if key in [
                "temperature", "max_tokens", "top_p", "frequency_penalty",
                "presence_penalty", "stop", "suffix", "echo", "best_of",
                "logprobs", "user"
            ]:
                payload[key] = value

        metrics.request_size = len(json.dumps(payload).encode("utf-8"))

        if stream:
            return self._text_completion_stream(payload, metrics)
        else:
            return await self._text_completion_non_stream(payload, metrics)

    async def _text_completion_non_stream(
        self,
        payload: Dict[str, Any],
        metrics: ProviderMetrics,
    ) -> ProviderResponse:
        """Handle non-streaming text completion."""
        client = await self.get_client()

        try:
            response = await client.post(
                f"{self.config.base_url}/completions",
                headers=self.get_headers(),
                json=payload,
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.response_size = len(json.dumps(data).encode("utf-8"))
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "OpenAI text completion successful",
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
                "Network error in OpenAI text completion",
                error=str(e),
                **metrics.to_dict(),
            )
            raise ProviderError(
                message=f"Network error in text completion: {e}",
                provider_type=self.provider_type,
                status_code=500,
            ) from e

    async def _text_completion_stream(
        self,
        payload: Dict[str, Any],
        metrics: ProviderMetrics,
    ) -> AsyncIterator[bytes]:
        """Handle streaming text completion."""
        client = await self.get_client()

        try:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/completions",
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
                "Network error in OpenAI streaming text completion",
                error=str(e),
                **metrics.to_dict(),
            )
            raise ProviderError(
                message=f"Network error in streaming text completion: {e}",
                provider_type=self.provider_type,
                status_code=500,
            ) from e

    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create embeddings using OpenAI API."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/embeddings",
            model=model,
        )
        self._request_count += 1

        # Build request payload
        payload = {
            "model": model,
            "input": input_text,
        }

        # Add optional parameters
        for key, value in kwargs.items():
            if key in ["encoding_format", "dimensions", "user"]:
                payload[key] = value

        metrics.request_size = len(json.dumps(payload).encode("utf-8"))

        client = await self.get_client()

        try:
            response = await client.post(
                f"{self.config.base_url}/embeddings",
                headers=self.get_headers(),
                json=payload,
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.response_size = len(json.dumps(data).encode("utf-8"))
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "OpenAI embeddings successful",
                model=model,
                input_count=len(input_text) if isinstance(input_text, list) else 1,
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
                "Network error in OpenAI embeddings",
                error=str(e),
                **metrics.to_dict(),
            )
            raise ProviderError(
                message=f"Network error in embeddings: {e}",
                provider_type=self.provider_type,
                status_code=500,
            ) from e

    def transform_ollama_to_provider(
        self,
        ollama_request: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama request format to OpenAI format."""
        if endpoint == "chat":
            return self._transform_ollama_chat_to_openai(ollama_request)
        elif endpoint == "generate":
            return self._transform_ollama_generate_to_openai(ollama_request)
        elif endpoint == "embeddings":
            return self._transform_ollama_embeddings_to_openai(ollama_request)
        else:
            # Default: pass through with minimal changes
            return ollama_request.copy()

    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform OpenAI response format to Ollama format."""
        if endpoint == "chat":
            return self._transform_openai_chat_to_ollama(provider_response)
        elif endpoint == "generate":
            return self._transform_openai_completion_to_ollama(provider_response)
        elif endpoint == "embeddings":
            return self._transform_openai_embeddings_to_ollama(provider_response)
        else:
            # Default: pass through with minimal changes
            return provider_response.copy()

    def _transform_ollama_chat_to_openai(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama chat request to OpenAI format."""
        openai_request = {
            "model": ollama_request.get("model", "gpt-3.5-turbo"),
            "messages": ollama_request.get("messages", []),
        }

        # Map Ollama options to OpenAI parameters
        options = ollama_request.get("options", {})
        if "temperature" in options:
            openai_request["temperature"] = options["temperature"]
        if "max_tokens" in options:
            openai_request["max_tokens"] = options["max_tokens"]
        if "top_p" in options:
            openai_request["top_p"] = options["top_p"]
        if "frequency_penalty" in options:
            openai_request["frequency_penalty"] = options["frequency_penalty"]
        if "presence_penalty" in options:
            openai_request["presence_penalty"] = options["presence_penalty"]
        if "stop" in options:
            openai_request["stop"] = options["stop"]

        # Handle streaming
        if ollama_request.get("stream", False):
            openai_request["stream"] = True

        # Handle response format
        if ollama_request.get("format") == "json":
            openai_request["response_format"] = {"type": "json_object"}

        return openai_request

    def _transform_ollama_generate_to_openai(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama generate request to OpenAI completions format."""
        openai_request = {
            "model": ollama_request.get("model", "gpt-3.5-turbo-instruct"),
            "prompt": ollama_request.get("prompt", ""),
        }

        # Map Ollama options to OpenAI parameters
        options = ollama_request.get("options", {})
        if "temperature" in options:
            openai_request["temperature"] = options["temperature"]
        if "max_tokens" in options:
            openai_request["max_tokens"] = options["max_tokens"]
        if "top_p" in options:
            openai_request["top_p"] = options["top_p"]
        if "frequency_penalty" in options:
            openai_request["frequency_penalty"] = options["frequency_penalty"]
        if "presence_penalty" in options:
            openai_request["presence_penalty"] = options["presence_penalty"]
        if "stop" in options:
            openai_request["stop"] = options["stop"]

        # Handle streaming
        if ollama_request.get("stream", False):
            openai_request["stream"] = True

        return openai_request

    def _transform_ollama_embeddings_to_openai(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama embeddings request to OpenAI format."""
        openai_request = {
            "model": ollama_request.get("model", "text-embedding-ada-002"),
            "input": ollama_request.get("input") or ollama_request.get("prompt", ""),
        }

        return openai_request

    def _transform_openai_chat_to_ollama(self, openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI chat response to Ollama format."""
        import time

        if "choices" not in openai_response or not openai_response["choices"]:
            return {"error": "No choices in OpenAI response"}

        choice = openai_response["choices"][0]
        message = choice.get("message", {})

        ollama_response = {
            "model": openai_response.get("model", "unknown"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content", ""),
            },
            "done": choice.get("finish_reason") is not None,
        }

        # Add usage information if available
        if "usage" in openai_response:
            usage = openai_response["usage"]
            ollama_response.update({
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
                "total_duration": 0,  # Not available from OpenAI
                "load_duration": 0,   # Not available from OpenAI
                "prompt_eval_duration": 0,  # Not available from OpenAI
                "eval_duration": 0,   # Not available from OpenAI
            })

        return ollama_response

    def _transform_openai_completion_to_ollama(self, openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI completion response to Ollama format."""
        import time

        if "choices" not in openai_response or not openai_response["choices"]:
            return {"error": "No choices in OpenAI response"}

        choice = openai_response["choices"][0]

        ollama_response = {
            "model": openai_response.get("model", "unknown"),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": choice.get("text", ""),
            "done": choice.get("finish_reason") is not None,
        }

        # Add usage information if available
        if "usage" in openai_response:
            usage = openai_response["usage"]
            ollama_response.update({
                "prompt_eval_count": usage.get("prompt_tokens", 0),
                "eval_count": usage.get("completion_tokens", 0),
                "total_duration": 0,  # Not available from OpenAI
                "load_duration": 0,   # Not available from OpenAI
                "prompt_eval_duration": 0,  # Not available from OpenAI
                "eval_duration": 0,   # Not available from OpenAI
            })

        return ollama_response

    def _transform_openai_embeddings_to_ollama(self, openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform OpenAI embeddings response to Ollama format."""
        if "data" not in openai_response or not openai_response["data"]:
            return {"error": "No data in OpenAI embeddings response"}

        # Extract embeddings from OpenAI format
        embeddings = [item["embedding"] for item in openai_response["data"]]

        # Ollama format depends on whether it was a single input or multiple
        if len(embeddings) == 1:
            return {"embedding": embeddings[0]}
        else:
            return {"embeddings": embeddings}