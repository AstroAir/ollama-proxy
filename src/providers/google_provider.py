"""Google Gemini API provider implementation.

This module provides an implementation of the AIProvider interface for
Google's Gemini API, supporting both direct API and Vertex AI endpoints.
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


class GoogleProvider(AIProvider):
    """Google Gemini API provider implementation."""

    def __init__(self, config: ProviderConfig):
        # Ensure the provider type is correct
        if config.provider_type != ProviderType.GOOGLE:
            raise ValueError(f"Expected Google provider type, got {config.provider_type}")

        # Set default capabilities if not provided
        if not config.capabilities:
            config = ProviderConfig(
                provider_type=config.provider_type,
                api_key=config.api_key,
                base_url=config.base_url or "https://generativelanguage.googleapis.com/v1beta",
                timeout=config.timeout,
                max_retries=config.max_retries,
                max_concurrent_requests=config.max_concurrent_requests,
                capabilities=frozenset([
                    ProviderCapability.CHAT_COMPLETION,
                    ProviderCapability.TEXT_COMPLETION,
                    ProviderCapability.EMBEDDINGS,
                    ProviderCapability.STREAMING,
                    ProviderCapability.VISION,  # Gemini Pro Vision supports vision
                ]),
                custom_headers=config.custom_headers,
                model_mapping=config.model_mapping,
            )

        super().__init__(config)

    def get_headers(self, extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
        """Get headers for Google API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"ollama-proxy/0.1.0 (google)",
        }

        # Add custom headers from config
        headers.update(self.config.custom_headers)

        # Add extra headers if provided
        if extra_headers:
            headers.update(extra_headers)

        return headers

    async def list_models(self) -> ProviderResponse:
        """List available Google Gemini models."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/models",
        )
        self._request_count += 1

        client = await self.get_client()

        try:
            response = await client.get(
                f"{self.config.base_url}/models?key={self.config.api_key}",
                headers=self.get_headers(),
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "Listed Google models",
                model_count=len(data.get("models", [])),
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
                "Network error listing Google models",
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
        """Create a chat completion using Google Gemini API."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/generateContent",
            model=model,
        )
        self._request_count += 1

        # Transform messages to Google format
        contents = self._prepare_contents(messages)

        # Build request payload
        payload: Dict[str, Any] = {
            "contents": contents,
        }

        # Add generation config
        generation_config: Dict[str, Any] = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]
        if "top_k" in kwargs:
            generation_config["topK"] = kwargs["top_k"]
        if "stop" in kwargs:
            stop = kwargs["stop"]
            if isinstance(stop, str):
                generation_config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                generation_config["stopSequences"] = stop

        if generation_config:
            payload["generationConfig"] = generation_config

        # Add safety settings (optional)
        if "safety_settings" in kwargs:
            payload["safetySettings"] = kwargs["safety_settings"]

        metrics.request_size = len(json.dumps(payload).encode("utf-8"))

        if stream:
            return self._chat_completion_stream(payload, model, metrics)
        else:
            return await self._chat_completion_non_stream(payload, model, metrics)

    def _prepare_contents(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare messages for Google Gemini API format."""
        contents = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # Map roles to Google format
            if role == "system":
                # Google doesn't have system role, so we'll add it as user message
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"System: {content}"}]
                })
            elif role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                contents.append({
                    "role": "model",  # Google uses "model" instead of "assistant"
                    "parts": [{"text": content}]
                })

        return contents

    async def _chat_completion_non_stream(
        self,
        payload: Dict[str, Any],
        model: str,
        metrics: ProviderMetrics,
    ) -> ProviderResponse:
        """Handle non-streaming chat completion."""
        client = await self.get_client()

        try:
            response = await client.post(
                f"{self.config.base_url}/models/{model}:generateContent?key={self.config.api_key}",
                headers=self.get_headers(),
                json=payload,
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.response_size = len(json.dumps(data).encode("utf-8"))
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "Google chat completion successful",
                model=model,
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
                "Network error in Google chat completion",
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
        model: str,
        metrics: ProviderMetrics,
    ) -> AsyncIterator[bytes]:
        """Handle streaming chat completion."""
        client = await self.get_client()

        try:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/models/{model}:streamGenerateContent?key={self.config.api_key}",
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
                "Network error in Google streaming chat completion",
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
        """Create a text completion using Google Gemini API."""
        # Convert text completion to chat completion
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, model, stream, **kwargs)

    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create embeddings using Google API."""
        metrics = ProviderMetrics(
            provider_type=self.provider_type,
            endpoint="/embedContent",
            model=model,
        )
        self._request_count += 1

        # Build request payload
        if isinstance(input_text, str):
            content = {"parts": [{"text": input_text}]}
        else:
            # For multiple inputs, we'll need to make multiple requests
            # For simplicity, just use the first one
            content = {"parts": [{"text": input_text[0] if input_text else ""}]}

        payload = {
            "content": content,
        }

        metrics.request_size = len(json.dumps(payload).encode("utf-8"))

        client = await self.get_client()

        try:
            response = await client.post(
                f"{self.config.base_url}/models/{model}:embedContent?key={self.config.api_key}",
                headers=self.get_headers(),
                json=payload,
            )

            if response.status_code != 200:
                await self.handle_error(response, metrics)

            data = response.json()
            metrics.response_size = len(json.dumps(data).encode("utf-8"))
            metrics.mark_complete(status_code=response.status_code)

            logger.debug(
                "Google embeddings successful",
                model=model,
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
                "Network error in Google embeddings",
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
        """Transform Ollama request format to Google format."""
        if endpoint == "chat":
            return self._transform_ollama_chat_to_google(ollama_request)
        elif endpoint == "generate":
            return self._transform_ollama_generate_to_google(ollama_request)
        elif endpoint == "embeddings":
            return self._transform_ollama_embeddings_to_google(ollama_request)
        else:
            # Default: pass through with minimal changes
            return ollama_request.copy()

    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Google response format to Ollama format."""
        if endpoint in ["chat", "generate"]:
            return self._transform_google_to_ollama(provider_response)
        elif endpoint == "embeddings":
            return self._transform_google_embeddings_to_ollama(provider_response)
        else:
            # Default: pass through with minimal changes
            return provider_response.copy()

    def _transform_ollama_chat_to_google(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama chat request to Google format."""
        messages = ollama_request.get("messages", [])
        contents = self._prepare_contents(messages)

        google_request: Dict[str, Any] = {
            "contents": contents,
        }

        # Add generation config
        generation_config: Dict[str, Any] = {}
        options = ollama_request.get("options", {})
        if "temperature" in options:
            generation_config["temperature"] = options["temperature"]
        if "max_tokens" in options:
            generation_config["maxOutputTokens"] = options["max_tokens"]
        if "top_p" in options:
            generation_config["topP"] = options["top_p"]
        if "top_k" in options:
            generation_config["topK"] = options["top_k"]
        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                generation_config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                generation_config["stopSequences"] = stop

        if generation_config:
            google_request["generationConfig"] = generation_config

        return google_request

    def _transform_ollama_generate_to_google(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama generate request to Google format."""
        prompt = ollama_request.get("prompt", "")
        system = ollama_request.get("system")

        # Create contents array
        contents = []
        if system:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {system}"}]
            })

        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })

        google_request: Dict[str, Any] = {
            "contents": contents,
        }

        # Add generation config
        generation_config: Dict[str, Any] = {}
        options = ollama_request.get("options", {})
        if "temperature" in options:
            generation_config["temperature"] = options["temperature"]
        if "max_tokens" in options:
            generation_config["maxOutputTokens"] = options["max_tokens"]
        if "top_p" in options:
            generation_config["topP"] = options["top_p"]
        if "top_k" in options:
            generation_config["topK"] = options["top_k"]
        if "stop" in options:
            stop = options["stop"]
            if isinstance(stop, str):
                generation_config["stopSequences"] = [stop]
            elif isinstance(stop, list):
                generation_config["stopSequences"] = stop

        if generation_config:
            google_request["generationConfig"] = generation_config

        return google_request

    def _transform_ollama_embeddings_to_google(self, ollama_request: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Ollama embeddings request to Google format."""
        input_text = ollama_request.get("input") or ollama_request.get("prompt", "")

        google_request = {
            "content": {
                "parts": [{"text": input_text}]
            }
        }

        return google_request

    def _transform_google_to_ollama(self, google_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Google response to Ollama format."""
        import time

        if "candidates" not in google_response or not google_response["candidates"]:
            return {"error": "No candidates in Google response"}

        candidate = google_response["candidates"][0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # Extract text from parts
        text = ""
        if parts:
            text = parts[0].get("text", "")

        ollama_response = {
            "model": "gemini",  # Google doesn't return model name in response
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "response": text,  # For generate endpoint
            "message": {  # For chat endpoint
                "role": "assistant",
                "content": text,
            },
            "done": candidate.get("finishReason") is not None,
        }

        # Add usage information if available
        if "usageMetadata" in google_response:
            usage = google_response["usageMetadata"]
            ollama_response.update({
                "prompt_eval_count": usage.get("promptTokenCount", 0),
                "eval_count": usage.get("candidatesTokenCount", 0),
                "total_duration": 0,  # Not available from Google
                "load_duration": 0,   # Not available from Google
                "prompt_eval_duration": 0,  # Not available from Google
                "eval_duration": 0,   # Not available from Google
            })

        return ollama_response

    def _transform_google_embeddings_to_ollama(self, google_response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Google embeddings response to Ollama format."""
        if "embedding" not in google_response:
            return {"error": "No embedding in Google response"}

        embedding = google_response["embedding"].get("values", [])

        return {"embedding": embedding}