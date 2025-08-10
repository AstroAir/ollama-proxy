"""Local Ollama API provider implementation.

This module provides an implementation of the AIProvider interface for
local Ollama instances, supporting direct communication with Ollama servers.
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


class OllamaProvider(AIProvider):
    """Local Ollama API provider implementation."""

    def __init__(self, config: ProviderConfig):
        """Initialize Ollama provider.

        Args:
            config: Provider configuration
        """
        # Set default capabilities if not provided
        if not config.capabilities:
            config = ProviderConfig(
                provider_type=config.provider_type,
                api_key=config.api_key or "",  # Ollama typically doesn't require API key
                base_url=config.base_url or "http://localhost:11434",
                timeout=config.timeout,
                max_retries=config.max_retries,
                max_concurrent_requests=config.max_concurrent_requests,
                capabilities=frozenset([
                    ProviderCapability.CHAT_COMPLETION,
                    ProviderCapability.TEXT_COMPLETION,
                    ProviderCapability.EMBEDDINGS,
                    ProviderCapability.STREAMING,
                ]),
                custom_headers=config.custom_headers,
                model_mapping=config.model_mapping,
            )

        super().__init__(config)

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OLLAMA

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        """Get supported capabilities."""
        return self.config.capabilities

    def get_headers(self, extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
        """Get headers for Ollama API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"ollama-proxy/0.1.0 (ollama-local)",
        }

        # Add API key if provided (some Ollama setups may use authentication)
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Add custom headers from config
        headers.update(self.config.custom_headers)

        # Add extra headers if provided
        if extra_headers:
            headers.update(extra_headers)

        return headers

    async def list_models(self) -> ProviderResponse:
        """List available models from Ollama."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)

        try:
            url = f"{self.config.base_url}/api/tags"
            headers = self.get_headers()
            
            response = await self._make_request("GET", url, headers=headers)
            ollama_response = response.json()
            
            # Convert Ollama response to OpenAI format
            models_data: Dict[str, Any] = {
                "object": "list",
                "data": []
            }
            data_list = models_data["data"]
            
            for model in ollama_response.get("models", []):
                data_list.append({
                    "id": model.get("name", ""),
                    "object": "model",
                    "owned_by": "ollama",
                    "created": model.get("modified_at", 0),
                    "size": model.get("size", 0),
                    "digest": model.get("digest", ""),
                    "details": model.get("details", {})
                })

            metrics.mark_complete(
                status_code=response.status_code,
                response_time=self._get_current_time() - start_time
            )
            
            return ProviderResponse(
                data=models_data,
                status_code=response.status_code,
                headers=dict(response.headers),
                metrics=metrics,
                provider_type=self.provider_type,
            )

        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "Failed to list Ollama models",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Failed to list models: {e}", provider_type=self.provider_type) from e

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Create a chat completion using Ollama."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)
        
        try:
            url = f"{self.config.base_url}/api/chat"
            
            # Map model name if configured
            actual_model = self.config.model_mapping.get(model, model)
            
            # Prepare Ollama chat request
            payload = {
                "model": actual_model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            
            headers = self.get_headers()
            
            if not stream:
                response = await self._make_request("POST", url, json=payload, headers=headers)
                ollama_response = response.json()
                
                # Convert Ollama response to OpenAI format
                openai_response = self._convert_ollama_chat_to_openai(ollama_response, model)
                
                metrics.mark_complete(
                    status_code=response.status_code,
                    response_time=self._get_current_time() - start_time
                )
                
                return ProviderResponse(
                    data=openai_response,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metrics=metrics,
                    provider_type=self.provider_type,
                )
            else:
                # Handle streaming response
                return self._handle_streaming_response(url, payload, headers, metrics, model)
                
        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "Ollama chat completion failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Chat completion failed: {e}", provider_type=self.provider_type) from e

    async def text_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Create a text completion using Ollama."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)
        
        try:
            url = f"{self.config.base_url}/api/generate"
            
            # Map model name if configured
            actual_model = self.config.model_mapping.get(model, model)
            
            # Prepare Ollama generate request
            payload = {
                "model": actual_model,
                "prompt": prompt,
                "stream": stream,
                **kwargs
            }
            
            headers = self.get_headers()
            
            if not stream:
                response = await self._make_request("POST", url, json=payload, headers=headers)
                ollama_response = response.json()
                
                # Convert Ollama response to OpenAI format
                openai_response = self._convert_ollama_generate_to_openai(ollama_response, model)
                
                metrics.mark_complete(
                    status_code=response.status_code,
                    response_time=self._get_current_time() - start_time
                )
                
                return ProviderResponse(
                    data=openai_response,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metrics=metrics,
                    provider_type=self.provider_type,
                )
            else:
                # Handle streaming response
                return self._handle_streaming_response(url, payload, headers, metrics, model)
                
        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "Ollama text completion failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Text completion failed: {e}", provider_type=self.provider_type) from e

    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create embeddings using Ollama."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)
        
        try:
            url = f"{self.config.base_url}/api/embeddings"
            
            # Map model name if configured
            actual_model = self.config.model_mapping.get(model, model)
            
            # Handle both string and list inputs
            if isinstance(input_text, str):
                prompt = input_text
            else:
                prompt = input_text[0] if input_text else ""
            
            payload = {
                "model": actual_model,
                "prompt": prompt,
                **kwargs
            }
            
            headers = self.get_headers()
            response = await self._make_request("POST", url, json=payload, headers=headers)
            ollama_response = response.json()
            
            # Convert Ollama embeddings response to OpenAI format
            embeddings_data = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": ollama_response.get("embedding", [])
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "total_tokens": len(prompt.split())
                }
            }
            
            metrics.mark_complete(
                status_code=response.status_code,
                response_time=self._get_current_time() - start_time
            )
            
            return ProviderResponse(
                data=embeddings_data,
                status_code=response.status_code,
                headers=dict(response.headers),
                metrics=metrics,
                provider_type=self.provider_type,
            )
            
        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "Ollama embeddings failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Embeddings creation failed: {e}", provider_type=self.provider_type) from e

    def _convert_ollama_chat_to_openai(self, ollama_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert Ollama chat response to OpenAI format."""
        message = ollama_response.get("message", {})
        content = message.get("content", "")

        return {
            "id": f"chatcmpl-{self._generate_id()}",
            "object": "chat.completion",
            "created": int(self._get_current_time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "completion_tokens": ollama_response.get("eval_count", 0),
                "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
            }
        }

    def _convert_ollama_generate_to_openai(self, ollama_response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert Ollama generate response to OpenAI format."""
        response_text = ollama_response.get("response", "")

        return {
            "id": f"cmpl-{self._generate_id()}",
            "object": "text_completion",
            "created": int(self._get_current_time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                "completion_tokens": ollama_response.get("eval_count", 0),
                "total_tokens": ollama_response.get("prompt_eval_count", 0) + ollama_response.get("eval_count", 0)
            }
        }

    async def _handle_streaming_response(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        metrics: ProviderMetrics,
        model: str,
    ) -> AsyncIterator[bytes]:
        """Handle streaming response from Ollama."""
        try:
            client = await self.get_client()
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        try:
                            # Parse Ollama streaming response
                            ollama_chunk = json.loads(line)

                            # Convert to OpenAI streaming format
                            if "api/chat" in url:
                                openai_chunk = self._convert_ollama_chat_stream_to_openai(ollama_chunk, model)
                            else:
                                openai_chunk = self._convert_ollama_generate_stream_to_openai(ollama_chunk, model)

                            yield f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

                            # Check if this is the final chunk
                            if ollama_chunk.get("done", False):
                                yield b"data: [DONE]\n\n"
                                break

                        except json.JSONDecodeError:
                            continue

                metrics.mark_complete(status_code=response.status_code)

        except Exception as e:
            metrics.mark_error(error=str(e))
            logger.error(
                "Ollama streaming failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Streaming failed: {e}", provider_type=self.provider_type) from e

    def transform_ollama_to_provider(
        self,
        ollama_request: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama request format to Ollama format (no-op)."""
        # Since this is the Ollama provider, no transformation needed
        return ollama_request.copy()

    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama response format to Ollama format (no-op)."""
        # Since this is the Ollama provider, no transformation needed
        return provider_response.copy()

    def _convert_ollama_chat_stream_to_openai(self, ollama_chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert Ollama chat streaming chunk to OpenAI format."""
        message = ollama_chunk.get("message", {})
        content = message.get("content", "")
        done = ollama_chunk.get("done", False)

        return {
            "id": f"chatcmpl-{self._generate_id()}",
            "object": "chat.completion.chunk",
            "created": int(self._get_current_time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": content
                    } if not done else {},
                    "finish_reason": "stop" if done else None
                }
            ]
        }

    def _convert_ollama_generate_stream_to_openai(self, ollama_chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert Ollama generate streaming chunk to OpenAI format."""
        response_text = ollama_chunk.get("response", "")
        done = ollama_chunk.get("done", False)

        return {
            "id": f"cmpl-{self._generate_id()}",
            "object": "text_completion",
            "created": int(self._get_current_time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": response_text,
                    "finish_reason": "stop" if done else None
                }
            ]
        }

    def _generate_id(self) -> str:
        """Generate a unique ID for responses."""
        import uuid
        return str(uuid.uuid4())[:8]
