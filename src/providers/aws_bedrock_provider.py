"""AWS Bedrock API provider implementation.

This module provides an implementation of the AIProvider interface for
AWS Bedrock, supporting various foundation models like Claude, Titan, etc.
"""

from __future__ import annotations

import json
import base64
from typing import Any, AsyncIterator, Dict, List, Union, Optional

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


class AWSBedrockProvider(AIProvider):
    """AWS Bedrock API provider implementation."""

    def __init__(self, config: ProviderConfig):
        """Initialize AWS Bedrock provider.

        Args:
            config: Provider configuration
        """
        # Set default capabilities if not provided
        if not config.capabilities:
            config = ProviderConfig(
                provider_type=config.provider_type,
                api_key=config.api_key,  # AWS Access Key ID
                base_url=config.base_url or "https://bedrock-runtime.us-east-1.amazonaws.com",
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
        self._aws_secret_key = config.custom_headers.get("aws_secret_access_key", "")
        self._aws_region = config.custom_headers.get("aws_region", "us-east-1")
        self._aws_session_token = config.custom_headers.get("aws_session_token")

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.AWS_BEDROCK

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        """Get supported capabilities."""
        return self.config.capabilities

    def _get_model_id(self, model: str) -> str:
        """Get the full Bedrock model ID from model name."""
        # Check if model is mapped
        if model in self.config.model_mapping:
            return self.config.model_mapping[model]
        
        # Common model mappings
        model_mappings = {
            "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
            "claude-2": "anthropic.claude-v2:1",
            "claude-instant": "anthropic.claude-instant-v1",
            "titan-text": "amazon.titan-text-express-v1",
            "titan-embed": "amazon.titan-embed-text-v1",
            "llama2-13b": "meta.llama2-13b-chat-v1",
            "llama2-70b": "meta.llama2-70b-chat-v1",
        }
        
        return model_mappings.get(model, model)

    def _prepare_bedrock_request(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """Prepare request payload based on model provider."""
        if model_id.startswith("anthropic.claude"):
            return self._prepare_claude_request(**kwargs)
        elif model_id.startswith("amazon.titan"):
            return self._prepare_titan_request(**kwargs)
        elif model_id.startswith("meta.llama"):
            return self._prepare_llama_request(**kwargs)
        else:
            # Generic format
            return kwargs

    def _prepare_claude_request(self, messages: Optional[List[Dict[str, Any]]] = None, prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Prepare request for Claude models."""
        if messages:
            # Convert messages to Claude format
            prompt_text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt_text += f"\n\nHuman: {content}"
                elif role == "assistant":
                    prompt_text += f"\n\nAssistant: {content}"
            prompt_text += "\n\nAssistant:"
        else:
            prompt_text = prompt or ""

        return {
            "prompt": prompt_text,
            "max_tokens_to_sample": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stop_sequences": kwargs.get("stop", []),
        }

    def _prepare_titan_request(self, messages: Optional[List[Dict[str, Any]]] = None, prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Prepare request for Titan models."""
        if messages:
            # Convert messages to simple prompt
            prompt_text = ""
            for msg in messages:
                content = msg.get("content", "")
                prompt_text += f"{content}\n"
        else:
            prompt_text = prompt or ""

        return {
            "inputText": prompt_text,
            "textGenerationConfig": {
                "maxTokenCount": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 1.0),
                "stopSequences": kwargs.get("stop", []),
            }
        }

    def _prepare_llama_request(self, messages: Optional[List[Dict[str, Any]]] = None, prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Prepare request for Llama models."""
        if messages:
            # Convert messages to Llama chat format
            prompt_text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    prompt_text += f"[INST] {content} [/INST]"
                elif role == "assistant":
                    prompt_text += f" {content} "
        else:
            prompt_text = prompt or ""

        return {
            "prompt": prompt_text,
            "max_gen_len": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
        }

    async def list_models(self) -> ProviderResponse:
        """List available models from AWS Bedrock."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)

        try:
            # Return common Bedrock models
            models_data = {
                "object": "list",
                "data": [
                    {"id": "claude-3-sonnet", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-3-haiku", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-3-opus", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-2", "object": "model", "owned_by": "anthropic"},
                    {"id": "claude-instant", "object": "model", "owned_by": "anthropic"},
                    {"id": "titan-text", "object": "model", "owned_by": "amazon"},
                    {"id": "titan-embed", "object": "model", "owned_by": "amazon"},
                    {"id": "llama2-13b", "object": "model", "owned_by": "meta"},
                    {"id": "llama2-70b", "object": "model", "owned_by": "meta"},
                ]
            }

            metrics.mark_complete(status_code=200, response_time=self._get_current_time() - start_time)
            
            return ProviderResponse(
                data=models_data,
                status_code=200,
                headers={},
                metrics=metrics,
                provider_type=self.provider_type,
            )

        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "Failed to list AWS Bedrock models",
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
        """Create a chat completion using AWS Bedrock."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)
        
        try:
            model_id = self._get_model_id(model)
            url = f"{self.config.base_url}/model/{model_id}/invoke"
            
            if stream:
                url = f"{self.config.base_url}/model/{model_id}/invoke-with-response-stream"
            
            # Prepare request payload based on model
            payload = self._prepare_bedrock_request(model_id, messages=messages, **kwargs)
            
            # AWS Bedrock requires AWS Signature V4 authentication
            headers = await self._get_aws_headers("POST", url, json.dumps(payload))
            
            if not stream:
                response = await self._make_request("POST", url, json=payload, headers=headers)
                
                # Parse Bedrock response based on model
                response_data = self._parse_bedrock_response(model_id, response.json())
                
                metrics.mark_complete(
                    status_code=response.status_code,
                    response_time=self._get_current_time() - start_time
                )
                
                return ProviderResponse(
                    data=response_data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metrics=metrics,
                    provider_type=self.provider_type,
                )
            else:
                # Handle streaming response
                return self._handle_streaming_response(url, payload, headers, metrics, model_id)
                
        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "AWS Bedrock chat completion failed",
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
        """Create a text completion using AWS Bedrock."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)
        
        try:
            model_id = self._get_model_id(model)
            url = f"{self.config.base_url}/model/{model_id}/invoke"
            
            if stream:
                url = f"{self.config.base_url}/model/{model_id}/invoke-with-response-stream"
            
            # Prepare request payload based on model
            payload = self._prepare_bedrock_request(model_id, prompt=prompt, **kwargs)
            
            headers = await self._get_aws_headers("POST", url, json.dumps(payload))
            
            if not stream:
                response = await self._make_request("POST", url, json=payload, headers=headers)
                
                # Parse Bedrock response based on model
                response_data = self._parse_bedrock_response(model_id, response.json())
                
                metrics.mark_complete(
                    status_code=response.status_code,
                    response_time=self._get_current_time() - start_time
                )
                
                return ProviderResponse(
                    data=response_data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metrics=metrics,
                    provider_type=self.provider_type,
                )
            else:
                # Handle streaming response
                return self._handle_streaming_response(url, payload, headers, metrics, model_id)
                
        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "AWS Bedrock text completion failed",
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
        """Create embeddings using AWS Bedrock."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)

        try:
            model_id = self._get_model_id(model)
            url = f"{self.config.base_url}/model/{model_id}/invoke"

            # Prepare embeddings request (typically for Titan Embeddings)
            if isinstance(input_text, str):
                input_text = [input_text]

            payload = {
                "inputText": input_text[0] if len(input_text) == 1 else input_text,
            }

            headers = await self._get_aws_headers("POST", url, json.dumps(payload))
            response = await self._make_request("POST", url, json=payload, headers=headers)

            # Convert Bedrock embeddings response to OpenAI format
            bedrock_response = response.json()
            embeddings_data = {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": bedrock_response.get("embedding", [])
                    }
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": len(str(input_text).split()),
                    "total_tokens": len(str(input_text).split())
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
                "AWS Bedrock embeddings failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Embeddings creation failed: {e}", provider_type=self.provider_type) from e

    def _parse_bedrock_response(self, model_id: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Bedrock response to OpenAI-compatible format."""
        if model_id.startswith("anthropic.claude"):
            return self._parse_claude_response(response_data)
        elif model_id.startswith("amazon.titan"):
            return self._parse_titan_response(response_data)
        elif model_id.startswith("meta.llama"):
            return self._parse_llama_response(response_data)
        else:
            return response_data

    def _parse_claude_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Claude response to OpenAI format."""
        completion = response_data.get("completion", "")
        return {
            "id": f"chatcmpl-{self._generate_id()}",
            "object": "chat.completion",
            "created": int(self._get_current_time()),
            "model": "claude",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": response_data.get("prompt_tokens", 0),
                "completion_tokens": response_data.get("completion_tokens", 0),
                "total_tokens": response_data.get("total_tokens", 0)
            }
        }

    def _parse_titan_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Titan response to OpenAI format."""
        results = response_data.get("results", [])
        content = results[0].get("outputText", "") if results else ""

        return {
            "id": f"chatcmpl-{self._generate_id()}",
            "object": "chat.completion",
            "created": int(self._get_current_time()),
            "model": "titan",
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
                "prompt_tokens": response_data.get("inputTextTokenCount", 0),
                "completion_tokens": len(content.split()),
                "total_tokens": response_data.get("inputTextTokenCount", 0) + len(content.split())
            }
        }

    def _parse_llama_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Llama response to OpenAI format."""
        generation = response_data.get("generation", "")

        return {
            "id": f"chatcmpl-{self._generate_id()}",
            "object": "chat.completion",
            "created": int(self._get_current_time()),
            "model": "llama",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generation
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": response_data.get("prompt_token_count", 0),
                "completion_tokens": response_data.get("generation_token_count", 0),
                "total_tokens": response_data.get("prompt_token_count", 0) + response_data.get("generation_token_count", 0)
            }
        }

    async def _get_aws_headers(self, method: str, url: str, payload: str) -> Dict[str, str]:
        """Generate AWS Signature V4 headers for Bedrock API."""
        # This is a simplified implementation
        # In production, you should use boto3 or implement full AWS Signature V4
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"ollama-proxy/0.1.0 (aws-bedrock)",
        }

        # For now, assume IAM role or AWS credentials are configured
        # In a full implementation, you would generate AWS Signature V4 here
        if self.config.api_key:
            headers["Authorization"] = f"AWS4-HMAC-SHA256 Credential={self.config.api_key}"

        return headers

    async def _handle_streaming_response(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        metrics: ProviderMetrics,
        model_id: str,
    ) -> AsyncIterator[bytes]:
        """Handle streaming response from AWS Bedrock."""
        try:
            client = await self.get_client()
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes():
                    if chunk:
                        # Parse Bedrock streaming format and convert to OpenAI format
                        yield self._convert_bedrock_stream_chunk(chunk, model_id)

                metrics.mark_complete(status_code=response.status_code)

        except Exception as e:
            metrics.mark_error(error=str(e))
            logger.error(
                "AWS Bedrock streaming failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Streaming failed: {e}", provider_type=self.provider_type) from e

    def transform_ollama_to_provider(
        self,
        ollama_request: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama request format to AWS Bedrock format."""
        # AWS Bedrock has different formats for different models
        # For now, return a copy with minimal transformation
        return ollama_request.copy()

    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform AWS Bedrock response format to Ollama format."""
        # AWS Bedrock has different formats for different models
        # For now, return a copy with minimal transformation
        return provider_response.copy()

    def _convert_bedrock_stream_chunk(self, chunk: bytes, model_id: str) -> bytes:
        """Convert Bedrock streaming chunk to OpenAI format."""
        try:
            # Parse Bedrock event stream format
            # This is a simplified implementation
            chunk_str = chunk.decode('utf-8')
            if chunk_str.startswith('data: '):
                data = json.loads(chunk_str[6:])

                # Convert to OpenAI streaming format
                openai_chunk = {
                    "id": f"chatcmpl-{self._generate_id()}",
                    "object": "chat.completion.chunk",
                    "created": int(self._get_current_time()),
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": data.get("completion", data.get("outputText", ""))
                            },
                            "finish_reason": None
                        }
                    ]
                }

                return f"data: {json.dumps(openai_chunk)}\n\n".encode('utf-8')

            return chunk

        except Exception:
            return chunk

    def _generate_id(self) -> str:
        """Generate a unique ID for responses."""
        import uuid
        return str(uuid.uuid4())[:8]
