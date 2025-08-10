"""Azure OpenAI API provider implementation.

This module provides an implementation of the AIProvider interface for
Azure OpenAI Service, supporting both chat completions and embeddings.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Union
from urllib.parse import urljoin

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


class AzureProvider(AIProvider):
    """Azure OpenAI API provider implementation."""

    def __init__(self, config: ProviderConfig):
        """Initialize Azure OpenAI provider.

        Args:
            config: Provider configuration
        """
        # Set default capabilities if not provided
        if not config.capabilities:
            config = ProviderConfig(
                provider_type=config.provider_type,
                api_key=config.api_key,
                base_url=config.base_url,
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

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.AZURE

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        """Get supported capabilities."""
        return self.config.capabilities

    def get_headers(self, extra_headers: Dict[str, str] | None = None) -> Dict[str, str]:
        """Get headers for Azure OpenAI API requests."""
        headers = {
            "api-key": self.config.api_key,
            "Content-Type": "application/json",
            "User-Agent": f"ollama-proxy/0.1.0 (azure-openai)",
        }

        # Add custom headers from config
        headers.update(self.config.custom_headers)

        # Add extra headers if provided
        if extra_headers:
            headers.update(extra_headers)

        return headers

    def _build_url(self, endpoint: str, deployment_name: str, api_version: str = "2024-02-01") -> str:
        """Build Azure OpenAI API URL.
        
        Args:
            endpoint: API endpoint (e.g., 'chat/completions', 'embeddings')
            deployment_name: Azure deployment name
            api_version: API version
            
        Returns:
            Complete API URL
        """
        if not self.config.base_url:
            raise ProviderError("Azure base URL is required", provider_type=self.provider_type)
            
        # Azure OpenAI URL format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={api_version}
        url = urljoin(
            self.config.base_url.rstrip('/'),
            f"/openai/deployments/{deployment_name}/{endpoint}"
        )
        return f"{url}?api-version={api_version}"

    def _extract_deployment_from_model(self, model: str) -> str:
        """Extract deployment name from model string.
        
        Azure models can be specified as 'deployment_name' or 'deployment_name@resource_name'
        """
        # Check if model is mapped to a deployment
        if model in self.config.model_mapping:
            return self.config.model_mapping[model]
        
        # Extract deployment name (before @ if present)
        return model.split('@')[0]

    async def list_models(self) -> ProviderResponse:
        """List available models from Azure OpenAI."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)

        try:
            # Azure OpenAI doesn't have a direct models endpoint like OpenAI
            # Return a basic response with common Azure OpenAI models
            models_data = {
                "object": "list",
                "data": [
                    {"id": "gpt-4", "object": "model", "owned_by": "azure-openai"},
                    {"id": "gpt-4-32k", "object": "model", "owned_by": "azure-openai"},
                    {"id": "gpt-35-turbo", "object": "model", "owned_by": "azure-openai"},
                    {"id": "gpt-35-turbo-16k", "object": "model", "owned_by": "azure-openai"},
                    {"id": "text-embedding-ada-002", "object": "model", "owned_by": "azure-openai"},
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
                "Failed to list Azure OpenAI models",
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
        """Create a chat completion using Azure OpenAI."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)
        
        try:
            deployment_name = self._extract_deployment_from_model(model)
            url = self._build_url("chat/completions", deployment_name)
            
            # Prepare request payload (similar to OpenAI format)
            payload = {
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            
            # Remove model from payload as Azure uses deployment name in URL
            payload.pop("model", None)
            
            headers = self.get_headers()
            
            if not stream:
                response = await self._make_request("POST", url, json=payload, headers=headers)
                metrics.mark_complete(
                    status_code=response.status_code,
                    response_time=self._get_current_time() - start_time
                )
                
                return ProviderResponse(
                    data=response.json(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metrics=metrics,
                    provider_type=self.provider_type,
                )
            else:
                # Handle streaming response
                return self._handle_streaming_response(url, payload, headers, metrics)
                
        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "Azure OpenAI chat completion failed",
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
        """Create a text completion using Azure OpenAI.
        
        Note: Azure OpenAI primarily supports chat completions.
        This method converts text completion to chat completion format.
        """
        # Convert text completion to chat completion format
        messages = [{"role": "user", "content": prompt}]
        return await self.chat_completion(messages, model, stream, **kwargs)

    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create embeddings using Azure OpenAI."""
        start_time = self._get_current_time()
        metrics = ProviderMetrics(provider_type=self.provider_type)
        
        try:
            deployment_name = self._extract_deployment_from_model(model)
            url = self._build_url("embeddings", deployment_name)
            
            payload = {
                "input": input_text,
                **kwargs
            }
            
            headers = self.get_headers()
            response = await self._make_request("POST", url, json=payload, headers=headers)
            
            metrics.mark_complete(
                status_code=response.status_code,
                response_time=self._get_current_time() - start_time
            )
            
            return ProviderResponse(
                data=response.json(),
                status_code=response.status_code,
                headers=dict(response.headers),
                metrics=metrics,
                provider_type=self.provider_type,
            )
            
        except Exception as e:
            metrics.mark_error(error=str(e), response_time=self._get_current_time() - start_time)
            logger.error(
                "Azure OpenAI embeddings failed",
                model=model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Embeddings creation failed: {e}", provider_type=self.provider_type) from e

    async def _handle_streaming_response(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        metrics: ProviderMetrics,
    ) -> AsyncIterator[bytes]:
        """Handle streaming response from Azure OpenAI."""
        try:
            client = await self.get_client()
            async with client.stream("POST", url, json=payload, headers=headers) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
                        
                metrics.mark_complete(status_code=response.status_code)
                
        except Exception as e:
            metrics.mark_error(error=str(e))
            logger.error(
                "Azure OpenAI streaming failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ProviderError(f"Streaming failed: {e}", provider_type=self.provider_type) from e

    def transform_ollama_to_provider(
        self,
        ollama_request: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama request format to Azure OpenAI format."""
        # Azure OpenAI uses the same format as OpenAI, so minimal transformation needed
        return ollama_request.copy()

    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Azure OpenAI response format to Ollama format."""
        # Azure OpenAI uses the same format as OpenAI, so minimal transformation needed
        return provider_response.copy()
