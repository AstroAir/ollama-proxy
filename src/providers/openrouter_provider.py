"""OpenRouter provider implementation.

This module provides an OpenRouter provider that implements the AIProvider interface
and wraps the existing OpenRouterClient for compatibility with the multi-provider system.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import structlog

from ..config import Settings
from ..exceptions import ErrorContext, ErrorType, ProxyError
from ..openrouter import OpenRouterClient
from .base import (
    AIProvider,
    ProviderCapability,
    ProviderConfig,
    ProviderError,
    ProviderMetrics,
    ProviderResponse,
    ProviderType,
)
from .transformers import RequestTransformer, ResponseTransformer

logger = structlog.get_logger(__name__)


class OpenRouterProvider(AIProvider):
    """OpenRouter provider implementation."""

    def __init__(self, config: ProviderConfig):
        """Initialize OpenRouter provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self._openrouter_client: Optional[OpenRouterClient] = None
        self._transformer = RequestTransformer()
        self._response_transformer = ResponseTransformer()

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OPENROUTER

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        """Get supported capabilities."""
        return frozenset([
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_COMPLETION,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.STREAMING,
        ])

    async def initialize(self) -> None:
        """Initialize the provider."""
        if self._openrouter_client is None:
            # Set environment variable for Settings
            os.environ["OPENROUTER_API_KEY"] = self.config.api_key

            # Create settings for OpenRouterClient
            settings = Settings()  # type: ignore[call-arg]

            self._openrouter_client = OpenRouterClient(settings)

        logger.info(
            "OpenRouter provider initialized",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    async def close(self) -> None:
        """Close the provider and cleanup resources."""
        if self._openrouter_client:
            await self._openrouter_client.close()
            self._openrouter_client = None
        logger.info("OpenRouter provider closed")

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Perform chat completion."""
        if not self._openrouter_client:
            await self.initialize()

        # Ensure client is initialized
        assert self._openrouter_client is not None, "OpenRouter client should be initialized"

        try:
            # Transform model name from Ollama format to OpenRouter format
            openrouter_model = model.replace(":latest", "")

            # If the model doesn't already have a provider prefix, add appropriate prefix
            if "/" not in openrouter_model:
                if openrouter_model.startswith("gpt"):
                    openrouter_model = f"openai/{openrouter_model}"
                elif openrouter_model.startswith("claude"):
                    openrouter_model = f"anthropic/{openrouter_model}"
                elif openrouter_model.startswith("gemini"):
                    openrouter_model = f"google/{openrouter_model}"
            else:
                # Replace ":" with "/" for models that already have a provider format
                openrouter_model = openrouter_model.replace(":", "/")

            # Prepare the request payload (without stream parameter)
            payload = {
                "model": openrouter_model,
                "messages": messages,
                **kwargs
            }

            if stream:
                # Return streaming response
                return self._openrouter_client.chat_completion_stream(payload)
            else:
                # Return non-streaming response
                response = await self._openrouter_client.chat_completion(payload, stream=stream)

                return ProviderResponse(
                    data=response.data or {},
                    status_code=response.status_code,
                    headers=response.headers or {},
                    metrics=ProviderMetrics(provider_type=self.provider_type),
                    provider_type=self.provider_type,
                )

        except Exception as e:
            logger.error(
                "Chat completion failed",
                provider_type=self.provider_type.value,
                model=model,
                error=str(e),
            )
            raise ProviderError(
                f"Chat completion failed: {e}",
                provider_type=self.provider_type,
                context=ErrorContext(
                    additional_data={
                        "operation": "chat_completion",
                        "provider": self.provider_type.value,
                        "model": model,
                    }
                ),
            )

    async def text_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Perform text completion."""
        raise NotImplementedError("OpenRouter provider text completion not yet implemented")

    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create embeddings."""
        raise NotImplementedError("OpenRouter provider embeddings not yet implemented")

    async def list_models(self) -> ProviderResponse:
        """List available models."""
        if not self._openrouter_client:
            await self.initialize()

        # Ensure client is initialized
        assert self._openrouter_client is not None, "OpenRouter client should be initialized"

        try:
            # Use the existing OpenRouterClient to fetch models
            response = await self._openrouter_client.fetch_models()

            # Transform the response to match the expected format
            models = []
            if response.data and "data" in response.data:
                for model in response.data["data"]:
                    model_id = model.get("id", "")
                    # Transform "google/gemini-pro" to "gemini-pro:latest"
                    if "/" in model_id:
                        model_name = model_id.split("/", 1)[1] + ":latest"
                    else:
                        model_name = model_id + ":latest"

                    models.append({
                        "id": model_id,
                        "name": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": model_id.split("/")[0] if "/" in model_id else "unknown",
                        "size": 0,  # OpenRouter doesn't provide model sizes, so default to 0
                        "digest": f"sha256:{model_id.replace('/', '_')}_digest",  # Generate a fake digest for compatibility
                        "modified_at": "2024-01-01T00:00:00Z",  # Default modified timestamp
                    })

            return ProviderResponse(
                data={"data": models},
                status_code=200,
                headers={},
                metrics=ProviderMetrics(provider_type=self.provider_type),
                provider_type=self.provider_type,
            )

        except Exception as e:
            logger.error(
                "Failed to list models",
                provider_type=self.provider_type.value,
                error=str(e),
            )
            raise ProviderError(
                f"Failed to list models: {e}",
                provider_type=self.provider_type,
                context=ErrorContext(
                    additional_data={
                        "operation": "list_models",
                        "provider": self.provider_type.value,
                    }
                ),
            )

    def transform_ollama_to_provider(
        self,
        ollama_request: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama request format to provider format."""
        return self._transformer.transform_chat_request(ollama_request, ProviderType.OPENROUTER)

    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform provider response format to Ollama format."""
        return self._response_transformer.transform_chat_response(provider_response, ProviderType.OPENROUTER)
