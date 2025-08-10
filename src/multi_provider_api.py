"""Multi-provider API router with enhanced functionality.

This module provides the main API endpoints that support multiple AI providers
with intelligent routing, fallback mechanisms, and comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .exceptions import ErrorContext, ErrorType, ProxyError
from .models import OllamaChatRequest, OllamaGenerateRequest, OllamaEmbeddingsRequest
from .multi_provider_config import MultiProviderSettings
from .providers.base import ProviderCapability, ProviderType
from .providers.factory import get_factory
from .providers.retry import get_health_manager
from .providers.router import ProviderRouter
from .providers.transformers import RequestTransformer, ResponseTransformer

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["multi-provider"])


class MultiProviderAPI:
    """Multi-provider API handler."""

    def __init__(self, settings: MultiProviderSettings):
        self.settings = settings
        self.router = ProviderRouter(
            routing_strategy=settings.routing_strategy,
            fallback_strategy=settings.fallback_strategy,
            enable_load_balancing=settings.enable_load_balancing,
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all configured providers."""
        if self._initialized:
            return

        factory = get_factory()
        provider_settings = self.settings.get_provider_settings()

        for provider_type, settings in provider_settings.items():
            if settings.enabled:
                try:
                    config = settings.to_provider_config(provider_type)
                    provider = factory.create_provider(provider_type, config)
                    # Initialize provider if it has an initialize method
                    if hasattr(provider, 'initialize'):
                        await provider.initialize()
                    logger.info(
                        "Initialized provider",
                        provider_type=provider_type.value,
                        base_url=config.base_url,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to initialize provider",
                        provider_type=provider_type.value,
                        error=str(e),
                    )

        self._initialized = True
        logger.info("Multi-provider API initialized")

    async def chat_completion(
        self,
        request: OllamaChatRequest,
        preferred_provider: Optional[ProviderType] = None,
    ) -> Union[Dict[str, Any], StreamingResponse]:
        """Handle chat completion with multi-provider support."""
        await self.initialize()

        try:
            # Route to appropriate provider
            provider = await self.router.route_request(
                capability=ProviderCapability.CHAT_COMPLETION,
                model=request.model,
                preferred_provider=preferred_provider,
            )

            # Transform request to provider format
            provider_request = RequestTransformer.transform_chat_request(
                request.model_dump(),
                provider.provider_type,
            )

            # Execute request with protection
            health_manager = get_health_manager()

            async def execute_request():
                return await provider.chat_completion(
                    messages=provider_request.get("messages", []),
                    model=provider_request.get("model", request.model),
                    stream=request.stream,
                    **{k: v for k, v in provider_request.items()
                       if k not in ["messages", "model", "stream"]},
                )

            if request.stream:
                # Handle streaming response
                return await self._handle_streaming_response(
                    execute_request, provider.provider_type, "chat"
                )
            else:
                # Handle non-streaming response
                provider_response = await health_manager.execute_with_protection(
                    execute_request,
                    provider.provider_type,
                )

                # Transform response to Ollama format
                ollama_response = ResponseTransformer.transform_chat_response(
                    provider_response.data,
                    provider.provider_type,
                )

                # Update metrics
                await self.router.update_provider_response_time(
                    provider.provider_type,
                    provider_response.metrics.duration_ms,
                )
                await self.router.release_provider(provider.provider_type)

                return ollama_response

        except Exception as e:
            logger.error(
                "Chat completion failed",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Try fallback if available
            if hasattr(e, 'provider_type'):
                fallback_provider = await self.router.handle_provider_error(
                    e.provider_type,
                    ProviderCapability.CHAT_COMPLETION,
                    e,
                    model=request.model,
                )

                if fallback_provider:
                    logger.info(
                        "Attempting fallback provider",
                        original_provider=e.provider_type.value,
                        fallback_provider=fallback_provider.provider_type.value,
                    )
                    # Recursive call with fallback provider
                    return await self.chat_completion(request, fallback_provider.provider_type)

            # No fallback available, raise the error
            if isinstance(e, ProxyError):
                raise
            else:
                raise ProxyError(
                    message=f"Chat completion failed: {e}",
                    error_type=ErrorType.INTERNAL_ERROR,
                ) from e

    async def generate_completion(
        self,
        request: OllamaGenerateRequest,
        preferred_provider: Optional[ProviderType] = None,
    ) -> Union[Dict[str, Any], StreamingResponse]:
        """Handle text generation with multi-provider support."""
        await self.initialize()

        try:
            # Route to appropriate provider
            provider = await self.router.route_request(
                capability=ProviderCapability.TEXT_COMPLETION,
                model=request.model,
                preferred_provider=preferred_provider,
            )

            # Transform request to provider format
            provider_request = RequestTransformer.transform_generate_request(
                request.model_dump(),
                provider.provider_type,
            )

            # Execute request with protection
            health_manager = get_health_manager()

            async def execute_request():
                if provider.provider_type == ProviderType.ANTHROPIC:
                    # Anthropic only supports chat, so use chat_completion
                    return await provider.chat_completion(
                        messages=provider_request.get("messages", []),
                        model=provider_request.get("model", request.model),
                        stream=request.stream or False,
                        **{k: v for k, v in provider_request.items()
                           if k not in ["messages", "model", "stream"]},
                    )
                else:
                    return await provider.text_completion(
                        prompt=provider_request.get("prompt", request.prompt),
                        model=provider_request.get("model", request.model),
                        stream=request.stream or False,
                        **{k: v for k, v in provider_request.items()
                           if k not in ["prompt", "model", "stream"]},
                    )

            if request.stream:
                # Handle streaming response
                return await self._handle_streaming_response(
                    execute_request, provider.provider_type, "generate"
                )
            else:
                # Handle non-streaming response
                provider_response = await health_manager.execute_with_protection(
                    execute_request,
                    provider.provider_type,
                )

                # Transform response to Ollama format
                ollama_response = ResponseTransformer.transform_generate_response(
                    provider_response.data,
                    provider.provider_type,
                )

                # Update metrics
                await self.router.update_provider_response_time(
                    provider.provider_type,
                    provider_response.metrics.duration_ms,
                )
                await self.router.release_provider(provider.provider_type)

                return ollama_response

        except Exception as e:
            logger.error(
                "Generate completion failed",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Try fallback if available
            if hasattr(e, 'provider_type'):
                fallback_provider = await self.router.handle_provider_error(
                    e.provider_type,
                    ProviderCapability.TEXT_COMPLETION,
                    e,
                    model=request.model,
                )

                if fallback_provider:
                    return await self.generate_completion(request, fallback_provider.provider_type)

            # No fallback available, raise the error
            if isinstance(e, ProxyError):
                raise
            else:
                raise ProxyError(
                    message=f"Generate completion failed: {e}",
                    error_type=ErrorType.INTERNAL_ERROR,
                ) from e

    async def create_embeddings(
        self,
        request: OllamaEmbeddingsRequest,
        preferred_provider: Optional[ProviderType] = None,
    ) -> Dict[str, Any]:
        """Handle embeddings creation with multi-provider support."""
        await self.initialize()

        try:
            # Route to appropriate provider
            provider = await self.router.route_request(
                capability=ProviderCapability.EMBEDDINGS,
                model=request.model,
                preferred_provider=preferred_provider,
            )

            # Transform request to provider format
            provider_request = RequestTransformer.transform_embeddings_request(
                request.model_dump(),
                provider.provider_type,
            )

            # Execute request with protection
            health_manager = get_health_manager()

            async def execute_request():
                return await provider.create_embeddings(
                    input_text=provider_request.get("input", request.prompt),
                    model=provider_request.get("model", request.model),
                    **{k: v for k, v in provider_request.items()
                       if k not in ["input", "model"]},
                )

            provider_response = await health_manager.execute_with_protection(
                execute_request,
                provider.provider_type,
            )

            # Transform response to Ollama format
            ollama_response = ResponseTransformer.transform_embeddings_response(
                provider_response.data,
                provider.provider_type,
            )

            # Update metrics
            await self.router.update_provider_response_time(
                provider.provider_type,
                provider_response.metrics.duration_ms,
            )
            await self.router.release_provider(provider.provider_type)

            return ollama_response

        except Exception as e:
            logger.error(
                "Embeddings creation failed",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Try fallback if available
            if hasattr(e, 'provider_type'):
                fallback_provider = await self.router.handle_provider_error(
                    e.provider_type,
                    ProviderCapability.EMBEDDINGS,
                    e,
                    model=request.model,
                )

                if fallback_provider:
                    return await self.create_embeddings(request, fallback_provider.provider_type)

            # No fallback available, raise the error
            if isinstance(e, ProxyError):
                raise
            else:
                raise ProxyError(
                    message=f"Embeddings creation failed: {e}",
                    error_type=ErrorType.INTERNAL_ERROR,
                ) from e

    async def _handle_streaming_response(
        self,
        execute_func,
        provider_type: ProviderType,
        endpoint: str,
    ) -> StreamingResponse:
        """Handle streaming responses from providers."""
        async def stream_generator():
            try:
                async for chunk in await execute_func():
                    yield chunk
            except Exception as e:
                logger.error(
                    "Streaming error",
                    provider_type=provider_type.value,
                    endpoint=endpoint,
                    error=str(e),
                )
                # Send error as SSE event
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n".encode()
            finally:
                await self.router.release_provider(provider_type)

        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async def list_models(self, provider_type: Optional[ProviderType] = None) -> Dict[str, Any]:
        """List models from all or specific providers."""
        await self.initialize()

        if provider_type:
            # List models from specific provider
            factory = get_factory()
            provider = factory.get_provider(provider_type)
            if provider is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Provider {provider_type.value} not configured"
                )

            response = await provider.list_models()
            return response.data
        else:
            # List models from all providers
            factory = get_factory()
            all_models: Dict[str, List[Dict[str, Any]]] = {"models": []}

            for instance_id, provider in factory.get_all_instances().items():
                try:
                    response = await provider.list_models()
                    provider_models = response.data.get("data", [])

                    # Add provider information to each model
                    for model in provider_models:
                        model["provider"] = provider.provider_type.value
                        all_models["models"].append(model)

                except Exception as e:
                    logger.warning(
                        "Failed to list models from provider",
                        provider_type=provider.provider_type.value,
                        error=str(e),
                    )

            return all_models

    async def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        await self.initialize()

        stats = self.router.get_provider_stats()
        health_status = get_health_manager().get_all_health_status()

        # Combine stats and health information
        combined_stats = {}
        for provider_type in ProviderType:
            provider_key = provider_type.value
            combined_stats[provider_key] = {
                **stats.get(provider_key, {}),
                **health_status.get(provider_type, {}),
            }

        return {
            "providers": combined_stats,
            "routing_strategy": self.settings.routing_strategy.value,
            "fallback_strategy": self.settings.fallback_strategy.value,
            "load_balancing_enabled": self.settings.enable_load_balancing,
        }