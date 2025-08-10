"""Provider routing and load balancing system.

This module handles routing requests to appropriate providers based on
various strategies including capability-based routing, load balancing,
and fallback mechanisms.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

import structlog

from ..multi_provider_config import FallbackStrategy, RoutingStrategy
from .base import AIProvider, ProviderCapability, ProviderError, ProviderType
from .factory import get_factory

logger = structlog.get_logger(__name__)


class ProviderRouter:
    """Router for managing requests across multiple AI providers."""

    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_BASED,
        fallback_strategy: FallbackStrategy = FallbackStrategy.NEXT_AVAILABLE,
        enable_load_balancing: bool = True,
    ):
        self.routing_strategy = routing_strategy
        self.fallback_strategy = fallback_strategy
        self.enable_load_balancing = enable_load_balancing

        # Provider health tracking
        self._provider_health: Dict[ProviderType, bool] = {}
        self._provider_load: Dict[ProviderType, int] = {}
        self._provider_response_times: Dict[ProviderType, List[float]] = {}
        self._last_used: Dict[ProviderType, float] = {}

        # Round-robin state
        self._round_robin_index = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def route_request(
        self,
        capability: ProviderCapability,
        model: Optional[str] = None,
        preferred_provider: Optional[ProviderType] = None,
        **kwargs: Any,
    ) -> AIProvider:
        """Route a request to the most appropriate provider.

        Args:
            capability: Required capability for the request
            model: Model name (if applicable)
            preferred_provider: Preferred provider type (if any)
            **kwargs: Additional routing parameters

        Returns:
            The selected provider instance

        Raises:
            ProviderError: If no suitable provider is available
        """
        async with self._lock:
            # Get available providers for the capability
            available_providers = await self._get_available_providers(capability)

            if not available_providers:
                raise ProviderError(
                    message=f"No providers available for capability: {capability.value}",
                    provider_type=ProviderType.OPENROUTER,  # Default
                    status_code=503,
                )

            # Apply preferred provider if specified and available
            if preferred_provider and preferred_provider in available_providers:
                provider = await self._get_provider_instance(preferred_provider)
                await self._update_provider_load(preferred_provider, 1)
                return provider

            # Select provider based on routing strategy
            selected_provider = await self._select_provider(
                available_providers, capability, model, **kwargs
            )

            provider = await self._get_provider_instance(selected_provider)
            await self._update_provider_load(selected_provider, 1)

            logger.debug(
                "Routed request to provider",
                provider_type=selected_provider.value,
                capability=capability.value,
                strategy=self.routing_strategy.value,
                model=model,
            )

            return provider

    async def handle_provider_error(
        self,
        failed_provider: ProviderType,
        capability: ProviderCapability,
        error: Exception,
        **kwargs: Any,
    ) -> Optional[AIProvider]:
        """Handle provider errors and attempt fallback.

        Args:
            failed_provider: The provider that failed
            capability: Required capability
            error: The error that occurred
            **kwargs: Additional parameters

        Returns:
            Fallback provider instance or None if no fallback available
        """
        async with self._lock:
            # Mark provider as unhealthy
            self._provider_health[failed_provider] = False

            logger.warning(
                "Provider failed, attempting fallback",
                failed_provider=failed_provider.value,
                error=str(error),
                fallback_strategy=self.fallback_strategy.value,
            )

            if self.fallback_strategy == FallbackStrategy.NONE:
                return None

            # Get alternative providers
            available_providers = await self._get_available_providers(capability)
            available_providers = [p for p in available_providers if p != failed_provider]

            if not available_providers:
                logger.error("No fallback providers available")
                return None

            if self.fallback_strategy == FallbackStrategy.NEXT_AVAILABLE:
                # Use the next available provider
                selected_provider = available_providers[0]
            elif self.fallback_strategy == FallbackStrategy.BEST_ALTERNATIVE:
                # Select the best alternative based on current strategy
                selected_provider = await self._select_provider(
                    available_providers, capability, **kwargs
                )
            else:
                return None

            provider = await self._get_provider_instance(selected_provider)
            await self._update_provider_load(selected_provider, 1)

            logger.info(
                "Fallback provider selected",
                fallback_provider=selected_provider.value,
                failed_provider=failed_provider.value,
            )

            return provider

    async def release_provider(self, provider_type: ProviderType) -> None:
        """Release a provider after request completion."""
        async with self._lock:
            await self._update_provider_load(provider_type, -1)

    async def update_provider_response_time(
        self, provider_type: ProviderType, response_time: float
    ) -> None:
        """Update provider response time metrics."""
        async with self._lock:
            if provider_type not in self._provider_response_times:
                self._provider_response_times[provider_type] = []

            # Keep only recent response times (last 100)
            times = self._provider_response_times[provider_type]
            times.append(response_time)
            if len(times) > 100:
                times.pop(0)

            # Mark provider as healthy if response was successful
            self._provider_health[provider_type] = True

    async def _get_available_providers(
        self, capability: ProviderCapability
    ) -> List[ProviderType]:
        """Get list of available providers for a capability."""
        from .init_providers import find_providers_with_capability
        from .factory import get_factory

        # Get providers that support the capability
        capable_providers = find_providers_with_capability(capability)

        # Filter by actual availability (instantiated in factory) and health status
        factory = get_factory()
        available_providers = []

        for provider_type in capable_providers:
            # Check if provider is actually instantiated
            provider_instance = factory.get_provider(provider_type)
            if provider_instance is None:
                continue

            # Check health status (assume healthy if not tracked yet)
            is_healthy = self._provider_health.get(provider_type, True)
            if is_healthy:
                available_providers.append(provider_type)

        return available_providers

    async def _select_provider(
        self,
        available_providers: List[ProviderType],
        capability: ProviderCapability,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderType:
        """Select a provider based on the routing strategy."""
        if len(available_providers) == 1:
            return available_providers[0]

        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_providers)
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return self._select_least_loaded(available_providers)
        elif self.routing_strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._select_fastest_response(available_providers)
        elif self.routing_strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._select_capability_based(available_providers, capability, model)
        else:
            # Default to first available
            return available_providers[0]

    async def _get_provider_instance(self, provider_type: ProviderType) -> AIProvider:
        """Get provider instance from factory."""
        factory = get_factory()
        provider = factory.get_provider(provider_type)

        if provider is None:
            raise ProviderError(
                message=f"Provider {provider_type.value} not initialized",
                provider_type=provider_type,
                status_code=503,
            )

        return provider

    async def _update_provider_load(self, provider_type: ProviderType, delta: int) -> None:
        """Update provider load counter."""
        current_load = self._provider_load.get(provider_type, 0)
        self._provider_load[provider_type] = max(0, current_load + delta)

    def _select_round_robin(self, providers: List[ProviderType]) -> ProviderType:
        """Select provider using round-robin strategy."""
        selected = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        return selected

    def _select_least_loaded(self, providers: List[ProviderType]) -> ProviderType:
        """Select provider with least current load."""
        min_load = float('inf')
        selected = providers[0]

        for provider in providers:
            load = self._provider_load.get(provider, 0)
            if load < min_load:
                min_load = load
                selected = provider

        return selected

    def _select_fastest_response(self, providers: List[ProviderType]) -> ProviderType:
        """Select provider with fastest average response time."""
        best_time = float('inf')
        selected = providers[0]

        for provider in providers:
            times = self._provider_response_times.get(provider, [])
            if times:
                avg_time = sum(times) / len(times)
                if avg_time < best_time:
                    best_time = avg_time
                    selected = provider
            else:
                # No data yet, consider it fast
                selected = provider
                break

        return selected

    def _select_capability_based(
        self,
        providers: List[ProviderType],
        capability: ProviderCapability,
        model: Optional[str] = None,
    ) -> ProviderType:
        """Select provider based on capability and model preferences."""
        # Simple heuristics for capability-based selection
        if model:
            model_lower = model.lower()

            # Route based on model name patterns, but prefer available providers
            # If OpenRouter is available, it can handle most models as a proxy
            if "gpt" in model_lower:
                if ProviderType.OPENAI in providers:
                    return ProviderType.OPENAI
                elif ProviderType.OPENROUTER in providers:
                    return ProviderType.OPENROUTER
            elif "claude" in model_lower:
                if ProviderType.ANTHROPIC in providers:
                    return ProviderType.ANTHROPIC
                elif ProviderType.OPENROUTER in providers:
                    return ProviderType.OPENROUTER
            elif "gemini" in model_lower:
                if ProviderType.GOOGLE in providers:
                    return ProviderType.GOOGLE
                elif ProviderType.OPENROUTER in providers:
                    return ProviderType.OPENROUTER

        # Default capability-based routing
        if capability == ProviderCapability.EMBEDDINGS:
            # Prefer OpenAI for embeddings, then OpenRouter, then Google
            if ProviderType.OPENAI in providers:
                return ProviderType.OPENAI
            elif ProviderType.OPENROUTER in providers:
                return ProviderType.OPENROUTER
            elif ProviderType.GOOGLE in providers:
                return ProviderType.GOOGLE
        elif capability == ProviderCapability.VISION:
            # Prefer providers with good vision support
            if ProviderType.GOOGLE in providers:
                return ProviderType.GOOGLE
            elif ProviderType.ANTHROPIC in providers:
                return ProviderType.ANTHROPIC
            elif ProviderType.OPENAI in providers:
                return ProviderType.OPENAI
            elif ProviderType.OPENROUTER in providers:
                return ProviderType.OPENROUTER

        # Default to least loaded if load balancing is enabled
        if self.enable_load_balancing:
            return self._select_least_loaded(providers)
        else:
            return providers[0]

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers."""
        stats = {}

        for provider_type in ProviderType:
            stats[provider_type.value] = {
                "healthy": self._provider_health.get(provider_type, True),
                "current_load": self._provider_load.get(provider_type, 0),
                "avg_response_time": self._get_avg_response_time(provider_type),
                "last_used": self._last_used.get(provider_type, 0),
            }

        return stats

    def _get_avg_response_time(self, provider_type: ProviderType) -> float:
        """Get average response time for a provider."""
        times = self._provider_response_times.get(provider_type, [])
        if not times:
            return 0.0
        return sum(times) / len(times)

    async def health_check_providers(self) -> Dict[ProviderType, bool]:
        """Perform health checks on all providers."""
        factory = get_factory()
        health_status = {}

        for provider_type in ProviderType:
            try:
                provider = factory.get_provider(provider_type)
                if provider is not None:
                    # Simple health check - try to list models
                    await provider.list_models()
                    health_status[provider_type] = True
                    self._provider_health[provider_type] = True
                else:
                    health_status[provider_type] = False
                    self._provider_health[provider_type] = False
            except Exception as e:
                logger.warning(
                    "Provider health check failed",
                    provider_type=provider_type.value,
                    error=str(e),
                )
                health_status[provider_type] = False
                self._provider_health[provider_type] = False

        return health_status