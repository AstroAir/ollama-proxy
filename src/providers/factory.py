"""Provider factory for creating AI provider instances.

This module provides factory functions for creating AI provider instances
based on configuration and provider type.
"""

from __future__ import annotations

from typing import Dict, Optional

import structlog

from .base import AIProvider, ProviderConfig, ProviderType
from .registry import get_registry

logger = structlog.get_logger(__name__)


class ProviderFactory:
    """Factory for creating AI provider instances."""

    def __init__(self):
        self._instances: Dict[str, AIProvider] = {}

    def create_provider(
        self,
        provider_type: ProviderType,
        config: ProviderConfig,
        instance_id: Optional[str] = None,
    ) -> AIProvider:
        """Create a provider instance.

        Args:
            provider_type: The type of provider to create
            config: Configuration for the provider
            instance_id: Optional instance identifier for caching

        Returns:
            The created provider instance

        Raises:
            ValueError: If the provider type is not registered
        """
        registry = get_registry()
        provider_class = registry.get_provider_class(provider_type)

        if provider_class is None:
            available_providers = registry.get_registered_providers()
            raise ValueError(
                f"Provider type '{provider_type.value}' is not registered. "
                f"Available providers: {[p.value for p in available_providers]}"
            )

        # Use instance_id for caching if provided, otherwise use provider_type
        cache_key = instance_id or provider_type.value

        if cache_key in self._instances:
            logger.debug(
                "Returning cached provider instance",
                provider_type=provider_type.value,
                instance_id=cache_key,
            )
            return self._instances[cache_key]

        # Create new instance
        try:
            provider = provider_class(config)
            self._instances[cache_key] = provider

            logger.info(
                "Created provider instance",
                provider_type=provider_type.value,
                instance_id=cache_key,
                capabilities=list(provider.capabilities),
            )

            return provider

        except Exception as e:
            logger.error(
                "Failed to create provider instance",
                provider_type=provider_type.value,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def get_provider(
        self,
        provider_type: ProviderType,
        instance_id: Optional[str] = None,
    ) -> Optional[AIProvider]:
        """Get an existing provider instance.

        Args:
            provider_type: The type of provider to get
            instance_id: Optional instance identifier

        Returns:
            The provider instance if it exists, None otherwise
        """
        cache_key = instance_id or provider_type.value
        return self._instances.get(cache_key)

    def remove_provider(
        self,
        provider_type: ProviderType,
        instance_id: Optional[str] = None,
    ) -> bool:
        """Remove a provider instance from the cache.

        Args:
            provider_type: The type of provider to remove
            instance_id: Optional instance identifier

        Returns:
            True if the provider was removed, False if it didn't exist
        """
        cache_key = instance_id or provider_type.value

        if cache_key in self._instances:
            provider = self._instances.pop(cache_key)
            logger.info(
                "Removed provider instance",
                provider_type=provider_type.value,
                instance_id=cache_key,
            )
            return True

        return False

    async def close_all(self) -> None:
        """Close all provider instances."""
        for instance_id, provider in self._instances.items():
            try:
                await provider.close()
                logger.debug(
                    "Closed provider instance",
                    provider_type=provider.provider_type.value,
                    instance_id=instance_id,
                )
            except Exception as e:
                logger.error(
                    "Error closing provider instance",
                    provider_type=provider.provider_type.value,
                    instance_id=instance_id,
                    error=str(e),
                )

        self._instances.clear()
        logger.info("Closed all provider instances")

    def get_all_instances(self) -> Dict[str, AIProvider]:
        """Get all provider instances.

        Returns:
            Dictionary mapping instance IDs to provider instances
        """
        return self._instances.copy()


# Global factory instance
_factory = ProviderFactory()


def get_provider(
    provider_type: ProviderType,
    config: Optional[ProviderConfig] = None,
    instance_id: Optional[str] = None,
) -> AIProvider:
    """Get or create a provider instance.

    Args:
        provider_type: The type of provider to get
        config: Configuration for creating a new provider (required if not cached)
        instance_id: Optional instance identifier

    Returns:
        The provider instance

    Raises:
        ValueError: If the provider is not cached and no config is provided
    """
    existing = _factory.get_provider(provider_type, instance_id)
    if existing is not None:
        return existing

    if config is None:
        raise ValueError(
            f"No cached provider found for {provider_type.value} and no config provided"
        )

    return _factory.create_provider(provider_type, config, instance_id)


def get_factory() -> ProviderFactory:
    """Get the global provider factory.

    Returns:
        The global provider factory instance
    """
    return _factory