"""Provider registry for managing AI provider implementations.

This module provides a centralized registry for AI providers, allowing
dynamic registration and discovery of provider implementations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import structlog

from .base import AIProvider, ProviderCapability, ProviderType

logger = structlog.get_logger(__name__)


class ProviderRegistry:
    """Registry for AI provider implementations."""

    def __init__(self):
        self._providers: Dict[ProviderType, Type[AIProvider]] = {}
        self._capabilities: Dict[ProviderType, frozenset[ProviderCapability]] = {}

    def register(
        self,
        provider_type: ProviderType,
        provider_class: Type[AIProvider],
        capabilities: Optional[frozenset[ProviderCapability]] = None,
    ) -> None:
        """Register a provider implementation.

        Args:
            provider_type: The type of provider being registered
            provider_class: The provider implementation class
            capabilities: Optional set of capabilities this provider supports
        """
        if provider_type in self._providers:
            logger.warning(
                "Overriding existing provider registration",
                provider_type=provider_type.value,
                existing_class=self._providers[provider_type].__name__,
                new_class=provider_class.__name__,
            )

        self._providers[provider_type] = provider_class

        if capabilities is not None:
            self._capabilities[provider_type] = capabilities

        logger.info(
            "Registered provider",
            provider_type=provider_type.value,
            provider_class=provider_class.__name__,
            capabilities=list(capabilities) if capabilities else None,
        )

    def get_provider_class(self, provider_type: ProviderType) -> Optional[Type[AIProvider]]:
        """Get the provider class for a given type.

        Args:
            provider_type: The type of provider to get

        Returns:
            The provider class if registered, None otherwise
        """
        return self._providers.get(provider_type)

    def is_registered(self, provider_type: ProviderType) -> bool:
        """Check if a provider type is registered.

        Args:
            provider_type: The provider type to check

        Returns:
            True if the provider is registered, False otherwise
        """
        return provider_type in self._providers

    def get_registered_providers(self) -> List[ProviderType]:
        """Get list of all registered provider types.

        Returns:
            List of registered provider types
        """
        return list(self._providers.keys())

    def get_providers_with_capability(
        self, capability: ProviderCapability
    ) -> List[ProviderType]:
        """Get providers that support a specific capability.

        Args:
            capability: The capability to search for

        Returns:
            List of provider types that support the capability
        """
        providers = []
        for provider_type, capabilities in self._capabilities.items():
            if capability in capabilities:
                providers.append(provider_type)
        return providers

    def get_capabilities(self, provider_type: ProviderType) -> Optional[frozenset[ProviderCapability]]:
        """Get capabilities for a specific provider.

        Args:
            provider_type: The provider type to get capabilities for

        Returns:
            Set of capabilities if known, None otherwise
        """
        return self._capabilities.get(provider_type)

    def unregister(self, provider_type: ProviderType) -> bool:
        """Unregister a provider.

        Args:
            provider_type: The provider type to unregister

        Returns:
            True if the provider was unregistered, False if it wasn't registered
        """
        if provider_type not in self._providers:
            return False

        del self._providers[provider_type]
        self._capabilities.pop(provider_type, None)

        logger.info("Unregistered provider", provider_type=provider_type.value)
        return True

    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        self._capabilities.clear()
        logger.info("Cleared all provider registrations")


# Global registry instance
_registry = ProviderRegistry()


def register_provider(
    provider_type: ProviderType,
    provider_class: Type[AIProvider],
    capabilities: Optional[frozenset[ProviderCapability]] = None,
) -> None:
    """Register a provider in the global registry.

    Args:
        provider_type: The type of provider being registered
        provider_class: The provider implementation class
        capabilities: Optional set of capabilities this provider supports
    """
    _registry.register(provider_type, provider_class, capabilities)


def get_registry() -> ProviderRegistry:
    """Get the global provider registry.

    Returns:
        The global provider registry instance
    """
    return _registry