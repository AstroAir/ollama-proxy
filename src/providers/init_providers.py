"""Provider initialization and registration.

This module handles the registration of all available AI providers
and provides utilities for provider discovery and initialization.
"""

from __future__ import annotations

import structlog

from .anthropic_provider import AnthropicProvider
from .base import ProviderCapability, ProviderType
from .google_provider import GoogleProvider
from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider
from .registry import register_provider

logger = structlog.get_logger(__name__)


def register_all_providers() -> None:
    """Register all available providers with their capabilities."""

    # Register OpenRouter provider
    register_provider(
        ProviderType.OPENROUTER,
        OpenRouterProvider,
        frozenset([
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_COMPLETION,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.STREAMING,
        ])
    )

    # Register OpenAI provider
    register_provider(
        ProviderType.OPENAI,
        OpenAIProvider,
        frozenset([
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_COMPLETION,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.STREAMING,
            ProviderCapability.FUNCTION_CALLING,
            ProviderCapability.VISION,
        ])
    )

    # Register Anthropic provider
    register_provider(
        ProviderType.ANTHROPIC,
        AnthropicProvider,
        frozenset([
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
        ])
    )

    # Register Google provider
    register_provider(
        ProviderType.GOOGLE,
        GoogleProvider,
        frozenset([
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_COMPLETION,
            ProviderCapability.EMBEDDINGS,
            ProviderCapability.STREAMING,
            ProviderCapability.VISION,
        ])
    )

    logger.info(
        "Registered all AI providers",
        providers=[
            ProviderType.OPENROUTER.value,
            ProviderType.OPENAI.value,
            ProviderType.ANTHROPIC.value,
            ProviderType.GOOGLE.value,
        ]
    )


def get_provider_capabilities() -> dict[ProviderType, frozenset[ProviderCapability]]:
    """Get capabilities for all registered providers."""
    from .registry import get_registry

    registry = get_registry()
    capabilities = {}

    for provider_type in registry.get_registered_providers():
        provider_caps = registry.get_capabilities(provider_type)
        if provider_caps:
            capabilities[provider_type] = provider_caps

    return capabilities


def find_providers_with_capability(capability: ProviderCapability) -> list[ProviderType]:
    """Find all providers that support a specific capability."""
    from .registry import get_registry

    registry = get_registry()
    return registry.get_providers_with_capability(capability)


def is_provider_available(provider_type: ProviderType) -> bool:
    """Check if a provider is available (registered)."""
    from .registry import get_registry

    registry = get_registry()
    return registry.is_registered(provider_type)


# Auto-register providers when module is imported
register_all_providers()