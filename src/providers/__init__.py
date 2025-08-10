"""AI Provider interfaces and implementations for ollama-proxy.

This module provides a unified interface for multiple AI providers including
OpenRouter, OpenAI, Anthropic Claude, and Google Gemini. The architecture
supports extensible provider registration and standardized request/response
transformation.
"""

from .base import (
    AIProvider,
    ProviderCapability,
    ProviderConfig,
    ProviderError,
    ProviderMetrics,
    ProviderResponse,
    ProviderType,
)
from .factory import ProviderFactory, get_provider
from .registry import ProviderRegistry, register_provider

# Import provider implementations
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .azure_provider import AzureProvider
from .aws_bedrock_provider import AWSBedrockProvider
from .ollama_provider import OllamaProvider

# Import initialization utilities
from .init_providers import (
    register_all_providers,
    get_provider_capabilities,
    find_providers_with_capability,
    is_provider_available,
)

__all__ = [
    # Base classes and interfaces
    "AIProvider",
    "ProviderCapability",
    "ProviderConfig",
    "ProviderError",
    "ProviderMetrics",
    "ProviderResponse",
    "ProviderType",
    # Factory and registry
    "ProviderFactory",
    "ProviderRegistry",
    "get_provider",
    "register_provider",
    # Provider implementations
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "AzureProvider",
    "AWSBedrockProvider",
    "OllamaProvider",
    # Initialization utilities
    "register_all_providers",
    "get_provider_capabilities",
    "find_providers_with_capability",
    "is_provider_available",
]