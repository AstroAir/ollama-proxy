"""Tests for multi-provider functionality.

This module contains comprehensive tests for the enhanced multi-provider
ollama-proxy system, including provider routing, fallback mechanisms,
and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.providers.base import (
    AIProvider,
    ProviderCapability,
    ProviderConfig,
    ProviderType,
    ProviderResponse,
    ProviderMetrics,
)
from src.providers.factory import ProviderFactory
from src.providers.registry import ProviderRegistry
from src.providers.router import ProviderRouter
from src.providers.transformers import RequestTransformer, ResponseTransformer
from src.multi_provider_config import MultiProviderSettings, RoutingStrategy, FallbackStrategy


class MockProvider(AIProvider):
    """Mock provider for testing."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.list_models_response = {"data": [{"id": "test-model", "object": "model"}]}
        self.chat_response = {"choices": [{"message": {"role": "assistant", "content": "Test response"}}]}
        self.embeddings_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    async def list_models(self) -> ProviderResponse:
        """Mock list models."""
        metrics = ProviderMetrics(provider_type=self.provider_type)
        metrics.mark_complete(status_code=200)
        return ProviderResponse(
            data=self.list_models_response,
            status_code=200,
            headers={},
            metrics=metrics,
            provider_type=self.provider_type,
        )

    async def chat_completion(self, messages, model, stream=False, **kwargs):
        """Mock chat completion."""
        metrics = ProviderMetrics(provider_type=self.provider_type, model=model)
        metrics.mark_complete(status_code=200)
        return ProviderResponse(
            data=self.chat_response,
            status_code=200,
            headers={},
            metrics=metrics,
            provider_type=self.provider_type,
        )

    async def text_completion(self, prompt, model, stream=False, **kwargs):
        """Mock text completion."""
        metrics = ProviderMetrics(provider_type=self.provider_type, model=model)
        metrics.mark_complete(status_code=200)
        return ProviderResponse(
            data={"choices": [{"text": "Test completion"}]},
            status_code=200,
            headers={},
            metrics=metrics,
            provider_type=self.provider_type,
        )

    async def create_embeddings(self, input_text, model, **kwargs):
        """Mock embeddings creation."""
        metrics = ProviderMetrics(provider_type=self.provider_type, model=model)
        metrics.mark_complete(status_code=200)
        return ProviderResponse(
            data=self.embeddings_response,
            status_code=200,
            headers={},
            metrics=metrics,
            provider_type=self.provider_type,
        )

    def transform_ollama_to_provider(self, ollama_request, endpoint):
        """Mock transformation."""
        return ollama_request

    def transform_provider_to_ollama(self, provider_response, endpoint):
        """Mock transformation."""
        return provider_response


@pytest.fixture
def mock_provider_config():
    """Create mock provider configuration."""
    return ProviderConfig(
        provider_type=ProviderType.OPENAI,
        api_key="test-key",
        base_url="https://api.test.com",
        capabilities=frozenset([
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_COMPLETION,
            ProviderCapability.EMBEDDINGS,
        ]),
    )


@pytest.fixture
def mock_provider(mock_provider_config):
    """Create mock provider instance."""
    return MockProvider(mock_provider_config)


@pytest.fixture
def provider_registry():
    """Create fresh provider registry."""
    registry = ProviderRegistry()
    registry.register(
        ProviderType.OPENAI,
        MockProvider,
        frozenset([
            ProviderCapability.CHAT_COMPLETION,
            ProviderCapability.TEXT_COMPLETION,
            ProviderCapability.EMBEDDINGS,
        ])
    )
    return registry


@pytest.fixture
def provider_factory():
    """Create provider factory."""
    return ProviderFactory()


@pytest.fixture
def provider_router():
    """Create provider router."""
    return ProviderRouter(
        routing_strategy=RoutingStrategy.CAPABILITY_BASED,
        fallback_strategy=FallbackStrategy.NEXT_AVAILABLE,
    )


class TestProviderBase:
    """Test provider base functionality."""

    def test_provider_config_validation(self):
        """Test provider configuration validation."""
        # Valid config
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            api_key="test-key",
            base_url="https://api.test.com",
        )
        assert config.provider_type == ProviderType.OPENAI
        assert config.api_key == "test-key"

        # Invalid config - empty API key
        with pytest.raises(ValueError, match="API key cannot be empty"):
            ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key="",
                base_url="https://api.test.com",
            )

    def test_provider_metrics(self):
        """Test provider metrics functionality."""
        metrics = ProviderMetrics(provider_type=ProviderType.OPENAI)

        assert metrics.provider_type == ProviderType.OPENAI
        assert metrics.status_code is None
        assert not metrics.is_successful

        metrics.mark_complete(status_code=200)
        assert metrics.status_code == 200
        assert metrics.is_successful

        metrics.increment_retry()
        assert metrics.retry_count == 1

    def test_provider_response(self, mock_provider):
        """Test provider response functionality."""
        metrics = ProviderMetrics(provider_type=ProviderType.OPENAI)
        response = ProviderResponse(
            data={"choices": [{"message": {"content": "test"}}]},
            status_code=200,
            headers={},
            metrics=metrics,
            provider_type=ProviderType.OPENAI,
        )

        assert response.is_success
        assert response.content == "test"


class TestProviderRegistry:
    """Test provider registry functionality."""

    def test_provider_registration(self, provider_registry):
        """Test provider registration."""
        assert provider_registry.is_registered(ProviderType.OPENAI)
        assert not provider_registry.is_registered(ProviderType.ANTHROPIC)

        providers = provider_registry.get_registered_providers()
        assert ProviderType.OPENAI in providers

    def test_capability_queries(self, provider_registry):
        """Test capability-based queries."""
        chat_providers = provider_registry.get_providers_with_capability(
            ProviderCapability.CHAT_COMPLETION
        )
        assert ProviderType.OPENAI in chat_providers

        capabilities = provider_registry.get_capabilities(ProviderType.OPENAI)
        assert ProviderCapability.CHAT_COMPLETION in capabilities


class TestProviderFactory:
    """Test provider factory functionality."""

    def test_provider_creation(self, provider_factory, mock_provider_config, provider_registry):
        """Test provider creation."""
        with patch('src.providers.factory.get_registry', return_value=provider_registry):
            provider = provider_factory.create_provider(
                ProviderType.OPENAI,
                mock_provider_config,
            )

            assert provider is not None
            assert provider.provider_type == ProviderType.OPENAI

    def test_provider_caching(self, provider_factory, mock_provider_config, provider_registry):
        """Test provider instance caching."""
        with patch('src.providers.factory.get_registry', return_value=provider_registry):
            provider1 = provider_factory.create_provider(
                ProviderType.OPENAI,
                mock_provider_config,
                instance_id="test-1",
            )

            provider2 = provider_factory.get_provider(
                ProviderType.OPENAI,
                instance_id="test-1",
            )

            assert provider1 is provider2


class TestRequestTransformers:
    """Test request transformation functionality."""

    def test_ollama_to_openai_chat_transform(self):
        """Test Ollama to OpenAI chat transformation."""
        ollama_request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "options": {"temperature": 0.7, "max_tokens": 100},
        }

        openai_request = RequestTransformer.transform_chat_request(
            ollama_request, ProviderType.OPENAI
        )

        assert openai_request["model"] == "gpt-3.5-turbo"
        assert openai_request["messages"] == [{"role": "user", "content": "Hello"}]
        assert openai_request["temperature"] == 0.7
        assert openai_request["max_tokens"] == 100

    def test_ollama_to_anthropic_chat_transform(self):
        """Test Ollama to Anthropic chat transformation."""
        ollama_request = {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"}
            ],
            "options": {"temperature": 0.7},
        }

        anthropic_request = RequestTransformer.transform_chat_request(
            ollama_request, ProviderType.ANTHROPIC
        )

        assert anthropic_request["model"] == "claude-3-sonnet"
        assert anthropic_request["system"] == "You are helpful"
        assert len(anthropic_request["messages"]) == 1
        assert anthropic_request["messages"][0]["role"] == "user"
        assert anthropic_request["temperature"] == 0.7


class TestResponseTransformers:
    """Test response transformation functionality."""

    def test_openai_to_ollama_chat_transform(self):
        """Test OpenAI to Ollama chat transformation."""
        openai_response = {
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello there!"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        ollama_response = ResponseTransformer.transform_chat_response(
            openai_response, ProviderType.OPENAI
        )

        assert ollama_response["model"] == "gpt-3.5-turbo"
        assert ollama_response["message"]["content"] == "Hello there!"
        assert ollama_response["done"] is True
        assert ollama_response["prompt_eval_count"] == 10
        assert ollama_response["eval_count"] == 5


@pytest.mark.asyncio
class TestProviderRouter:
    """Test provider routing functionality."""

    async def test_capability_based_routing(self, provider_router):
        """Test capability-based routing."""
        with patch('src.providers.init_providers.find_providers_with_capability') as mock_find:
            mock_find.return_value = [ProviderType.OPENAI]

            with patch('src.providers.factory.get_factory') as mock_get_factory:
                mock_factory = MagicMock()
                mock_provider = MagicMock()
                mock_factory.get_provider.return_value = mock_provider
                mock_get_factory.return_value = mock_factory

                with patch.object(provider_router, '_get_provider_instance') as mock_get:
                    mock_get.return_value = mock_provider

                    result = await provider_router.route_request(
                        ProviderCapability.CHAT_COMPLETION
                    )

                    assert result == mock_provider
                    mock_find.assert_called_once_with(ProviderCapability.CHAT_COMPLETION)


@pytest.mark.asyncio
class TestMultiProviderIntegration:
    """Test complete multi-provider integration."""

    async def test_multi_provider_settings(self):
        """Test multi-provider settings."""
        settings = MultiProviderSettings(
            openai_enabled=True,
            openai_api_key="test-key",
            anthropic_enabled=True,
            anthropic_api_key="test-key-2",
            routing_strategy=RoutingStrategy.ROUND_ROBIN,
            fallback_strategy=FallbackStrategy.NEXT_AVAILABLE,
        )

        provider_settings = settings.get_provider_settings()

        assert ProviderType.OPENAI in provider_settings
        assert ProviderType.ANTHROPIC in provider_settings
        assert provider_settings[ProviderType.OPENAI].api_key == "test-key"
        assert provider_settings[ProviderType.ANTHROPIC].api_key == "test-key-2"

    async def test_health_check_integration(self, mock_provider):
        """Test health check integration."""
        from src.health import HealthChecker

        health_checker = HealthChecker()

        with patch('src.health.get_factory') as mock_factory:
            mock_factory_instance = MagicMock()
            mock_factory_instance.get_provider.return_value = mock_provider
            mock_factory.return_value = mock_factory_instance

            health_status = await health_checker.check_provider_health(ProviderType.OPENAI)

            assert health_status["healthy"] is True
            assert health_status["status"] == "operational"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])