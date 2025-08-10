"""Tests for new AI providers (Azure, AWS Bedrock, Ollama).

This module contains comprehensive tests for the newly implemented
AI providers including Azure OpenAI, AWS Bedrock, and local Ollama.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.providers.base import (
    ProviderCapability,
    ProviderConfig,
    ProviderType,
    ProviderResponse,
    ProviderMetrics,
)
from src.providers.azure_provider import AzureProvider
from src.providers.aws_bedrock_provider import AWSBedrockProvider
from src.providers.ollama_provider import OllamaProvider


class TestAzureProvider:
    """Test Azure OpenAI provider implementation."""

    @pytest.fixture
    def azure_config(self):
        """Create Azure provider configuration."""
        return ProviderConfig(
            provider_type=ProviderType.AZURE,
            api_key="test-azure-key",
            base_url="https://test-resource.openai.azure.com",
            timeout=300,
            max_retries=3,
            max_concurrent_requests=10,
        )

    @pytest.fixture
    def azure_provider(self, azure_config):
        """Create Azure provider instance."""
        return AzureProvider(azure_config)

    def test_azure_provider_initialization(self, azure_provider):
        """Test Azure provider initialization."""
        assert azure_provider.provider_type == ProviderType.AZURE
        assert ProviderCapability.CHAT_COMPLETION in azure_provider.capabilities
        assert ProviderCapability.EMBEDDINGS in azure_provider.capabilities

    def test_azure_headers(self, azure_provider):
        """Test Azure-specific headers."""
        headers = azure_provider.get_headers()
        assert "api-key" in headers
        assert headers["api-key"] == "test-azure-key"
        assert "Content-Type" in headers

    def test_azure_url_building(self, azure_provider):
        """Test Azure URL building."""
        url = azure_provider._build_url("chat/completions", "gpt-4-deployment")
        expected = "https://test-resource.openai.azure.com/openai/deployments/gpt-4-deployment/chat/completions?api-version=2024-02-01"
        assert url == expected

    def test_deployment_extraction(self, azure_provider):
        """Test deployment name extraction from model."""
        # Simple deployment name
        assert azure_provider._extract_deployment_from_model("gpt-4") == "gpt-4"
        
        # Deployment with resource
        assert azure_provider._extract_deployment_from_model("gpt-4@test-resource") == "gpt-4"
        
        # Model mapping - create new config with mapping
        from dataclasses import replace
        new_config = replace(azure_provider.config, model_mapping={"custom-model": "actual-deployment"})
        azure_provider.config = new_config
        assert azure_provider._extract_deployment_from_model("custom-model") == "actual-deployment"

    @pytest.mark.asyncio
    async def test_azure_list_models(self, azure_provider):
        """Test Azure list models."""
        response = await azure_provider.list_models()
        
        assert isinstance(response, ProviderResponse)
        assert response.status_code == 200
        assert "data" in response.data
        assert len(response.data["data"]) > 0

    @pytest.mark.asyncio
    async def test_azure_chat_completion(self, azure_provider):
        """Test Azure chat completion."""
        with patch.object(azure_provider, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "Test response"}}]
            }
            mock_response.headers = {}
            mock_request.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            response = await azure_provider.chat_completion(messages, "gpt-4")
            
            assert isinstance(response, ProviderResponse)
            assert response.status_code == 200
            mock_request.assert_called_once()


class TestAWSBedrockProvider:
    """Test AWS Bedrock provider implementation."""

    @pytest.fixture
    def bedrock_config(self):
        """Create AWS Bedrock provider configuration."""
        return ProviderConfig(
            provider_type=ProviderType.AWS_BEDROCK,
            api_key="test-access-key",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
            timeout=300,
            max_retries=3,
            max_concurrent_requests=10,
            custom_headers={
                "aws_secret_access_key": "test-secret-key",
                "aws_region": "us-east-1",
            }
        )

    @pytest.fixture
    def bedrock_provider(self, bedrock_config):
        """Create AWS Bedrock provider instance."""
        return AWSBedrockProvider(bedrock_config)

    def test_bedrock_provider_initialization(self, bedrock_provider):
        """Test AWS Bedrock provider initialization."""
        assert bedrock_provider.provider_type == ProviderType.AWS_BEDROCK
        assert ProviderCapability.CHAT_COMPLETION in bedrock_provider.capabilities
        assert ProviderCapability.TEXT_COMPLETION in bedrock_provider.capabilities

    def test_bedrock_model_id_mapping(self, bedrock_provider):
        """Test Bedrock model ID mapping."""
        # Test common model mappings
        assert bedrock_provider._get_model_id("claude-3-sonnet") == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert bedrock_provider._get_model_id("titan-text") == "amazon.titan-text-express-v1"
        
        # Test custom mapping - create new config with mapping
        from dataclasses import replace
        new_config = replace(bedrock_provider.config, model_mapping={"custom-model": "custom.model.id"})
        bedrock_provider.config = new_config
        assert bedrock_provider._get_model_id("custom-model") == "custom.model.id"

    def test_bedrock_request_preparation(self, bedrock_provider):
        """Test Bedrock request preparation for different models."""
        # Test Claude request
        claude_request = bedrock_provider._prepare_claude_request(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100
        )
        assert "prompt" in claude_request
        assert "max_tokens_to_sample" in claude_request
        assert "Human: Hello" in claude_request["prompt"]

        # Test Titan request
        titan_request = bedrock_provider._prepare_titan_request(
            prompt="Hello world",
            max_tokens=100
        )
        assert "inputText" in titan_request
        assert "textGenerationConfig" in titan_request

    def test_bedrock_response_parsing(self, bedrock_provider):
        """Test Bedrock response parsing."""
        # Test Claude response parsing
        claude_response = {"completion": "Hello! How can I help you?"}
        parsed = bedrock_provider._parse_claude_response(claude_response)
        assert "choices" in parsed
        assert parsed["choices"][0]["message"]["content"] == "Hello! How can I help you?"

        # Test Titan response parsing
        titan_response = {"results": [{"outputText": "Hello! How can I help you?"}]}
        parsed = bedrock_provider._parse_titan_response(titan_response)
        assert "choices" in parsed
        assert parsed["choices"][0]["message"]["content"] == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_bedrock_list_models(self, bedrock_provider):
        """Test AWS Bedrock list models."""
        response = await bedrock_provider.list_models()
        
        assert isinstance(response, ProviderResponse)
        assert response.status_code == 200
        assert "data" in response.data
        assert len(response.data["data"]) > 0


class TestOllamaProvider:
    """Test local Ollama provider implementation."""

    @pytest.fixture
    def ollama_config(self):
        """Create Ollama provider configuration."""
        return ProviderConfig(
            provider_type=ProviderType.OLLAMA,
            api_key="",  # Ollama typically doesn't require API key
            base_url="http://localhost:11434",
            timeout=300,
            max_retries=3,
            max_concurrent_requests=10,
        )

    @pytest.fixture
    def ollama_provider(self, ollama_config):
        """Create Ollama provider instance."""
        return OllamaProvider(ollama_config)

    def test_ollama_provider_initialization(self, ollama_provider):
        """Test Ollama provider initialization."""
        assert ollama_provider.provider_type == ProviderType.OLLAMA
        assert ProviderCapability.CHAT_COMPLETION in ollama_provider.capabilities
        assert ProviderCapability.TEXT_COMPLETION in ollama_provider.capabilities
        assert ProviderCapability.EMBEDDINGS in ollama_provider.capabilities

    def test_ollama_headers(self, ollama_provider):
        """Test Ollama headers (no auth by default)."""
        headers = ollama_provider.get_headers()
        assert "Content-Type" in headers
        assert "Authorization" not in headers  # No API key by default

    def test_ollama_response_conversion(self, ollama_provider):
        """Test Ollama response conversion to OpenAI format."""
        # Test chat response conversion
        ollama_chat_response = {
            "message": {"content": "Hello! How can I help you?"},
            "prompt_eval_count": 10,
            "eval_count": 15,
        }
        converted = ollama_provider._convert_ollama_chat_to_openai(ollama_chat_response, "llama2")
        assert "choices" in converted
        assert converted["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        assert converted["usage"]["prompt_tokens"] == 10
        assert converted["usage"]["completion_tokens"] == 15

        # Test generate response conversion
        ollama_generate_response = {
            "response": "Hello! How can I help you?",
            "prompt_eval_count": 10,
            "eval_count": 15,
        }
        converted = ollama_provider._convert_ollama_generate_to_openai(ollama_generate_response, "llama2")
        assert "choices" in converted
        assert converted["choices"][0]["text"] == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_ollama_list_models(self, ollama_provider):
        """Test Ollama list models."""
        with patch.object(ollama_provider, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {
                        "name": "llama2:7b",
                        "modified_at": "2023-01-01T00:00:00Z",
                        "size": 3800000000,
                        "digest": "sha256:abc123",
                        "details": {"format": "gguf"}
                    }
                ]
            }
            mock_response.headers = {}
            mock_request.return_value = mock_response

            response = await ollama_provider.list_models()
            
            assert isinstance(response, ProviderResponse)
            assert response.status_code == 200
            assert "data" in response.data
            assert len(response.data["data"]) == 1
            assert response.data["data"][0]["id"] == "llama2:7b"

    @pytest.mark.asyncio
    async def test_ollama_chat_completion(self, ollama_provider):
        """Test Ollama chat completion."""
        with patch.object(ollama_provider, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {"content": "Hello! How can I help you?"},
                "prompt_eval_count": 10,
                "eval_count": 15,
            }
            mock_response.headers = {}
            mock_request.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            response = await ollama_provider.chat_completion(messages, "llama2")
            
            assert isinstance(response, ProviderResponse)
            assert response.status_code == 200
            assert "choices" in response.data
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_ollama_embeddings(self, ollama_provider):
        """Test Ollama embeddings."""
        with patch.object(ollama_provider, '_make_request') as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            mock_response.headers = {}
            mock_request.return_value = mock_response

            response = await ollama_provider.create_embeddings("Hello world", "nomic-embed-text")
            
            assert isinstance(response, ProviderResponse)
            assert response.status_code == 200
            assert "data" in response.data
            assert len(response.data["data"]) == 1
            assert response.data["data"][0]["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]
