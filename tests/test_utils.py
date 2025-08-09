"""Tests for utility functions."""

from __future__ import annotations

import pytest

from src.utils import (
    ModelMapping,
    build_ollama_to_openrouter_map,
    build_openrouter_to_ollama_map,
    filter_models,
    openrouter_id_to_ollama_name,
    resolve_model_name,
)


class TestOpenRouterIdToOllamaName:
    """Test openrouter_id_to_ollama_name function."""

    def test_simple_model_id(self):
        """Test simple model ID conversion."""
        result = openrouter_id_to_ollama_name("gpt-4")
        assert result == "gpt-4:latest"

    def test_vendor_prefixed_model_id(self):
        """Test vendor-prefixed model ID conversion."""
        result = openrouter_id_to_ollama_name("openai/gpt-4")
        assert result == "gpt-4:latest"

    def test_complex_model_id(self):
        """Test complex model ID with special characters."""
        result = openrouter_id_to_ollama_name("anthropic/claude-3.5-sonnet")
        assert result == "claude-3.5-sonnet:latest"

    def test_model_id_with_multiple_slashes(self):
        """Test model ID with multiple slashes."""
        result = openrouter_id_to_ollama_name("vendor/sub/model-name")
        assert result == "model-name:latest"


class TestModelMapping:
    """Test ModelMapping dataclass."""

    def test_from_openrouter_model(self):
        """Test creating ModelMapping from OpenRouter model data."""
        model_data = {"id": "openai/gpt-4",
                      "name": "GPT-4", "context_length": 8192}

        mapping = ModelMapping.from_openrouter_model(model_data)
        assert mapping.ollama_name == "gpt-4:latest"
        assert mapping.openrouter_id == "openai/gpt-4"
        assert mapping.display_name == "GPT-4"
        assert mapping.context_length == 8192


class TestBuildMappings:
    """Test mapping building functions."""

    @pytest.fixture
    def sample_models(self):
        """Sample model data for testing."""
        return [
            {"id": "openai/gpt-4", "name": "GPT-4"},
            {"id": "anthropic/claude-3", "name": "Claude 3"},
            {"id": "meta-llama/llama-2-7b", "name": "Llama 2 7B"},
        ]

    def test_build_ollama_to_openrouter_map(self, sample_models):
        """Test building Ollama to OpenRouter mapping."""
        mapping = build_ollama_to_openrouter_map(sample_models)

        assert len(mapping) == 3
        assert mapping["gpt-4:latest"] == "openai/gpt-4"
        assert mapping["claude-3:latest"] == "anthropic/claude-3"
        assert mapping["llama-2-7b:latest"] == "meta-llama/llama-2-7b"

    def test_build_openrouter_to_ollama_map(self, sample_models):
        """Test building OpenRouter to Ollama mapping."""
        mapping = build_openrouter_to_ollama_map(sample_models)

        assert len(mapping) == 3
        assert mapping["openai/gpt-4"] == "gpt-4:latest"
        assert mapping["anthropic/claude-3"] == "claude-3:latest"
        assert mapping["meta-llama/llama-2-7b"] == "llama-2-7b:latest"


class TestFilterModels:
    """Test model filtering function."""

    @pytest.fixture
    def sample_models(self):
        """Sample model data for testing."""
        return [
            {"id": "openai/gpt-4", "name": "GPT-4"},
            {"id": "anthropic/claude-3", "name": "Claude 3"},
            {"id": "meta-llama/llama-2-7b", "name": "Llama 2 7B"},
        ]

    def test_filter_models_empty_filter(self, sample_models):
        """Test filtering with empty filter set."""
        result = filter_models(sample_models, set())
        assert len(result) == 3
        assert result == sample_models

    def test_filter_models_with_filter(self, sample_models):
        """Test filtering with specific models."""
        filter_set = {"gpt-4:latest", "claude-3:latest"}
        result = filter_models(sample_models, filter_set)

        assert len(result) == 2
        assert any(m["id"] == "openai/gpt-4" for m in result)
        assert any(m["id"] == "anthropic/claude-3" for m in result)
        assert not any(m["id"] == "meta-llama/llama-2-7b" for m in result)

    def test_filter_models_no_matches(self, sample_models):
        """Test filtering with no matching models."""
        filter_set = {"nonexistent:latest"}
        result = filter_models(sample_models, filter_set)
        assert len(result) == 0


class TestResolveModelName:
    """Test model name resolution function."""

    @pytest.fixture
    def sample_mapping(self):
        """Sample Ollama to OpenRouter mapping."""
        return {
            "gpt-4:latest": "openai/gpt-4",
            "claude-3:latest": "anthropic/claude-3",
            "llama-2-7b:latest": "meta-llama/llama-2-7b",
            "llama-2-13b:latest": "meta-llama/llama-2-13b",
        }

    def test_resolve_exact_match(self, sample_mapping):
        """Test resolving exact model name match."""
        ollama_name, openrouter_id = resolve_model_name(
            "gpt-4:latest", sample_mapping)
        assert ollama_name == "gpt-4:latest"
        assert openrouter_id == "openai/gpt-4"

    def test_resolve_prefix_match(self, sample_mapping):
        """Test resolving model name by prefix."""
        ollama_name, openrouter_id = resolve_model_name(
            "gpt-4", sample_mapping)
        assert ollama_name == "gpt-4:latest"
        assert openrouter_id == "openai/gpt-4"

    def test_resolve_prefix_match_multiple(self, sample_mapping):
        """Test resolving model name with multiple prefix matches."""
        # The function only matches exact prefixes up to the colon, so "llama-2" won't match "llama-2-7b:latest"
        # This should return None since there's no exact "llama-2:*" match
        ollama_name, openrouter_id = resolve_model_name(
            "llama-2", sample_mapping)
        assert ollama_name is None
        assert openrouter_id is None

    def test_resolve_prefix_match_exact_prefix(self, sample_mapping):
        """Test resolving model name with exact prefix match."""
        # This should work since we're looking for "llama-2-7b" and we have "llama-2-7b:latest"
        ollama_name, openrouter_id = resolve_model_name(
            "llama-2-7b", sample_mapping)
        assert ollama_name == "llama-2-7b:latest"
        assert openrouter_id == "meta-llama/llama-2-7b"

    def test_resolve_no_match(self, sample_mapping):
        """Test resolving non-existent model name."""
        ollama_name, openrouter_id = resolve_model_name(
            "nonexistent", sample_mapping)
        assert ollama_name is None
        assert openrouter_id is None

    def test_resolve_none_input(self, sample_mapping):
        """Test resolving None input."""
        ollama_name, openrouter_id = resolve_model_name(None, sample_mapping)
        assert ollama_name is None
        assert openrouter_id is None

    def test_resolve_with_version_no_match(self, sample_mapping):
        """Test resolving model name with version that doesn't match."""
        ollama_name, openrouter_id = resolve_model_name(
            "gpt-4:v2", sample_mapping)
        assert ollama_name is None
        assert openrouter_id is None


if __name__ == "__main__":
    pytest.main([__file__])
