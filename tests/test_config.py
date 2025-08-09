"""Tests for configuration management."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import ModelFilter, Settings, load_model_filter


class TestSettings:
    """Test Settings class."""

    def test_settings_with_required_fields(self):
        """Test settings creation with required fields."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890"}):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.openrouter_api_key == "test-api-key-1234567890"
            assert settings.host == "0.0.0.0"
            assert settings.port == 11434

    def test_settings_validation_empty_api_key(self):
        """Test validation fails with empty API key."""
        with pytest.raises(ValidationError):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
                Settings()  # type: ignore[call-arg]

    def test_settings_validation_invalid_port(self):
        """Test validation fails with invalid port."""
        with pytest.raises(ValidationError):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890", "PORT": "70000"}):
                Settings()  # type: ignore[call-arg]

    def test_settings_validation_invalid_log_level(self):
        """Test validation fails with invalid log level."""
        with pytest.raises(ValidationError):
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-api-key-1234567890", "LOG_LEVEL": "INVALID"}):
                Settings()  # type: ignore[call-arg]

    def test_settings_custom_values(self):
        """Test settings with custom values."""
        env_vars = {
            "OPENROUTER_API_KEY": "test-api-key-1234567890",
            "HOST": "127.0.0.1",
            "PORT": "8080",
            "LOG_LEVEL": "DEBUG",
            "DEBUG": "true",
        }
        with patch.dict(os.environ, env_vars):
            settings = Settings()  # type: ignore[call-arg]
            assert settings.host == "127.0.0.1"
            assert settings.port == 8080
            assert settings.log_level.value == "DEBUG"
            assert settings.debug is True


class TestModelFilter:
    """Test ModelFilter class."""

    def test_model_filter_empty(self):
        """Test empty model filter."""
        filter_obj = ModelFilter()
        assert len(filter_obj.models) == 0
        assert filter_obj.path is None

    def test_model_filter_load_from_file(self):
        """Test loading model filter from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("model1:latest\n")
            f.write("model2:latest\n")
            f.write("# comment\n")
            f.write("\n")  # empty line
            f.write("model3:latest\n")
            temp_path = f.name

        try:
            filter_obj = ModelFilter.from_file(temp_path)

            assert len(filter_obj.models) == 3
            assert "model1:latest" in filter_obj.models
            assert "model2:latest" in filter_obj.models
            assert "model3:latest" in filter_obj.models
            assert filter_obj.path == Path(temp_path)
        finally:
            os.unlink(temp_path)

    def test_model_filter_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError, match="Model filter file not found"):
            ModelFilter.from_file("/nonexistent/file.txt")


class TestLoadModelFilter:
    """Test load_model_filter function."""

    def test_load_model_filter_no_path(self):
        """Test loading model filter with no path."""
        env_vars = {"OPENROUTER_API_KEY": "test-api-key-1234567890", "MODELS_FILTER_PATH": ""}
        with patch.dict(os.environ, env_vars):
            settings = Settings()  # type: ignore[call-arg]
            filter_obj = load_model_filter(settings)
            assert len(filter_obj.models) == 0

    def test_load_model_filter_nonexistent_file(self):
        """Test loading model filter with nonexistent file."""
        env_vars = {
            "OPENROUTER_API_KEY": "test-api-key-1234567890",
            "MODELS_FILTER_PATH": "/nonexistent/file.txt"
        }
        with patch.dict(os.environ, env_vars):
            settings = Settings()  # type: ignore[call-arg]
            filter_obj = load_model_filter(settings)
            assert len(filter_obj.models) == 0

    def test_load_model_filter_existing_file(self):
        """Test loading model filter with existing file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test-model:latest\n")
            temp_path = f.name

        try:
            env_vars = {
                "OPENROUTER_API_KEY": "test-api-key-1234567890",
                "MODELS_FILTER_PATH": temp_path
            }
            with patch.dict(os.environ, env_vars):
                settings = Settings()  # type: ignore[call-arg]
                filter_obj = load_model_filter(settings)
                assert len(filter_obj.models) == 1
                assert "test-model:latest" in filter_obj.models
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
