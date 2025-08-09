"""Tests for enhanced configuration with pattern matching and modern features."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import Environment, LogLevel, ModelFilter, Settings


class TestLogLevel:
    """Test LogLevel enum with pattern matching features."""

    def test_from_string_basic(self):
        """Test basic string conversion."""
        assert LogLevel.from_string("DEBUG") == LogLevel.DEBUG
        assert LogLevel.from_string("INFO") == LogLevel.INFO
        assert LogLevel.from_string("WARNING") == LogLevel.WARNING
        assert LogLevel.from_string("ERROR") == LogLevel.ERROR
        assert LogLevel.from_string("CRITICAL") == LogLevel.CRITICAL

    def test_from_string_aliases(self):
        """Test alias support."""
        assert LogLevel.from_string("DBG") == LogLevel.DEBUG
        assert LogLevel.from_string("D") == LogLevel.DEBUG
        assert LogLevel.from_string("WARN") == LogLevel.WARNING
        assert LogLevel.from_string("W") == LogLevel.WARNING
        assert LogLevel.from_string("ERR") == LogLevel.ERROR
        assert LogLevel.from_string("E") == LogLevel.ERROR
        assert LogLevel.from_string("CRIT") == LogLevel.CRITICAL
        assert LogLevel.from_string("FATAL") == LogLevel.CRITICAL
        assert LogLevel.from_string("C") == LogLevel.CRITICAL
        assert LogLevel.from_string("F") == LogLevel.CRITICAL

    def test_from_string_case_insensitive(self):
        """Test case insensitive conversion."""
        assert LogLevel.from_string("debug") == LogLevel.DEBUG
        assert LogLevel.from_string("Info") == LogLevel.INFO
        assert LogLevel.from_string("warning") == LogLevel.WARNING

    def test_from_string_invalid(self):
        """Test invalid log level raises error."""
        with pytest.raises(ValueError, match="Invalid log level"):
            LogLevel.from_string("INVALID")

    def test_numeric_level(self):
        """Test numeric level property."""
        assert LogLevel.DEBUG.numeric_level == 10
        assert LogLevel.INFO.numeric_level == 20
        assert LogLevel.WARNING.numeric_level == 30
        assert LogLevel.ERROR.numeric_level == 40
        assert LogLevel.CRITICAL.numeric_level == 50

    def test_is_enabled_for(self):
        """Test level comparison."""
        assert LogLevel.DEBUG.is_enabled_for(LogLevel.DEBUG)
        assert LogLevel.INFO.is_enabled_for(LogLevel.DEBUG)
        assert LogLevel.WARNING.is_enabled_for(LogLevel.INFO)
        assert not LogLevel.DEBUG.is_enabled_for(LogLevel.INFO)
        assert not LogLevel.INFO.is_enabled_for(LogLevel.WARNING)


class TestEnvironment:
    """Test Environment enum with enhanced features."""

    def test_from_string_basic(self):
        """Test basic string conversion."""
        assert Environment.from_string("development") == Environment.DEVELOPMENT
        assert Environment.from_string("staging") == Environment.STAGING
        assert Environment.from_string("production") == Environment.PRODUCTION

    def test_from_string_aliases(self):
        """Test alias support."""
        assert Environment.from_string("dev") == Environment.DEVELOPMENT
        assert Environment.from_string("local") == Environment.DEVELOPMENT
        assert Environment.from_string("stage") == Environment.STAGING
        assert Environment.from_string("test") == Environment.STAGING
        assert Environment.from_string("testing") == Environment.STAGING
        assert Environment.from_string("prod") == Environment.PRODUCTION
        assert Environment.from_string("live") == Environment.PRODUCTION

    def test_from_string_unknown_defaults_to_development(self):
        """Test unknown environment defaults to development."""
        assert Environment.from_string("unknown") == Environment.DEVELOPMENT

    def test_environment_properties(self):
        """Test environment property methods."""
        dev = Environment.DEVELOPMENT
        staging = Environment.STAGING
        prod = Environment.PRODUCTION

        assert dev.is_development
        assert not dev.is_staging
        assert not dev.is_production

        assert not staging.is_development
        assert staging.is_staging
        assert not staging.is_production

        assert not prod.is_development
        assert not prod.is_staging
        assert prod.is_production

    def test_debug_enabled(self):
        """Test debug enabled property."""
        assert Environment.DEVELOPMENT.debug_enabled
        assert not Environment.STAGING.debug_enabled
        assert not Environment.PRODUCTION.debug_enabled

    def test_log_level_default(self):
        """Test default log level for environments."""
        assert Environment.DEVELOPMENT.log_level_default == LogLevel.DEBUG
        assert Environment.STAGING.log_level_default == LogLevel.INFO
        assert Environment.PRODUCTION.log_level_default == LogLevel.WARNING


class TestModelFilter:
    """Test ModelFilter with enhanced pattern matching."""

    def test_empty_filter(self):
        """Test empty filter creation."""
        filter_obj = ModelFilter.empty()
        assert len(filter_obj.models) == 0
        assert len(filter_obj.patterns) == 0
        assert len(filter_obj.exclude_patterns) == 0
        assert not filter_obj

    def test_from_file_basic(self):
        """Test loading basic filter from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("model1:latest\n")
            f.write("model2:latest\n")
            f.write("# comment\n")
            f.write("\n")  # empty line
            temp_path = f.name

        try:
            filter_obj = ModelFilter.from_file(temp_path)
            assert len(filter_obj.models) == 2
            assert "model1:latest" in filter_obj.models
            assert "model2:latest" in filter_obj.models
            assert filter_obj.path == Path(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_file_with_patterns(self):
        """Test loading filter with wildcard patterns."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("exact-model:latest\n")
            f.write("gpt-*\n")  # wildcard pattern
            f.write("!claude-*\n")  # exclusion pattern
            temp_path = f.name

        try:
            filter_obj = ModelFilter.from_file(temp_path)
            assert len(filter_obj.models) == 1
            assert "exact-model:latest" in filter_obj.models
            assert len(filter_obj.patterns) == 1
            assert "gpt-*" in filter_obj.patterns
            assert len(filter_obj.exclude_patterns) == 1
            assert "claude-*" in filter_obj.exclude_patterns
        finally:
            os.unlink(temp_path)

    def test_is_allowed_exact_match(self):
        """Test exact model matching."""
        filter_obj = ModelFilter(models=frozenset(["allowed-model:latest"]))
        assert filter_obj.is_allowed("allowed-model:latest")
        assert not filter_obj.is_allowed("forbidden-model:latest")

    def test_is_allowed_pattern_match(self):
        """Test pattern matching."""
        filter_obj = ModelFilter(patterns=frozenset(["gpt-*"]))
        assert filter_obj.is_allowed("gpt-4:latest")
        assert filter_obj.is_allowed("gpt-3.5-turbo:latest")
        assert not filter_obj.is_allowed("claude-3:latest")

    def test_is_allowed_exclusion_pattern(self):
        """Test exclusion pattern matching."""
        filter_obj = ModelFilter(
            patterns=frozenset(["*"]),  # Allow all
            exclude_patterns=frozenset(["claude-*"]),  # Except claude models
        )
        assert filter_obj.is_allowed("gpt-4:latest")
        assert not filter_obj.is_allowed("claude-3:latest")

    def test_is_allowed_empty_filter(self):
        """Test empty filter allows all."""
        filter_obj = ModelFilter.empty()
        assert filter_obj.is_allowed("any-model:latest")

    def test_get_statistics(self):
        """Test filter statistics."""
        filter_obj = ModelFilter(
            models=frozenset(["model1", "model2"]),
            patterns=frozenset(["gpt-*"]),
            exclude_patterns=frozenset(["claude-*"]),
        )
        stats = filter_obj.get_statistics()

        assert stats["exact_models"] == 2
        assert stats["patterns"] == 1
        assert stats["exclude_patterns"] == 1
        assert stats["total_rules"] == 4
        assert stats["has_filters"] is True


class TestSettings:
    """Test Settings with enhanced validation using environment variables."""

    def test_api_key_validation_basic(self, monkeypatch):
        """Test basic API key validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-api-key-12345")
        settings = Settings(OPENROUTER_API_KEY="valid-api-key-12345")
        assert settings.openrouter_api_key == "valid-api-key-12345"

    def test_api_key_validation_openai_style(self, monkeypatch):
        """Test OpenAI-style API key validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-1234567890abcdef1234567890abcdef")
        settings = Settings(OPENROUTER_API_KEY="sk-1234567890abcdef1234567890abcdef")
        assert settings.openrouter_api_key.startswith("sk-")

    def test_api_key_validation_openrouter_style(self, monkeypatch):
        """Test OpenRouter-style API key validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-1234567890abcdef")
        settings = Settings(OPENROUTER_API_KEY="or-1234567890abcdef")
        assert settings.openrouter_api_key.startswith("or-")

    def test_api_key_validation_too_short(self, monkeypatch):
        """Test API key too short validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "short")
        with pytest.raises(ValidationError, match="too short"):
            Settings(OPENROUTER_API_KEY="short")

    def test_api_key_validation_empty(self, monkeypatch):
        """Test empty API key validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            Settings(OPENROUTER_API_KEY="")

    def test_api_key_validation_invalid_characters(self, monkeypatch):
        """Test API key with invalid characters."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "invalid@key#with$special%chars")
        with pytest.raises(ValidationError, match="invalid characters"):
            Settings(OPENROUTER_API_KEY="invalid@key#with$special%chars")

    def test_host_validation_valid(self, monkeypatch):
        """Test valid host validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-test-key-12345")

        monkeypatch.setenv("HOST", "localhost")
        settings1 = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings1.host == "localhost"

        monkeypatch.setenv("HOST", "0.0.0.0")
        settings2 = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings2.host == "0.0.0.0"

        monkeypatch.setenv("HOST", "192.168.1.1")
        settings3 = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings3.host == "192.168.1.1"

    def test_host_validation_invalid_ipv4(self, monkeypatch):
        """Test invalid IPv4 address validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-test-key-12345")
        monkeypatch.setenv("HOST", "999.999.999.999")
        with pytest.raises(ValidationError, match="Invalid IPv4 address"):
            Settings(OPENROUTER_API_KEY="valid-test-key-12345")

    def test_port_validation_valid(self, monkeypatch):
        """Test valid port validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-test-key-12345")
        monkeypatch.setenv("PORT", "8080")
        settings = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings.port == 8080

    def test_port_validation_invalid_range(self, monkeypatch):
        """Test invalid port range validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-test-key-12345")

        monkeypatch.setenv("PORT", "0")
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Settings(OPENROUTER_API_KEY="valid-test-key-12345")

        monkeypatch.setenv("PORT", "70000")
        with pytest.raises(ValidationError, match="less than or equal to 65535"):
            Settings(OPENROUTER_API_KEY="valid-test-key-12345")

    def test_log_level_validation(self, monkeypatch):
        """Test log level validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-test-key-12345")

        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings.log_level == LogLevel.DEBUG

        monkeypatch.setenv("LOG_LEVEL", "INFO")
        settings2 = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings2.log_level == LogLevel.INFO

    def test_environment_validation(self, monkeypatch):
        """Test environment validation."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-test-key-12345")

        monkeypatch.setenv("ENVIRONMENT", "production")
        settings = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings.environment == Environment.PRODUCTION

        monkeypatch.setenv("ENVIRONMENT", "development")
        settings2 = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert settings2.environment == Environment.DEVELOPMENT

    def test_computed_fields(self, monkeypatch):
        """Test computed field properties."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-test-key-12345")

        monkeypatch.setenv("ENVIRONMENT", "development")
        dev_settings = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert not dev_settings.is_model_filtering_enabled
        assert dev_settings.environment.is_development
        assert not dev_settings.environment.is_production

        monkeypatch.setenv("ENVIRONMENT", "production")
        prod_settings = Settings(OPENROUTER_API_KEY="valid-test-key-12345")
        assert not prod_settings.environment.is_development
        assert prod_settings.environment.is_production


if __name__ == "__main__":
    pytest.main([__file__])
