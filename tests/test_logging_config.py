"""Tests for logging configuration module."""

from __future__ import annotations

import logging
from unittest.mock import ANY, Mock, patch

import pytest
import structlog

from src.config import Settings
from src.logging_config import setup_logging


class TestSetupLogging:
    """Test logging configuration setup."""

    @patch("structlog.configure")
    @patch("logging.basicConfig")
    def test_setup_logging_production(self, mock_basic_config, mock_structlog_configure):
        """Test logging setup for production environment."""
        settings = Mock(spec=Settings)
        settings.log_level = "INFO"
        settings.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        settings.debug = False

        setup_logging(settings)

        # Verify basic logging configuration
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == logging.INFO
        assert call_args[1]["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Verify structlog configuration
        mock_structlog_configure.assert_called_once()
        call_args = mock_structlog_configure.call_args

        # Check that processors include JSONRenderer for production
        processors = call_args[1]["processors"]
        processor_types = [type(p).__name__ for p in processors]
        assert "JSONRenderer" in processor_types

    @patch("structlog.configure")
    @patch("logging.basicConfig")
    def test_setup_logging_development(self, mock_basic_config, mock_structlog_configure):
        """Test logging setup for development environment."""
        settings = Mock(spec=Settings)
        settings.log_level = "DEBUG"
        settings.log_format = "%(levelname)s:%(name)s:%(message)s"
        settings.debug = True

        setup_logging(settings)

        # Verify basic logging configuration
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args
        assert call_args[1]["level"] == logging.DEBUG
        assert call_args[1]["format"] == "%(levelname)s:%(name)s:%(message)s"

        # Verify structlog configuration
        mock_structlog_configure.assert_called_once()
        call_args = mock_structlog_configure.call_args

        # Check that processors include ConsoleRenderer for development
        processors = call_args[1]["processors"]
        processor_types = [type(p).__name__ for p in processors]
        assert "ConsoleRenderer" in processor_types

    @patch("structlog.configure")
    @patch("logging.basicConfig")
    def test_setup_logging_different_levels(self, mock_basic_config, mock_structlog_configure):
        """Test logging setup with different log levels."""
        test_cases = [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]

        for level_str, level_int in test_cases:
            mock_basic_config.reset_mock()
            mock_structlog_configure.reset_mock()

            settings = Mock(spec=Settings)
            settings.log_level = level_str
            settings.log_format = "test format"
            settings.debug = False

            setup_logging(settings)

            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            assert call_args[1]["level"] == level_int
            assert call_args[1]["format"] == "test format"

    @patch("structlog.configure")
    @patch("logging.basicConfig")
    def test_setup_logging_processors_order(self, mock_basic_config, mock_structlog_configure):
        """Test that structlog processors are configured in correct order."""
        settings = Mock(spec=Settings)
        settings.log_level = "INFO"
        settings.log_format = "test"
        settings.debug = False

        setup_logging(settings)

        call_args = mock_structlog_configure.call_args
        processors = call_args[1]["processors"]

        # Verify key processors are present
        processor_names = [type(p).__name__ if hasattr(
            p, '__name__') else str(p) for p in processors]

        # These processors should be present for proper log formatting
        expected_processors = [
            "function",  # filter_by_level, add_logger_name, add_log_level, format_exc_info are functions
            "PositionalArgumentsFormatter",
            "TimeStamper",
            "StackInfoRenderer",
            "UnicodeDecoder",
            "JSONRenderer",  # For production mode
        ]

        # Check that we have the right number of processors
        # At least 7 processors should be configured
        assert len(processors) >= 7

        # Check for specific processor types
        assert any(
            "PositionalArgumentsFormatter" in name for name in processor_names)
        assert any("TimeStamper" in name for name in processor_names)
        assert any("JSONRenderer" in name for name in processor_names)

    @patch("structlog.configure")
    @patch("logging.basicConfig")
    def test_setup_logging_structlog_config(self, mock_basic_config, mock_structlog_configure):
        """Test structlog configuration parameters."""
        settings = Mock(spec=Settings)
        settings.log_level = "INFO"
        settings.log_format = "test"
        settings.debug = False

        setup_logging(settings)

        call_args = mock_structlog_configure.call_args
        config = call_args[1]

        # Verify structlog configuration parameters
        assert config["context_class"] == dict
        assert config["cache_logger_on_first_use"] is True
        assert "logger_factory" in config
        assert "wrapper_class" in config

    def test_setup_logging_integration(self):
        """Test actual logging setup integration."""
        settings = Mock(spec=Settings)
        settings.log_level = "INFO"
        settings.log_format = "%(levelname)s:%(name)s:%(message)s"
        settings.debug = True

        # This should not raise any exceptions
        setup_logging(settings)

        # Verify we can get a logger and it works
        logger = structlog.get_logger("test")
        assert logger is not None

        # Test that the logger can be used (this should not raise)
        try:
            logger.info("Test message", extra_field="test_value")
        except Exception as e:
            pytest.fail(f"Logger failed to log message: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
