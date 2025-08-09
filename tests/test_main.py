"""Tests for the main entry point module."""

from __future__ import annotations

import sys
from unittest.mock import Mock, patch

import pytest

from src.main import main, parse_args


class TestParseArgs:
    """Test command line argument parsing."""

    def test_parse_args_defaults(self):
        """Test parsing with no arguments."""
        with patch("sys.argv", ["ollama-proxy"]):
            args = parse_args()
            assert args.host is None
            assert args.port is None
            assert args.reload is False
            assert args.api_key is None
            assert args.models_filter is None
            assert args.log_level is None

    def test_parse_args_all_options(self):
        """Test parsing with all arguments provided."""
        test_args = [
            "ollama-proxy",
            "--host", "127.0.0.1",
            "--port", "8080",
            "--reload",
            "--api-key", "test-key-123",
            "--models-filter", "/path/to/filter.txt",
            "--log-level", "DEBUG"
        ]
        with patch("sys.argv", test_args):
            args = parse_args()
            assert args.host == "127.0.0.1"
            assert args.port == 8080
            assert args.reload is True
            assert args.api_key == "test-key-123"
            assert args.models_filter == "/path/to/filter.txt"
            assert args.log_level == "DEBUG"

    def test_parse_args_invalid_log_level(self):
        """Test parsing with invalid log level."""
        test_args = ["ollama-proxy", "--log-level", "INVALID"]
        with patch("sys.argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()


class TestMain:
    """Test the main function."""

    @patch("src.main.uvicorn.run")
    @patch("src.main.create_app")
    @patch("src.main.get_settings")
    @patch("src.main.parse_args")
    def test_main_success(self, mock_parse_args, mock_get_settings, mock_create_app, mock_uvicorn_run):
        """Test successful main execution."""
        # Setup mocks
        mock_args = Mock()
        mock_args.api_key = None
        mock_args.host = None
        mock_args.port = None
        mock_args.models_filter = None
        mock_args.log_level = None
        mock_args.reload = None
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = "test-key-123"
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 11434
        mock_settings.reload = False
        mock_settings.log_level = "INFO"
        mock_get_settings.return_value = mock_settings

        mock_app = Mock()
        mock_create_app.return_value = mock_app

        # Call main
        main()

        # Verify calls
        mock_parse_args.assert_called_once()
        mock_get_settings.assert_called_once()
        mock_create_app.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(
            mock_app,
            host="0.0.0.0",
            port=11434,
            reload=False,
            log_level="info",
            access_log=True,
        )

    @patch("src.main.get_settings")
    @patch("src.main.parse_args")
    def test_main_missing_api_key(self, mock_parse_args, mock_get_settings):
        """Test main with missing API key."""
        # Setup mocks
        mock_args = Mock()
        mock_args.api_key = None
        mock_args.host = None
        mock_args.port = None
        mock_args.models_filter = None
        mock_args.log_level = None
        mock_args.reload = None
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = ""  # Empty API key
        mock_get_settings.return_value = mock_settings

        # Call main and expect SystemExit
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1

    @patch("src.main.parse_args")
    def test_main_config_error(self, mock_parse_args):
        """Test main with configuration error."""
        mock_args = Mock()
        mock_parse_args.return_value = mock_args

        # Mock get_settings to raise an exception
        with patch("src.main.get_settings", side_effect=Exception("Config error")):
            with pytest.raises(SystemExit) as exc_info:
                main()
            
            assert exc_info.value.code == 1

    @patch("src.main.uvicorn.run")
    @patch("src.main.create_app")
    @patch("src.main.get_settings")
    @patch("src.main.parse_args")
    def test_main_with_overrides(self, mock_parse_args, mock_get_settings, mock_create_app, mock_uvicorn_run):
        """Test main with command line overrides."""
        # Setup mocks with command line overrides
        mock_args = Mock()
        mock_args.api_key = "override-key"
        mock_args.host = "127.0.0.1"
        mock_args.port = 8080
        mock_args.models_filter = "/custom/filter.txt"
        mock_args.log_level = "DEBUG"
        mock_args.reload = True
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = "original-key"
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 11434
        mock_settings.reload = False
        mock_settings.log_level = "INFO"
        mock_get_settings.return_value = mock_settings

        mock_app = Mock()
        mock_create_app.return_value = mock_app

        # Call main
        main()

        # Verify settings were overridden
        assert mock_settings.openrouter_api_key == "override-key"
        assert mock_settings.host == "127.0.0.1"
        assert mock_settings.port == 8080
        assert mock_settings.models_filter_path == "/custom/filter.txt"
        assert mock_settings.log_level == "DEBUG"
        assert mock_settings.reload is True

        # Verify uvicorn was called with overridden values
        mock_uvicorn_run.assert_called_once_with(
            mock_app,
            host="127.0.0.1",
            port=8080,
            reload=True,
            log_level="debug",
            access_log=True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
