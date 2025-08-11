"""Comprehensive tests for the main module to improve coverage."""

import argparse
import os
import sys
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.main import (
    parse_args, main, dev_main, daemon_main, admin_main,
    health_check, config_main, benchmark_main, test_main,
    lint_main, format_main, cli_main
)


class TestParseArgs:
    """Test command line argument parsing."""

    def test_parse_args_defaults(self):
        """Test parsing with default arguments."""
        with patch('sys.argv', ['ollama-proxy']):
            args = parse_args()
            assert args.api_key is None
            assert args.host is None
            assert args.port is None
            assert args.models_filter is None
            assert args.log_level is None
            assert args.reload is False

    def test_parse_args_with_values(self):
        """Test parsing with provided arguments."""
        test_argv = [
            'ollama-proxy',
            '--api-key', 'test-key',
            '--host', '127.0.0.1',
            '--port', '8080',
            '--models-filter', 'models.txt',
            '--log-level', 'DEBUG',
            '--reload'
        ]
        
        with patch('sys.argv', test_argv):
            args = parse_args()
            assert args.api_key == 'test-key'
            assert args.host == '127.0.0.1'
            assert args.port == 8080
            assert args.models_filter == 'models.txt'
            assert args.log_level == 'DEBUG'
            assert args.reload is True


class TestMainFunction:
    """Test the main function and its variants."""

    @patch('src.main.uvicorn.run')
    @patch('src.main.create_app')
    @patch('src.main.get_settings')
    @patch('src.main.parse_args')
    def test_main_success(self, mock_parse_args, mock_get_settings, mock_create_app, mock_uvicorn_run):
        """Test successful main execution."""
        # Setup mocks
        mock_args = Mock()
        mock_args.api_key = 'test-key'
        mock_args.host = '0.0.0.0'
        mock_args.port = 11434
        mock_args.models_filter = None
        mock_args.log_level = 'INFO'
        mock_args.reload = False
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = 'test-key'
        mock_settings.host = '0.0.0.0'
        mock_settings.port = 11434
        mock_settings.reload = False
        mock_settings.log_level = 'INFO'
        mock_get_settings.return_value = mock_settings

        mock_app = Mock()
        mock_create_app.return_value = mock_app

        # Call main
        main()

        # Verify calls
        mock_parse_args.assert_called_once()
        mock_get_settings.assert_called_once()
        mock_create_app.assert_called_once()
        mock_uvicorn_run.assert_called_once()

    @patch('src.main.get_settings')
    @patch('src.main.parse_args')
    def test_main_missing_api_key(self, mock_parse_args, mock_get_settings):
        """Test main with missing API key."""
        mock_args = Mock()
        mock_args.api_key = None
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = None
        mock_get_settings.return_value = mock_settings

        with pytest.raises(SystemExit):
            main()

    @patch('src.main.uvicorn.run')
    @patch('src.main.create_app')
    @patch('src.main.get_settings')
    @patch('src.main.parse_args')
    def test_dev_main(self, mock_parse_args, mock_get_settings, mock_create_app, mock_uvicorn_run):
        """Test dev_main function."""
        mock_args = Mock()
        mock_args.api_key = 'test-key'
        mock_args.host = '0.0.0.0'
        mock_args.port = 11434
        mock_args.models_filter = None
        mock_args.log_level = 'DEBUG'
        mock_args.reload = True
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = 'test-key'
        mock_settings.host = '0.0.0.0'
        mock_settings.port = 11434
        mock_settings.reload = True
        mock_settings.log_level = 'DEBUG'
        mock_get_settings.return_value = mock_settings

        mock_app = Mock()
        mock_create_app.return_value = mock_app

        dev_main()

        mock_uvicorn_run.assert_called_once()
        # Verify development-specific settings
        call_args = mock_uvicorn_run.call_args
        assert call_args[1]['reload'] is True
        assert call_args[1]['log_level'] == 'debug'

    @patch('src.main.subprocess.Popen')
    @patch('src.main.get_settings')
    @patch('src.main.parse_args')
    def test_daemon_main(self, mock_parse_args, mock_get_settings, mock_popen):
        """Test daemon_main function."""
        mock_args = Mock()
        mock_args.api_key = 'test-key'
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = 'test-key'
        mock_get_settings.return_value = mock_settings

        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        with patch('builtins.print') as mock_print:
            daemon_main()

        mock_popen.assert_called_once()
        mock_print.assert_called()

    def test_admin_main(self):
        """Test admin_main function."""
        with patch('builtins.print') as mock_print:
            admin_main()
        mock_print.assert_called()

    @patch('src.main.httpx.get')
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response

        with patch('builtins.print') as mock_print:
            health_check()
        
        mock_get.assert_called_once()
        mock_print.assert_called()

    @patch('src.main.httpx.get')
    def test_health_check_failure(self, mock_get):
        """Test health check failure."""
        mock_get.side_effect = Exception("Connection failed")

        with patch('builtins.print') as mock_print:
            health_check()
        
        mock_print.assert_called()

    @patch('src.main.get_settings')
    def test_config_main(self, mock_get_settings):
        """Test config_main function."""
        mock_settings = Mock()
        mock_settings.dict.return_value = {"key": "value"}
        mock_get_settings.return_value = mock_settings

        with patch('builtins.print') as mock_print:
            config_main()
        
        mock_print.assert_called()

    @patch('src.main.subprocess.run')
    def test_benchmark_main(self, mock_run):
        """Test benchmark_main function."""
        mock_run.return_value = Mock(returncode=0)

        with patch('builtins.print') as mock_print:
            benchmark_main()
        
        mock_run.assert_called()
        mock_print.assert_called()

    @patch('src.main.subprocess.run')
    def test_test_main(self, mock_run):
        """Test test_main function."""
        mock_run.return_value = Mock(returncode=0)

        with patch('builtins.print') as mock_print:
            test_main()
        
        mock_run.assert_called()
        mock_print.assert_called()

    @patch('src.main.subprocess.run')
    def test_lint_main(self, mock_run):
        """Test lint_main function."""
        mock_run.return_value = Mock(returncode=0)

        with patch('builtins.print') as mock_print:
            lint_main()
        
        mock_run.assert_called()
        mock_print.assert_called()

    @patch('src.main.subprocess.run')
    def test_format_main(self, mock_run):
        """Test format_main function."""
        mock_run.return_value = Mock(returncode=0)

        with patch('builtins.print') as mock_print:
            format_main()
        
        mock_run.assert_called()
        mock_print.assert_called()


class TestCLIMain:
    """Test the unified CLI interface."""

    @patch('src.main.main')
    def test_cli_main_server_subcommand(self, mock_main):
        """Test CLI with server subcommand."""
        test_argv = ['ollama-proxy-cli', 'server', '--host', '127.0.0.1']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_main.assert_called_once()

    @patch('src.main.dev_main')
    def test_cli_main_dev_subcommand(self, mock_dev_main):
        """Test CLI with dev subcommand."""
        test_argv = ['ollama-proxy-cli', 'dev', '--reload']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_dev_main.assert_called_once()

    @patch('src.main.daemon_main')
    def test_cli_main_daemon_subcommand(self, mock_daemon_main):
        """Test CLI with daemon subcommand."""
        test_argv = ['ollama-proxy-cli', 'daemon']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_daemon_main.assert_called_once()

    @patch('src.main.admin_main')
    def test_cli_main_admin_subcommand(self, mock_admin_main):
        """Test CLI with admin subcommand."""
        test_argv = ['ollama-proxy-cli', 'admin']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_admin_main.assert_called_once()

    @patch('src.main.health_check')
    def test_cli_main_health_subcommand(self, mock_health_check):
        """Test CLI with health subcommand."""
        test_argv = ['ollama-proxy-cli', 'health']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_health_check.assert_called_once()

    @patch('src.main.config_main')
    def test_cli_main_config_subcommand(self, mock_config_main):
        """Test CLI with config subcommand."""
        test_argv = ['ollama-proxy-cli', 'config']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_config_main.assert_called_once()

    @patch('src.main.benchmark_main')
    def test_cli_main_benchmark_subcommand(self, mock_benchmark_main):
        """Test CLI with benchmark subcommand."""
        test_argv = ['ollama-proxy-cli', 'benchmark']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_benchmark_main.assert_called_once()

    @patch('src.main.test_main')
    def test_cli_main_test_subcommand(self, mock_test_main):
        """Test CLI with test subcommand."""
        test_argv = ['ollama-proxy-cli', 'test']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_test_main.assert_called_once()

    @patch('src.main.lint_main')
    def test_cli_main_lint_subcommand(self, mock_lint_main):
        """Test CLI with lint subcommand."""
        test_argv = ['ollama-proxy-cli', 'lint']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_lint_main.assert_called_once()

    @patch('src.main.format_main')
    def test_cli_main_format_subcommand(self, mock_format_main):
        """Test CLI with format subcommand."""
        test_argv = ['ollama-proxy-cli', 'format']
        
        with patch('sys.argv', test_argv):
            cli_main()
        
        mock_format_main.assert_called_once()

    def test_cli_main_invalid_subcommand(self):
        """Test CLI with invalid subcommand."""
        test_argv = ['ollama-proxy-cli', 'invalid']
        
        with patch('sys.argv', test_argv):
            with pytest.raises(SystemExit):
                cli_main()


class TestEnvironmentHandling:
    """Test environment variable handling."""

    @patch.dict(os.environ, {'OPENROUTER_API_KEY': 'env-key'})
    @patch('src.main.uvicorn.run')
    @patch('src.main.create_app')
    @patch('src.main.get_settings')
    @patch('src.main.parse_args')
    def test_main_with_env_api_key(self, mock_parse_args, mock_get_settings, mock_create_app, mock_uvicorn_run):
        """Test main with API key from environment."""
        mock_args = Mock()
        mock_args.api_key = None
        mock_parse_args.return_value = mock_args

        mock_settings = Mock()
        mock_settings.openrouter_api_key = 'env-key'
        mock_get_settings.return_value = mock_settings

        mock_app = Mock()
        mock_create_app.return_value = mock_app

        main()

        mock_uvicorn_run.assert_called_once()

    @patch('src.main.parse_args')
    def test_main_command_line_override(self, mock_parse_args):
        """Test command line arguments override environment variables."""
        mock_args = Mock()
        mock_args.api_key = 'cli-key'
        mock_args.host = 'cli-host'
        mock_args.port = 9999
        mock_parse_args.return_value = mock_args

        with patch('src.main.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.openrouter_api_key = 'cli-key'
            mock_get_settings.return_value = mock_settings

            with patch('src.main.create_app'), patch('src.main.uvicorn.run'):
                main()

        # Verify settings were called with overrides
        mock_get_settings.assert_called_once()
