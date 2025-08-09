"""Tests for the FastAPI application factory and middleware."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import create_app, setup_logging
from src.config import Settings
from src.exceptions import NetworkError, OpenRouterError
from src.openrouter import OpenRouterResponse


class TestSetupLogging:
    """Test logging setup function."""

    @patch("src.app.structlog.configure")
    @patch("src.app.logging.basicConfig")
    def test_setup_logging(self, mock_basic_config, mock_structlog_configure):
        """Test logging configuration setup."""
        settings = Mock(spec=Settings)
        settings.log_level = "INFO"
        settings.log_format = "%(asctime)s - %(levelname)s - %(message)s"

        setup_logging(settings)

        # Verify basic logging was configured
        mock_basic_config.assert_called_once()
        args, kwargs = mock_basic_config.call_args
        assert kwargs["level"] == 20  # logging.INFO
        assert kwargs["format"] == "%(asctime)s - %(levelname)s - %(message)s"

        # Verify structlog was configured
        mock_structlog_configure.assert_called_once()


class TestCreateApp:
    """Test FastAPI application creation."""

    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI instance."""
        with patch("src.app.lifespan"):
            app = create_app()
            assert isinstance(app, FastAPI)
            assert app.title == "Ollama Proxy"
            assert app.description == "A proxy server that translates Ollama API calls to OpenRouter"
            assert app.version == "0.1.0"

    def test_create_app_includes_router(self):
        """Test that the app includes the API router."""
        with patch("src.app.lifespan"):
            app = create_app()
            # Check that routes are registered
            route_paths = [getattr(route, 'path', None)
                           for route in app.routes if hasattr(route, 'path')]
            assert "/" in route_paths

    def test_create_app_has_cors_middleware(self):
        """Test that CORS middleware is configured."""
        with patch("src.app.lifespan"):
            app = create_app()
            # Check that CORS middleware is in the middleware stack
            middleware_types = [getattr(middleware.cls, '__name__', None)
                                for middleware in app.user_middleware if hasattr(middleware, 'cls')]
            assert "CORSMiddleware" in middleware_types

    def test_create_app_has_exception_handlers(self):
        """Test that exception handlers are registered."""
        with patch("src.app.lifespan"):
            app = create_app()
            # Check that exception handlers are registered
            assert OpenRouterError in app.exception_handlers
            assert Exception in app.exception_handlers


class TestAppMiddleware:
    """Test application middleware functionality."""

    @pytest.fixture
    def app(self):
        """Create test app with mocked lifespan."""
        with patch("src.app.lifespan"):
            return create_app()

    def test_request_logging_middleware(self, app):
        """Test request logging middleware."""
        with patch("src.app.structlog.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            client = TestClient(app)
            response = client.get("/")

            # Verify logger was called for request and response
            assert mock_logger.info.call_count >= 2

    def test_openrouter_error_handler(self, app):
        """Test OpenRouter error exception handler."""
        # Create a route that raises OpenRouterError
        @app.get("/test-error")
        async def test_error():
            raise OpenRouterError("Test error", status_code=400)

        with patch("src.app.structlog.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            client = TestClient(app)
            response = client.get("/test-error")

            assert response.status_code == 400
            data = response.json()
            assert data["error"] == "Test error"
            assert data["type"] == "openrouter_error"

            # Verify error was logged
            mock_logger.error.assert_called()

    def test_generic_error_handler(self, app):
        """Test generic exception handler."""
        # Create a route that raises a generic exception
        @app.get("/test-generic-error")
        async def test_generic_error():
            raise ValueError("Generic test error")

        with patch("src.app.structlog.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Configure TestClient to not raise exceptions
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/test-generic-error")

            assert response.status_code == 500
            data = response.json()
            assert data["error"] == "Internal server error"
            assert data["type"] == "internal_error"

            # Verify error was logged
            mock_logger.error.assert_called()


class TestLifespan:
    """Test application lifespan management."""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.app.OpenRouterClient")
    @patch("src.app.setup_logging")
    @patch("src.app.get_settings")
    def test_lifespan_success(self, mock_get_settings, mock_setup_logging, mock_openrouter_client):
        """Test successful lifespan initialization."""
        # Setup mocks
        mock_settings = Mock(spec=Settings)
        mock_settings.models_filter = None
        mock_settings.models_filter_path = None
        mock_get_settings.return_value = mock_settings

        mock_client_instance = AsyncMock()
        mock_openrouter_client.return_value = mock_client_instance

        # Mock successful model fetch
        mock_response_data = {
            "data": [
                {"id": "openai/gpt-4", "name": "GPT-4"},
                {"id": "anthropic/claude-3", "name": "Claude 3"},
            ]
        }
        mock_response = OpenRouterResponse(
            data=mock_response_data,
            status_code=200,
            headers={},
            metrics=AsyncMock()
        )
        mock_client_instance.fetch_models.return_value = mock_response

        # Create app (this will trigger lifespan)
        app = create_app()

        with TestClient(app) as client:
            # App should start successfully
            response = client.get("/")
            assert response.status_code == 200

        # Verify setup was called
        mock_setup_logging.assert_called_once_with(mock_settings)
        mock_client_instance.fetch_models.assert_called_once()
        mock_client_instance.close.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.app.OpenRouterClient")
    @patch("src.app.setup_logging")
    @patch("src.app.get_settings")
    def test_lifespan_openrouter_error(self, mock_get_settings, mock_setup_logging, mock_openrouter_client):
        """Test lifespan with OpenRouter error."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_client_instance = AsyncMock()
        mock_openrouter_client.return_value = mock_client_instance

        # Mock OpenRouter error
        mock_client_instance.fetch_models.side_effect = OpenRouterError(
            "API Error")

        # App creation should raise the error
        with pytest.raises(OpenRouterError):
            app = create_app()
            with TestClient(app):
                pass

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.app.OpenRouterClient")
    @patch("src.app.setup_logging")
    @patch("src.app.get_settings")
    def test_lifespan_network_error(self, mock_get_settings, mock_setup_logging, mock_openrouter_client):
        """Test lifespan with network error."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_client_instance = AsyncMock()
        mock_openrouter_client.return_value = mock_client_instance

        # Mock network error
        mock_client_instance.fetch_models.side_effect = NetworkError(
            "Network Error")

        # App creation should raise the error
        with pytest.raises(NetworkError):
            app = create_app()
            with TestClient(app):
                pass

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.app.OpenRouterClient")
    @patch("src.app.setup_logging")
    @patch("src.app.get_settings")
    def test_lifespan_generic_error(self, mock_get_settings, mock_setup_logging, mock_openrouter_client):
        """Test lifespan with generic error."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_client_instance = AsyncMock()
        mock_openrouter_client.return_value = mock_client_instance

        # Mock generic error
        mock_client_instance.fetch_models.side_effect = ValueError(
            "Generic Error")

        # App creation should raise the error
        with pytest.raises(ValueError):
            app = create_app()
            with TestClient(app):
                pass

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.app.OpenRouterClient")
    @patch("src.app.setup_logging")
    @patch("src.app.get_settings")
    def test_lifespan_cleanup_on_error(self, mock_get_settings, mock_setup_logging, mock_openrouter_client):
        """Test that cleanup happens even when there's an error."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_client_instance = AsyncMock()
        mock_openrouter_client.return_value = mock_client_instance

        # Mock error during model fetch
        mock_client_instance.fetch_models.side_effect = Exception("Test error")

        # App creation should raise the error but still call cleanup
        with pytest.raises(Exception):
            app = create_app()
            with TestClient(app):
                pass

        # Verify cleanup was called
        mock_client_instance.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
