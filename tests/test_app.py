"""Tests for the FastAPI application factory and middleware."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app import create_app
from src.config import Settings
from src.exceptions import NetworkError, OpenRouterError
from src.openrouter import OpenRouterResponse



class TestCreateApp:
    """Test FastAPI application creation."""

    def test_create_app_returns_fastapi_instance(self):
        """Test that create_app returns a FastAPI instance."""
        with patch("src.app.lifespan"):
            app = create_app()
            assert isinstance(app, FastAPI)
            assert app.title == "Ollama Proxy"
            assert app.description == "Multi-provider AI proxy with intelligent routing and fallback"
            assert app.version == "0.2.0"

    def test_create_app_includes_multi_provider_routes(self):
        """Test that the app includes multi-provider API routes."""
        with patch("src.app.lifespan"):
            app = create_app()
            # Check that multi-provider routes are registered
            route_paths = [getattr(route, 'path', None)
                           for route in app.routes if hasattr(route, 'path')]
            assert "/api/tags" in route_paths
            assert "/api/chat" in route_paths
            assert "/api/generate" in route_paths
            assert "/api/embeddings" in route_paths
            assert "/api/providers" in route_paths

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
            from src.exceptions import ProxyError
            from starlette.exceptions import HTTPException
            app = create_app()
            # Check that exception handlers are registered
            assert ProxyError in app.exception_handlers
            assert HTTPException in app.exception_handlers
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




if __name__ == "__main__":
    pytest.main([__file__])
