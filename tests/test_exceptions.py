"""Tests for enhanced exceptions with pattern matching and error context."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.exceptions import (
    ConfigurationError,
    ErrorCode,
    ErrorContext,
    ErrorType,
    ModelError,
    ModelForbiddenError,
    ModelNotFoundError,
    NetworkError,
    OpenRouterError,
    ProxyError,
    ValidationError,
    handle_pydantic_validation_error,
    map_openrouter_error,
)


class TestErrorContext:
    """Test ErrorContext dataclass."""

    def test_default_creation(self):
        """Test default error context creation."""
        context = ErrorContext()
        assert isinstance(context.timestamp, datetime)
        assert context.request_id is None
        assert context.user_id is None
        assert context.correlation_id is None
        assert isinstance(context.additional_data, dict)

    def test_custom_creation(self):
        """Test error context with custom values."""
        context = ErrorContext(
            request_id="test-request",
            user_id="test-user",
            correlation_id="test-correlation",
            additional_data={"key": "value"},
        )
        assert context.request_id == "test-request"
        assert context.user_id == "test-user"
        assert context.correlation_id == "test-correlation"
        assert context.additional_data["key"] == "value"

    def test_to_dict(self):
        """Test dictionary conversion."""
        context = ErrorContext(request_id="test")
        result = context.to_dict()

        assert "timestamp" in result
        assert "request_id" in result
        assert result["request_id"] == "test"
        assert isinstance(result["timestamp"], str)  # ISO format


class TestProxyError:
    """Test base ProxyError class."""

    def test_basic_creation(self):
        """Test basic error creation."""
        error = ProxyError(
            message="Test error",
            error_type=ErrorType.INTERNAL_ERROR,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
        )
        assert error.message == "Test error"
        assert error.error_type == ErrorType.INTERNAL_ERROR
        assert error.error_code == ErrorCode.INTERNAL_ERROR
        assert error.status_code == 500

    def test_to_dict(self):
        """Test dictionary conversion."""
        error = ProxyError(
            message="Test error",
            error_type=ErrorType.VALIDATION_ERROR,
            error_code=ErrorCode.INVALID_REQUEST,
            status_code=400,
            details={"field": "test"},
        )
        result = error.to_dict()

        assert result["error"] == "Test error"
        assert result["type"] == ErrorType.VALIDATION_ERROR
        assert result["status_code"] == 400
        assert result["code"] == ErrorCode.INVALID_REQUEST.value
        assert result["details"]["field"] == "test"


class TestModelError:
    """Test ModelError and its subclasses."""

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("test-model")
        assert error.model_name == "test-model"
        assert error.status_code == 404
        assert error.error_code == ErrorCode.MODEL_NOT_FOUND
        assert "test-model" in error.message

    def test_model_forbidden_error(self):
        """Test ModelForbiddenError."""
        error = ModelForbiddenError("test-model")
        assert error.model_name == "test-model"
        assert error.status_code == 403
        assert error.error_code == ErrorCode.MODEL_FORBIDDEN
        assert "test-model" in error.message


class TestOpenRouterError:
    """Test OpenRouterError with enhanced features."""

    def test_basic_creation(self):
        """Test basic OpenRouter error creation."""
        error = OpenRouterError(
            message="API error", status_code=400, response_data={"error": "Bad request"}
        )
        assert error.message == "API error"
        assert error.status_code == 400
        assert error.response_data["error"] == "Bad request"

    def test_status_code_mapping(self):
        """Test automatic status code to error code mapping."""
        # Test 400 - Bad Request
        error_400 = OpenRouterError(message="Bad request", status_code=400)
        assert error_400.error_code == ErrorCode.INVALID_REQUEST

        # Test 401 - Unauthorized
        error_401 = OpenRouterError(message="Unauthorized", status_code=401)
        assert error_401.error_code == ErrorCode.OPENROUTER_AUTH_ERROR

        # Test 403 - Forbidden
        error_403 = OpenRouterError(message="Forbidden", status_code=403)
        assert error_403.error_code == ErrorCode.OPENROUTER_AUTH_ERROR

        # Test 404 - Not Found
        error_404 = OpenRouterError(message="Not found", status_code=404)
        assert error_404.error_code == ErrorCode.MODEL_NOT_FOUND

        # Test 429 - Rate Limited
        error_429 = OpenRouterError(message="Rate limited", status_code=429)
        assert error_429.error_code == ErrorCode.OPENROUTER_RATE_LIMIT

        # Test 500 - Server Error
        error_500 = OpenRouterError(message="Server error", status_code=500)
        assert error_500.error_code == ErrorCode.OPENROUTER_API_ERROR

    def test_get_retry_after(self):
        """Test retry-after extraction."""
        # Rate limit error with retry-after
        error_with_retry = OpenRouterError(
            message="Rate limited", status_code=429, response_data={"retry_after": 60}
        )
        assert error_with_retry.get_retry_after() == 60

        # Non-rate-limit error
        error_without_retry = OpenRouterError(message="Bad request", status_code=400)
        assert error_without_retry.get_retry_after() is None

    def test_is_retryable(self):
        """Test retryable error detection."""
        # Rate limit error - retryable
        rate_limit_error = OpenRouterError(message="Rate limited", status_code=429)
        assert rate_limit_error.is_retryable()

        # Server error - retryable
        server_error = OpenRouterError(message="Server error", status_code=500)
        assert server_error.is_retryable()

        # Client error - not retryable
        client_error = OpenRouterError(message="Bad request", status_code=400)
        assert not client_error.is_retryable()

        # Auth error - not retryable
        auth_error = OpenRouterError(message="Unauthorized", status_code=401)
        assert not auth_error.is_retryable()


class TestValidationError:
    """Test ValidationError class."""

    def test_basic_creation(self):
        """Test basic validation error creation."""
        error = ValidationError(
            message="Invalid field", field="test_field", value="invalid_value"
        )
        assert error.message == "Invalid field"
        assert error.field == "test_field"
        assert error.value == "invalid_value"
        assert error.status_code == 400
        assert error.error_code == ErrorCode.INVALID_REQUEST

    def test_details_in_dict(self):
        """Test details inclusion in dictionary."""
        error = ValidationError(
            message="Invalid field", field="test_field", value="invalid_value"
        )
        result = error.to_dict()

        assert result["details"]["field"] == "test_field"
        assert result["details"]["value"] == "invalid_value"


class TestConfigurationError:
    """Test ConfigurationError class."""

    def test_basic_creation(self):
        """Test basic configuration error creation."""
        error = ConfigurationError(message="Invalid config", config_key="test_key")
        assert error.message == "Invalid config"
        assert error.config_key == "test_key"
        assert error.status_code == 500
        assert error.error_code == ErrorCode.CONFIGURATION_ERROR


class TestNetworkError:
    """Test NetworkError class."""

    def test_basic_creation(self):
        """Test basic network error creation."""
        original_error = ConnectionError("Connection failed")
        error = NetworkError(message="Network issue", original_error=original_error)
        assert error.message == "Network issue"
        assert error.original_error == original_error
        assert error.status_code == 503
        assert error.error_code == ErrorCode.NETWORK_ERROR

    def test_details_with_original_error(self):
        """Test details inclusion with original error."""
        original_error = ConnectionError("Connection failed")
        error = NetworkError(message="Network issue", original_error=original_error)
        result = error.to_dict()

        assert result["details"]["original_error"] == "Connection failed"
        assert result["details"]["error_type"] == "ConnectionError"


class TestMapOpenRouterError:
    """Test map_openrouter_error function."""

    def test_basic_error_mapping(self):
        """Test basic error mapping."""
        error = map_openrouter_error(
            status_code=400, response_data={"message": "Bad request"}
        )
        assert isinstance(error, OpenRouterError)
        assert error.status_code == 400
        assert "Bad request" in error.message

    def test_nested_error_message_extraction(self):
        """Test nested error message extraction."""
        # Test nested error structure
        response_data = {
            "error": {"message": "Invalid model", "code": "model_not_found"}
        }
        error = map_openrouter_error(status_code=404, response_data=response_data)
        assert "Invalid model" in error.message

    def test_error_with_code_and_message(self):
        """Test error with both code and message."""
        response_data = {
            "error": {"code": "rate_limit_exceeded", "message": "Too many requests"}
        }
        error = map_openrouter_error(status_code=429, response_data=response_data)
        # Status code formatting takes precedence over the extracted code:message format
        assert "Rate limit exceeded: Too many requests" in error.message

    def test_status_code_specific_messages(self):
        """Test status code specific message formatting."""
        # Test 401
        error_401 = map_openrouter_error(401, {"message": "Invalid key"})
        assert "Authentication failed: Invalid key" in error_401.message

        # Test 403
        error_403 = map_openrouter_error(403, {"message": "Forbidden"})
        assert "Access forbidden: Forbidden" in error_403.message

        # Test 404
        error_404 = map_openrouter_error(404, {"message": "Not found"})
        assert "Resource not found: Not found" in error_404.message

        # Test 429
        error_429 = map_openrouter_error(429, {"message": "Rate limited"})
        assert "Rate limit exceeded: Rate limited" in error_429.message

        # Test 500
        error_500 = map_openrouter_error(500, {"message": "Server error"})
        assert "Server error (500): Server error" in error_500.message

    def test_fallback_message_extraction(self):
        """Test fallback message extraction."""
        # Test with detail field
        response_data = {"detail": "Detailed error message"}
        error = map_openrouter_error(400, response_data)
        assert "Detailed error message" in error.message

        # Test with no recognizable error fields
        response_data = {"unknown_field": "some value"}
        error = map_openrouter_error(400, response_data)
        assert "API error (400)" in error.message


class TestHandlePydanticValidationError:
    """Test handle_pydantic_validation_error function."""

    def test_pydantic_error_conversion(self):
        """Test conversion of Pydantic validation errors."""

        # Mock a Pydantic validation error
        class MockPydanticError(Exception):
            def errors(self):
                return [
                    {
                        "loc": ("field_name",),
                        "msg": "field required",
                        "type": "value_error.missing",
                    }
                ]

        mock_error = MockPydanticError()
        validation_error = handle_pydantic_validation_error(mock_error)

        assert isinstance(validation_error, ValidationError)
        assert "field_name" in validation_error.message
        assert "field required" in validation_error.message
        assert validation_error.field == "field_name"

    def test_generic_error_conversion(self):
        """Test conversion of generic errors."""
        generic_error = ValueError("Generic validation error")
        validation_error = handle_pydantic_validation_error(generic_error)

        assert isinstance(validation_error, ValidationError)
        assert "Generic validation error" in validation_error.message


if __name__ == "__main__":
    pytest.main([__file__])
