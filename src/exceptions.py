"""Custom exceptions and error handling for ollama-proxy with modern Python features."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Any, TypeAlias

# Type aliases for better code clarity
StatusCode: TypeAlias = int
ErrorMessage: TypeAlias = str
ErrorDetails: TypeAlias = dict[str, Any]


class ErrorType(StrEnum):
    """Error type classifications."""

    VALIDATION_ERROR = "validation_error"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_FORBIDDEN = "model_forbidden"
    OPENROUTER_ERROR = "openrouter_error"
    NETWORK_ERROR = "network_error"
    INTERNAL_ERROR = "internal_error"
    CONFIGURATION_ERROR = "configuration_error"


class ErrorCode(Enum):
    """Specific error codes for better error handling."""

    # Model errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_FORBIDDEN = "MODEL_FORBIDDEN"
    MODEL_RESOLUTION_FAILED = "MODEL_RESOLUTION_FAILED"

    # OpenRouter errors
    OPENROUTER_API_ERROR = "OPENROUTER_API_ERROR"
    OPENROUTER_NETWORK_ERROR = "OPENROUTER_NETWORK_ERROR"
    OPENROUTER_AUTH_ERROR = "OPENROUTER_AUTH_ERROR"
    OPENROUTER_RATE_LIMIT = "OPENROUTER_RATE_LIMIT"

    # Request errors
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_FORMAT = "INVALID_FORMAT"

    # System errors
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"


class ProxyError(Exception):
    """Base exception for ollama-proxy errors."""

    def __init__(
        self,
        message: str,
        error_type: ErrorType,
        error_code: ErrorCode | None = None,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for JSON response."""
        result = {
            "error": self.message,
            "type": self.error_type,
            "status_code": self.status_code,
        }

        if self.error_code:
            result["code"] = self.error_code.value

        if self.details:
            result["details"] = self.details

        return result


class ModelError(ProxyError):
    """Model-related errors."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        error_code: ErrorCode | None = None,
        status_code: int = 404,
    ):
        super().__init__(
            message=message,
            error_type=(
                ErrorType.MODEL_NOT_FOUND
                if status_code == 404
                else ErrorType.MODEL_FORBIDDEN
            ),
            error_code=error_code,
            status_code=status_code,
            details={"model": model_name} if model_name else None,
        )
        self.model_name = model_name


class ModelNotFoundError(ModelError):
    """Model not found error."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            model_name=model_name,
            error_code=ErrorCode.MODEL_NOT_FOUND,
            status_code=404,
        )


class ModelForbiddenError(ModelError):
    """Model access forbidden error."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' is not allowed by the filter",
            model_name=model_name,
            error_code=ErrorCode.MODEL_FORBIDDEN,
            status_code=403,
        )


@dataclass(slots=True, frozen=True)
class ErrorContext:
    """Enhanced error context with metadata."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str | None = None
    user_id: str | None = None
    correlation_id: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "user_id": self.user_id,
            "correlation_id": self.correlation_id,
            "additional_data": self.additional_data,
        }


class OpenRouterError(ProxyError):
    """OpenRouter API errors with enhanced pattern matching and context."""

    def __init__(
        self,
        message: ErrorMessage,
        status_code: StatusCode | None = None,
        response_data: ErrorDetails | None = None,
        error_code: ErrorCode | None = None,
        context: ErrorContext | None = None,
    ):
        # Enhanced pattern matching for status code mapping
        if not error_code and status_code:
            error_code = self._map_status_code_to_error_code(status_code)

        super().__init__(
            message=message,
            error_type=ErrorType.OPENROUTER_ERROR,
            error_code=error_code,
            status_code=status_code or 500,
            details=response_data,
        )
        self.response_data = response_data or {}
        self.context = context or ErrorContext()

    @staticmethod
    def _map_status_code_to_error_code(status_code: StatusCode) -> ErrorCode:
        """Map HTTP status codes to error codes using pattern matching."""
        match status_code:
            case 400:
                return ErrorCode.INVALID_REQUEST
            case 401:
                return ErrorCode.OPENROUTER_AUTH_ERROR
            case 403:
                return ErrorCode.OPENROUTER_AUTH_ERROR
            case 404:
                return ErrorCode.MODEL_NOT_FOUND
            case 429:
                return ErrorCode.OPENROUTER_RATE_LIMIT
            case code if 500 <= code < 600:
                return ErrorCode.OPENROUTER_API_ERROR
            case _:
                return ErrorCode.OPENROUTER_API_ERROR

    def get_retry_after(self) -> int | None:
        """Extract retry-after value from rate limit errors."""
        match self.error_code:
            case ErrorCode.OPENROUTER_RATE_LIMIT:
                if isinstance(self.response_data, dict):
                    return self.response_data.get("retry_after")
            case _:
                return None

    def is_retryable(self) -> bool:
        """Check if this error is retryable using pattern matching."""
        match self.error_code:
            case ErrorCode.OPENROUTER_RATE_LIMIT | ErrorCode.NETWORK_ERROR:
                return True
            case ErrorCode.OPENROUTER_API_ERROR if self.status_code >= 500:
                return True
            case _:
                return False


class ValidationError(ProxyError):
    """Request validation errors."""

    def __init__(self, message: str, field: str | None = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION_ERROR,
            error_code=ErrorCode.INVALID_REQUEST,
            status_code=400,
            details=details,
        )
        self.field = field
        self.value = value


class ConfigurationError(ProxyError):
    """Configuration-related errors."""

    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(
            message=message,
            error_type=ErrorType.CONFIGURATION_ERROR,
            error_code=ErrorCode.CONFIGURATION_ERROR,
            status_code=500,
            details={"config_key": config_key} if config_key else None,
        )
        self.config_key = config_key


class NetworkError(ProxyError):
    """Network-related errors."""

    def __init__(self, message: str, original_error: Exception | None = None):
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__

        super().__init__(
            message=message,
            error_type=ErrorType.NETWORK_ERROR,
            error_code=ErrorCode.NETWORK_ERROR,
            status_code=503,
            details=details,
        )
        self.original_error = original_error


def map_openrouter_error(
    status_code: StatusCode,
    response_data: ErrorDetails | None = None,
    context: ErrorContext | None = None,
) -> OpenRouterError:
    """Map OpenRouter API errors to our error types using pattern matching."""

    # Extract error message using pattern matching
    message = "OpenRouter API error"

    if response_data:
        match response_data:
            case {"error": {"message": str(msg)}}:
                message = msg
            case {"error": str(msg)}:
                message = msg
            case {"message": str(msg)}:
                message = msg
            case {"detail": str(msg)}:
                message = msg
            case {"error": {"code": str(code), "message": str(msg)}}:
                message = f"{code}: {msg}"
            case _:
                # Try to extract any string value that looks like an error message
                for key in ["error", "message", "detail", "description"]:
                    if key in response_data and isinstance(response_data[key], str):
                        message = response_data[key]
                        break

    # Enhanced message formatting based on status code
    match status_code:
        case 401:
            message = f"Authentication failed: {message}"
        case 403:
            message = f"Access forbidden: {message}"
        case 404:
            message = f"Resource not found: {message}"
        case 429:
            message = f"Rate limit exceeded: {message}"
        case server_code if 500 <= server_code < 600:
            message = f"Server error ({server_code}): {message}"
        case _:
            message = f"API error ({status_code}): {message}"

    return OpenRouterError(
        message=message,
        status_code=status_code,
        response_data=response_data,
        context=context,
    )


def handle_pydantic_validation_error(exc: Exception) -> ValidationError:
    """Convert Pydantic validation errors to our format."""
    try:
        # Try to extract field information from Pydantic error
        if hasattr(exc, "errors"):
            errors = exc.errors()
            if errors:
                first_error = errors[0]
                field = ".".join(str(loc) for loc in first_error.get("loc", []))
                message = first_error.get("msg", str(exc))
                return ValidationError(
                    message=f"Validation error in field '{field}': {message}",
                    field=field,
                )
    except Exception:
        pass

    return ValidationError(f"Request validation failed: {str(exc)}")
