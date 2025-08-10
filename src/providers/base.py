"""Base classes and interfaces for AI providers.

This module defines the core abstractions for AI providers, including the base
provider interface, configuration classes, and common data structures used
across all provider implementations.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
import structlog

from ..exceptions import ErrorCode, ErrorContext, ErrorType, ProxyError

logger = structlog.get_logger(__name__)


class ProviderType(StrEnum):
    """Supported AI provider types."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    AWS_BEDROCK = "aws_bedrock"
    OLLAMA = "ollama"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


class ProviderCapability(StrEnum):
    """Capabilities that providers can support."""

    CHAT_COMPLETION = "chat_completion"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    IMAGE_GENERATION = "image_generation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    VISION = "vision"


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Configuration for an AI provider."""

    provider_type: ProviderType
    api_key: str
    base_url: str
    timeout: int = 300
    max_retries: int = 3
    max_concurrent_requests: int = 100
    capabilities: frozenset[ProviderCapability] = field(default_factory=frozenset)
    custom_headers: Dict[str, str] = field(default_factory=dict)
    model_mapping: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Allow empty API key for Ollama provider (it typically doesn't require one)
        if not self.api_key.strip() and self.provider_type != ProviderType.OLLAMA:
            raise ValueError(f"API key cannot be empty for {self.provider_type}")
        if not self.base_url.strip():
            raise ValueError(f"Base URL cannot be empty for {self.provider_type}")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")


@dataclass(slots=True)
class ProviderMetrics:
    """Metrics for provider requests."""

    provider_type: ProviderType
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    model: Optional[str] = None
    endpoint: Optional[str] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    retry_count: int = 0
    request_size: int = 0
    response_size: int = 0

    def mark_complete(self, status_code: Optional[int] = None, error: Optional[str] = None, response_time: Optional[float] = None) -> None:
        """Mark request as complete."""
        self.end_time = response_time or time.time()
        if status_code is not None:
            self.status_code = status_code
        if error is not None:
            self.error = error

    def mark_error(self, error: str, response_time: Optional[float] = None) -> None:
        """Mark request as failed with error."""
        self.end_time = response_time or time.time()
        self.error = error

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1

    @property
    def duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def is_successful(self) -> bool:
        """Check if request was successful."""
        return self.status_code is not None and 200 <= self.status_code < 300

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "provider_type": self.provider_type.value,
            "request_id": self.request_id,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "error": self.error,
            "retry_count": self.retry_count,
            "request_size": self.request_size,
            "response_size": self.response_size,
            "is_successful": self.is_successful,
        }


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Standardized response from AI providers."""

    data: Dict[str, Any]
    status_code: int
    headers: Dict[str, str]
    metrics: ProviderMetrics
    provider_type: ProviderType

    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status_code < 300

    @property
    def content(self) -> Optional[str]:
        """Extract content from response."""
        # Try different common response formats
        if "choices" in self.data and self.data["choices"]:
            choice = self.data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]

        if "content" in self.data:
            return self.data["content"]

        if "text" in self.data:
            return self.data["text"]

        return None

    @property
    def usage(self) -> Optional[Dict[str, Any]]:
        """Extract usage information."""
        return self.data.get("usage")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data,
            "status_code": self.status_code,
            "headers": self.headers,
            "metrics": self.metrics.to_dict(),
            "provider_type": self.provider_type.value,
            "is_success": self.is_success,
        }


class ProviderError(ProxyError):
    """Base exception for provider-specific errors."""

    def __init__(
        self,
        message: str,
        provider_type: ProviderType,
        status_code: int = 500,
        error_code: Optional[ErrorCode] = None,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_type=ErrorType.OPENROUTER_ERROR,  # Default to OpenRouter error type
            error_code=error_code,
            status_code=status_code,
        )
        self.provider_type = provider_type
        self.context = context
        self.original_error = original_error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "error": str(self),
            "provider_type": self.provider_type.value,
            "status_code": self.status_code,
        }

        if self.error_code:
            result["error_code"] = self.error_code.value

        if self.context and self.context.request_id:
            result["request_id"] = self.context.request_id

        return result


class AIProvider(ABC):
    """Abstract base class for AI providers.

    This class defines the interface that all AI providers must implement.
    It provides common functionality for HTTP client management, error handling,
    and request/response transformation.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._error_count = 0
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._rate_limiter = asyncio.Semaphore(100)  # Default rate limit

    @property
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        return self.config.provider_type

    @property
    def capabilities(self) -> frozenset[ProviderCapability]:
        """Get supported capabilities."""
        return self.config.capabilities

    @property
    def request_count(self) -> int:
        """Get total request count."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """Get total error count."""
        return self._error_count

    @property
    def error_rate(self) -> float:
        """Get error rate as percentage."""
        if self._request_count == 0:
            return 0.0
        return (self._error_count / self._request_count) * 100

    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a specific capability."""
        return capability in self.capabilities

    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.config.timeout,
                    write=30.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=self.config.max_concurrent_requests,
                    max_keepalive_connections=min(20, self.config.max_concurrent_requests // 5),
                    keepalive_expiry=30.0,
                ),
                follow_redirects=True,
                max_redirects=3,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a rate-limited HTTP request."""
        async with self._semaphore:  # Limit concurrent requests
            async with self._rate_limiter:  # Rate limiting
                client = await self.get_client()

                try:
                    response = await client.request(method, url, **kwargs)
                    self._request_count += 1
                    response.raise_for_status()
                    return response
                except Exception as e:
                    self._error_count += 1
                    logger.error(
                        "HTTP request failed",
                        method=method,
                        url=url,
                        error=str(e),
                        provider=self.provider_type.value,
                    )
                    raise

    def _get_current_time(self) -> float:
        """Get current timestamp."""
        return time.time()

    def get_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"ollama-proxy/0.1.0 ({self.provider_type.value})",
        }

        # Add custom headers from config
        headers.update(self.config.custom_headers)

        # Add extra headers if provided
        if extra_headers:
            headers.update(extra_headers)

        return headers

    @abstractmethod
    async def list_models(self) -> ProviderResponse:
        """List available models."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Create a chat completion."""
        pass

    @abstractmethod
    async def text_completion(
        self,
        prompt: str,
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[ProviderResponse, AsyncIterator[bytes]]:
        """Create a text completion."""
        pass

    @abstractmethod
    async def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Create embeddings."""
        pass

    @abstractmethod
    def transform_ollama_to_provider(
        self,
        ollama_request: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform Ollama request format to provider format."""
        pass

    @abstractmethod
    def transform_provider_to_ollama(
        self,
        provider_response: Dict[str, Any],
        endpoint: str,
    ) -> Dict[str, Any]:
        """Transform provider response format to Ollama format."""
        pass

    async def handle_error(
        self,
        response: httpx.Response,
        metrics: ProviderMetrics,
    ) -> None:
        """Handle HTTP error responses."""
        self._error_count += 1

        try:
            error_data = response.json()
        except Exception:
            error_data = {"message": response.text or f"HTTP {response.status_code} error"}

        error_message = self._extract_error_message(error_data)

        logger.error(
            "Provider API error",
            provider_type=self.provider_type.value,
            status_code=response.status_code,
            error_message=error_message,
            url=str(response.url),
            **metrics.to_dict(),
        )

        raise ProviderError(
            message=f"{self.provider_type.value} API error: {error_message}",
            provider_type=self.provider_type,
            status_code=response.status_code,
            context=ErrorContext(request_id=metrics.request_id),
        )

    def _extract_error_message(self, error_data: Dict[str, Any]) -> str:
        """Extract error message from provider response."""
        # Try common error message fields
        for field in ["message", "error", "detail", "description"]:
            if field in error_data:
                value = error_data[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict) and "message" in value:
                    return value["message"]

        return str(error_data)