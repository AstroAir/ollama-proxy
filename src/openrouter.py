"""Enhanced OpenRouter API client with latest API support and modern Python patterns."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Self, TypeAlias

import httpx
import structlog

from .config import Settings
from .exceptions import ErrorContext, NetworkError, OpenRouterError

# Type aliases for better code clarity
ModelID: TypeAlias = str
RequestPayload: TypeAlias = dict[str, Any]
ResponseData: TypeAlias = dict[str, Any]
Headers: TypeAlias = dict[str, str]

logger = structlog.get_logger(__name__)


class OpenRouterEndpoint(StrEnum):
    """OpenRouter API endpoints with latest API support."""

    MODELS = "/models"
    CHAT_COMPLETIONS = "/chat/completions"
    COMPLETIONS = "/completions"
    EMBEDDINGS = "/embeddings"
    GENERATION = "/generation"
    # New endpoints from latest API
    PROVIDERS = "/providers"
    LIMITS = "/auth/key"
    COSTS = "/generation/{id}/cost"


@dataclass(slots=True, kw_only=True)
class RequestMetrics:
    """Enhanced request metrics with comprehensive monitoring and modern features."""

    endpoint: OpenRouterEndpoint | str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    model: ModelID | None = None
    stream: bool = False
    request_size: int = 0
    response_size: int = 0
    status_code: int | None = None
    error: str | None = None
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    retry_count: int = 0
    cache_hit: bool = False

    def mark_complete(
        self, status_code: int | None = None, error: str | None = None
    ) -> None:
        """Mark request as complete with optional status and error."""
        self.end_time = time.time()
        self.status_code = status_code
        self.error = error

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.retry_count += 1

    def mark_cache_hit(self) -> None:
        """Mark as cache hit."""
        self.cache_hit = True

    @property
    def duration_ms(self) -> float:
        """Get request duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def is_complete(self) -> bool:
        """Check if request is complete."""
        return self.end_time is not None

    @property
    def is_successful(self) -> bool:
        """Check if request was successful using pattern matching."""
        match self.status_code:
            case code if code and 200 <= code < 300:
                return True
            case _:
                return False

    @property
    def performance_category(self) -> str:
        """Categorize performance using pattern matching."""
        duration = self.duration_ms
        match duration:
            case d if d < 100:
                return "excellent"
            case d if d < 500:
                return "good"
            case d if d < 1000:
                return "acceptable"
            case d if d < 5000:
                return "slow"
            case _:
                return "very_slow"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging with enhanced metrics."""
        return {
            "request_id": self.request_id,
            "endpoint": str(self.endpoint),
            "duration_ms": self.duration_ms,
            "model": self.model,
            "stream": self.stream,
            "request_size": self.request_size,
            "response_size": self.response_size,
            "status_code": self.status_code,
            "error": self.error,
            "is_complete": self.is_complete,
            "is_successful": self.is_successful,
            "performance_category": self.performance_category,
            "retry_count": self.retry_count,
            "cache_hit": self.cache_hit,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class OpenRouterResponse:
    """Enhanced response wrapper with metadata and modern pattern matching."""

    data: ResponseData
    status_code: int
    headers: Headers
    metrics: RequestMetrics

    @property
    def is_success(self) -> bool:
        """Check if response was successful using pattern matching."""
        match self.status_code:
            case code if 200 <= code < 300:
                return True
            case _:
                return False

    @property
    def usage(self) -> dict[str, Any] | None:
        """Get usage information if available with enhanced parsing."""
        match self.data:
            case {"usage": dict() as usage_data}:
                return usage_data
            case _:
                return None

    @property
    def model_used(self) -> str | None:
        """Get the actual model used in the response."""
        match self.data:
            case {"model": str() as model}:
                return model
            case _:
                return None

    @property
    def finish_reason(self) -> str | None:
        """Get finish reason from first choice."""
        choices = self.get_choices()
        if choices:
            match choices[0]:
                case {"finish_reason": str() as reason}:
                    return reason
                case _:
                    return None
        return None

    def get_choices(self) -> list[dict[str, Any]]:
        """Get choices from response with enhanced error handling."""
        match self.data:
            case {"choices": list() as choices}:
                return choices
            case _:
                return []

    def get_content(self) -> str | None:
        """Get content from first choice message."""
        choices = self.get_choices()
        if choices:
            match choices[0]:
                case {"message": {"content": str() as content}}:
                    return content
                case {"delta": {"content": str() as content}}:
                    return content
                case _:
                    return None
        return None

    def get_error_info(self) -> dict[str, Any] | None:
        """Extract error information if present."""
        match self.data:
            case {"error": dict() as error_info}:
                return error_info
            case {"error": str() as error_msg}:
                return {"message": error_msg}
            case _:
                return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "status_code": self.status_code,
            "headers": self.headers,
            "metrics": self.metrics.to_dict(),
            "is_success": self.is_success,
            "model_used": self.model_used,
            "finish_reason": self.finish_reason,
        }


class OpenRouterClient:
    """Enhanced OpenRouter API client with latest API support and comprehensive monitoring."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.timeout = settings.openrouter_timeout
        self._client: httpx.AsyncClient | None = None
        self._request_count = 0
        self._error_count = 0

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

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get or create HTTP client with enhanced configuration and monitoring."""
        if self._client is None:
            # Enhanced client configuration for better performance and reliability
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,  # Connection timeout
                    read=self.timeout,  # Read timeout
                    write=30.0,  # Write timeout
                    pool=5.0,  # Pool timeout
                ),
                limits=httpx.Limits(
                    max_connections=self.settings.max_concurrent_requests,
                    max_keepalive_connections=min(
                        20, self.settings.max_concurrent_requests // 5
                    ),
                    keepalive_expiry=30.0,
                ),
                # Enhanced retry configuration
                transport=httpx.AsyncHTTPTransport(retries=3, verify=True),
                # Follow redirects with limit
                follow_redirects=True,
                max_redirects=3,
            )

        try:
            yield self._client
        except Exception as e:
            # Enhanced error handling with logging
            logger.error(
                "HTTP client error",
                error=str(e),
                error_type=type(e).__name__,
                request_count=self._request_count,
            )
            # Close client on error to ensure clean state
            if self._client:
                await self._client.aclose()
                self._client = None
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_headers(
        self,
        extra_headers: Headers | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        request_id: str | None = None,
    ) -> Headers:
        """Get enhanced headers for OpenRouter API requests with latest requirements."""
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ollama-proxy/0.1.0 (https://github.com/ollama-proxy)",
            # OpenRouter-specific headers for better service and rankings
            "X-Title": site_name or "Ollama Proxy",
        }

        # Add optional site URL for OpenRouter rankings (helps with rate limits)
        if site_url:
            headers["HTTP-Referer"] = site_url

        # Enhanced request tracking with better ID generation
        req_id = request_id or f"ollama-proxy-{uuid.uuid4().hex[:12]}"
        headers["X-Request-ID"] = req_id

        # Add correlation headers for better debugging
        headers["X-Client-Version"] = "0.1.0"
        headers["X-Client-Type"] = "ollama-proxy"

        # Pattern matching for header customization based on environment
        match self.settings.environment.value:
            case "development":
                headers["X-Debug-Mode"] = "true"
            case "production":
                headers["X-Environment"] = "production"
            case _:
                pass

        if extra_headers:
            headers.update(extra_headers)

        return headers

    async def _handle_response_error(
        self, response: httpx.Response, metrics: RequestMetrics
    ) -> None:
        """Enhanced error handling with comprehensive logging and metrics."""
        self._error_count += 1

        try:
            error_data = response.json()
            if isinstance(error_data, dict) and "error" in error_data:
                error_info = error_data["error"]
                if isinstance(error_info, dict):
                    message = error_info.get("message", str(error_info))
                    error_code = error_info.get("code")
                    error_type = error_info.get("type")
                else:
                    message = str(error_info)
                    error_code = None
                    error_type = None
            else:
                message = error_data.get("message", str(error_data))
                error_code = None
                error_type = None
        except (json.JSONDecodeError, AttributeError):
            message = response.text or f"HTTP {response.status_code} error"
            error_code = None
            error_type = None

        # Enhanced logging with metrics
        logger.error(
            "OpenRouter API error",
            status_code=response.status_code,
            message=message,
            error_code=error_code,
            error_type=error_type,
            url=str(response.url),
            **metrics.to_dict(),
        )

        # Create error context for better debugging
        context = ErrorContext(
            request_id=metrics.request_id,
            additional_data={
                "url": str(response.url),
                "method": response.request.method if response.request else "UNKNOWN",
                "duration_ms": metrics.duration_ms,
                "retry_count": metrics.retry_count,
            },
        )

        # Pattern matching for error type creation with enhanced messages
        match response.status_code:
            case 400:
                error_msg = f"Bad request: {message}"
            case 401:
                error_msg = f"Authentication failed: {message}"
            case 403:
                error_msg = f"Access forbidden: {message}"
            case 404:
                error_msg = f"Resource not found: {message}"
            case 429:
                error_msg = f"Rate limit exceeded: {message}"
            case code if 500 <= code < 600:
                error_msg = f"Server error ({code}): {message}"
            case _:
                error_msg = f"API error ({response.status_code}): {message}"

        raise OpenRouterError(
            message=error_msg,
            status_code=response.status_code,
            response_data=(
                {"code": error_code, "type": error_type} if error_code else {}
            ),
            context=context,
        )

    async def fetch_models(self) -> OpenRouterResponse:
        """Fetch available models from OpenRouter with enhanced monitoring."""
        metrics = RequestMetrics(endpoint=OpenRouterEndpoint.MODELS)
        self._request_count += 1

        async with self._get_client() as client:
            try:
                response = await client.get(
                    f"{self.base_url}{OpenRouterEndpoint.MODELS}",
                    headers=self._get_headers(),
                )

                if response.status_code != 200:
                    await self._handle_response_error(response, metrics)

                data = response.json()
                logger.info(
                    "Successfully fetched models",
                    model_count=len(data.get("data", [])),
                    **metrics.to_dict(),
                )

                return OpenRouterResponse(
                    data=data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metrics=metrics,
                )

            except httpx.RequestError as e:
                self._error_count += 1
                logger.error(
                    "Network error fetching models", error=str(e), **metrics.to_dict()
                )
                raise NetworkError(
                    f"Network error fetching models: {e}", original_error=e
                )

    async def chat_completion(
        self, payload: RequestPayload, stream: bool = False, **kwargs: Any
    ) -> OpenRouterResponse | AsyncIterator[bytes]:
        """Create a chat completion request with enhanced monitoring and latest API features."""
        metrics = RequestMetrics(
            endpoint=OpenRouterEndpoint.CHAT_COMPLETIONS,
            model=payload.get("model"),
            stream=stream,
        )
        self._request_count += 1

        # Enhanced payload processing with latest OpenRouter features
        enhanced_payload = self._enhance_chat_payload(payload, **kwargs)

        endpoint = f"{self.base_url}{OpenRouterEndpoint.CHAT_COMPLETIONS}"
        headers = self._get_headers(request_id=metrics.request_id)

        # Add streaming header if needed
        if stream:
            headers["Accept"] = "text/event-stream"
            enhanced_payload["stream"] = True

        # Set request size for metrics
        metrics.request_size = len(json.dumps(enhanced_payload).encode("utf-8"))

        async with self._get_client() as client:
            try:
                response = await client.post(
                    endpoint, headers=headers, json=enhanced_payload
                )

                if response.status_code != 200:
                    await self._handle_response_error(response, metrics)

                if stream:
                    return self._stream_response(response, metrics)
                else:
                    data = response.json()
                    metrics.response_size = len(json.dumps(data).encode("utf-8"))
                    metrics.mark_complete(status_code=response.status_code)

                    logger.debug(
                        "Chat completion successful",
                        model=payload.get("model"),
                        **metrics.to_dict(),
                    )
                    return OpenRouterResponse(
                        data=data,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        metrics=metrics,
                    )

            except httpx.RequestError as e:
                self._error_count += 1
                metrics.mark_complete(error=str(e))
                logger.error(
                    "Network error in chat completion",
                    error=str(e),
                    **metrics.to_dict(),
                )
                raise NetworkError(
                    f"Network error in chat completion: {e}", original_error=e
                )

    def _enhance_chat_payload(
        self, payload: RequestPayload, **kwargs: Any
    ) -> RequestPayload:
        """Enhance chat payload with latest OpenRouter API features."""
        enhanced = payload.copy()

        # Add latest OpenRouter-specific parameters if provided
        if "provider" in kwargs:
            enhanced["provider"] = kwargs["provider"]

        if "models" in kwargs:
            enhanced["models"] = kwargs["models"]

        if "route" in kwargs:
            enhanced["route"] = kwargs["route"]

        if "transforms" in kwargs:
            enhanced["transforms"] = kwargs["transforms"]

        # Enhanced parameter validation and defaults
        match enhanced.get("model"):
            case str() as model if "/" not in model:
                # Add default provider if not specified
                logger.warning(
                    f"Model '{model}' may need provider prefix for optimal routing"
                )
            case _:
                pass

        # Set reasonable defaults for better performance
        if "max_tokens" not in enhanced:
            enhanced["max_tokens"] = 4096

        if "temperature" not in enhanced:
            enhanced["temperature"] = 0.7

        return enhanced

    async def _stream_response(
        self, response: httpx.Response, metrics: RequestMetrics
    ) -> AsyncIterator[bytes]:
        """Handle streaming response from OpenRouter with enhanced monitoring."""
        try:
            chunk_count = 0
            async for chunk in response.aiter_bytes():
                if chunk:
                    chunk_count += 1
                    yield chunk

            logger.debug(
                "Streaming response completed",
                chunk_count=chunk_count,
                **metrics.to_dict(),
            )
        except Exception as e:
            self._error_count += 1
            logger.error(
                "Error in streaming response", error=str(e), **metrics.to_dict()
            )
            raise OpenRouterError(f"Streaming error: {e}")

    async def fetch_embeddings(self, payload: dict[str, Any]) -> OpenRouterResponse:
        """Fetch embeddings from OpenRouter with enhanced monitoring."""
        metrics = RequestMetrics(
            endpoint=OpenRouterEndpoint.EMBEDDINGS, model=payload.get("model")
        )
        self._request_count += 1

        async with self._get_client() as client:
            try:
                response = await client.post(
                    f"{self.base_url}{OpenRouterEndpoint.EMBEDDINGS}",
                    headers=self._get_headers(),
                    json=payload,
                )

                if response.status_code != 200:
                    await self._handle_response_error(response, metrics)

                data = response.json()
                logger.debug(
                    "Embeddings request successful",
                    model=payload.get("model"),
                    **metrics.to_dict(),
                )

                return OpenRouterResponse(
                    data=data,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    metrics=metrics,
                )

            except httpx.RequestError as e:
                self._error_count += 1
                logger.error(
                    "Network error fetching embeddings",
                    error=str(e),
                    **metrics.to_dict(),
                )
                raise NetworkError(
                    f"Network error fetching embeddings: {e}", original_error=e
                )


# Legacy functions for backward compatibility
async def fetch_models(api_key: str) -> dict[str, Any]:
    """Legacy function for backward compatibility."""
    import os

    from .config import Settings

    # Temporarily set the environment variable for this call
    original_key = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = api_key
    try:
        settings = Settings()  # type: ignore[call-arg]
    finally:
        # Restore original environment
        if original_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_key
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)
    client = OpenRouterClient(settings)
    try:
        response = await client.fetch_models()
        return response.data
    finally:
        await client.close()


async def chat_completion(
    api_key: str, payload: dict[str, Any], stream: bool = False
) -> Any:
    """Legacy function for backward compatibility."""
    import os

    from .config import Settings

    # Temporarily set the environment variable for this call
    original_key = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = api_key
    try:
        settings = Settings()  # type: ignore[call-arg]
    finally:
        # Restore original environment
        if original_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_key
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)
    client = OpenRouterClient(settings)
    try:
        result = await client.chat_completion(payload, stream)
        if stream:
            return result  # AsyncIterator[bytes]
        else:
            # Type guard to ensure we have OpenRouterResponse
            if isinstance(result, OpenRouterResponse):
                return result.data
            else:
                # This shouldn't happen, but handle gracefully
                return result
    finally:
        if not stream:  # Don't close if streaming
            await client.close()


async def fetch_embeddings(api_key: str, payload: dict) -> dict[str, Any]:
    """Legacy function for backward compatibility."""
    import os

    from .config import Settings

    # Temporarily set the environment variable for this call
    original_key = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = api_key
    try:
        settings = Settings()  # type: ignore[call-arg]
    finally:
        # Restore original environment
        if original_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_key
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)
    client = OpenRouterClient(settings)
    try:
        response = await client.fetch_embeddings(payload)
        return response.data
    finally:
        await client.close()
