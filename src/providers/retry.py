"""Enhanced error handling and retry logic for AI providers.

This module provides sophisticated retry mechanisms, circuit breakers,
and error handling strategies for improved reliability across multiple
AI providers.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import httpx
import structlog

from .base import ProviderError, ProviderType

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RetryStrategy(StrEnum):
    """Retry strategies for failed requests."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


class CircuitBreakerState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_status_codes: frozenset[int] = frozenset([429, 500, 502, 503, 504])
    retryable_exceptions: frozenset[type[Exception]] = frozenset([
        httpx.TimeoutException,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.NetworkError,
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open state
    request_volume_threshold: int = 10


class CircuitBreaker:
    """Circuit breaker for provider reliability."""

    def __init__(self, config: CircuitBreakerConfig, provider_type: ProviderType):
        self.config = config
        self.provider_type = provider_type
        self.state = CircuitBreakerState.CLOSED

        # Counters
        self.failure_count = 0
        self.success_count = 0
        self.request_count = 0
        self.last_failure_time = 0.0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise ProviderError(
                        message=f"Circuit breaker open for {self.provider_type.value}",
                        provider_type=self.provider_type,
                        status_code=503,
                    )
                else:
                    # Transition to half-open
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(
                        "Circuit breaker transitioning to half-open",
                        provider_type=self.provider_type.value,
                    )

        try:
            result = await func()
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful request."""
        async with self._lock:
            self.request_count += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(
                        "Circuit breaker closed (recovered)",
                        provider_type=self.provider_type.value,
                    )
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)

    async def _on_failure(self) -> None:
        """Handle failed request."""
        async with self._lock:
            self.request_count += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            if (
                self.state == CircuitBreakerState.CLOSED
                and self.failure_count >= self.config.failure_threshold
                and self.request_count >= self.config.request_volume_threshold
            ):
                self.state = CircuitBreakerState.OPEN
                logger.warning(
                    "Circuit breaker opened",
                    provider_type=self.provider_type.value,
                    failure_count=self.failure_count,
                )
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(
                    "Circuit breaker reopened from half-open",
                    provider_type=self.provider_type.value,
                )

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "request_count": self.request_count,
            "last_failure_time": self.last_failure_time,
        }


class RetryHandler:
    """Handles retry logic for provider requests."""

    def __init__(self, config: RetryConfig):
        self.config = config

    async def retry_with_backoff(
        self,
        func: Callable[[], Awaitable[T]],
        provider_type: ProviderType,
    ) -> T:
        """Execute function with retry and backoff logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.debug(
                        "Retrying request after delay",
                        provider_type=provider_type.value,
                        attempt=attempt,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)

                return await func()

            except Exception as e:
                last_exception = e

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.debug(
                        "Error is not retryable, giving up",
                        provider_type=provider_type.value,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

                if attempt == self.config.max_retries:
                    logger.error(
                        "Max retries exceeded",
                        provider_type=provider_type.value,
                        attempts=attempt + 1,
                        error=str(e),
                    )
                    break

                logger.warning(
                    "Request failed, will retry",
                    provider_type=provider_type.value,
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__,
                )

        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise ProviderError(
                message=f"All retries failed for {provider_type.value}",
                provider_type=provider_type,
                status_code=500,
            )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        else:  # IMMEDIATE
            delay = 0.0

        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter and delay > 0:
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter

        return delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable."""
        # Check exception type
        for retryable_type in self.config.retryable_exceptions:
            if isinstance(error, retryable_type):
                return True

        # Check status code for HTTP errors
        if isinstance(error, ProviderError):
            return error.status_code in self.config.retryable_status_codes

        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in self.config.retryable_status_codes

        return False


class ProviderHealthManager:
    """Manages health status and circuit breakers for providers."""

    def __init__(self):
        self._circuit_breakers: Dict[ProviderType, CircuitBreaker] = {}
        self._retry_handlers: Dict[ProviderType, RetryHandler] = {}
        self._health_status: Dict[ProviderType, bool] = {}

    def get_circuit_breaker(
        self,
        provider_type: ProviderType,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create circuit breaker for provider."""
        if provider_type not in self._circuit_breakers:
            cb_config = config or CircuitBreakerConfig()
            self._circuit_breakers[provider_type] = CircuitBreaker(cb_config, provider_type)

        return self._circuit_breakers[provider_type]

    def get_retry_handler(
        self,
        provider_type: ProviderType,
        config: Optional[RetryConfig] = None,
    ) -> RetryHandler:
        """Get or create retry handler for provider."""
        if provider_type not in self._retry_handlers:
            retry_config = config or RetryConfig()
            self._retry_handlers[provider_type] = RetryHandler(retry_config)

        return self._retry_handlers[provider_type]

    async def execute_with_protection(
        self,
        func: Callable[[], Awaitable[T]],
        provider_type: ProviderType,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> T:
        """Execute function with both retry and circuit breaker protection."""
        circuit_breaker = self.get_circuit_breaker(provider_type, circuit_breaker_config)
        retry_handler = self.get_retry_handler(provider_type, retry_config)

        async def protected_func() -> T:
            return await retry_handler.retry_with_backoff(func, provider_type)

        return await circuit_breaker.call(protected_func)

    def is_healthy(self, provider_type: ProviderType) -> bool:
        """Check if provider is healthy."""
        if provider_type in self._circuit_breakers:
            cb = self._circuit_breakers[provider_type]
            return cb.state != CircuitBreakerState.OPEN

        return self._health_status.get(provider_type, True)

    def get_all_health_status(self) -> Dict[ProviderType, Dict[str, Any]]:
        """Get health status for all providers."""
        status = {}

        for provider_type in ProviderType:
            cb_state = None
            if provider_type in self._circuit_breakers:
                cb = self._circuit_breakers[provider_type]
                cb_state = cb.get_state()

            status[provider_type] = {
                "healthy": self.is_healthy(provider_type),
                "circuit_breaker": cb_state,
            }

        return status


# Global health manager instance
_health_manager = ProviderHealthManager()


def get_health_manager() -> ProviderHealthManager:
    """Get the global health manager instance."""
    return _health_manager