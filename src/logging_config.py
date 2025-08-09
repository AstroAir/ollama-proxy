"""Enhanced logging configuration with structured logging and monitoring."""

from __future__ import annotations

import logging
import sys
from typing import Any, List

import structlog
from structlog.typing import FilteringBoundLogger

from .config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure comprehensive structured logging."""

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format,
        stream=sys.stdout,
    )

    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add appropriate renderer based on environment
    if settings.debug:
        # Pretty console output for development
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestLogger:
    """Request logging utilities."""

    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger

    def log_request_start(
        self, method: str, path: str, query: str | None = None, **kwargs: Any
    ) -> None:
        """Log request start."""
        self.logger.info(
            "Request started", method=method, path=path, query=query, **kwargs
        )

    def log_request_success(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log successful request completion."""
        self.logger.info(
            "Request completed successfully",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_request_error(
        self,
        method: str,
        path: str,
        error: str,
        status_code: int | None = None,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log request error."""
        self.logger.error(
            "Request failed",
            method=method,
            path=path,
            error=error,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )


class OpenRouterLogger:
    """OpenRouter API logging utilities."""

    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger

    def log_api_call(
        self,
        endpoint: str,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> None:
        """Log OpenRouter API call."""
        self.logger.info(
            "OpenRouter API call",
            endpoint=endpoint,
            model=model,
            stream=stream,
            **kwargs,
        )

    def log_api_success(
        self,
        endpoint: str,
        model: str | None = None,
        duration_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log successful OpenRouter API call."""
        self.logger.info(
            "OpenRouter API call successful",
            endpoint=endpoint,
            model=model,
            duration_ms=duration_ms,
            **kwargs,
        )

    def log_api_error(
        self,
        endpoint: str,
        error: str,
        status_code: int | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log OpenRouter API error."""
        self.logger.error(
            "OpenRouter API error",
            endpoint=endpoint,
            error=error,
            status_code=status_code,
            model=model,
            **kwargs,
        )


class ModelLogger:
    """Model operation logging utilities."""

    def __init__(self, logger: FilteringBoundLogger):
        self.logger = logger

    def log_model_resolution(
        self, requested: str, resolved: str | None, openrouter_id: str | None = None
    ) -> None:
        """Log model name resolution."""
        if resolved:
            self.logger.debug(
                "Model resolved successfully",
                requested=requested,
                resolved=resolved,
                openrouter_id=openrouter_id,
            )
        else:
            self.logger.warning("Model resolution failed", requested=requested)

    def log_model_filter(self, model: str, allowed: bool, filter_size: int) -> None:
        """Log model filtering result."""
        if allowed:
            self.logger.debug(
                "Model allowed by filter", model=model, filter_size=filter_size
            )
        else:
            self.logger.warning(
                "Model blocked by filter", model=model, filter_size=filter_size
            )

    def log_models_loaded(
        self, total_count: int, filtered_count: int | None = None
    ) -> None:
        """Log model loading results."""
        self.logger.info(
            "Models loaded",
            total_count=total_count,
            filtered_count=filtered_count or total_count,
        )


# Convenience functions for getting specialized loggers
def get_request_logger() -> RequestLogger:
    """Get a request logger instance."""
    return RequestLogger(get_logger("request"))


def get_openrouter_logger() -> OpenRouterLogger:
    """Get an OpenRouter logger instance."""
    return OpenRouterLogger(get_logger("openrouter"))


def get_model_logger() -> ModelLogger:
    """Get a model logger instance."""
    return ModelLogger(get_logger("model"))
