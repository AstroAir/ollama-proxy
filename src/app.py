"""Main application factory and dependency injection setup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import router
from .config import AppState, Settings, get_settings
from .exceptions import NetworkError, OpenRouterError
from .openrouter import OpenRouterClient, OpenRouterResponse
from .utils import build_ollama_to_openrouter_map, build_openrouter_to_ollama_map


def setup_logging(settings: Settings) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format,
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    settings = get_settings()
    setup_logging(settings)

    logger = structlog.get_logger(__name__)
    logger.info("Starting ollama-proxy application", version="0.1.0")

    # Initialize OpenRouter client
    openrouter_client = OpenRouterClient(settings)

    try:
        # Fetch models and build mappings
        logger.info("Fetching models from OpenRouter...")
        model_response = await openrouter_client.fetch_models()
        all_models = model_response.data.get("data", [])

        # Build model mappings
        ollama_to_openrouter_map = build_ollama_to_openrouter_map(all_models)
        openrouter_to_ollama_map = build_openrouter_to_ollama_map(all_models)

        # Initialize application state
        app_state = AppState.create(settings)
        app_state.update_models(
            all_models, ollama_to_openrouter_map, openrouter_to_ollama_map
        )
        app_state.openrouter_client = openrouter_client

        # Store state in app
        app.state.app_state = app_state

        logger.info(
            "Application initialized successfully",
            model_count=len(all_models),
            mapping_count=len(ollama_to_openrouter_map),
            filter_count=len(app_state.model_filter.models),
        )

        yield

    except (OpenRouterError, NetworkError) as e:
        logger.error(
            "Failed to initialize OpenRouter client",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
    except Exception as e:
        logger.error(
            "Failed to initialize application",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
    finally:
        # Cleanup
        await openrouter_client.close()
        logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Ollama Proxy",
        description="A proxy server that translates Ollama API calls to OpenRouter",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = structlog.get_logger(__name__)

        logger.info(
            "Incoming request",
            method=request.method,
            path=request.url.path,
            query=str(request.url.query) if request.url.query else None,
        )

        try:
            response = await call_next(request)

            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )

            return response

        except Exception as exc:
            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(exc),
            )
            raise

    # Add global exception handlers
    @app.exception_handler(OpenRouterError)
    async def openrouter_error_handler(request: Request, exc: OpenRouterError):
        logger = structlog.get_logger(__name__)
        logger.error(
            "OpenRouter API error",
            path=request.url.path,
            status_code=exc.status_code,
            message=str(exc),
        )

        return JSONResponse(
            status_code=exc.status_code or 500,
            content={"error": str(exc), "type": "openrouter_error"},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger = structlog.get_logger(__name__)
        logger.error(
            "Unhandled exception", path=request.url.path, error=str(exc), exc_info=True
        )

        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error",
                     "type": "internal_error"},
        )

    # Include API routes
    app.include_router(router)

    return app


# Create the app instance
app = create_app()
