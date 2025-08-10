"""Main application factory with multi-provider support.

This module creates the main FastAPI application with support for multiple
AI providers, enhanced error handling, health checks, and monitoring.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .exceptions import ProxyError
from .health import router as health_router
from .multi_provider_api import MultiProviderAPI
from .multi_provider_config import MultiProviderSettings
from .providers.base import ProviderType
from .providers.factory import get_factory

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ollama-proxy with multi-provider support")

    # Initialize providers
    settings = app.state.settings
    multi_api = app.state.multi_api
    await multi_api.initialize()

    # Start health check background task
    health_check_task = asyncio.create_task(periodic_health_check(settings))
    app.state.health_check_task = health_check_task

    logger.info("Ollama-proxy startup complete")

    yield

    # Shutdown
    logger.info("Shutting down ollama-proxy")

    # Cancel health check task
    if hasattr(app.state, 'health_check_task'):
        app.state.health_check_task.cancel()
        try:
            await app.state.health_check_task
        except asyncio.CancelledError:
            pass

    # Close all provider connections
    factory = get_factory()
    await factory.close_all()

    logger.info("Ollama-proxy shutdown complete")


async def periodic_health_check(settings: MultiProviderSettings) -> None:
    """Periodic health check for all providers."""
    from .health import get_health_checker

    health_checker = get_health_checker()

    while True:
        try:
            await asyncio.sleep(settings.health_check_interval)

            # Perform health check
            health_status = await health_checker.check_system_health()

            providers = health_status.providers or {}
            logger.debug(
                "Periodic health check completed",
                status=health_status.status,
                healthy_providers=sum(
                    1 for p in providers.values()
                    if p.get("healthy", False)
                ),
                total_providers=len(providers),
            )

        except asyncio.CancelledError:
            logger.info("Health check task cancelled")
            break
        except Exception as e:
            logger.error(
                "Error in periodic health check",
                error=str(e),
                error_type=type(e).__name__,
            )


def create_app(settings: Optional[MultiProviderSettings] = None) -> FastAPI:
    """Create FastAPI application with multi-provider support."""
    # Use default settings if none provided
    if settings is None:
        settings = MultiProviderSettings()

    app = FastAPI(
        title="Ollama Proxy",
        description="Multi-provider AI proxy with intelligent routing and fallback",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Store settings and multi-provider API in app state
    app.state.settings = settings
    app.state.multi_api = MultiProviderAPI(settings)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add health check routes
    app.include_router(health_router)

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
    @app.exception_handler(ProxyError)
    async def proxy_error_handler(request: Request, exc: ProxyError):
        logger = structlog.get_logger(__name__)
        logger.error(
            "Proxy error",
            path=request.url.path,
            status_code=exc.status_code,
            error_type=exc.error_type.value if exc.error_type else None,
            message=str(exc),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "type": exc.error_type.value if exc.error_type else "proxy_error",
            },
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

    # Basic compatibility routes
    @app.get("/", response_class=PlainTextResponse)
    async def root() -> str:
        """Root endpoint for basic health check."""
        return "Ollama is running"

    @app.get("/api/version")
    async def get_version() -> Dict[str, str]:
        """Get API version information."""
        return {"version": "0.2.0"}

    # Multi-provider API endpoints
    @app.get("/api/tags")
    async def list_models() -> Dict[str, Any]:
        """List all available models from all providers."""
        multi_api = app.state.multi_api
        return await multi_api.list_models()

    @app.get("/api/tags/{provider_type}")
    async def list_provider_models(provider_type: str) -> Dict[str, Any]:
        """List models from a specific provider."""
        try:
            from .providers.base import ProviderType
            provider_enum = ProviderType(provider_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider type: {provider_type}"
            )

        multi_api = app.state.multi_api
        return await multi_api.list_models(provider_enum)

    @app.post("/api/chat")
    async def chat_completion(request: Request) -> Dict[str, Any]:
        """Chat completion with multi-provider support."""
        from .models import OllamaChatRequest

        try:
            body = await request.json()
            chat_request = OllamaChatRequest(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        # Check for provider preference in headers
        preferred_provider = None
        provider_header = request.headers.get("X-Provider")
        if provider_header:
            try:
                preferred_provider = ProviderType(provider_header.lower())
            except ValueError:
                logger.warning(
                    "Invalid provider specified in header",
                    provider=provider_header,
                )

        multi_api = app.state.multi_api
        return await multi_api.chat_completion(chat_request, preferred_provider)

    @app.post("/api/generate")
    async def generate_completion(request: Request) -> Dict[str, Any]:
        """Text generation with multi-provider support."""
        from .models import OllamaGenerateRequest

        try:
            body = await request.json()
            generate_request = OllamaGenerateRequest(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        # Check for provider preference in headers
        preferred_provider = None
        provider_header = request.headers.get("X-Provider")
        if provider_header:
            try:
                preferred_provider = ProviderType(provider_header.lower())
            except ValueError:
                logger.warning(
                    "Invalid provider specified in header",
                    provider=provider_header,
                )

        multi_api = app.state.multi_api
        return await multi_api.generate_completion(generate_request, preferred_provider)

    @app.post("/api/embeddings")
    async def create_embeddings(request: Request) -> Dict[str, Any]:
        """Embeddings creation with multi-provider support."""
        from .models import OllamaEmbeddingsRequest

        try:
            body = await request.json()
            embeddings_request = OllamaEmbeddingsRequest(**body)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

        # Check for provider preference in headers
        preferred_provider = None
        provider_header = request.headers.get("X-Provider")
        if provider_header:
            try:
                preferred_provider = ProviderType(provider_header.lower())
            except ValueError:
                logger.warning(
                    "Invalid provider specified in header",
                    provider=provider_header,
                )

        multi_api = app.state.multi_api
        return await multi_api.create_embeddings(embeddings_request, preferred_provider)

    @app.get("/api/providers")
    async def get_provider_info() -> Dict[str, Any]:
        """Get information about all configured providers."""
        multi_api = app.state.multi_api
        return await multi_api.get_provider_stats()

    @app.get("/api/providers/{provider_type}/stats")
    async def get_provider_stats(provider_type: str) -> Dict[str, Any]:
        """Get statistics for a specific provider."""
        try:
            provider_enum = ProviderType(provider_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider type: {provider_type}"
            )

        factory = get_factory()
        provider = factory.get_provider(provider_enum)

        if provider is None:
            raise HTTPException(
                status_code=404,
                detail=f"Provider {provider_type} not configured"
            )

        return {
            "provider_type": provider_type,
            "request_count": provider.request_count,
            "error_count": provider.error_count,
            "error_rate": provider.error_rate,
            "capabilities": list(provider.capabilities),
        }

    return app
