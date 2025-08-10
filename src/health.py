"""Health check and monitoring utilities.

This module provides health check endpoints, monitoring capabilities,
and system status reporting for the multi-provider proxy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException

from .providers.base import ProviderType
from .providers.factory import get_factory
from .providers.retry import get_health_manager

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@dataclass
class HealthStatus:
    """Health status information."""

    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: float
    version: str = "0.1.0"
    uptime: float = 0.0
    providers: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "version": self.version,
            "uptime": self.uptime,
            "providers": self.providers or {},
            "metrics": self.metrics or {},
        }


class HealthChecker:
    """Health checker for the proxy system."""

    def __init__(self):
        self.start_time = time.time()
        self._last_check: Dict[ProviderType, float] = {}
        self._check_results: Dict[ProviderType, bool] = {}

    async def check_system_health(self) -> HealthStatus:
        """Check overall system health."""
        current_time = time.time()
        uptime = current_time - self.start_time

        # Check provider health
        provider_health = await self._check_all_providers()

        # Determine overall status
        healthy_providers = sum(1 for status in provider_health.values() if status.get("healthy", False))
        total_providers = len(provider_health)

        if healthy_providers == 0:
            overall_status = "unhealthy"
        elif healthy_providers < total_providers:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        # Get system metrics
        metrics = await self._get_system_metrics()

        return HealthStatus(
            status=overall_status,
            timestamp=current_time,
            uptime=uptime,
            providers=provider_health,
            metrics=metrics,
        )

    async def check_provider_health(self, provider_type: ProviderType) -> Dict[str, Any]:
        """Check health of a specific provider."""
        factory = get_factory()
        health_manager = get_health_manager()

        try:
            provider = factory.get_provider(provider_type)
            if provider is None:
                return {
                    "healthy": False,
                    "status": "not_configured",
                    "error": f"Provider {provider_type.value} not configured",
                    "last_check": time.time(),
                }

            # Simple health check - try to list models
            start_time = time.time()
            await provider.list_models()
            response_time = (time.time() - start_time) * 1000

            self._last_check[provider_type] = time.time()
            self._check_results[provider_type] = True

            return {
                "healthy": True,
                "status": "operational",
                "response_time_ms": response_time,
                "last_check": self._last_check[provider_type],
                "circuit_breaker": health_manager.get_circuit_breaker(provider_type).get_state(),
                "request_count": provider.request_count,
                "error_count": provider.error_count,
                "error_rate": provider.error_rate,
            }

        except Exception as e:
            self._last_check[provider_type] = time.time()
            self._check_results[provider_type] = False

            logger.warning(
                "Provider health check failed",
                provider_type=provider_type.value,
                error=str(e),
                error_type=type(e).__name__,
            )

            return {
                "healthy": False,
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "last_check": self._last_check[provider_type],
            }

    async def _check_all_providers(self) -> Dict[str, Any]:
        """Check health of all configured providers."""
        factory = get_factory()
        provider_health = {}

        # Get all provider instances
        instances = factory.get_all_instances()

        # Check each provider
        tasks = []
        for instance_id, provider in instances.items():
            task = self.check_provider_health(provider.provider_type)
            tasks.append((provider.provider_type.value, task))

        # Execute health checks concurrently
        for provider_name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=10.0)
                provider_health[provider_name] = result
            except asyncio.TimeoutError:
                provider_health[provider_name] = {
                    "healthy": False,
                    "status": "timeout",
                    "error": "Health check timed out",
                    "last_check": time.time(),
                }
            except Exception as e:
                provider_health[provider_name] = {
                    "healthy": False,
                    "status": "error",
                    "error": str(e),
                    "last_check": time.time(),
                }

        return provider_health

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        factory = get_factory()
        instances = factory.get_all_instances()

        total_requests = sum(provider.request_count for provider in instances.values())
        total_errors = sum(provider.error_count for provider in instances.values())

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0.0,
            "active_providers": len(instances),
            "uptime": time.time() - self.start_time,
        }


# Global health checker instance
_health_checker = HealthChecker()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    health_status = await _health_checker.check_system_health()

    if health_status.status == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status.to_dict())

    return health_status.to_dict()


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with provider information."""
    return (await _health_checker.check_system_health()).to_dict()


@router.get("/providers")
async def provider_health_check() -> Dict[str, Any]:
    """Health check for all providers."""
    return await _health_checker._check_all_providers()


@router.get("/providers/{provider_type}")
async def single_provider_health_check(provider_type: str) -> Dict[str, Any]:
    """Health check for a specific provider."""
    try:
        provider_enum = ProviderType(provider_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider type: {provider_type}"
        )

    return await _health_checker.check_provider_health(provider_enum)


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system metrics."""
    return await _health_checker._get_system_metrics()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker