"""Provider-specific metrics collection and analysis.

This module provides detailed metrics collection for AI providers,
including performance tracking, error analysis, and cost monitoring.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import structlog

from .base import ProviderType, ProviderCapability

logger = structlog.get_logger(__name__)


@dataclass
class ProviderMetrics:
    """Comprehensive metrics for a provider."""
    
    provider_type: ProviderType
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Timing metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=dict)
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    
    # Capability metrics
    capability_usage: Dict[ProviderCapability, int] = field(default_factory=dict)
    
    # Model usage
    model_usage: Dict[str, int] = field(default_factory=dict)
    
    # Cost tracking (if available)
    estimated_cost: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    
    # Health status
    is_healthy: bool = True
    last_health_check: Optional[float] = None
    consecutive_failures: int = 0
    
    # Rate limiting
    rate_limit_hits: int = 0
    last_rate_limit: Optional[float] = None

    def record_request(
        self,
        success: bool,
        response_time: float,
        capability: ProviderCapability,
        model: Optional[str] = None,
        error: Optional[str] = None,
        tokens_used: Optional[Dict[str, int]] = None,
    ) -> None:
        """Record a request and update metrics."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1
            
            if error:
                self.error_counts[error] = self.error_counts.get(error, 0) + 1
                self.last_error = error
                self.last_error_time = time.time()
        
        # Update timing metrics
        self.response_times.append(response_time)
        self._update_timing_stats()
        
        # Update capability usage
        self.capability_usage[capability] = self.capability_usage.get(capability, 0) + 1
        
        # Update model usage
        if model:
            self.model_usage[model] = self.model_usage.get(model, 0) + 1
        
        # Update token usage
        if tokens_used:
            for token_type, count in tokens_used.items():
                self.token_usage[token_type] = self.token_usage.get(token_type, 0) + count
        
        # Update health status
        self._update_health_status()

    def _update_timing_stats(self) -> None:
        """Update timing statistics."""
        if not self.response_times:
            return
            
        times = list(self.response_times)
        self.avg_response_time = sum(times) / len(times)
        self.min_response_time = min(times)
        self.max_response_time = max(times)

    def _update_health_status(self) -> None:
        """Update health status based on recent performance."""
        # Consider unhealthy if too many consecutive failures
        if self.consecutive_failures >= 5:
            self.is_healthy = False
        elif self.consecutive_failures == 0 and not self.is_healthy:
            self.is_healthy = True

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        return 100.0 - self.get_success_rate()

    def get_p95_response_time(self) -> float:
        """Get 95th percentile response time."""
        if not self.response_times:
            return 0.0
        
        times = sorted(self.response_times)
        index = int(len(times) * 0.95)
        return times[min(index, len(times) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "provider_type": self.provider_type.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "error_rate": self.get_error_rate(),
            "avg_response_time": self.avg_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0.0,
            "max_response_time": self.max_response_time,
            "p95_response_time": self.get_p95_response_time(),
            "error_counts": dict(self.error_counts),
            "last_error": self.last_error,
            "last_error_time": self.last_error_time,
            "capability_usage": {cap.value: count for cap, count in self.capability_usage.items()},
            "model_usage": dict(self.model_usage),
            "estimated_cost": self.estimated_cost,
            "token_usage": dict(self.token_usage),
            "is_healthy": self.is_healthy,
            "last_health_check": self.last_health_check,
            "consecutive_failures": self.consecutive_failures,
            "rate_limit_hits": self.rate_limit_hits,
            "last_rate_limit": self.last_rate_limit,
        }


class MetricsCollector:
    """Collects and manages metrics for all providers."""
    
    def __init__(self):
        self._provider_metrics: Dict[ProviderType, ProviderMetrics] = {}
        self._global_metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "start_time": time.time(),
        }
        self._lock = asyncio.Lock()

    async def record_provider_request(
        self,
        provider_type: ProviderType,
        success: bool,
        response_time: float,
        capability: ProviderCapability,
        model: Optional[str] = None,
        error: Optional[str] = None,
        tokens_used: Optional[Dict[str, int]] = None,
    ) -> None:
        """Record a provider request."""
        async with self._lock:
            # Get or create provider metrics
            if provider_type not in self._provider_metrics:
                self._provider_metrics[provider_type] = ProviderMetrics(provider_type)
            
            metrics = self._provider_metrics[provider_type]
            metrics.record_request(
                success=success,
                response_time=response_time,
                capability=capability,
                model=model,
                error=error,
                tokens_used=tokens_used,
            )
            
            # Update global metrics
            self._global_metrics["total_requests"] += 1
            if not success:
                self._global_metrics["total_errors"] += 1

    async def get_provider_metrics(self, provider_type: ProviderType) -> Optional[ProviderMetrics]:
        """Get metrics for a specific provider."""
        async with self._lock:
            return self._provider_metrics.get(provider_type)

    async def get_all_provider_metrics(self) -> Dict[ProviderType, ProviderMetrics]:
        """Get metrics for all providers."""
        async with self._lock:
            return self._provider_metrics.copy()

    async def get_global_metrics(self) -> Dict[str, Any]:
        """Get global system metrics."""
        async with self._lock:
            uptime = time.time() - self._global_metrics["start_time"]
            total_requests = self._global_metrics["total_requests"]
            total_errors = self._global_metrics["total_errors"]
            
            return {
                "uptime_seconds": uptime,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "global_success_rate": ((total_requests - total_errors) / total_requests * 100) if total_requests > 0 else 100.0,
                "requests_per_second": total_requests / uptime if uptime > 0 else 0.0,
                "active_providers": len(self._provider_metrics),
                "healthy_providers": sum(1 for m in self._provider_metrics.values() if m.is_healthy),
            }

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all providers."""
        async with self._lock:
            unhealthy_providers: List[str] = []
            provider_health: Dict[str, Any] = {}
            healthy_count = 0
            summary = {
                "overall_healthy": True,
                "provider_health": provider_health,
                "unhealthy_providers": unhealthy_providers,
                "total_providers": len(self._provider_metrics),
                "healthy_count": healthy_count,
            }
            
            for provider_type, metrics in self._provider_metrics.items():
                is_healthy = metrics.is_healthy
                provider_health[provider_type.value] = {
                    "healthy": is_healthy,
                    "success_rate": metrics.get_success_rate(),
                    "consecutive_failures": metrics.consecutive_failures,
                    "last_error": metrics.last_error,
                }

                if is_healthy:
                    healthy_count += 1
                else:
                    unhealthy_providers.append(provider_type.value)
                    summary["overall_healthy"] = False

            # Update the summary with final counts
            summary["healthy_count"] = healthy_count
            
            return summary

    async def reset_metrics(self, provider_type: Optional[ProviderType] = None) -> None:
        """Reset metrics for a provider or all providers."""
        async with self._lock:
            if provider_type:
                if provider_type in self._provider_metrics:
                    self._provider_metrics[provider_type] = ProviderMetrics(provider_type)
            else:
                self._provider_metrics.clear()
                self._global_metrics = {
                    "total_requests": 0,
                    "total_errors": 0,
                    "start_time": time.time(),
                }

    async def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring systems."""
        async with self._lock:
            return {
                "timestamp": time.time(),
                "global_metrics": await self.get_global_metrics(),
                "provider_metrics": {
                    provider_type.value: metrics.to_dict()
                    for provider_type, metrics in self._provider_metrics.items()
                },
                "health_summary": await self.get_health_summary(),
            }


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
