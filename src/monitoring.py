"""Enhanced monitoring and metrics collection with modern Python patterns."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional, TypeAlias

import structlog

# Type aliases for better code clarity
MetricName: TypeAlias = str
MetricValue: TypeAlias = float
Labels: TypeAlias = Dict[str, str]
Timestamp: TypeAlias = float

logger = structlog.get_logger(__name__)


class MetricType(StrEnum):
    """Types of metrics with modern enum features."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

    @property
    def is_cumulative(self) -> bool:
        """Check if metric type is cumulative using pattern matching."""
        match self:
            case MetricType.COUNTER | MetricType.HISTOGRAM:
                return True
            case MetricType.GAUGE | MetricType.SUMMARY:
                return False
            case _:
                return False


@dataclass(slots=True, frozen=True)
class MetricPoint:
    """Individual metric data point with enhanced features."""

    name: MetricName
    value: MetricValue
    labels: Labels
    timestamp: Timestamp = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "type": self.metric_type.value,
        }

    @property
    def age_seconds(self) -> float:
        """Get age of metric point in seconds."""
        return time.time() - self.timestamp


@dataclass(slots=True)
class PerformanceStats:
    """Performance statistics with pattern matching analysis."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0

    # Request size statistics
    total_request_bytes: int = 0
    total_response_bytes: int = 0

    # Error tracking
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Recent performance samples (for percentile calculation)
    recent_durations: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def add_request(
        self,
        duration_ms: float,
        success: bool,
        request_bytes: int = 0,
        response_bytes: int = 0,
        error_type: Optional[str] = None,
    ) -> None:
        """Add request statistics with comprehensive tracking."""
        self.total_requests += 1
        self.total_duration_ms += duration_ms
        self.recent_durations.append(duration_ms)

        # Update min/max
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)

        # Track success/failure
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] += 1

        # Track bytes
        self.total_request_bytes += request_bytes
        self.total_response_bytes += response_bytes

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def average_duration_ms(self) -> float:
        """Calculate average request duration."""
        if self.total_requests == 0:
            return 0.0
        return self.total_duration_ms / self.total_requests

    @property
    def requests_per_second(self) -> float:
        """Estimate current requests per second (rough approximation)."""
        if not self.recent_durations:
            return 0.0
        # Simple approximation based on recent samples
        return min(len(self.recent_durations), 60)  # Cap at reasonable value

    def get_percentile(self, percentile: float) -> float:
        """Calculate percentile from recent durations."""
        if not self.recent_durations:
            return 0.0

        sorted_durations = sorted(self.recent_durations)
        index = int(len(sorted_durations) * (percentile / 100))
        return sorted_durations[min(index, len(sorted_durations) - 1)]

    def get_performance_category(self) -> str:
        """Categorize overall performance using pattern matching."""
        avg_duration = self.average_duration_ms
        success_rate = self.success_rate

        match (avg_duration, success_rate):
            case (duration, rate) if duration < 100 and rate > 99:
                return "excellent"
            case (duration, rate) if duration < 500 and rate > 95:
                return "good"
            case (duration, rate) if duration < 1000 and rate > 90:
                return "acceptable"
            case (duration, rate) if duration < 5000 and rate > 80:
                return "poor"
            case _:
                return "critical"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring systems."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_duration_ms": self.average_duration_ms,
            "min_duration_ms": (
                self.min_duration_ms if self.min_duration_ms != float("inf") else 0
            ),
            "max_duration_ms": self.max_duration_ms,
            "p50_duration_ms": self.get_percentile(50),
            "p95_duration_ms": self.get_percentile(95),
            "p99_duration_ms": self.get_percentile(99),
            "requests_per_second": self.requests_per_second,
            "total_request_bytes": self.total_request_bytes,
            "total_response_bytes": self.total_response_bytes,
            "error_counts": dict(self.error_counts),
            "performance_category": self.get_performance_category(),
        }


class MetricsCollector:
    """Enhanced metrics collector with modern patterns and thread safety."""

    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self._metrics: List[MetricPoint] = []
        self._stats_by_endpoint: Dict[str, PerformanceStats] = defaultdict(
            PerformanceStats
        )
        self._global_stats = PerformanceStats()
        self._lock = threading.RLock()

        logger.info("Metrics collector initialized", max_metrics=max_metrics)

    def record_metric(
        self,
        name: MetricName,
        value: MetricValue,
        labels: Optional[Labels] = None,
        metric_type: MetricType = MetricType.GAUGE,
    ) -> None:
        """Record a metric point with thread safety."""
        labels = labels or {}
        point = MetricPoint(
            name=name, value=value, labels=labels, metric_type=metric_type
        )

        with self._lock:
            self._metrics.append(point)

            # Prevent memory bloat
            if len(self._metrics) > self.max_metrics:
                # Remove oldest 10% of metrics
                remove_count = self.max_metrics // 10
                self._metrics = self._metrics[remove_count:]

        logger.debug("Metric recorded", name=name, value=value, labels=labels)

    def record_request(
        self,
        endpoint: str,
        duration_ms: float,
        success: bool,
        request_bytes: int = 0,
        response_bytes: int = 0,
        error_type: Optional[str] = None,
        labels: Optional[Labels] = None,
    ) -> None:
        """Record request statistics with comprehensive tracking."""
        with self._lock:
            # Update endpoint-specific stats
            self._stats_by_endpoint[endpoint].add_request(
                duration_ms=duration_ms,
                success=success,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                error_type=error_type,
            )

            # Update global stats
            self._global_stats.add_request(
                duration_ms=duration_ms,
                success=success,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                error_type=error_type,
            )

        # Record as metrics for external systems
        base_labels = {"endpoint": endpoint}
        if labels:
            base_labels.update(labels)

        self.record_metric(
            "request_duration_ms", duration_ms, base_labels, MetricType.HISTOGRAM
        )
        self.record_metric("request_total", 1, base_labels, MetricType.COUNTER)

        if success:
            self.record_metric(
                "request_success_total", 1, base_labels, MetricType.COUNTER
            )
        else:
            error_labels = {**base_labels, "error_type": error_type or "unknown"}
            self.record_metric(
                "request_error_total", 1, error_labels, MetricType.COUNTER
            )

    def get_metrics(self, max_age_seconds: Optional[float] = None) -> List[MetricPoint]:
        """Get metrics with optional age filtering."""
        with self._lock:
            if max_age_seconds is None:
                return self._metrics.copy()

            cutoff_time = time.time() - max_age_seconds
            return [m for m in self._metrics if m.timestamp >= cutoff_time]

    def get_endpoint_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics for specific endpoint or all endpoints."""
        with self._lock:
            if endpoint:
                return self._stats_by_endpoint[endpoint].to_dict()

            return {
                "global": self._global_stats.to_dict(),
                "by_endpoint": {
                    ep: stats.to_dict() for ep, stats in self._stats_by_endpoint.items()
                },
            }

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status using pattern matching."""
        with self._lock:
            global_category = self._global_stats.get_performance_category()

            # Determine overall health
            match global_category:
                case "excellent" | "good":
                    health_status = "healthy"
                case "acceptable":
                    health_status = "degraded"
                case "poor":
                    health_status = "unhealthy"
                case "critical":
                    health_status = "critical"
                case _:
                    health_status = "unknown"

            return {
                "status": health_status,
                "performance_category": global_category,
                "total_requests": self._global_stats.total_requests,
                "success_rate": self._global_stats.success_rate,
                "average_duration_ms": self._global_stats.average_duration_ms,
                "active_endpoints": len(self._stats_by_endpoint),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def reset_stats(self) -> None:
        """Reset all statistics (useful for testing)."""
        with self._lock:
            self._metrics.clear()
            self._stats_by_endpoint.clear()
            self._global_stats = PerformanceStats()

        logger.info("Metrics collector reset")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


@asynccontextmanager
async def track_request(endpoint: str, labels: Optional[Labels] = None):
    """Async context manager for tracking request performance."""
    collector = get_metrics_collector()
    start_time = time.time()
    success = False
    error_type = None

    try:
        yield
        success = True
    except Exception as e:
        error_type = type(e).__name__
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        collector.record_request(
            endpoint=endpoint,
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
            labels=labels,
        )


def record_metric(
    name: MetricName,
    value: MetricValue,
    labels: Optional[Labels] = None,
    metric_type: MetricType = MetricType.GAUGE,
) -> None:
    """Convenience function to record a metric."""
    collector = get_metrics_collector()
    collector.record_metric(name, value, labels, metric_type)
