"""Tests for monitoring and metrics collection."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from src.monitoring import (
    MetricPoint,
    MetricType,
    MetricsCollector,
    PerformanceStats,
    get_metrics_collector,
    record_metric,
    track_request,
)


class TestMetricType:
    """Test MetricType enum."""

    def test_metric_type_values(self):
        """Test metric type enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"

    def test_is_cumulative(self):
        """Test is_cumulative property."""
        assert MetricType.COUNTER.is_cumulative is True
        assert MetricType.HISTOGRAM.is_cumulative is True
        assert MetricType.GAUGE.is_cumulative is False
        assert MetricType.SUMMARY.is_cumulative is False


class TestMetricPoint:
    """Test MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating a metric point."""
        labels = {"endpoint": "/api/test", "method": "GET"}
        point = MetricPoint(
            name="test_metric",
            value=42.0,
            labels=labels,
            metric_type=MetricType.COUNTER
        )

        assert point.name == "test_metric"
        assert point.value == 42.0
        assert point.labels == labels
        assert point.metric_type == MetricType.COUNTER
        assert isinstance(point.timestamp, float)

    def test_metric_point_to_dict(self):
        """Test converting metric point to dictionary."""
        labels = {"test": "value"}
        point = MetricPoint(
            name="test_metric",
            value=123.45,
            labels=labels,
            timestamp=1234567890.0,
            metric_type=MetricType.GAUGE
        )

        result = point.to_dict()
        expected = {
            "name": "test_metric",
            "value": 123.45,
            "labels": labels,
            "timestamp": 1234567890.0,
            "type": "gauge"
        }
        assert result == expected

    def test_metric_point_age_seconds(self):
        """Test age calculation."""
        with patch("time.time", return_value=1000.0):
            point = MetricPoint(
                name="test",
                value=1.0,
                labels={},
                timestamp=950.0
            )

        with patch("time.time", return_value=1010.0):
            assert point.age_seconds == 60.0


class TestPerformanceStats:
    """Test PerformanceStats dataclass."""

    def test_performance_stats_initialization(self):
        """Test performance stats initialization."""
        stats = PerformanceStats()
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_duration_ms == 0.0
        assert stats.min_duration_ms == float("inf")
        assert stats.max_duration_ms == 0.0

    def test_add_successful_request(self):
        """Test adding successful request."""
        stats = PerformanceStats()
        stats.add_request(
            duration_ms=100.0,
            success=True,
            request_bytes=1024,
            response_bytes=2048
        )

        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.failed_requests == 0
        assert stats.total_duration_ms == 100.0
        assert stats.min_duration_ms == 100.0
        assert stats.max_duration_ms == 100.0
        assert stats.total_request_bytes == 1024
        assert stats.total_response_bytes == 2048

    def test_add_failed_request(self):
        """Test adding failed request."""
        stats = PerformanceStats()
        stats.add_request(
            duration_ms=250.0,
            success=False,
            error_type="NetworkError"
        )

        assert stats.total_requests == 1
        assert stats.successful_requests == 0
        assert stats.failed_requests == 1
        assert stats.error_counts["NetworkError"] == 1

    def test_success_rate(self):
        """Test success rate calculation."""
        stats = PerformanceStats()

        # No requests
        assert stats.success_rate == 0.0

        # Add some requests
        stats.add_request(100.0, True)
        stats.add_request(200.0, True)
        stats.add_request(150.0, False)

        assert stats.success_rate == 66.66666666666666

    def test_average_duration(self):
        """Test average duration calculation."""
        stats = PerformanceStats()

        # No requests
        assert stats.average_duration_ms == 0.0

        # Add requests
        stats.add_request(100.0, True)
        stats.add_request(200.0, True)
        stats.add_request(300.0, False)

        assert stats.average_duration_ms == 200.0

    def test_get_percentile(self):
        """Test percentile calculation."""
        stats = PerformanceStats()

        # No data
        assert stats.get_percentile(50) == 0.0

        # Add data
        for duration in [100, 200, 300, 400, 500]:
            stats.add_request(float(duration), True)

        assert stats.get_percentile(50) == 300.0  # Median
        assert stats.get_percentile(90) == 500.0

    def test_performance_category(self):
        """Test performance categorization."""
        stats = PerformanceStats()

        # Excellent performance
        stats.add_request(50.0, True)
        assert stats.get_performance_category() == "excellent"

        # Reset and test good performance
        stats = PerformanceStats()
        stats.add_request(300.0, True)
        assert stats.get_performance_category() == "good"

        # Reset and test poor performance
        stats = PerformanceStats()
        stats.add_request(2000.0, True)
        stats.add_request(2000.0, False)  # Lower success rate
        # With 50% success rate and high latency, this should be "critical"
        assert stats.get_performance_category() == "critical"


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_metrics=1000)
        assert collector.max_metrics == 1000
        assert len(collector._metrics) == 0

    def test_record_metric(self):
        """Test recording a metric."""
        collector = MetricsCollector()
        labels = {"endpoint": "/test"}

        collector.record_metric(
            "test_metric",
            42.0,
            labels,
            MetricType.COUNTER
        )

        assert len(collector._metrics) == 1
        metric = collector._metrics[0]
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.labels == labels
        assert metric.metric_type == MetricType.COUNTER

    def test_record_request(self):
        """Test recording request statistics."""
        collector = MetricsCollector()

        collector.record_request(
            endpoint="/api/test",
            duration_ms=150.0,
            success=True,
            request_bytes=1024,
            response_bytes=2048
        )

        # Check endpoint stats
        stats = collector.get_endpoint_stats("/api/test")
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["average_duration_ms"] == 150.0

    def test_get_metrics_with_age_filter(self):
        """Test getting metrics with age filtering."""
        collector = MetricsCollector()

        # Manually create metric points with specific timestamps
        old_metric = MetricPoint(
            name="old_metric",
            value=1.0,
            labels={},
            timestamp=1000.0,
            metric_type=MetricType.GAUGE
        )
        new_metric = MetricPoint(
            name="new_metric",
            value=2.0,
            labels={},
            timestamp=1100.0,
            metric_type=MetricType.GAUGE
        )

        # Add metrics directly to the collector
        collector._metrics.extend([old_metric, new_metric])

        # Get recent metrics (last 50 seconds from time 1120)
        # old_metric at 1000 is 120 seconds old (too old)
        # new_metric at 1100 is 20 seconds old (recent)
        with patch("time.time", return_value=1120.0):
            recent_metrics = collector.get_metrics(max_age_seconds=50)

        # Filter by name to find our specific metrics
        old_metrics = [m for m in recent_metrics if m.name == "old_metric"]
        new_metrics = [m for m in recent_metrics if m.name == "new_metric"]

        assert len(old_metrics) == 0  # Should be filtered out
        assert len(new_metrics) == 1  # Should be included

    def test_memory_management(self):
        """Test memory management with max_metrics limit."""
        collector = MetricsCollector(max_metrics=10)

        # Add more metrics than the limit
        for i in range(15):
            collector.record_metric(f"metric_{i}", float(i))

        # Should have removed oldest metrics
        assert len(collector._metrics) <= 10

    def test_get_health_status(self):
        """Test health status determination."""
        collector = MetricsCollector()

        # Add some good performance data
        collector.record_request("/test", 50.0, True)
        collector.record_request("/test", 60.0, True)

        health = collector.get_health_status()
        assert health["status"] == "healthy"
        assert health["performance_category"] == "excellent"
        assert health["total_requests"] == 2
        assert health["success_rate"] == 100.0

    def test_reset_stats(self):
        """Test resetting statistics."""
        collector = MetricsCollector()

        # Add some data
        collector.record_metric("test", 1.0)
        collector.record_request("/test", 100.0, True)

        # Reset
        collector.reset_stats()

        assert len(collector._metrics) == 0
        stats = collector.get_endpoint_stats()
        assert stats["global"]["total_requests"] == 0


class TestGlobalFunctions:
    """Test global utility functions."""

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2

    def test_record_metric_function(self):
        """Test global record_metric function."""
        # Reset global collector
        import src.monitoring
        src.monitoring._metrics_collector = None

        record_metric("test_metric", 123.0, {"test": "value"})

        collector = get_metrics_collector()
        metrics = collector.get_metrics()
        assert len(metrics) >= 1

        # Find our metric
        test_metrics = [m for m in metrics if m.name == "test_metric"]
        assert len(test_metrics) == 1
        assert test_metrics[0].value == 123.0

    @pytest.mark.asyncio
    async def test_track_request_success(self):
        """Test track_request context manager for successful request."""
        # Reset global collector
        import src.monitoring
        src.monitoring._metrics_collector = None

        async with track_request("/api/test", {"method": "GET"}):
            # Simulate some work
            await asyncio.sleep(0.01)

        collector = get_metrics_collector()
        stats = collector.get_endpoint_stats("/api/test")
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1
        assert stats["failed_requests"] == 0

    @pytest.mark.asyncio
    async def test_track_request_failure(self):
        """Test track_request context manager for failed request."""
        # Reset global collector
        import src.monitoring
        src.monitoring._metrics_collector = None

        with pytest.raises(ValueError):
            async with track_request("/api/test"):
                raise ValueError("Test error")

        collector = get_metrics_collector()
        stats = collector.get_endpoint_stats("/api/test")
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 1


if __name__ == "__main__":
    import asyncio
    pytest.main([__file__])
