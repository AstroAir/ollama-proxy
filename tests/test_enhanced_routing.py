"""Tests for enhanced routing and queue management features.

This module tests the improved routing logic, model-based routing,
queue management, and metrics collection.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.providers.base import ProviderCapability, ProviderConfig, ProviderType
from src.providers.router import ProviderRouter
from src.providers.queue_manager import (
    RequestQueueManager,
    QueuedRequest,
    RequestPriority,
    get_queue_manager,
)
from src.providers.metrics_collector import MetricsCollector, ProviderMetrics
from src.multi_provider_config import RoutingStrategy, FallbackStrategy


class TestEnhancedRouting:
    """Test enhanced routing capabilities."""

    @pytest.fixture
    def router(self):
        """Create router instance."""
        return ProviderRouter(
            routing_strategy=RoutingStrategy.CAPABILITY_BASED,
            fallback_strategy=FallbackStrategy.NEXT_AVAILABLE,
            enable_load_balancing=True,
        )

    def test_model_routing_rules(self, router):
        """Test model-based routing rules."""
        # Test OpenAI models
        assert router.get_provider_for_model("gpt-4") == ProviderType.OPENAI
        assert router.get_provider_for_model("gpt-3.5-turbo") == ProviderType.OPENAI
        assert router.get_provider_for_model("text-embedding-ada-002") == ProviderType.OPENAI

        # Test Anthropic models
        assert router.get_provider_for_model("claude-3-sonnet") == ProviderType.ANTHROPIC
        assert router.get_provider_for_model("claude-2.1") == ProviderType.ANTHROPIC

        # Test Google models
        assert router.get_provider_for_model("gemini-pro") == ProviderType.GOOGLE
        assert router.get_provider_for_model("palm-2") == ProviderType.GOOGLE

        # Test Azure models
        assert router.get_provider_for_model("gpt-4@test.openai.azure.com") == ProviderType.AZURE

        # Test AWS Bedrock models
        assert router.get_provider_for_model("anthropic.claude-3-sonnet-20240229-v1:0") == ProviderType.AWS_BEDROCK
        assert router.get_provider_for_model("amazon.titan-text-express-v1") == ProviderType.AWS_BEDROCK

        # Test Ollama models
        assert router.get_provider_for_model("llama2") == ProviderType.OLLAMA
        assert router.get_provider_for_model("mistral") == ProviderType.OLLAMA

        # Test unknown model
        assert router.get_provider_for_model("unknown-model") is None

    @pytest.mark.asyncio
    async def test_model_based_routing_selection(self, router):
        """Test that model-based routing is prioritized in selection."""
        available_providers = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE]
        
        with patch.object(router, '_get_available_providers', return_value=available_providers):
            # Should select OpenAI for GPT models
            selected = await router._select_provider(
                available_providers,
                ProviderCapability.CHAT_COMPLETION,
                model="gpt-4"
            )
            assert selected == ProviderType.OPENAI

            # Should select Anthropic for Claude models
            selected = await router._select_provider(
                available_providers,
                ProviderCapability.CHAT_COMPLETION,
                model="claude-3-sonnet"
            )
            assert selected == ProviderType.ANTHROPIC

    def test_cost_optimized_routing(self, router):
        """Test cost-optimized routing strategy."""
        providers = [ProviderType.OPENAI, ProviderType.OLLAMA, ProviderType.ANTHROPIC]
        
        # Should prefer Ollama (local, cheapest)
        selected = router._select_cost_optimized(providers)
        assert selected == ProviderType.OLLAMA

        # Without Ollama, should prefer OpenRouter
        providers = [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.OPENROUTER]
        selected = router._select_cost_optimized(providers)
        assert selected == ProviderType.OPENROUTER


class TestQueueManager:
    """Test request queue management."""

    @pytest_asyncio.fixture
    async def queue_manager(self):
        """Create and start queue manager."""
        manager = RequestQueueManager(
            max_queue_size=100,
            max_concurrent_requests=10,
            enable_prioritization=True,
        )
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_queue_enqueue_dequeue(self, queue_manager):
        """Test basic enqueue and dequeue operations."""
        # Enqueue a request
        request_id = await queue_manager.enqueue_request(
            capability=ProviderCapability.CHAT_COMPLETION,
            model="gpt-4",
            priority=RequestPriority.NORMAL,
        )
        
        assert request_id is not None
        
        # Dequeue the request
        request = await queue_manager.dequeue_request()
        assert request is not None
        assert request.id == request_id
        assert request.capability == ProviderCapability.CHAT_COMPLETION
        assert request.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue_manager):
        """Test that higher priority requests are dequeued first."""
        # Enqueue requests with different priorities
        low_id = await queue_manager.enqueue_request(
            capability=ProviderCapability.CHAT_COMPLETION,
            priority=RequestPriority.LOW,
        )
        
        high_id = await queue_manager.enqueue_request(
            capability=ProviderCapability.CHAT_COMPLETION,
            priority=RequestPriority.HIGH,
        )
        
        normal_id = await queue_manager.enqueue_request(
            capability=ProviderCapability.CHAT_COMPLETION,
            priority=RequestPriority.NORMAL,
        )
        
        # Should dequeue in priority order: HIGH, NORMAL, LOW
        request1 = await queue_manager.dequeue_request()
        assert request1.id == high_id
        
        request2 = await queue_manager.dequeue_request()
        assert request2.id == normal_id
        
        request3 = await queue_manager.dequeue_request()
        assert request3.id == low_id

    @pytest.mark.asyncio
    async def test_queue_full_handling(self, queue_manager):
        """Test queue full handling."""
        # Fill up the queue
        for i in range(queue_manager.max_queue_size):
            await queue_manager.enqueue_request(
                capability=ProviderCapability.CHAT_COMPLETION,
            )
        
        # Next request should raise QueueFull
        with pytest.raises(asyncio.QueueFull):
            await queue_manager.enqueue_request(
                capability=ProviderCapability.CHAT_COMPLETION,
            )

    @pytest.mark.asyncio
    async def test_request_processing(self, queue_manager):
        """Test request processing with concurrency control."""
        async def mock_processor(request):
            await asyncio.sleep(0.1)  # Simulate processing time
            return f"processed_{request.id}"
        
        # Enqueue a request
        request_id = await queue_manager.enqueue_request(
            capability=ProviderCapability.CHAT_COMPLETION,
        )
        
        # Dequeue and process
        request = await queue_manager.dequeue_request()
        result = await queue_manager.process_request(request, mock_processor)
        
        assert result == f"processed_{request_id}"

    def test_queue_stats(self, queue_manager):
        """Test queue statistics."""
        stats = queue_manager.get_queue_stats()
        
        assert "total_queued" in stats
        assert "total_processed" in stats
        assert "total_expired" in stats
        assert "total_rejected" in stats
        assert "active_requests" in stats
        assert "queue_sizes" in stats
        assert "max_queue_size" in stats


class TestMetricsCollector:
    """Test metrics collection and analysis."""

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector."""
        return MetricsCollector()

    @pytest.mark.asyncio
    async def test_provider_metrics_recording(self, metrics_collector):
        """Test recording provider metrics."""
        # Record successful request
        await metrics_collector.record_provider_request(
            provider_type=ProviderType.OPENAI,
            success=True,
            response_time=0.5,
            capability=ProviderCapability.CHAT_COMPLETION,
            model="gpt-4",
            tokens_used={"input": 10, "output": 20},
        )
        
        # Get metrics
        metrics = await metrics_collector.get_provider_metrics(ProviderType.OPENAI)
        assert metrics is not None
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.get_success_rate() == 100.0

    @pytest.mark.asyncio
    async def test_error_tracking(self, metrics_collector):
        """Test error tracking in metrics."""
        # Record failed request
        await metrics_collector.record_provider_request(
            provider_type=ProviderType.OPENAI,
            success=False,
            response_time=1.0,
            capability=ProviderCapability.CHAT_COMPLETION,
            error="Rate limit exceeded",
        )
        
        metrics = await metrics_collector.get_provider_metrics(ProviderType.OPENAI)
        assert metrics.failed_requests == 1
        assert metrics.consecutive_failures == 1
        assert "Rate limit exceeded" in metrics.error_counts
        assert metrics.last_error == "Rate limit exceeded"

    @pytest.mark.asyncio
    async def test_health_status_tracking(self, metrics_collector):
        """Test health status tracking."""
        # Record multiple failures
        for i in range(6):  # Threshold is 5 consecutive failures
            await metrics_collector.record_provider_request(
                provider_type=ProviderType.OPENAI,
                success=False,
                response_time=1.0,
                capability=ProviderCapability.CHAT_COMPLETION,
                error="Connection error",
            )
        
        metrics = await metrics_collector.get_provider_metrics(ProviderType.OPENAI)
        assert not metrics.is_healthy  # Should be marked unhealthy
        
        # Record successful request
        await metrics_collector.record_provider_request(
            provider_type=ProviderType.OPENAI,
            success=True,
            response_time=0.5,
            capability=ProviderCapability.CHAT_COMPLETION,
        )
        
        metrics = await metrics_collector.get_provider_metrics(ProviderType.OPENAI)
        assert metrics.is_healthy  # Should be healthy again

    @pytest.mark.asyncio
    async def test_global_metrics(self, metrics_collector):
        """Test global metrics collection."""
        # Record some requests
        await metrics_collector.record_provider_request(
            provider_type=ProviderType.OPENAI,
            success=True,
            response_time=0.5,
            capability=ProviderCapability.CHAT_COMPLETION,
        )
        
        await metrics_collector.record_provider_request(
            provider_type=ProviderType.ANTHROPIC,
            success=False,
            response_time=1.0,
            capability=ProviderCapability.CHAT_COMPLETION,
            error="Error",
        )
        
        global_metrics = await metrics_collector.get_global_metrics()
        assert global_metrics["total_requests"] == 2
        assert global_metrics["total_errors"] == 1
        assert global_metrics["global_success_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_health_summary(self, metrics_collector):
        """Test health summary generation."""
        # Record requests for multiple providers
        await metrics_collector.record_provider_request(
            provider_type=ProviderType.OPENAI,
            success=True,
            response_time=0.5,
            capability=ProviderCapability.CHAT_COMPLETION,
        )
        
        # Make Anthropic unhealthy
        for i in range(6):
            await metrics_collector.record_provider_request(
                provider_type=ProviderType.ANTHROPIC,
                success=False,
                response_time=1.0,
                capability=ProviderCapability.CHAT_COMPLETION,
                error="Error",
            )
        
        health_summary = await metrics_collector.get_health_summary()
        assert not health_summary["overall_healthy"]
        assert ProviderType.ANTHROPIC.value in health_summary["unhealthy_providers"]
        assert health_summary["healthy_count"] == 1
        assert health_summary["total_providers"] == 2
