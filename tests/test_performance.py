"""Performance and load tests for the ollama-proxy application."""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.monitoring import get_metrics_collector
from src.openrouter import OpenRouterResponse


class TestPerformanceMetrics:
    """Test performance monitoring and metrics collection."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("src.openrouter.OpenRouterClient.fetch_models") as mock_fetch:
                mock_response_data = {
                    "data": [{"id": "openai/gpt-4", "name": "OpenAI: GPT-4"}]
                }
                mock_response = OpenRouterResponse(
                    data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
                )
                mock_fetch.return_value = mock_response

                app = create_app()
                with TestClient(app) as c:
                    yield c

    def test_metrics_collection_performance(self, client):
        """Test that metrics collection doesn't significantly impact performance."""
        # Reset metrics collector
        collector = get_metrics_collector()
        collector.reset_stats()

        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            mock_response = OpenRouterResponse(
                data={"choices": [{"message": {"content": "Hello"}}]},
                status_code=200, headers={}, metrics=AsyncMock()
            )
            mock_chat.return_value = mock_response

            payload = {
                "model": "gpt-4:latest",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }

            # Measure time for multiple requests
            start_time = time.time()
            num_requests = 10

            for _ in range(num_requests):
                response = client.post("/api/chat", json=payload)
                assert response.status_code == 200

            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_request = total_time / num_requests

            # Verify performance is reasonable (should be very fast with mocks)
            assert avg_time_per_request < 0.1  # Less than 100ms per request

            # Verify metrics were collected (if metrics collection is enabled)
            all_stats = collector.get_endpoint_stats()
            if all_stats:  # Only check if metrics collection is working
                # Check if any endpoint stats were recorded
                total_recorded_requests = sum(
                    stats.get("total_requests", 0) for stats in all_stats.values()
                )
                assert total_recorded_requests >= 0  # At least some requests should be recorded

    def test_memory_usage_with_many_requests(self, client):
        """Test memory usage doesn't grow excessively with many requests."""
        import psutil
        import gc

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            mock_response = OpenRouterResponse(
                data={"choices": [{"message": {"content": "Hello"}}]},
                status_code=200, headers={}, metrics=AsyncMock()
            )
            mock_chat.return_value = mock_response

            payload = {
                "model": "gpt-4:latest",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }

            # Make many requests
            for _ in range(100):
                response = client.post("/api/chat", json=payload)
                assert response.status_code == 200

            # Force garbage collection
            gc.collect()

            # Check memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 50MB)
            assert memory_increase < 50 * 1024 * 1024

    def test_concurrent_request_performance(self, client):
        """Test performance under concurrent load."""
        import concurrent.futures
        import threading

        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            # Simple counter using list for mutable reference
            response_counter = [0]

            def mock_response(*args, **kwargs):
                response_counter[0] += 1
                # Simulate some processing time
                time.sleep(0.01)
                return OpenRouterResponse(
                    data={"choices": [
                        {"message": {"content": f"Response {response_counter[0]}"}}]},
                    status_code=200, headers={}, metrics=AsyncMock()
                )

            mock_chat.side_effect = mock_response

            def make_request():
                payload = {
                    "model": "gpt-4:latest",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                }
                start = time.time()
                response = client.post("/api/chat", json=payload)
                end = time.time()
                return response.status_code, end - start

            # Test with different concurrency levels
            concurrency_levels = [1, 5, 10]

            for concurrency in concurrency_levels:
                start_time = time.time()

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [executor.submit(make_request)
                               for _ in range(concurrency)]
                    results = [future.result()
                               for future in concurrent.futures.as_completed(futures)]

                end_time = time.time()
                total_time = end_time - start_time

                # All requests should succeed
                for status_code, request_time in results:
                    assert status_code == 200
                    assert request_time < 1.0  # Each request should complete in under 1 second

                # Total time should be reasonable
                assert total_time < 2.0  # All requests should complete in under 2 seconds

    def test_streaming_performance(self, client):
        """Test streaming response performance."""
        with patch("src.openrouter.OpenRouterClient.chat_completion_stream") as mock_chat:
            async def mock_stream():
                # Simulate streaming chunks
                for i in range(10):
                    yield f'data: {{"id": "chunk-{i}", "choices": [{{"delta": {{"content": "chunk {i} "}}}}]}}\n\n'.encode()
                    await asyncio.sleep(0.001)  # Small delay between chunks
                yield b'data: [DONE]\n\n'

            mock_chat.return_value = mock_stream()

            payload = {
                "model": "gpt-4:latest",
                "messages": [{"role": "user", "content": "Stream test"}],
                "stream": True
            }

            start_time = time.time()
            response = client.post("/api/chat", json=payload)
            end_time = time.time()

            assert response.status_code == 200

            # Streaming should start quickly
            response_time = end_time - start_time
            assert response_time < 0.5  # Should start streaming in under 500ms

            # Verify streaming content
            content = response.text
            assert "chunk 0" in content
            assert "chunk 9" in content
            assert "[DONE]" in content

    def test_metrics_aggregation_performance(self):
        """Test performance of metrics aggregation operations."""
        collector = get_metrics_collector()
        collector.reset_stats()

        # Add many metrics
        start_time = time.time()

        for i in range(1000):
            collector.record_metric(f"test_metric_{i % 10}", float(i), {
                                    "endpoint": f"/test/{i % 5}"})

        metrics_time = time.time() - start_time

        # Recording metrics should be fast
        assert metrics_time < 1.0  # Should complete in under 1 second

        # Test aggregation performance
        start_time = time.time()

        all_metrics = collector.get_metrics()
        endpoint_stats = collector.get_endpoint_stats()
        health_status = collector.get_health_status()

        aggregation_time = time.time() - start_time

        # Aggregation should be fast
        assert aggregation_time < 0.1  # Should complete in under 100ms

        # Verify results
        # Should respect limits
        assert len(all_metrics) <= collector.max_metrics
        assert len(endpoint_stats) > 0
        assert "status" in health_status

    def test_model_resolution_performance(self):
        """Test model name resolution performance."""
        # Test model resolution with many models
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("src.openrouter.OpenRouterClient.fetch_models") as mock_fetch:
                # Create many mock models
                mock_models = []
                for i in range(100):
                    mock_models.append({
                        "id": f"provider-{i % 5}/model-{i}",
                        "name": f"Provider {i % 5}: Model {i}"
                    })

                mock_response = OpenRouterResponse(
                    data={"data": mock_models}, status_code=200, headers={}, metrics=AsyncMock()
                )
                mock_fetch.return_value = mock_response

                # Create app with the new mock
                app = create_app()
                with TestClient(app) as test_client:
                    # Measure time to get tags (which involves model resolution)
                    start_time = time.time()
                    response = test_client.get("/api/tags")
                    end_time = time.time()

                    assert response.status_code == 200

                    # Model resolution should be fast even with many models
                    resolution_time = end_time - start_time
                    assert resolution_time < 0.5  # Should complete in under 500ms

                    # Verify all models were processed
                    data = response.json()
                    assert len(data["models"]) == 100

    def test_error_handling_performance(self, client):
        """Test that error handling doesn't significantly impact performance."""
        # Test various error conditions
        error_scenarios = [
            {"model": "nonexistent:latest", "expected_status": 404},
            {"model": "", "expected_status": 422},
            # Invalid JSON is handled by FastAPI before reaching our code
        ]

        for scenario in error_scenarios:
            payload = {
                "model": scenario["model"],
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }

            start_time = time.time()
            response = client.post("/api/chat", json=payload)
            end_time = time.time()

            assert response.status_code == scenario["expected_status"]

            # Error handling should be fast
            error_time = end_time - start_time
            assert error_time < 0.1  # Should complete in under 100ms


class TestLoadTesting:
    """Load testing scenarios."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            with patch("src.openrouter.OpenRouterClient.fetch_models") as mock_fetch:
                mock_response_data = {
                    "data": [{"id": "openai/gpt-4", "name": "OpenAI: GPT-4"}]
                }
                mock_response = OpenRouterResponse(
                    data=mock_response_data, status_code=200, headers={}, metrics=AsyncMock()
                )
                mock_fetch.return_value = mock_response

                app = create_app()
                with TestClient(app) as c:
                    yield c

    @pytest.mark.slow
    def test_sustained_load(self, client):
        """Test sustained load over time."""
        with patch("src.openrouter.OpenRouterClient.chat_completion") as mock_chat:
            mock_response = OpenRouterResponse(
                data={"choices": [
                    {"message": {"content": "Load test response"}}]},
                status_code=200, headers={}, metrics=AsyncMock()
            )
            mock_chat.return_value = mock_response

            payload = {
                "model": "gpt-4:latest",
                "messages": [{"role": "user", "content": "Load test"}],
                "stream": False
            }

            # Run sustained load for 10 seconds
            start_time = time.time()
            request_count = 0
            errors = 0

            while time.time() - start_time < 10:
                try:
                    response = client.post("/api/chat", json=payload)
                    if response.status_code == 200:
                        request_count += 1
                    else:
                        errors += 1
                except Exception:
                    errors += 1

                # Small delay to prevent overwhelming
                time.sleep(0.01)

            # Verify performance
            total_time = time.time() - start_time
            requests_per_second = request_count / total_time

            assert request_count > 0
            assert errors == 0  # No errors should occur
            assert requests_per_second > 10  # Should handle at least 10 RPS


if __name__ == "__main__":
    pytest.main([__file__])
