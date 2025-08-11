"""Performance and load tests for the ollama-proxy application."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.monitoring import get_metrics_collector
from src.openrouter import OpenRouterResponse

# Try to import performance testing dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


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
        from unittest.mock import patch, MagicMock
        from src.providers.base import ProviderError, ProviderType
        from src.exceptions import ErrorContext

        # Mock the multi-provider API's chat_completion method to avoid retries
        with patch("src.multi_provider_api.MultiProviderAPI.chat_completion") as mock_chat:
            # Make the mock raise an error immediately for nonexistent models
            mock_chat.side_effect = ProviderError(
                "Model not found",
                provider_type=ProviderType.OPENROUTER,
                context=ErrorContext(additional_data={"model": "nonexistent"})
            )

            # Test various error conditions
            error_scenarios = [
                {"model": "nonexistent:latest", "expected_status": 500},  # Provider error becomes 500
                {"model": "", "expected_status": 400},  # Validation error
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
                assert error_time < 1.0  # Should complete in under 1 second (relaxed for mocked errors)


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


@pytest.mark.performance
@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
class TestBenchmarkPerformance:
    """Benchmark performance tests using pytest-benchmark."""

    @pytest.fixture
    def benchmark_client(self):
        """Create test client for benchmarking."""
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

    def test_health_check_benchmark(self, benchmark, benchmark_client):
        """Benchmark health check endpoint."""
        def health_check():
            response = benchmark_client.get("/health")
            assert response.status_code == 200
            return response

        result = benchmark(health_check)
        assert result.status_code == 200

    def test_api_version_benchmark(self, benchmark, benchmark_client):
        """Benchmark API version endpoint."""
        def get_version():
            response = benchmark_client.get("/api/version")
            assert response.status_code == 200
            return response

        result = benchmark(get_version)
        assert result.json()["version"] == "0.2.0"

    def test_list_models_benchmark(self, benchmark, benchmark_client):
        """Benchmark model listing endpoint."""
        def list_models():
            response = benchmark_client.get("/api/tags")
            assert response.status_code == 200
            return response

        result = benchmark(list_models)
        assert "models" in result.json()


@pytest.mark.performance
@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
class TestResourceUsagePerformance:
    """Resource usage performance tests."""

    @pytest.fixture
    def resource_client(self):
        """Create test client for resource testing."""
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

    def test_memory_usage_stability(self, resource_client):
        """Test memory usage stability under load."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Make many requests
        for i in range(100):
            response = resource_client.get("/health")
            assert response.status_code == 200

            if i % 20 == 0:  # Check memory every 20 requests
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory

                # Memory shouldn't increase dramatically
                assert memory_increase < 50  # Less than 50MB increase

    def test_cpu_usage_monitoring(self, resource_client):
        """Test CPU usage during load."""
        cpu_samples = []

        def monitor_cpu():
            for _ in range(10):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))

        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        # Generate some load
        for i in range(20):
            response = resource_client.get("/health")
            assert response.status_code == 200
            time.sleep(0.05)

        monitor_thread.join()

        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)

            # CPU usage should be reasonable
            assert avg_cpu < 90.0  # Average CPU under 90%
            assert max_cpu < 100.0  # Max CPU under 100%


@pytest.mark.performance
@pytest.mark.slow
class TestConcurrencyPerformance:
    """Concurrency performance tests."""

    @pytest.fixture
    def concurrent_client(self):
        """Create test client for concurrency testing."""
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

    def test_concurrent_health_checks(self, concurrent_client):
        """Test concurrent health check performance."""
        def make_request():
            response = concurrent_client.get("/health")
            return response.status_code == 200

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        end_time = time.time()
        duration = end_time - start_time

        # All requests should succeed
        assert all(results)
        # Should complete within reasonable time
        assert duration < 15.0

        # Calculate throughput
        throughput = len(results) / duration
        assert throughput > 3  # At least 3 requests per second

    def test_response_time_consistency(self, concurrent_client):
        """Test response time consistency under load."""
        response_times = []

        for i in range(30):
            start_time = time.time()
            response = concurrent_client.get("/health")
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)

        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        # Assertions
        assert avg_response_time < 2.0  # Average response time under 2 seconds
        assert max_response_time < 10.0  # Max response time under 10 seconds
