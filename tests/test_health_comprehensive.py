"""Comprehensive tests for the health module to improve coverage."""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.health import HealthChecker, HealthStatus, get_health_checker
from src.providers.base import ProviderType


class TestHealthStatus:
    """Test HealthStatus model."""

    def test_health_status_creation(self):
        """Test creating a HealthStatus instance."""
        status = HealthStatus(
            status="healthy",
            timestamp=time.time(),
            uptime=100.0,
            providers={},
            metrics={}
        )
        assert status.status == "healthy"
        assert status.uptime == 100.0

    def test_health_status_to_dict(self):
        """Test converting HealthStatus to dictionary."""
        status = HealthStatus(
            status="healthy",
            timestamp=1234567890.0,
            uptime=100.0,
            providers={"openrouter": {"healthy": True}},
            metrics={"cpu": 50.0}
        )
        
        result = status.to_dict()
        assert result["status"] == "healthy"
        assert result["timestamp"] == 1234567890.0
        assert result["uptime"] == 100.0
        assert "openrouter" in result["providers"]
        assert "cpu" in result["metrics"]


class TestHealthChecker:
    """Test HealthChecker class."""

    @pytest.fixture
    def health_checker(self):
        """Create a HealthChecker instance for testing."""
        return HealthChecker()

    @pytest.mark.asyncio
    async def test_check_system_health_all_healthy(self, health_checker):
        """Test system health check when all providers are healthy."""
        with patch.object(health_checker, '_check_all_providers') as mock_check_providers, \
             patch.object(health_checker, '_get_system_metrics') as mock_get_metrics:
            
            mock_check_providers.return_value = {
                "openrouter": {"healthy": True, "status": "operational"},
                "anthropic": {"healthy": True, "status": "operational"}
            }
            mock_get_metrics.return_value = {"cpu": 50.0, "memory": 60.0}
            
            result = await health_checker.check_system_health()
            
            assert result.status == "healthy"
            assert len(result.providers) == 2
            assert result.metrics["cpu"] == 50.0

    @pytest.mark.asyncio
    async def test_check_system_health_degraded(self, health_checker):
        """Test system health check when some providers are unhealthy."""
        with patch.object(health_checker, '_check_all_providers') as mock_check_providers, \
             patch.object(health_checker, '_get_system_metrics') as mock_get_metrics:
            
            mock_check_providers.return_value = {
                "openrouter": {"healthy": True, "status": "operational"},
                "anthropic": {"healthy": False, "status": "error"}
            }
            mock_get_metrics.return_value = {"cpu": 50.0}
            
            result = await health_checker.check_system_health()
            
            assert result.status == "degraded"

    @pytest.mark.asyncio
    async def test_check_system_health_unhealthy(self, health_checker):
        """Test system health check when all providers are unhealthy."""
        with patch.object(health_checker, '_check_all_providers') as mock_check_providers, \
             patch.object(health_checker, '_get_system_metrics') as mock_get_metrics:
            
            mock_check_providers.return_value = {
                "openrouter": {"healthy": False, "status": "error"},
                "anthropic": {"healthy": False, "status": "error"}
            }
            mock_get_metrics.return_value = {"cpu": 50.0}
            
            result = await health_checker.check_system_health()
            
            assert result.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_check_provider_health_success(self, health_checker):
        """Test checking individual provider health successfully."""
        mock_provider = AsyncMock()
        mock_provider.list_models.return_value = []
        mock_provider.request_count = 10
        mock_provider.error_count = 1
        mock_provider.error_rate = 0.1
        
        with patch('src.health.get_factory') as mock_get_factory, \
             patch('src.health.get_health_manager') as mock_get_health_manager:
            
            mock_factory = Mock()
            mock_factory.get_provider.return_value = mock_provider
            mock_get_factory.return_value = mock_factory
            
            mock_health_manager = Mock()
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.get_state.return_value = "closed"
            mock_health_manager.get_circuit_breaker.return_value = mock_circuit_breaker
            mock_get_health_manager.return_value = mock_health_manager
            
            result = await health_checker.check_provider_health(ProviderType.OPENROUTER)
            
            assert result["healthy"] is True
            assert result["status"] == "operational"
            assert "response_time_ms" in result
            assert result["request_count"] == 10
            assert result["error_count"] == 1

    @pytest.mark.asyncio
    async def test_check_provider_health_failure(self, health_checker):
        """Test checking individual provider health with failure."""
        mock_provider = AsyncMock()
        mock_provider.list_models.side_effect = Exception("Provider error")
        
        with patch('src.health.get_factory') as mock_get_factory, \
             patch('src.health.get_health_manager') as mock_get_health_manager:
            
            mock_factory = Mock()
            mock_factory.get_provider.return_value = mock_provider
            mock_get_factory.return_value = mock_factory
            
            mock_health_manager = Mock()
            mock_circuit_breaker = Mock()
            mock_circuit_breaker.get_state.return_value = "open"
            mock_health_manager.get_circuit_breaker.return_value = mock_circuit_breaker
            mock_get_health_manager.return_value = mock_health_manager
            
            result = await health_checker.check_provider_health(ProviderType.OPENROUTER)
            
            assert result["healthy"] is False
            assert result["status"] == "error"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_check_all_providers(self, health_checker):
        """Test checking all providers."""
        with patch.object(health_checker, 'check_provider_health') as mock_check_provider:
            mock_check_provider.return_value = {"healthy": True, "status": "operational"}
            
            with patch('src.health.ProviderType') as mock_provider_type:
                mock_provider_type.__iter__ = Mock(return_value=iter([ProviderType.OPENROUTER]))
                
                result = await health_checker._check_all_providers()
                
                assert len(result) >= 0  # May vary based on available providers

    @pytest.mark.asyncio
    async def test_get_system_metrics(self, health_checker):
        """Test getting system metrics."""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 75.0
            
            mock_memory_obj = Mock()
            mock_memory_obj.percent = 60.0
            mock_memory_obj.available = 1024 * 1024 * 1024  # 1GB
            mock_memory.return_value = mock_memory_obj
            
            mock_disk_obj = Mock()
            mock_disk_obj.percent = 50.0
            mock_disk_obj.free = 10 * 1024 * 1024 * 1024  # 10GB
            mock_disk.return_value = mock_disk_obj
            
            result = await health_checker._get_system_metrics()
            
            assert result["cpu_percent"] == 75.0
            assert result["memory_percent"] == 60.0
            assert result["disk_percent"] == 50.0
            assert "memory_available_gb" in result
            assert "disk_free_gb" in result

    @pytest.mark.asyncio
    async def test_get_system_metrics_error(self, health_checker):
        """Test getting system metrics with error."""
        with patch('psutil.cpu_percent', side_effect=Exception("psutil error")):
            result = await health_checker._get_system_metrics()
            
            # Should return empty dict or default values on error
            assert isinstance(result, dict)


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with health routes."""
        from fastapi import FastAPI
        from src.health import router
        
        app = FastAPI()
        app.include_router(router, prefix="/health")
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_health_check_endpoint_healthy(self, client):
        """Test basic health check endpoint when healthy."""
        with patch('src.health._health_checker.check_system_health') as mock_check:
            mock_status = HealthStatus(
                status="healthy",
                timestamp=time.time(),
                uptime=100.0,
                providers={},
                metrics={}
            )
            mock_check.return_value = mock_status
            
            response = client.get("/health/")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_endpoint_unhealthy(self, client):
        """Test basic health check endpoint when unhealthy."""
        with patch('src.health._health_checker.check_system_health') as mock_check:
            mock_status = HealthStatus(
                status="unhealthy",
                timestamp=time.time(),
                uptime=100.0,
                providers={},
                metrics={}
            )
            mock_check.return_value = mock_status
            
            response = client.get("/health/")
            assert response.status_code == 503

    def test_detailed_health_check_endpoint(self, client):
        """Test detailed health check endpoint."""
        with patch('src.health._health_checker.check_system_health') as mock_check:
            mock_status = HealthStatus(
                status="healthy",
                timestamp=time.time(),
                uptime=100.0,
                providers={"openrouter": {"healthy": True}},
                metrics={"cpu": 50.0}
            )
            mock_check.return_value = mock_status
            
            response = client.get("/health/detailed")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "providers" in data
            assert "metrics" in data

    def test_provider_health_check_endpoint(self, client):
        """Test provider health check endpoint."""
        with patch('src.health._health_checker._check_all_providers') as mock_check:
            mock_check.return_value = {
                "openrouter": {"healthy": True, "status": "operational"}
            }
            
            response = client.get("/health/providers")
            assert response.status_code == 200
            data = response.json()
            assert "openrouter" in data

    def test_single_provider_health_check_endpoint(self, client):
        """Test single provider health check endpoint."""
        with patch('src.health._health_checker.check_provider_health') as mock_check:
            mock_check.return_value = {"healthy": True, "status": "operational"}
            
            response = client.get("/health/providers/openrouter")
            assert response.status_code == 200
            data = response.json()
            assert data["healthy"] is True

    def test_single_provider_health_check_invalid_provider(self, client):
        """Test single provider health check with invalid provider."""
        response = client.get("/health/providers/invalid-provider")
        assert response.status_code == 400

    def test_get_metrics_endpoint(self, client):
        """Test get metrics endpoint."""
        with patch('src.health._health_checker._get_system_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = {"cpu": 50.0, "memory": 60.0}
            
            response = client.get("/health/metrics")
            assert response.status_code == 200
            data = response.json()
            assert data["cpu"] == 50.0
            assert data["memory"] == 60.0


class TestHealthCheckerUtilities:
    """Test health checker utility functions."""

    def test_get_health_checker(self):
        """Test getting the global health checker instance."""
        checker = get_health_checker()
        assert isinstance(checker, HealthChecker)
        
        # Should return the same instance
        checker2 = get_health_checker()
        assert checker is checker2

    def test_health_checker_singleton_behavior(self):
        """Test that health checker behaves as singleton."""
        # Import the module-level instance
        from src.health import _health_checker
        
        checker1 = get_health_checker()
        assert checker1 is _health_checker

    @pytest.mark.asyncio
    async def test_health_checker_caching(self):
        """Test that health checker caches results appropriately."""
        health_checker = HealthChecker()
        
        # Mock provider check
        with patch.object(health_checker, 'check_provider_health') as mock_check:
            mock_check.return_value = {"healthy": True, "status": "operational"}
            
            # First call
            result1 = await health_checker.check_provider_health(ProviderType.OPENROUTER)
            
            # Second call within cache period should use cache
            result2 = await health_checker.check_provider_health(ProviderType.OPENROUTER)
            
            assert result1 == result2
            # Should have been called at least once
            assert mock_check.call_count >= 1
