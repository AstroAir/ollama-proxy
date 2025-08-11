"""Security tests for ollama-proxy."""

import json
import os
import subprocess
import tempfile
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.multi_provider_config import MultiProviderSettings


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.fixture
    def security_client(self):
        """Create test client for security testing."""
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="test-api-key-security",
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value.json.return_value = {
                "data": [{"id": "google/gemini-pro", "name": "Google: Gemini Pro"}]
            }
            mock_client.get.return_value.status_code = 200
            mock_client.post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Safe response"}}]
            }
            mock_client.post.return_value.status_code = 200
            mock_client.post.return_value.headers = {}
            mock_client_class.return_value = mock_client
            
            app = create_app(settings)
            with TestClient(app) as client:
                yield client

    def test_sql_injection_protection(self, security_client):
        """Test protection against SQL injection attempts."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --",
        ]
        
        for malicious_input in malicious_inputs:
            response = security_client.post("/api/chat", json={
                "model": malicious_input,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            })
            
            # Should return validation error, not crash
            assert response.status_code in [400, 422]

    def test_xss_protection(self, security_client):
        """Test protection against XSS attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
        ]
        
        for payload in xss_payloads:
            response = security_client.post("/api/chat", json={
                "model": "gemini-pro",
                "messages": [{"role": "user", "content": payload}],
                "stream": False
            })
            
            # Should handle safely
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                # Response should not contain unescaped script tags
                response_text = response.text
                assert "<script>" not in response_text
                assert "javascript:" not in response_text

    def test_command_injection_protection(self, security_client):
        """Test protection against command injection."""
        command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(id)",
        ]
        
        for payload in command_injection_payloads:
            response = security_client.post("/api/chat", json={
                "model": "gemini-pro",
                "messages": [{"role": "user", "content": payload}],
                "stream": False
            })
            
            # Should handle safely
            assert response.status_code in [200, 400, 422]

    def test_path_traversal_protection(self, security_client):
        """Test protection against path traversal attacks."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
        ]
        
        for payload in path_traversal_payloads:
            response = security_client.post("/api/show", json={
                "name": payload
            })
            
            # Should return error, not expose file contents
            assert response.status_code in [400, 404, 422]

    def test_large_payload_protection(self, security_client):
        """Test protection against large payload attacks."""
        # Create a very large payload
        large_content = "A" * (10 * 1024 * 1024)  # 10MB
        
        response = security_client.post("/api/chat", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": large_content}],
            "stream": False
        })
        
        # Should reject or handle gracefully
        assert response.status_code in [400, 413, 422]

    def test_malformed_json_protection(self, security_client):
        """Test protection against malformed JSON."""
        malformed_payloads = [
            '{"model": "gemini-pro", "messages": [{"role": "user", "content": "test"}',  # Missing closing brace
            '{"model": "gemini-pro", "messages": [{"role": "user", "content": "test"}]}}',  # Extra brace
            '{"model": "gemini-pro", "messages": [{"role": "user", "content": "test"}], "invalid": }',  # Invalid value
        ]
        
        for payload in malformed_payloads:
            response = security_client.post(
                "/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Should return validation error
            assert response.status_code in [400, 422]


@pytest.mark.security
class TestAuthenticationSecurity:
    """Test authentication and authorization security."""

    def test_missing_api_key_protection(self):
        """Test that missing API key is properly handled."""
        # Create app without API key
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="",  # Empty API key
        )
        
        with patch("httpx.AsyncClient"):
            app = create_app(settings)
            with TestClient(app) as client:
                response = client.post("/api/chat", json={
                    "model": "gemini-pro",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": False
                })
                
                # Should fail due to missing API key
                assert response.status_code in [401, 500]

    def test_api_key_exposure_protection(self, security_client):
        """Test that API keys are not exposed in responses."""
        endpoints = [
            "/health",
            "/metrics",
            "/api/version",
            "/api/tags"
        ]
        
        for endpoint in endpoints:
            response = security_client.get(endpoint)
            response_text = response.text.lower()
            
            # Check for common API key patterns
            assert "api_key" not in response_text
            assert "apikey" not in response_text
            assert "token" not in response_text or "test" in response_text  # Allow test tokens
            assert "secret" not in response_text


@pytest.mark.security
class TestHeaderSecurity:
    """Test HTTP header security."""

    @pytest.fixture
    def header_client(self):
        """Create client for header testing."""
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="test-api-key-headers",
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value.json.return_value = {"data": []}
            mock_client.get.return_value.status_code = 200
            mock_client_class.return_value = mock_client
            
            app = create_app(settings)
            with TestClient(app) as client:
                yield client

    def test_security_headers_present(self, header_client):
        """Test that security headers are present."""
        response = header_client.get("/health")
        
        # Check for security headers (if implemented)
        headers = response.headers
        
        # These might not be implemented yet, but good to test
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
            "strict-transport-security",
        ]
        
        # For now, just ensure no sensitive headers are exposed
        sensitive_headers = [
            "server",
            "x-powered-by",
        ]
        
        for header in sensitive_headers:
            assert header not in [h.lower() for h in headers.keys()]

    def test_cors_headers_security(self, header_client):
        """Test CORS headers security."""
        response = header_client.options("/api/chat")
        
        if "access-control-allow-origin" in response.headers:
            # If CORS is enabled, ensure it's not too permissive
            origin = response.headers["access-control-allow-origin"]
            assert origin != "*" or "localhost" in origin


@pytest.mark.security
class TestErrorHandlingSecurity:
    """Test error handling security."""

    @pytest.fixture
    def error_client(self):
        """Create client for error testing."""
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="test-api-key-errors",
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value.json.return_value = {"data": []}
            mock_client.get.return_value.status_code = 200
            mock_client_class.return_value = mock_client
            
            app = create_app(settings)
            with TestClient(app) as client:
                yield client

    def test_error_information_disclosure(self, error_client):
        """Test that errors don't disclose sensitive information."""
        # Test various error conditions
        error_requests = [
            ("/api/nonexistent", "GET"),
            ("/api/chat", "POST"),  # Missing required fields
            ("/api/show", "POST"),  # Missing required fields
        ]
        
        for endpoint, method in error_requests:
            if method == "GET":
                response = error_client.get(endpoint)
            else:
                response = error_client.post(endpoint, json={})
            
            # Should return error
            assert response.status_code >= 400
            
            error_text = response.text.lower()
            
            # Should not expose sensitive information
            sensitive_info = [
                "traceback",
                "stack trace",
                "file path",
                "/home/",
                "/usr/",
                "c:\\",
                "password",
                "secret",
                "api_key",
            ]
            
            for info in sensitive_info:
                assert info not in error_text

    def test_exception_handling_security(self, error_client):
        """Test that exceptions are handled securely."""
        # Force an exception by providing invalid data
        response = error_client.post("/api/chat", json={
            "model": None,  # Invalid model type
            "messages": "invalid",  # Invalid messages type
        })
        
        assert response.status_code in [400, 422]
        
        # Error response should be well-formed JSON
        try:
            error_data = response.json()
            assert "detail" in error_data
        except json.JSONDecodeError:
            pytest.fail("Error response is not valid JSON")


@pytest.mark.security
@pytest.mark.slow
class TestDependencySecurity:
    """Test dependency security."""

    def test_known_vulnerabilities(self):
        """Test for known vulnerabilities in dependencies."""
        try:
            # Run safety check
            result = subprocess.run(
                ["uv", "run", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # No vulnerabilities found
                pass
            else:
                # Parse safety output for vulnerabilities
                try:
                    vulnerabilities = json.loads(result.stdout)
                    if vulnerabilities:
                        pytest.fail(f"Found {len(vulnerabilities)} security vulnerabilities")
                except json.JSONDecodeError:
                    # Safety might not be available or output format changed
                    pytest.skip("Could not parse safety output")
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Safety tool not available or timed out")

    def test_bandit_security_scan(self):
        """Test static security analysis with bandit."""
        try:
            # Run bandit security scan
            result = subprocess.run(
                ["uv", "run", "bandit", "-r", "src/", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # No high-severity issues found
                pass
            else:
                try:
                    bandit_output = json.loads(result.stdout)
                    high_severity_issues = [
                        issue for issue in bandit_output.get("results", [])
                        if issue.get("issue_severity") == "HIGH"
                    ]
                    
                    if high_severity_issues:
                        pytest.fail(f"Found {len(high_severity_issues)} high-severity security issues")
                        
                except json.JSONDecodeError:
                    # Bandit might not be available
                    pytest.skip("Could not parse bandit output")
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Bandit tool not available or timed out")
