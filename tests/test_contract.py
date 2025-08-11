"""Contract tests for ollama-proxy API compatibility."""

import json
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from jsonschema import validate, ValidationError

from src.app import create_app
from src.multi_provider_config import MultiProviderSettings


# JSON Schema definitions for API contracts
OLLAMA_API_SCHEMAS = {
    "version_response": {
        "type": "object",
        "properties": {
            "version": {"type": "string"}
        },
        "required": ["version"]
    },
    
    "tags_response": {
        "type": "object",
        "properties": {
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "model": {"type": "string"},
                        "modified_at": {"type": "string"},
                        "size": {"type": "integer"},
                        "digest": {"type": "string"},
                        "details": {
                            "type": "object",
                            "properties": {
                                "parent_model": {"type": "string"},
                                "format": {"type": "string"},
                                "family": {"type": "string"},
                                "families": {"type": "array", "items": {"type": "string"}},
                                "parameter_size": {"type": "string"},
                                "quantization_level": {"type": "string"}
                            }
                        }
                    },
                    "required": ["name", "model", "modified_at", "size", "digest"]
                }
            }
        },
        "required": ["models"]
    },
    
    "chat_request": {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                }
            },
            "stream": {"type": "boolean"},
            "options": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "top_p": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_tokens": {"type": "integer", "minimum": 1},
                    "stop": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "required": ["model", "messages"]
    },
    
    "chat_response": {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "created_at": {"type": "string"},
            "message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["role", "content"]
            },
            "done": {"type": "boolean"},
            "total_duration": {"type": "integer"},
            "load_duration": {"type": "integer"},
            "prompt_eval_count": {"type": "integer"},
            "prompt_eval_duration": {"type": "integer"},
            "eval_count": {"type": "integer"},
            "eval_duration": {"type": "integer"}
        },
        "required": ["model", "created_at", "message", "done"]
    },
    
    "generate_request": {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "prompt": {"type": "string"},
            "stream": {"type": "boolean"},
            "options": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "top_p": {"type": "number"},
                    "max_tokens": {"type": "integer"}
                }
            }
        },
        "required": ["model", "prompt"]
    },
    
    "generate_response": {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "created_at": {"type": "string"},
            "response": {"type": "string"},
            "done": {"type": "boolean"},
            "context": {"type": "array", "items": {"type": "integer"}},
            "total_duration": {"type": "integer"},
            "load_duration": {"type": "integer"},
            "prompt_eval_count": {"type": "integer"},
            "prompt_eval_duration": {"type": "integer"},
            "eval_count": {"type": "integer"},
            "eval_duration": {"type": "integer"}
        },
        "required": ["model", "created_at", "response", "done"]
    },
    
    "show_request": {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    },
    
    "show_response": {
        "type": "object",
        "properties": {
            "modelfile": {"type": "string"},
            "parameters": {"type": "string"},
            "template": {"type": "string"},
            "details": {
                "type": "object",
                "properties": {
                    "parent_model": {"type": "string"},
                    "format": {"type": "string"},
                    "family": {"type": "string"},
                    "families": {"type": "array", "items": {"type": "string"}},
                    "parameter_size": {"type": "string"},
                    "quantization_level": {"type": "string"}
                }
            }
        },
        "required": ["modelfile"]
    },
    
    "embeddings_request": {
        "type": "object",
        "properties": {
            "model": {"type": "string"},
            "prompt": {"type": "string"}
        },
        "required": ["model", "prompt"]
    },
    
    "embeddings_response": {
        "type": "object",
        "properties": {
            "embedding": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "required": ["embedding"]
    }
}


@pytest.mark.contract
class TestOllamaAPIContract:
    """Test API contract compliance with Ollama API specification."""

    @pytest.fixture
    def contract_client(self):
        """Create test client for contract testing."""
        from unittest.mock import patch, AsyncMock
        from src.openrouter import OpenRouterResponse
        
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="test-api-key-contract",
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            
            # Mock model fetching
            mock_client.get.return_value.json.return_value = {
                "data": [
                    {"id": "google/gemini-pro", "name": "Google: Gemini Pro"},
                    {"id": "openai/gpt-4", "name": "OpenAI: GPT-4"},
                ]
            }
            mock_client.get.return_value.status_code = 200
            
            # Mock chat completions
            mock_client.post.return_value.json.return_value = {
                "id": "chatcmpl-123",
                "choices": [{"message": {"content": "Test response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
            mock_client.post.return_value.status_code = 200
            mock_client.post.return_value.headers = {}
            
            mock_client_class.return_value = mock_client
            
            app = create_app(settings)
            with TestClient(app) as client:
                yield client

    def validate_schema(self, data: Dict[str, Any], schema_name: str) -> None:
        """Validate data against a schema."""
        schema = OLLAMA_API_SCHEMAS.get(schema_name)
        if not schema:
            pytest.fail(f"Schema '{schema_name}' not found")
        
        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            pytest.fail(f"Schema validation failed for {schema_name}: {e.message}")

    def test_version_endpoint_contract(self, contract_client):
        """Test /api/version endpoint contract."""
        response = contract_client.get("/api/version")
        assert response.status_code == 200
        
        data = response.json()
        self.validate_schema(data, "version_response")

    def test_tags_endpoint_contract(self, contract_client):
        """Test /api/tags endpoint contract."""
        response = contract_client.get("/api/tags")
        assert response.status_code == 200
        
        data = response.json()
        self.validate_schema(data, "tags_response")

    def test_chat_endpoint_contract(self, contract_client):
        """Test /api/chat endpoint contract."""
        # Test request schema
        chat_request = {
            "model": "gemini-pro",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "stream": False
        }
        
        self.validate_schema(chat_request, "chat_request")
        
        response = contract_client.post("/api/chat", json=chat_request)
        assert response.status_code == 200
        
        # Test response schema
        data = response.json()
        self.validate_schema(data, "chat_response")

    def test_generate_endpoint_contract(self, contract_client):
        """Test /api/generate endpoint contract."""
        # Test request schema
        generate_request = {
            "model": "gemini-pro",
            "prompt": "Hello world",
            "stream": False
        }
        
        self.validate_schema(generate_request, "generate_request")
        
        response = contract_client.post("/api/generate", json=generate_request)
        assert response.status_code == 200
        
        # Test response schema
        data = response.json()
        self.validate_schema(data, "generate_response")

    def test_show_endpoint_contract(self, contract_client):
        """Test /api/show endpoint contract."""
        # Test request schema
        show_request = {"name": "gemini-pro"}
        
        self.validate_schema(show_request, "show_request")
        
        response = contract_client.post("/api/show", json=show_request)
        assert response.status_code == 200
        
        # Test response schema
        data = response.json()
        self.validate_schema(data, "show_response")

    def test_embeddings_endpoint_contract(self, contract_client):
        """Test /api/embeddings endpoint contract."""
        # Test request schema
        embeddings_request = {
            "model": "text-embedding-ada-002",
            "prompt": "Hello world"
        }
        
        self.validate_schema(embeddings_request, "embeddings_request")
        
        response = contract_client.post("/api/embeddings", json=embeddings_request)
        assert response.status_code == 200
        
        # Test response schema
        data = response.json()
        self.validate_schema(data, "embeddings_response")

    def test_unsupported_endpoints_contract(self, contract_client):
        """Test that unsupported endpoints return proper error format."""
        unsupported_endpoints = [
            "/api/create",
            "/api/copy", 
            "/api/pull",
            "/api/push"
        ]
        
        for endpoint in unsupported_endpoints:
            response = contract_client.post(endpoint, json={})
            assert response.status_code == 501
            
            data = response.json()
            assert "detail" in data
            assert isinstance(data["detail"], str)

    def test_error_response_format(self, contract_client):
        """Test error response format consistency."""
        # Test with invalid model
        response = contract_client.post("/api/chat", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_content_type_headers(self, contract_client):
        """Test that responses have correct content-type headers."""
        endpoints = [
            ("/api/version", "GET"),
            ("/api/tags", "GET"),
            ("/health", "GET"),
            ("/metrics", "GET")
        ]
        
        for endpoint, method in endpoints:
            if method == "GET":
                response = contract_client.get(endpoint)
            else:
                response = contract_client.post(endpoint, json={})
            
            assert response.status_code in [200, 400, 404, 501]
            assert "application/json" in response.headers.get("content-type", "")


@pytest.mark.contract
class TestAPICompatibility:
    """Test compatibility with official Ollama client expectations."""

    @pytest.fixture
    def compatibility_client(self):
        """Create client for compatibility testing."""
        from unittest.mock import patch, AsyncMock
        
        settings = MultiProviderSettings(
            openrouter_enabled=True,
            openrouter_api_key="test-api-key-compat",
        )
        
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.return_value.json.return_value = {
                "data": [{"id": "google/gemini-pro", "name": "Google: Gemini Pro"}]
            }
            mock_client.get.return_value.status_code = 200
            mock_client.post.return_value.json.return_value = {
                "choices": [{"message": {"content": "Response"}}]
            }
            mock_client.post.return_value.status_code = 200
            mock_client.post.return_value.headers = {}
            mock_client_class.return_value = mock_client
            
            app = create_app(settings)
            with TestClient(app) as client:
                yield client

    def test_root_endpoint_compatibility(self, compatibility_client):
        """Test root endpoint returns expected message."""
        response = compatibility_client.get("/")
        assert response.status_code == 200
        assert response.text == "Ollama is running"

    def test_model_name_mapping_compatibility(self, compatibility_client):
        """Test model name mapping works as expected."""
        # Test that Ollama-style model names are accepted
        response = compatibility_client.post("/api/chat", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        })
        
        assert response.status_code == 200

    def test_streaming_parameter_compatibility(self, compatibility_client):
        """Test streaming parameter is handled correctly."""
        # Test with stream=false
        response = compatibility_client.post("/api/chat", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        })
        
        assert response.status_code == 200
        
        # Test with stream=true (should still work)
        response = compatibility_client.post("/api/chat", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        })
        
        assert response.status_code == 200

    def test_options_parameter_compatibility(self, compatibility_client):
        """Test options parameter handling."""
        response = compatibility_client.post("/api/chat", json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100
            }
        })
        
        assert response.status_code == 200
