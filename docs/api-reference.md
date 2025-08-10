# API Reference

This document provides a detailed reference for the Ollama API endpoints supported by the Ollama Proxy. The proxy aims to be a drop-in replacement for the official Ollama server, so it maintains compatibility with the most common endpoints.

## General Information

- **Base URL**: `http://<host>:<port>`
- **Default Port**: `11434`
- **Authentication**: All requests require a valid OpenRouter API key configured on the proxy server
- **Content-Type**: All POST requests should use `Content-Type: application/json`

## Supported Endpoints

### Health & Monitoring

#### `GET /`

Returns a simple health check message to confirm that the server is running.

- **Success Response (200 OK)**:

    ```text
    Ollama is running
    ```

#### `GET /api/version`

Returns the version of the proxy.

- **Success Response (200 OK)**:

    ```json
    {
      "version": "0.1.0-openrouter"
    }
    ```

#### `GET /health`

Returns detailed health information about the proxy.

- **Success Response (200 OK)**:

    ```json
    {
      "status": "healthy",
      "uptime_seconds": 1234.56,
      "model_count": 42,
      "filtered_model_count": 10,
      "request_count": 123,
      "error_count": 0,
      "error_rate": 0.0,
      "last_model_refresh": 1640995200.0,
      "environment": "production"
    }
    ```

#### `GET /metrics`

Returns metrics for monitoring and observability.

- **Success Response (200 OK)**:

    ```json
    {
      "metrics": [...],
      "statistics": {...},
      "timestamp": 1640995200.0
    }
    ```

### Model Management

#### `GET /api/tags`

Lists all available models that are accessible through the proxy. The list is fetched from OpenRouter and can be filtered using the [model filter configuration](configuration.md#model-filtering).

- **Success Response (200 OK)**:

    ```json
    {
      "models": [
        {
          "name": "google/gemini-pro:latest",
          "modified_at": "2023-12-12T14:00:00Z",
          "size": 7000000000,
          "digest": "sha256:abcdef1234567890",
          "details": {
            "format": "gguf",
            "family": "gemini",
            "families": ["gemini"],
            "parameter_size": "7B",
            "quantization_level": "Unknown"
          }
        }
      ]
    }
    ```

#### `POST /api/show`

Provides detailed information about a specific model. Note that much of the information is stubbed since it is not available from the OpenRouter API.

- **Request Body**:

    ```json
    {
      "name": "google/gemini-pro:latest"
    }
    ```

- **Success Response (200 OK)**:

    ```json
    {
      "license": "",
      "modelfile": "",
      "parameters": "",
      "template": "",
      "details": {
        "parent_model": "",
        "format": "",
        "family": "gemini",
        "families": ["gemini"],
        "parameter_size": "Unknown",
        "quantization_level": ""
      },
      "model_info": {},
      "tensors": []
    }
    ```

### Inference

#### `POST /api/chat`

Handles chat completion requests. This is the primary endpoint for interacting with models.

- **Request Body**:

    ```json
    {
      "model": "google/gemini-pro:latest",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "Why is the sky blue?"
        }
      ],
      "stream": false,
      "options": {
        "temperature": 0.7,
        "top_p": 0.9
      }
    }
    ```

- **Response (Non-streaming)**:

    ```json
    {
      "model": "google/gemini-pro:latest",
      "created_at": "2023-12-12T14:00:00Z",
      "message": {
        "role": "assistant",
        "content": "The sky is blue because of Rayleigh scattering..."
      },
      "done": true,
      "total_duration": 0,
      "load_duration": 0,
      "prompt_eval_count": null,
      "prompt_eval_duration": 0,
      "eval_count": 0,
      "eval_duration": 0
    }
    ```

- **Response (Streaming)**: A stream of JSON objects, each representing a token or a final summary.

    ```json
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":"The"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":" sky"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":" is"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":" blue"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":" because"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":" of"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":" Rayleigh"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":" scattering"},"done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","message":{"role":"assistant","content":"."},"done":true}
    ```

#### `POST /api/generate`

Handles text generation requests (a simpler version of `/api/chat`).

- **Request Body**:

    ```json
    {
      "model": "google/gemini-pro:latest",
      "prompt": "Once upon a time",
      "system": "You are a creative writer.",
      "stream": false,
      "options": {
        "temperature": 0.8
      }
    }
    ```

- **Response (Non-streaming)**:

    ```json
    {
      "model": "google/gemini-pro:latest",
      "created_at": "2023-12-12T14:00:00Z",
      "response": " there was a brave knight...",
      "done": true,
      "context": [],
      "total_duration": 0,
      "load_duration": 0,
      "prompt_eval_count": null,
      "prompt_eval_duration": 0,
      "eval_count": 0,
      "eval_duration": 0
    }
    ```

- **Response (Streaming)**: A stream of JSON objects.

    ```json
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","response":" there","done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","response":" was","done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","response":" a","done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","response":" brave","done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","response":" knight","done":false}
    {"model":"google/gemini-pro:latest","created_at":"2023-12-12T14:00:00Z","response":"...","done":true}
    ```

### Embeddings

#### `POST /api/embed` and `POST /api/embeddings`

Generates embeddings for a given input. Both endpoints are supported for compatibility.

- **Request Body for `/api/embed`**:

    ```json
    {
      "model": "text-embedding-ada-002",
      "input": "This is a test sentence."
    }
    ```

- **Request Body for `/api/embeddings`**:

    ```json
    {
      "model": "text-embedding-ada-002",
      "prompt": "This is a test sentence."
    }
    ```

- **Success Response (200 OK)**:

    ```json
    {
      "embedding": [0.1, 0.2, 0.3, ...]
    }
    ```

### Process Management

#### `GET /api/ps`

Lists running models (stubbed implementation).

- **Success Response (200 OK)**:

    ```json
    {
      "models": [],
      "created_at": "2023-12-12T14:00:00Z"
    }
    ```

### Multi-Provider Endpoints

#### `GET /api/providers`

Lists all configured providers and their status.

- **Success Response (200 OK)**:

    ```json
    {
      "providers": [
        {
          "type": "openrouter",
          "enabled": true,
          "healthy": true,
          "priority": 1,
          "request_count": 1234,
          "error_count": 5,
          "error_rate": 0.004,
          "avg_response_time_ms": 850.5
        },
        {
          "type": "openai",
          "enabled": true,
          "healthy": true,
          "priority": 2,
          "request_count": 567,
          "error_count": 2,
          "error_rate": 0.003,
          "avg_response_time_ms": 650.2
        }
      ]
    }
    ```

#### `GET /api/providers/{provider_type}/stats`

Get detailed statistics for a specific provider.

- **Success Response (200 OK)**:

    ```json
    {
      "provider_type": "openai",
      "enabled": true,
      "healthy": true,
      "priority": 2,
      "request_count": 567,
      "successful_requests": 565,
      "failed_requests": 2,
      "error_rate": 0.003,
      "avg_response_time_ms": 650.2,
      "min_response_time_ms": 200.1,
      "max_response_time_ms": 2500.8,
      "circuit_breaker_state": "closed",
      "last_health_check": "2023-12-12T14:00:00Z",
      "models_available": 25
    }
    ```

#### `GET /api/tags/{provider_type}`

Lists models available from a specific provider.

- **Success Response (200 OK)**:

    ```json
    {
      "models": [
        {
          "name": "gpt-4:latest",
          "provider": "openai",
          "modified_at": "2023-12-12T14:00:00Z",
          "size": 0,
          "digest": "sha256:abcdef1234567890",
          "details": {
            "format": "api",
            "family": "gpt",
            "families": ["gpt"],
            "parameter_size": "Unknown",
            "quantization_level": "Unknown"
          }
        }
      ]
    }
    ```

## Error Responses

The proxy returns standardized error responses for various conditions:

### Model Not Found (404)

```json
{
  "error": "Model 'nonexistent-model' not found.",
  "type": "model_not_found"
}
```

### Model Forbidden (403)

```json
{
  "error": "Model 'forbidden-model' is not allowed by the filter.",
  "type": "model_forbidden"
}
```

### OpenRouter API Error (502)

```json
{
  "error": "OpenRouter API error: 401 Unauthorized",
  "type": "openrouter_error"
}
```

### Internal Server Error (500)

```json
{
  "error": "Internal server error",
  "type": "internal_error"
}
```

## Unsupported Endpoints

The following Ollama API endpoints are not supported by the proxy and will return an `HTTP 501 Not Implemented` error:

- `POST /api/create`
- `POST /api/copy`
- `DELETE /api/delete`
- `POST /api/pull`
- `POST /api/push`
- `POST /api/blobs/{digest}`
- `HEAD /api/blobs/{digest}`

These endpoints are related to local model management, which is not applicable when using the OpenRouter proxy.
