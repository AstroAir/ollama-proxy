# Ollama Proxy - Project Context for AI Assistant

## Overview

This project is a modern, high-performance proxy server called **Ollama Proxy**. Its primary function is to translate API calls from the [Ollama](https://ollama.com/) format to the [OpenRouter](https://openrouter.ai/) API format. This allows users to leverage the wide variety of AI models available through OpenRouter using familiar Ollama client tools and libraries without any modifications.

The application is built using modern Python (3.12+), leveraging `asyncio` for performance and structured with FastAPI for the web framework. It uses `uv` for fast dependency management and packaging.

## Key Technologies and Architecture

* **Language**: Python 3.12+
* **Framework**: FastAPI (ASGI)
* **Async**: `asyncio`, `httpx`
* **Packaging/Dependencies**: `uv`, `pyproject.toml` (PEP 621)
* **Configuration**: `pydantic-settings`
* **Logging**: `structlog` (structured JSON logging)
* **Testing**: `pytest`, `pytest-asyncio`, `pytest-cov`
* **Code Quality**: `black`, `isort`, `flake8`, `mypy`, `pre-commit`
* **Deployment**: Docker, Docker Compose
* **API Communication**: `httpx`

### Core Components

1. **Main Application (`src/app.py`)**: Sets up the FastAPI application, configures middleware (CORS, logging), registers exception handlers, and manages the application lifecycle via a lifespan context manager. It initializes the OpenRouter client and builds model mappings on startup.
2. **Configuration (`src/config.py`)**: Uses `pydantic-settings` to manage configuration from environment variables, `.env` files, and command-line arguments. Defines `Settings` and `ModelFilter` dataclasses.
3. **API Endpoints (`src/api.py`)**: Defines FastAPI routes for various Ollama API endpoints (`/`, `/api/version`, `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/embed`, `/api/embeddings`, `/health`, `/metrics`). Translates Ollama requests to OpenRouter format, calls the OpenRouter client, and translates responses back.
4. **OpenRouter Client (`src/openrouter.py`)**: An `httpx.AsyncClient`-based wrapper for interacting with the OpenRouter API. Handles authentication, request/response processing, error handling, and provides methods for chat completions, model listing, and embeddings.
5. **Models (`src/models.py`)**: Pydantic models defining the schemas for both Ollama API requests/responses and internal representations.
6. **Utilities (`src/utils.py`)**: Helper functions for model name mapping and filtering.
7. **Exceptions (`src/exceptions.py`)**: Custom exception hierarchy for structured error handling.
8. **Monitoring (`src/monitoring.py`)**: (Likely) Handles metrics collection and reporting (file content not fully read but inferred from imports).
9. **Entry Point (`src/main.py`)**: Parses command-line arguments, loads settings, and starts the Uvicorn ASGI server.

## Building, Running, and Development

### Prerequisites

* Python 3.12+
* `uv` (for dependency management and project tasks)
* An OpenRouter API key.

### Setup

1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate` (or use `uv` directly).
3. Install dependencies:
    * **Production**: `make install` (or `uv sync --no-dev`)
    * **Development**: `make install-dev` (or `uv sync --all-extras --dev`)

### Configuration

Configuration is done via environment variables or a `.env` file (see `.env.example`).

Key variables include:

* `OPENROUTER_API_KEY` (Required)
* `HOST` (default: 0.0.0.0)
* `PORT` (default: 11434)
* `LOG_LEVEL` (default: INFO)
* `MODELS_FILTER` (path to a file listing allowed models, see `models-filter.txt.example`)
* `OPENROUTER_BASE_URL` (default: <https://openrouter.ai/api/v1>)

### Running the Server

* **Development**: `make dev` (or `uv run python -m src.main --reload --log-level DEBUG`)
* **Production**: `make run` (or `uv run python -m src.main`)
* **With Docker**: `make docker-build && make docker-run` (requires setting `OPENROUTER_API_KEY`)

### Running Tests

* `make test` (or `uv run pytest`)
* `make test-cov` (or `uv run pytest --cov=src`)

### Code Quality Checks

* **Linting**: `make lint` (or `uv run flake8 src tests`)
* **Formatting**: `make format` (or `uv run black src tests && uv run isort src tests`)
* **Type Checking**: `make type-check` (or `uv run mypy src`)
* **All Checks**: `make check-all`

## Development Conventions

* **Code Style**: Enforced by `black` (formatter), `isort` (import sorter), `flake8` (linter), and `mypy` (type checker). Pre-commit hooks (`pre-commit install`) are used to automate checks.
* **Testing**: Uses `pytest` with `pytest-asyncio` for async tests. Tests are located in the `tests/` directory.
* **Documentation**: Inline code comments and docstrings are used. External documentation is located in the `docs/` directory (not fully explored here).
* **Configuration Validation**: Heavy use of Pydantic for validating settings and request/response data.
* **Error Handling**: Custom exceptions with structured data for consistent API error responses.
* **Logging**: Structured logging with `structlog` for better observability.
* **Dependencies**: Managed by `uv` and defined in `pyproject.toml`.
* **Modern Python Features**: Extensive use of type hints, dataclasses, `match` statements (structural pattern matching), context managers, and async/await.

## Key Files

* `README.md`: Project overview and usage instructions.
* `pyproject.toml`: Project metadata, dependencies, build system, and tool configurations.
* `Makefile`: Common development commands.
* `Dockerfile`, `docker-compose.yml`: Containerization definitions.
* `src/main.py`: Application entry point.
* `src/app.py`: FastAPI application factory and setup.
* `src/config.py`: Configuration management.
* `src/api.py`: API route definitions and handlers.
* `src/openrouter.py`: OpenRouter API client.
* `src/models.py`: Data models for API requests/responses.
* `src/utils.py`: Utility functions.
* `src/exceptions.py`: Custom exceptions.
* `tests/`: Unit and integration tests.
* `.env.example`, `models-filter.txt.example`: Configuration examples.
* `.pre-commit-config.yaml`: Pre-commit hook configuration.

## Guidance Examples for Common Tasks

### 1. Adding a New Ollama API Endpoint

To add support for a new Ollama endpoint (e.g., `/api/new-feature`):

1. **Define Models (if needed)**: Add request/response Pydantic models in `src/models.py`.
2. **Implement Handler**: Create a new async function in `src/api.py` decorated with the appropriate FastAPI decorator (e.g., `@router.post("/api/new-feature")`). This function should:
    * Parse the request using your new models.
    * Translate the request to the equivalent OpenRouter API call format.
    * Use the `OpenRouterClient` (obtained via dependency injection: `openrouter_client: OpenRouterClient = Depends(get_openrouter_client)`) to make the API call.
    * Translate the OpenRouter response back to the Ollama format.
    * Return the Ollama-formatted response.
    * Handle errors appropriately using custom exceptions from `src/exceptions.py`.
3. **Dependency Injection**: If your endpoint needs access to application state or the OpenRouter client, use FastAPI's `Depends` as shown in existing endpoints.
4. **Testing**: Add unit and/or integration tests in the `tests/` directory for your new endpoint and handler logic.

### 2. Modifying Request/Response Translation Logic

The core translation logic resides in `src/api.py` and `src/openrouter.py`.

* **Ollama to OpenRouter**: This happens in the API handler functions in `src/api.py` (e.g., `_build_chat_payload` for `/api/chat`).
* **OpenRouter to Ollama**: This also happens in the API handler functions in `src/api.py` (e.g., processing the response in `_handle_non_streaming_chat` or the streaming generator in `_handle_streaming_chat`).

Locate the relevant handler function and modify the data transformation logic as needed.

### 3. Adding a New Configuration Option

1. **Update Settings Model**: Add a new field to the `Settings` class in `src/config.py`. Use Pydantic field definitions for validation, default values, and descriptions.
2. **Use in Code**: Access the setting via the `AppState` object (available in API handlers via `app_state: AppState = Depends(get_app_state)`) or directly in the `Settings` object where it's injected (e.g., `OpenRouterClient`).
3. **Documentation**: Update `.env.example` and potentially user-facing documentation if the setting is meant to be used by end-users.

### 4. Extending the OpenRouter Client

To add a new capability to the OpenRouter client (e.g., support a new OpenRouter API endpoint):

1. **Add Endpoint Constant**: Add the new endpoint path to the `OpenRouterEndpoint` enum in `src/openrouter.py`.
2. **Add Method**: Add a new async method to the `OpenRouterClient` class. Follow the patterns of existing methods like `chat_completion` or `fetch_models`. Use the internal `_get_client` context manager for the HTTP client, `_get_headers` for authentication, and `_handle_response_error` for consistent error handling. Update metrics as appropriate.
3. **Testing**: Add tests for the new client method in `tests/test_openrouter_client.py`.

### 5. Debugging an API Request

1. **Enable Debug Logging**: Run the server with `LOG_LEVEL=DEBUG`.
2. **Check Logs**: Logs are structured JSON by default, making them easy to parse. Look for log entries related to the request path and request ID.
3. **Add Temporary Logs**: Use `structlog.get_logger(__name__)` to get a logger in your module and add `logger.debug("Checkpoint message", key=value)` calls to trace execution flow and data states.
4. **Use the Metrics Endpoint**: Access `/metrics` to get performance and error statistics.
5. **Use the Health Endpoint**: Access `/health` to check the overall status of the application.
