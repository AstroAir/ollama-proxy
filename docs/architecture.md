# Architecture Overview

This document provides a deep dive into the architecture of the Ollama Proxy, its key components, and the design principles behind it.

## Project Structure

The project is organized into two main directories: `src` for the application's source code and `tests` for the test suite.

```
src/
├── __init__.py
├── main.py              # Entry point and CLI
├── app.py               # FastAPI application factory
├── config.py            # Configuration management
├── models.py            # Pydantic models and schemas
├── api.py               # API route handlers
├── openrouter.py        # OpenRouter client
├── utils.py             # Utility functions
├── exceptions.py        # Custom exceptions
├── logging_config.py    # Logging configuration
└── monitoring.py        # Monitoring and metrics collection
```

## Key Components

### 1. FastAPI Application (`app.py`)

The core of the proxy is a [FastAPI](https://fastapi.tiangolo.com/) application. FastAPI was chosen for its high performance, asynchronous support, and automatic generation of interactive API documentation.

The `create_app` function in `app.py` is a factory that initializes and configures the FastAPI application, including setting up logging, state management, and API routers.

Key features of the FastAPI application:

- **Lifespan Management**: Uses FastAPI's lifespan feature to initialize resources on startup and clean them up on shutdown
- **Middleware**: Includes custom middleware for request logging and error handling
- **Exception Handlers**: Global exception handlers for consistent error responses
- **CORS Support**: Cross-Origin Resource Sharing support for web client integration

### 2. Configuration (`config.py`)

Configuration is managed by [Pydantic's `BaseSettings`](https://docs.pydantic.dev/latest/usage/settings/). This allows for a robust and type-safe configuration system that can read settings from environment variables and `.env` files. The `get_settings` function provides a cached instance of the settings, ensuring they are loaded only once.

Key features of the configuration system:

- **Validation**: Strong validation of configuration values with custom validators
- **Type Safety**: Full type hints for all configuration options
- **Environment-based**: Support for different environments (development, staging, production)
- **Computed Properties**: Derived configuration values based on other settings
- **Model Filtering**: Advanced model filtering with pattern matching support

### 3. API Endpoints (`api.py`)

All API logic is contained within `api.py`. This module defines the API routes that mimic the Ollama API. Each route handler is responsible for:

- Receiving the incoming request.
- Validating the request body using Pydantic models defined in `models.py`.
- Calling the OpenRouter client to perform the requested action.
- Translating the response from OpenRouter back into the Ollama format.
- Handling any errors that may occur during the process.

Key features of the API implementation:

- **Dependency Injection**: Uses FastAPI's dependency injection for clean, testable code
- **Enhanced Error Handling**: Context managers for consistent error handling
- **Streaming Support**: Full support for streaming responses from OpenRouter
- **Model Resolution**: Intelligent model name resolution and validation
- **Request Tracking**: Request ID generation and tracking for observability

### 4. OpenRouter Client (`openrouter.py`)

This component is responsible for all communication with the OpenRouter API. It uses the `httpx` library for making asynchronous HTTP requests, which is essential for the proxy's non-blocking, high-performance design.

The `OpenRouterClient` class encapsulates the logic for making requests to OpenRouter's chat, generation, and embedding endpoints.

Key features of the OpenRouter client:

- **Asynchronous HTTP**: Uses `httpx.AsyncClient` for high-performance async requests
- **Connection Pooling**: Efficient connection reuse for better performance
- **Timeout Management**: Configurable timeouts for different types of requests
- **Error Handling**: Comprehensive error handling for various HTTP status codes
- **Streaming Support**: Support for both streaming and non-streaming responses

### 5. Data Models (`models.py`)

This file contains all the Pydantic models that define the data structures for API requests and responses. These models are used for data validation, serialization, and ensuring type safety throughout the application. They are crucial for maintaining compatibility with the Ollama API.

Key features of the data models:

- **Comprehensive Coverage**: Models for all supported Ollama API endpoints
- **Validation**: Strict validation of request and response data
- **Type Safety**: Full type hints for all model fields
- **Extensibility**: Easy to extend with new fields or models as needed
- **Documentation**: Clear field descriptions for better understanding

### 6. Monitoring and Metrics (`monitoring.py`)

The monitoring system provides insights into the proxy's performance and health.

Key features of the monitoring system:

- **Metrics Collection**: Collects key metrics like request counts, response times, and error rates
- **Health Checks**: Comprehensive health status reporting
- **Endpoint Statistics**: Detailed statistics for each API endpoint
- **Performance Tracking**: Tracks performance metrics over time
- **Export-ready**: Metrics formatted for easy integration with monitoring systems

## Design Principles

### Dependency Injection

The application makes extensive use of FastAPI's dependency injection system. This promotes clean, decoupled code. For example, the `OpenRouterClient` and `AppState` are provided to the route handlers as dependencies, making the code easier to test and maintain.

Benefits of dependency injection:

- **Testability**: Easy to mock dependencies in unit tests
- **Reusability**: Components can be reused in different contexts
- **Maintainability**: Changes to dependencies don't require changes to dependent code
- **Flexibility**: Easy to swap implementations (e.g., different HTTP clients)

### Structured Logging

The proxy uses the `structlog` library to produce structured logs in JSON format. This is invaluable for observability in a production environment, as it allows for easy parsing, searching, and filtering of logs. Contextual information, such as request IDs, is automatically added to the logs.

Benefits of structured logging:

- **Searchability**: Easy to search and filter logs using tools like ELK stack
- **Analysis**: Structured data can be easily analyzed for trends and patterns
- **Correlation**: Request IDs allow correlating log entries across different components
- **Automation**: Structured logs can be easily processed by automated systems

### Robust Error Handling

A custom exception hierarchy is defined in `exceptions.py`. This allows for a clear and consistent way of handling errors. A custom middleware catches these exceptions and converts them into the appropriate HTTP responses, ensuring that the client always receives a meaningful error message.

Error handling features:

- **Hierarchical Exceptions**: Clear exception hierarchy for different error types
- **Context Preservation**: Error context is preserved for better debugging
- **Consistent Responses**: All errors result in consistent JSON responses
- **Error Classification**: Errors are classified by type for easier handling

### Asynchronous Operations

To handle a high number of concurrent requests efficiently, the entire request-response cycle is asynchronous. The use of `async/await` with `httpx` and FastAPI ensures that the server is non-blocking and can handle I/O-bound operations (like making requests to OpenRouter) without getting blocked.

Benefits of asynchronous operations:

- **High Concurrency**: Can handle many concurrent requests with minimal resources
- **Better Resource Utilization**: More efficient use of CPU and memory
- **Scalability**: Scales better under high load
- **Performance**: Lower latency for I/O-bound operations

### State Management

The application uses FastAPI's state management to store shared resources like the OpenRouter client and model mappings. This ensures that resources are properly initialized and shared across requests.

State management features:

- **Application Lifecycle**: Proper initialization and cleanup of resources
- **Thread Safety**: Safe access to shared resources in concurrent environments
- **Resource Sharing**: Efficient sharing of expensive resources across requests

## Monitoring and Observability

The application is designed with monitoring in mind:

- **Structured Logs**: As mentioned, logs are in JSON format for easy analysis.
- **Health Check**: The `/health` endpoint provides a simple way to check the status of the proxy.
- **Metrics**: The `/metrics` endpoint exposes key metrics, such as request counts, error rates, and response times. This can be integrated with monitoring tools like Prometheus to provide a real-time view of the application's performance.

### Metrics Collection

The proxy collects various metrics to help monitor its performance:

1. **Request Metrics**:
   - Total request count
   - Request rate
   - Response time distribution
   - Error rates by type

2. **Endpoint Metrics**:
   - Per-endpoint request counts
   - Per-endpoint response times
   - Per-endpoint error rates

3. **System Metrics**:
   - Memory usage
   - CPU usage
   - Uptime

### Health Monitoring

The health check system provides comprehensive information about the proxy's status:

- **Application Health**: Overall health status (healthy, degraded, unhealthy)
- **Uptime**: How long the application has been running
- **Model Availability**: Number of available models
- **Request Statistics**: Request and error counts
- **Performance Metrics**: Error rates and response times

### Integration with Monitoring Systems

The proxy's metrics endpoint is designed to integrate easily with popular monitoring systems:

- **Prometheus**: Metrics are exposed in a Prometheus-compatible format
- **Datadog**: Structured logs can be easily shipped to Datadog
- **CloudWatch**: AWS CloudWatch integration for cloud deployments
- **ELK Stack**: Elasticsearch, Logstash, and Kibana integration for on-premises deployments

## Security Considerations

The architecture includes several security features:

- **Input Validation**: All inputs are validated using Pydantic models
- **API Key Management**: Secure handling of OpenRouter API keys
- **Rate Limiting**: Configurable request rate limiting
- **Error Sanitization**: Sensitive information is not exposed in error messages
- **Secure Dependencies**: Regular updates to dependencies to address security vulnerabilities

## Performance Optimization

Several techniques are used to optimize performance:

- **Connection Reuse**: HTTP connections to OpenRouter are reused for better performance
- **Caching**: Model mappings and other data are cached to reduce processing time
- **Asynchronous Processing**: Non-blocking I/O operations for better throughput
- **Memory Efficiency**: Efficient data structures and memory management
- **Resource Pooling**: Shared resources like HTTP clients are pooled for reuse
