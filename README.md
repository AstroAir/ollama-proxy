# Ollama Proxy

A modern, high-performance proxy server that translates Ollama API calls to OpenRouter, enabling seamless access to a wide variety of AI models through the familiar Ollama interface.

## Overview

Ollama Proxy acts as a bridge between any Ollama-compatible client and the OpenRouter API. This allows you to use your favorite tools and applications that support Ollama with the extensive range of models offered by OpenRouter, without needing to modify your client-side code.

## Features

-   **🔄 Seamless Translation**: Converts Ollama API calls to the OpenRouter format.
-   **🚀 High Performance**: Built with modern Python and `asyncio` for speed.
-   **⚙️ Flexible Configuration**: Configure via environment variables, `.env` files, or CLI arguments.
-   **🔍 Model Filtering**: Control which OpenRouter models are exposed.
-   **📊 Structured Logging**: JSON logs for better observability.
-   **🐳 Docker Support**: Easy to deploy with Docker and Docker Compose.

## Getting Started

### Prerequisites

-   Python 3.11+
-   An [OpenRouter API key](https://openrouter.ai/keys)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ollama-proxy.git
    cd ollama-proxy
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

3.  **Configure your API key:**
    Create a `.env` file and add your key:
    ```env
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    ```

4.  **Run the server:**
    ```bash
    ollama-proxy
    ```

For more detailed instructions, see the [full documentation](docs/index.md).

## Quick Start Examples

### Basic Chat Request
```bash
# Non-streaming chat
curl http://localhost:11434/api/chat -d '{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": false
}'

# Streaming chat
curl http://localhost:11434/api/chat -d '{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}'
```

### List Available Models
```bash
curl http://localhost:11434/api/tags
```

### Using with Python
```python
import requests

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": False
})
print(response.json())
```

## Common Issues

### API Key Issues
- **Error**: "OpenRouter API key is required"
  - **Solution**: Set `OPENROUTER_API_KEY` environment variable or use `--api-key` flag

### Model Not Found
- **Error**: "Model 'xyz' not found"
  - **Solution**: Check available models with `curl http://localhost:11434/api/tags`
  - Use the exact model name from the list (e.g., "gpt-4:latest")

### Connection Issues
- **Error**: Connection refused on port 11434
  - **Solution**: Ensure the proxy is running and check the host/port configuration
  - Use `--host 0.0.0.0 --port 8080` to customize binding

## Documentation

For detailed information about configuration, API compatibility, deployment, and architecture, please refer to our [full documentation](docs/index.md).

-   [**Introduction**](docs/index.md)
-   [**Configuration Guide**](docs/CONFIGURATION.md)
-   [**API Reference**](docs/API_REFERENCE.md)
-   [**Usage Examples**](docs/USAGE_EXAMPLES.md)
-   [**Deployment Guide**](docs/DEPLOYMENT.md)
-   [**Architecture Overview**](docs/ARCHITECTURE.md)
-   [**Troubleshooting Guide**](docs/TROUBLESHOOTING.md)

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
