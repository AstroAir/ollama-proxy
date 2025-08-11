---
title: Ollama Proxy
hide:
  - toc
---

A modern, high-performance proxy server that translates Ollama API calls to OpenRouter, enabling seamless access to a wide variety of AI models through the familiar Ollama interface.

[![PyPI](https://img.shields.io/pypi/v/ollama-proxy)](https://pypi.org/project/ollama-proxy/)
[![License](https://img.shields.io/github/license/your-username/ollama-proxy)](https://github.com/AstroAir/ollama-proxy/blob/main/LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/ollama-proxy)](https://pypi.org/project/ollama-proxy/)

---

## Overview

Ollama Proxy acts as a bridge between any Ollama-compatible client and the OpenRouter API. This allows you to use your favorite tools and applications that support Ollama with the extensive range of models offered by OpenRouter, without needing to modify your client-side code.

## Features

- **🔄 Seamless Translation**: Converts Ollama API calls to the OpenRouter format.
- **🚀 High Performance**: Built with modern Python and `asyncio` for speed.
- **⚙️ Flexible Configuration**: Configure via environment variables, `.env` files, or CLI arguments.
- **🔍 Model Filtering**: Control which OpenRouter models are exposed.
- **📊 Structured Logging**: JSON logs for better observability.
- **🐳 Docker Support**: Easy to deploy with Docker and Docker Compose.

## Quick Start

### Prerequisites

- Python 3.12+
- An [OpenRouter API key](https://openrouter.ai/keys)

### Installation {#installation}

1. **Clone the repository:**

    ```bash
    git clone https://github.com/AstroAir/ollama-proxy.git
    cd ollama-proxy
    ```

2. **Install dependencies:**

    ```bash
    pip install -e .
    ```

3. **Configure your API key:**
    Create a `.env` file and add your key:

    ```env
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    ```

4. **Run the server:**

    ```bash
    ollama-proxy
    ```

Once the server is running, you can configure your Ollama client to point to `http://localhost:11434` (or your custom host and port).

## Documentation

Explore our comprehensive documentation to learn more about configuring and using the Ollama Proxy:

### Core Documentation

- [**Configuration Guide**](configuration.md): Learn how to customize the proxy's behavior.
- [**API Reference**](api-reference.md): See the full list of supported Ollama API endpoints.
- [**Usage Examples**](usage-examples.md): Practical examples of how to use the proxy with various tools.
- [**Deployment Guide**](deployment.md): Find out how to deploy the proxy in a production environment.
- [**Architecture Overview**](architecture.md): Get a deeper understanding of the project's design.
- [**Troubleshooting Guide**](troubleshooting.md): Find solutions to common problems.

### Advanced Features

- [**Multi-Provider Support**](multi-provider.md): Configure and use multiple AI providers with intelligent routing.
- [**CLI Tools & Administration**](cli-tools.md): Comprehensive guide to command-line tools and administrative interfaces.

## Contributing

Contributions are welcome! Please see the [Contributing Guide](contributing.md) for details on how to contribute to the project.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/AstroAir/ollama-proxy/blob/main/LICENSE) file for details.
