# Ollama Proxy

A modern, high-performance multi-provider proxy server that translates Ollama API calls to multiple AI providers, enabling seamless access to a wide variety of AI models through the familiar Ollama interface.

## Overview

Ollama Proxy acts as a bridge between any Ollama-compatible client and multiple AI providers including OpenRouter, OpenAI, Anthropic Claude, Google Gemini, Azure OpenAI, AWS Bedrock, and local Ollama instances. This allows you to use your favorite tools and applications that support Ollama with the extensive range of models offered by these providers, without needing to modify your client-side code.

## Features

- **🔄 Multi-Provider Support**: Seamlessly connects to OpenRouter, OpenAI, Anthropic, Google, Azure OpenAI, AWS Bedrock, and local Ollama
- **🧠 Intelligent Routing**: Model-based routing, load balancing, and automatic failover between providers
- **🚀 High Performance**: Built with modern Python and `asyncio` for maximum throughput and scalability
- **⚙️ Flexible Configuration**: Configure via environment variables, `.env` files, or CLI arguments
- **🔍 Advanced Filtering**: Control which models are exposed from each provider
- **📊 Comprehensive Monitoring**: Detailed metrics, health checks, and performance tracking
- **🔧 Scalability Features**: Connection pooling, request queuing, rate limiting, and async optimization
- **🛡️ Robust Error Handling**: Circuit breakers, retry logic, and graceful degradation
- **🐳 Docker Support**: Easy to deploy with Docker and Docker Compose

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [OpenRouter API key](https://openrouter.ai/keys)

### Quick Start

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AstroAir/ollama-proxy.git
   cd ollama-proxy
   ```

2. **Set up development environment (recommended):**

   ```bash
   # Unix/Linux/macOS
   ./scripts/dev-setup.sh

   # Windows
   scripts\dev-setup.bat

   # Or use make
   make quickstart
   ```

3. **Configure your API key:**
   Edit the `.env` file created during setup:

   ```env
   OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```

4. **Start the server:**

   ```bash
   # Multiple ways to start:
   ollama-proxy                    # Basic start
   ollama-proxy-dev               # Development mode
   ollama-proxy-cli server        # CLI interface
   make dev                       # Using make
   python scripts/launcher.py     # Cross-platform launcher
   ```

### Alternative Installation Methods

#### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --all-extras --dev
```

#### Using pip

```bash
pip install -e .
# For development
pip install -e ".[test]"
```

#### Using Docker

```bash
docker build -t ollama-proxy .
docker run -p 11434:11434 -e OPENROUTER_API_KEY=your_key ollama-proxy
```

For more detailed instructions, see the [full documentation](docs/index.md).

## Usage

### Entry Points

Ollama Proxy provides multiple entry points for different use cases:

#### Main Entry Points

- `ollama-proxy` - Standard server mode
- `ollama-proxy-server` - Alias for standard mode
- `ollama-proxy-dev` - Development mode with auto-reload
- `ollama-proxy-daemon` - Daemon mode for background services

#### Administrative Tools

- `ollama-proxy-admin` - Administrative interface
- `ollama-proxy-health` - Health check utility
- `ollama-proxy-config` - Configuration management
- `ollama-proxy-benchmark` - Performance benchmarking

#### Development Tools

- `ollama-proxy-test` - Test runner
- `ollama-proxy-lint` - Code linting
- `ollama-proxy-format` - Code formatting

#### Unified CLI Interface

- `ollama-proxy-cli` - Unified command-line interface with subcommands

### Command Examples

#### Starting the Server

```bash
# Basic server start
ollama-proxy

# Development mode with auto-reload
ollama-proxy-dev

# Custom host and port
ollama-proxy --host 0.0.0.0 --port 8080

# With specific API key
ollama-proxy --api-key sk-or-your-key-here

# Daemon mode
ollama-proxy-daemon
```

#### Using the Unified CLI

```bash
# Start server (default subcommand)
ollama-proxy-cli server --host 0.0.0.0 --port 8080

# Development mode
ollama-proxy-cli dev

# Administrative tasks
ollama-proxy-cli admin status
ollama-proxy-cli admin config --format json
ollama-proxy-cli admin models

# Health checks
ollama-proxy-cli health --json
ollama-proxy-cli health --host production.com

# Configuration management
ollama-proxy-cli config --show --format yaml
ollama-proxy-cli config --validate

# Performance testing
ollama-proxy-cli benchmark --requests 1000 --concurrency 50

# Development tasks
ollama-proxy-cli test --coverage
ollama-proxy-cli lint --fix
ollama-proxy-cli format --check
```

#### Cross-Platform Scripts

```bash
# Unix/Linux/macOS
./scripts/start-server.sh --dev
./scripts/test-runner.sh --coverage --html-report
./scripts/maintenance.sh health

# Windows
scripts\start-server.bat --dev
scripts\test-runner.bat --coverage --html-report
scripts\maintenance.bat health

# Cross-platform Python launcher
python scripts/launcher.py --dev
python scripts/launcher.py --host 0.0.0.0 --port 8080
```

#### Using Make/Batch Commands

```bash
# Unix/Linux/macOS with Make
make quickstart          # Complete setup
make dev                 # Start development server
make test-cov           # Run tests with coverage
make health-check       # Check server health
make benchmark          # Run benchmarks

# Windows with make.bat
make.bat quickstart     # Complete setup
make.bat dev           # Start development server
make.bat test-cov      # Run tests with coverage
make.bat health-check  # Check server health
```

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

## Configuration

### Environment Variables

Ollama Proxy can be configured using environment variables or a `.env` file:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key | - | ✅ |
| `HOST` | Host to bind to | `0.0.0.0` | ❌ |
| `PORT` | Port to listen on | `11434` | ❌ |
| `LOG_LEVEL` | Logging level | `INFO` | ❌ |
| `ENVIRONMENT` | Environment (development/production) | `development` | ❌ |
| `MODELS_FILTER_PATH` | Path to model filter file | `models-filter.txt` | ❌ |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` | ❌ |
| `OPENROUTER_TIMEOUT` | Request timeout in seconds | `300` | ❌ |
| `MAX_CONCURRENT_REQUESTS` | Maximum concurrent requests | `100` | ❌ |
| `DEBUG` | Enable debug mode | `false` | ❌ |
| `RELOAD` | Enable auto-reload | `false` | ❌ |

### Configuration File Example

Create a `.env` file in the project root:

```env
# Required
OPENROUTER_API_KEY=sk-or-your-api-key-here

# Server Configuration
HOST=0.0.0.0
PORT=11434
LOG_LEVEL=INFO
ENVIRONMENT=production

# OpenRouter Settings
OPENROUTER_TIMEOUT=300
MAX_CONCURRENT_REQUESTS=100

# Model Filtering
MODELS_FILTER_PATH=models-filter.txt

# Development Settings
DEBUG=false
RELOAD=false
```

### Model Filtering

Create a `models-filter.txt` file to control which models are available:

```text
# OpenRouter model IDs to expose
openai/gpt-4
openai/gpt-3.5-turbo
anthropic/claude-3-sonnet
meta-llama/llama-2-70b-chat
```

### Command Line Arguments

All configuration options can be overridden via command line arguments:

```bash
ollama-proxy \
  --host 127.0.0.1 \
  --port 8080 \
  --api-key sk-or-your-key \
  --log-level DEBUG \
  --models-filter custom-models.txt \
  --reload
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

## Development

### Development Setup

1. **Quick setup with scripts:**

   ```bash
   # Unix/Linux/macOS
   ./scripts/dev-setup.sh

   # Windows
   scripts\dev-setup.bat

   # Or using make
   make quickstart
   ```

2. **Manual setup:**

   ```bash
   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync --all-extras --dev

   # Set up pre-commit hooks
   uv run pre-commit install

   # Create .env file
   cp .env.example .env
   # Edit .env and set your OPENROUTER_API_KEY
   ```

### Development Workflow

```bash
# Start development server
make dev
# or
ollama-proxy-dev
# or
python scripts/launcher.py --dev

# Run tests
make test
# or with coverage
make test-cov
# or watch mode
./scripts/test-runner.sh --watch

# Code quality checks
make check-all
# or individual checks
make lint
make format-check
make type-check

# Format code
make format
```

### Available Scripts

The `scripts/` directory contains various utility scripts:

- **`dev-setup.sh/.bat`** - Complete development environment setup
- **`start-server.sh/.bat/.ps1`** - Cross-platform server startup
- **`test-runner.sh/.bat`** - Advanced test runner with multiple options
- **`build-deploy.sh`** - Build and deployment automation
- **`maintenance.sh/.bat`** - Server maintenance and monitoring
- **`launcher.py`** - Universal Python launcher (works on all platforms)

### Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
./scripts/test-runner.sh --unit --fast
./scripts/test-runner.sh --integration --verbose

# Watch mode for development
./scripts/test-runner.sh --watch

# Performance tests
make benchmark
```

### Code Quality

```bash
# Run all quality checks
make check-all

# Individual checks
make lint          # Linting with flake8
make format-check  # Check formatting
make type-check    # Type checking with mypy

# Auto-fix issues
make format        # Format with black and isort
ollama-proxy-lint --fix  # Auto-fix linting issues
```

### Building and Deployment

```bash
# Build package
make build

# Build Docker image
make docker-build

# Run deployment checks
./scripts/build-deploy.sh check

# Create release
./scripts/build-deploy.sh release --version 1.0.0
```

### Monitoring and Maintenance

```bash
# Check server health
make health-check
ollama-proxy-health

# Show configuration
ollama-proxy-config --show

# Performance monitoring
make benchmark
./scripts/maintenance.sh monitor

# Cleanup
make cleanup
./scripts/maintenance.sh cleanup
```

## Documentation

For detailed information about configuration, API compatibility, deployment, and architecture, please refer to our [full documentation](docs/index.md).

### Core Documentation

- [**Introduction**](docs/index.md)
- [**Configuration Guide**](docs/configuration.md)
- [**API Reference**](docs/api-reference.md)
- [**Usage Examples**](docs/usage-examples.md)
- [**Deployment Guide**](docs/deployment.md)
- [**Architecture Overview**](docs/architecture.md)
- [**Troubleshooting Guide**](docs/troubleshooting.md)

### Advanced Features

- [**Multi-Provider Support**](docs/multi-provider.md) - Configure multiple AI providers with intelligent routing
- [**CLI Tools & Administration**](docs/cli-tools.md) - Comprehensive command-line tools and administrative interfaces

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
