# Configuration Guide

The Ollama Proxy is highly configurable, allowing you to tailor its behavior to your specific needs. You can configure the application through environment variables, a `.env` file, or command-line arguments.

## Configuration Methods

### 1. Environment Variables

You can set configuration options using environment variables. For example:

```bash
export OPENROUTER_API_KEY="your_api_key"
export PORT=8080
ollama-proxy
```

### 2. `.env` File

Create a `.env` file in the project's root directory to store your configuration. This is the recommended approach for development.

```env
# .env
OPENROUTER_API_KEY="your_openrouter_api_key_here"
HOST=0.0.0.0
PORT=11434
LOG_LEVEL=INFO
```

### 3. Command-Line Arguments

You can override settings from the environment or `.env` file using command-line arguments when you start the server.

```bash
ollama-proxy --port 8080 --log-level DEBUG
```

**Priority Order:** Command-line arguments > Environment variables > `.env` file.

## All Configuration Options

### Environment Variables

| Variable                  | Default                        | Description                                                                                                 |
| ------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `OPENROUTER_API_KEY`      | **Required**                   | Your API key for OpenRouter. This is essential for the proxy to function.                                   |
| `HOST`                    | `0.0.0.0`                      | The host address the server will bind to. Use `0.0.0.0` to accept connections from any IP.                  |
| `PORT`                    | `11434`                        | The port the server will listen on. `11434` is the default Ollama port.                                      |
| `LOG_LEVEL`               | `INFO`                         | The logging level. Options are `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.                               |
| `MODELS_FILTER_PATH`      | `models-filter.txt`            | The path to the file used for filtering models. See [Model Filtering](#model-filtering) for more details.     |
| `OPENROUTER_BASE_URL`     | `https://openrouter.ai/api/v1` | The base URL for the OpenRouter API. You typically won't need to change this.                                |
| `OPENROUTER_TIMEOUT`      | `300`                          | The timeout in seconds for requests made to the OpenRouter API.                                             |
| `MAX_CONCURRENT_REQUESTS` | `100`                          | The maximum number of concurrent requests the proxy will handle.                                            |
| `DEBUG`                   | `false`                        | Set to `true` to enable debug mode, which provides more verbose logging.                                    |
| `RELOAD`                  | `false`                        | Set to `true` to enable auto-reloading for development. The server will restart when code changes are detected. |

### Command-Line Options

To see all available command-line options, run:

```bash
ollama-proxy --help
```

- `--host`: Overrides the `HOST` environment variable.
- `--port`: Overrides the `PORT` environment variable.
- `--api-key`: Overrides the `OPENROUTER_API_KEY` environment variable.
- `--models-filter`: Overrides the `MODELS_FILTER_PATH` environment variable.
- `--log-level`: Overrides the `LOG_LEVEL` environment variable.
- `--reload`: Overrides the `RELOAD` environment variable.

## Model Filtering

You can control which OpenRouter models are available through the proxy by creating a model filter file. By default, the proxy looks for a file named `models-filter.txt` in the root directory.

To use model filtering:

1. Create a file (e.g., `models-filter.txt`).
2. Add the desired model names to the file, one per line. You should use the Ollama-compatible model name (e.g., `google/gemini-pro:latest`).

### Basic Model Filtering

**Example `models-filter.txt`:**

```text
# This is a comment and will be ignored
gpt-4:latest
claude-3-5-sonnet:latest
llama-2-7b-chat:latest
gemini-pro:latest
```

### Advanced Model Filtering

The model filter supports more advanced patterns:

```text
# Exact model names
gpt-4:latest
claude-3-5-sonnet:latest

# Wildcard patterns
llama*:latest
*gemini*:*
claude*:*

# Exclude specific models (prefix with !)
!llama-2-13b-chat:latest
!gpt-3-5-turbo:*

# Comments for organization
# OpenAI models
gpt-4:latest
gpt-3-5-turbo:latest

# Anthropic models
claude-3-5-sonnet:latest
claude-3-haiku:latest
```

### Filter File Locations

You can specify a custom filter file location:

```bash
# Using environment variable
export MODELS_FILTER_PATH="/path/to/your/custom-filter.txt"
ollama-proxy

# Using command-line argument
ollama-proxy --models-filter /path/to/your/custom-filter.txt
```

### Testing Your Filter

To test your filter configuration:

1. Start the proxy with your filter:

   ```bash
   ollama-proxy --models-filter ./your-filter.txt
   ```

2. Check which models are available:

   ```bash
   curl http://localhost:11434/api/tags
   ```

3. Or use the Ollama CLI:

   ```bash
   ollama list
   ```

If the filter file is empty or does not exist, all models from OpenRouter will be available.

## Environment-Specific Configuration

### Development Configuration

For development, you might want more verbose logging and auto-reload:

```env
# .env.development
OPENROUTER_API_KEY="your_dev_api_key"
HOST=localhost
PORT=11434
LOG_LEVEL=DEBUG
RELOAD=true
MODELS_FILTER_PATH=./dev-models-filter.txt
```

### Production Configuration

For production, you might want stricter settings:

```env
# .env.production
OPENROUTER_API_KEY="your_production_api_key"
HOST=0.0.0.0
PORT=11434
LOG_LEVEL=WARNING
MAX_CONCURRENT_REQUESTS=200
MODELS_FILTER_PATH=./prod-models-filter.txt
```

## Configuration Validation

The proxy validates configuration at startup. If there are any issues, you'll see error messages indicating what needs to be fixed:

```bash
# Example error for missing API key
Error loading configuration: 1 validation error for Settings
openrouter_api_key
  Field required [type=missing, input_value={}, input_type=dict]
```

To verify your configuration is correct before starting the server:

```bash
# Test with a dry run (if supported)
ollama-proxy --dry-run

# Or just check the help
ollama-proxy --help
```
