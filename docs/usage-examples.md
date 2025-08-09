# Usage Examples

This guide provides practical examples of how to use the Ollama Proxy with various tools and applications. These examples will help you get started quickly and show you how to integrate the proxy into your existing workflows.

## Table of Contents

- [Basic Usage with Ollama CLI](#basic-usage-with-ollama-cli)
- [Using with Ollama Clients](#using-with-ollama-clients)
- [Programming Language Examples](#programming-language-examples)
- [Advanced Configuration Examples](#advanced-configuration-examples)
- [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Basic Usage with Ollama CLI

### Listing Available Models

Once the proxy is running, you can list all available models:

```bash
# List models through the proxy
ollama list

# Or using curl directly
curl http://localhost:11434/api/tags
```

### Chat with a Model

You can chat with any model available through OpenRouter:

```bash
# Start a chat session
ollama run gemini-pro:latest

# Or send a single message
echo "Why is the sky blue?" | ollama run gemini-pro:latest
```

### Generate Text

Generate text using a model:

```bash
# Simple text generation
ollama run gemini-pro:latest "Write a short poem about programming"

# Using prompt files
ollama run gemini-pro:latest -f ./prompt.txt
```

## Using with Ollama Clients

### Python Client Example

If you're using the Ollama Python client, you can configure it to use the proxy:

```python
import ollama

# Configure client to use proxy
client = ollama.Client(host='http://localhost:11434')

# List models
models = client.list()
print(models)

# Chat with a model
response = client.chat(
    model='gemini-pro:latest',
    messages=[{
        'role': 'user',
        'content': 'Why is the sky blue?',
    }]
)
print(response['message']['content'])
```

### JavaScript/Node.js Client Example

For the JavaScript client:

```javascript
import ollama from 'ollama';

// The client will automatically use http://localhost:11434
// unless you specify a different host

// List models
const models = await ollama.list();
console.log(models);

// Chat with a model
const response = await ollama.chat({
  model: 'gemini-pro:latest',
  messages: [{ role: 'user', content: 'Why is the sky blue?' }]
});
console.log(response.message.content);
```

## Programming Language Examples

### Python Direct API Usage

You can also interact with the proxy directly using HTTP requests:

```python
import requests
import json

# Proxy endpoint
proxy_url = 'http://localhost:11434'

# List available models
response = requests.get(f'{proxy_url}/api/tags')
models = response.json()
print("Available models:", [m['name'] for m in models['models']])

# Chat completion
chat_data = {
    "model": "gemini-pro:latest",
    "messages": [
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms"
        }
    ]
}

response = requests.post(
    f'{proxy_url}/api/chat',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(chat_data)
)

if response.status_code == 200:
    result = response.json()
    print("Response:", result['message']['content'])
else:
    print(f"Error: {response.status_code} - {response.text}")
```

### Streaming Responses

To handle streaming responses in Python:

```python
import requests
import json

proxy_url = 'http://localhost:11434'

# Streaming chat completion
chat_data = {
    "model": "gemini-pro:latest",
    "messages": [
        {
            "role": "user",
            "content": "Write a story about a robot learning to paint"
        }
    ],
    "stream": True
}

with requests.post(
    f'{proxy_url}/api/chat',
    headers={'Content-Type': 'application/json'},
    data=json.dumps(chat_data),
    stream=True
) as response:
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode('utf-8'))
            if 'message' in chunk and 'content' in chunk['message']:
                print(chunk['message']['content'], end='', flush=True)
            if chunk.get('done', False):
                break
```

### cURL Examples

You can also use cURL to interact with the proxy:

```bash
# List models
curl http://localhost:11434/api/tags

# Chat completion
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-pro:latest",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ]
  }'

# Streaming chat completion
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-pro:latest",
    "messages": [
      {
        "role": "user",
        "content": "Count from 1 to 100."
      }
    ],
    "stream": true
  }'
```

## Advanced Configuration Examples

### Model Filtering

Create a custom `models-filter.txt` to limit which models are available:

```text
# Only allow these specific models
gemini-pro:latest
gpt-4:latest
claude-3-5-sonnet:latest

# Allow all models from a specific family using wildcards
llama*:*latest

# Exclude specific models (prefix with !)
!llama-2-13b-chat:latest
```

Then start the proxy with your filter:

```bash
ollama-proxy --models-filter ./models-filter.txt
```

### Environment-based Configuration

Create a `.env` file for different environments:

```env
# Development environment
OPENROUTER_API_KEY=your_dev_api_key_here
LOG_LEVEL=DEBUG
HOST=localhost
PORT=11434
MODELS_FILTER_PATH=./dev-models-filter.txt
```

For production:

```env
# Production environment
OPENROUTER_API_KEY=your_prod_api_key_here
LOG_LEVEL=WARNING
HOST=0.0.0.0
PORT=11434
MODELS_FILTER_PATH=./prod-models-filter.txt
MAX_CONCURRENT_REQUESTS=200
```

### Docker Compose with Custom Configuration

Create a `docker-compose.prod.yml` for production deployment:

```yaml
version: '3.8'
services:
  ollama-proxy:
    build: .
    ports:
      - "11434:11434"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - LOG_LEVEL=INFO
      - MAX_CONCURRENT_REQUESTS=200
    volumes:
      - ./prod-models-filter.txt:/app/models-filter.txt
    restart: unless-stopped
```

## Troubleshooting Common Issues

### Connection Issues

If you can't connect to the proxy:

1. Check if the proxy is running:

   ```bash
   # Check if the port is listening
   netstat -tlnp | grep :11434
   
   # Or use lsof
   lsof -i :11434
   ```

2. Verify the proxy logs:

   ```bash
   # If running with Docker
   docker logs ollama-proxy-container
   
   # If running directly
   # Check your terminal where you started the proxy
   ```

### Model Not Found Errors

If you get "model not found" errors:

1. List available models to see what's actually available:

   ```bash
   curl http://localhost:11434/api/tags
   ```

2. Check your model filter configuration:

   ```bash
   # View your filter file
   cat models-filter.txt
   ```

3. Try without filtering to see all models:

   ```bash
   ollama-proxy --models-filter ""
   ```

### API Key Issues

If you get authentication errors:

1. Verify your API key:

   ```bash
   echo $OPENROUTER_API_KEY
   ```

2. Test the key directly with OpenRouter:

   ```bash
   curl https://openrouter.ai/api/v1/models \
     -H "Authorization: Bearer $OPENROUTER_API_KEY"
   ```

### Performance Issues

If responses are slow:

1. Check the proxy logs for any errors or warnings.

2. Monitor the proxy's health endpoint:

   ```bash
   curl http://localhost:11434/health
   ```

3. Check the metrics endpoint for performance data:

   ```bash
   curl http://localhost:11434/metrics
   ```
