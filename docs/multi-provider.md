# Multi-Provider Support

The Ollama Proxy includes advanced multi-provider support, allowing you to use multiple AI providers simultaneously with intelligent routing, fallback mechanisms, and comprehensive monitoring.

## Overview

The multi-provider system supports:

- **Multiple AI Providers**: OpenAI, Anthropic Claude, Google Gemini, OpenRouter, Azure OpenAI, AWS Bedrock, and local Ollama
- **Intelligent Routing**: Model-based routing, capability-based, round-robin, least-loaded, fastest-response, and cost-optimized strategies
- **Fallback Mechanisms**: Automatic failover to alternative providers with configurable strategies
- **Enhanced Error Handling**: Circuit breakers, retry logic, and comprehensive error recovery
- **Health Monitoring**: Real-time health checks, provider status monitoring, and detailed metrics collection
- **Request/Response Transformation**: Seamless format conversion between Ollama and provider APIs
- **Scalability Features**: Connection pooling, request queuing, rate limiting, and async optimization
- **Advanced Configuration**: Flexible provider settings, routing rules, and environment-specific configurations

## Configuration

### Environment Variables

The multi-provider system supports comprehensive configuration through environment variables:

```bash
# Basic settings
HOST=0.0.0.0
PORT=11434
ENVIRONMENT=production
LOG_LEVEL=info

# Routing configuration
ROUTING_STRATEGY=capability_based  # round_robin, least_loaded, fastest_response, capability_based
FALLBACK_STRATEGY=next_available   # none, next_available, retry_same, best_alternative
ENABLE_LOAD_BALANCING=true
HEALTH_CHECK_INTERVAL=60

# OpenRouter (existing)
OPENROUTER_ENABLED=true
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_TIMEOUT=300
OPENROUTER_PRIORITY=1

# OpenAI
OPENAI_ENABLED=true
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_TIMEOUT=300
OPENAI_PRIORITY=2

# Anthropic Claude
ANTHROPIC_ENABLED=true
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_TIMEOUT=300
ANTHROPIC_PRIORITY=3

# Google Gemini
GOOGLE_ENABLED=true
GOOGLE_API_KEY=your_google_key
GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GOOGLE_TIMEOUT=300
GOOGLE_PRIORITY=4

# Azure OpenAI
AZURE_ENABLED=true
AZURE_API_KEY=your_azure_key
AZURE_BASE_URL=https://your-resource.openai.azure.com
AZURE_TIMEOUT=300
AZURE_PRIORITY=5

# AWS Bedrock
AWS_BEDROCK_ENABLED=true
AWS_BEDROCK_ACCESS_KEY=your_aws_access_key
AWS_BEDROCK_SECRET_KEY=your_aws_secret_key
AWS_BEDROCK_REGION=us-east-1
AWS_BEDROCK_TIMEOUT=300
AWS_BEDROCK_PRIORITY=6

# Local Ollama
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_PRIORITY=7
```

### Configuration File

Create a `.env` file in the project root with your provider credentials and preferences:

```env
# Required - at least one provider must be enabled
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Optional routing configuration
ROUTING_STRATEGY=capability_based
FALLBACK_STRATEGY=next_available
ENABLE_LOAD_BALANCING=true
```

## Usage

### Starting the Multi-Provider Proxy

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start the proxy
python -m src.main
```

### API Endpoints

The multi-provider proxy maintains full compatibility with the original Ollama API while adding new capabilities:

#### Original Ollama Endpoints (Multi-Provider)

```bash
# Chat completion with automatic provider selection
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Text generation with fallback support
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-sonnet",
    "prompt": "Write a haiku about coding"
  }'
```

#### Provider Selection

You can specify a preferred provider using the `X-Provider` header:

```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -H "X-Provider: openai" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

#### New Enhanced Endpoints

```bash
# List models from all providers
curl http://localhost:11434/api/tags

# List models from specific provider
curl http://localhost:11434/api/tags/openai

# Get provider information and statistics
curl http://localhost:11434/api/providers

# Get specific provider stats
curl http://localhost:11434/api/providers/openai/stats

# Health check endpoints
curl http://localhost:11434/health
curl http://localhost:11434/health/detailed
curl http://localhost:11434/health/providers
curl http://localhost:11434/health/providers/openai
```

## Routing Strategies

### 1. Model-Based Routing (Priority)

Automatically routes requests based on model names and patterns:

- **OpenAI Models**: `gpt-4*`, `gpt-3.5*`, `text-embedding-ada*` → OpenAI provider
- **Anthropic Models**: `claude-3*`, `claude-2*`, `claude-instant*` → Anthropic provider
- **Google Models**: `gemini*`, `palm*`, `bison*` → Google provider
- **Azure Models**: `*@*.openai.azure.com` → Azure OpenAI provider
- **AWS Bedrock Models**: `anthropic.claude*`, `amazon.titan*`, `meta.llama*` → AWS Bedrock provider
- **Local Models**: `llama*`, `mistral*`, `codellama*` → Ollama provider

### 2. Capability-Based Routing (Default)

Routes requests based on provider capabilities and model preferences:

- Analyzes the requested model and operation type
- Selects the most suitable provider based on capabilities
- Considers model availability and provider specializations
- Falls back to alternative providers if the primary choice fails

### 3. Cost-Optimized Routing

Routes requests to minimize costs:

- **Priority Order**: Ollama (local) → OpenRouter → Google → Anthropic → OpenAI → Azure → AWS Bedrock
- Considers provider pricing and cost per token
- Balances cost with performance requirements

### 4. Round-Robin Routing

Distributes requests evenly across all available providers:

- Simple load distribution
- Good for balanced workloads
- Ensures all providers are utilized equally

### 5. Least-Loaded Routing

Routes to the provider with the lowest current load:

- Monitors active requests per provider
- Selects provider with minimum concurrent requests
- Optimizes for performance under varying loads

### 4. Fastest-Response Routing

Routes to the provider with the best recent response times:

- Tracks response time statistics
- Selects provider with lowest average response time
- Adapts to changing provider performance

## Fallback Strategies

### Next Available

If the primary provider fails, automatically try the next available provider:

```bash
FALLBACK_STRATEGY=next_available
```

### Retry Same

Retry the same provider with exponential backoff before trying alternatives:

```bash
FALLBACK_STRATEGY=retry_same
```

### Best Alternative

Select the best alternative provider based on the routing strategy:

```bash
FALLBACK_STRATEGY=best_alternative
```

## Health Monitoring

The multi-provider system includes comprehensive health monitoring:

### Provider Health Checks

- Automatic health checks for all enabled providers
- Circuit breaker pattern to prevent cascading failures
- Real-time status monitoring and reporting

### Health Endpoints

```bash
# Overall system health
curl http://localhost:11434/health

# Detailed health information
curl http://localhost:11434/health/detailed

# Provider-specific health
curl http://localhost:11434/health/providers
curl http://localhost:11434/health/providers/openai
```

### Circuit Breakers

Each provider has an independent circuit breaker that:

- Monitors error rates and response times
- Automatically disables failing providers
- Gradually re-enables providers when they recover
- Prevents resource waste on failing providers

## Scalability Features

### Connection Pooling

Advanced HTTP connection management for optimal performance:

- **Per-Provider Connection Pools**: Each provider maintains its own connection pool
- **Configurable Pool Sizes**: Adjust based on provider limits and requirements
- **Keep-Alive Connections**: Reuse connections to reduce latency
- **Connection Limits**: Prevent resource exhaustion with configurable limits

```bash
# Configure connection pooling
OPENAI_MAX_CONCURRENT_REQUESTS=50
ANTHROPIC_MAX_CONCURRENT_REQUESTS=30
GOOGLE_MAX_CONCURRENT_REQUESTS=40
```

### Request Queuing

Intelligent request queuing for high-load scenarios:

- **Priority-Based Queuing**: Critical requests processed first
- **Queue Size Limits**: Prevent memory exhaustion
- **Request Timeouts**: Automatic cleanup of expired requests
- **Concurrency Control**: Limit concurrent requests per provider

```bash
# Configure request queuing
MAX_QUEUE_SIZE=1000
MAX_CONCURRENT_REQUESTS=100
ENABLE_PRIORITIZATION=true
```

### Rate Limiting

Built-in rate limiting to respect provider limits:

- **Per-Provider Rate Limits**: Respect individual provider constraints
- **Adaptive Rate Limiting**: Automatically adjust based on provider responses
- **Queue Management**: Queue requests when rate limits are hit
- **Backoff Strategies**: Exponential backoff for rate-limited requests

### Async Optimization

Fully asynchronous architecture for maximum throughput:

- **Non-Blocking I/O**: All network operations are asynchronous
- **Concurrent Request Processing**: Handle multiple requests simultaneously
- **Resource Efficiency**: Minimal memory and CPU overhead
- **Graceful Degradation**: Maintain performance under high load

## Error Handling

The multi-provider system includes enhanced error handling:

### Automatic Retries

- Configurable retry logic with exponential backoff
- Provider-specific retry policies
- Intelligent retry decision making

### Error Classification

- Transient errors (network issues, timeouts)
- Permanent errors (authentication, model not found)
- Provider-specific error handling

### Graceful Degradation

- Automatic fallback to alternative providers
- Partial functionality maintenance during outages
- User-friendly error messages

## Monitoring and Metrics

### Provider Statistics

Track detailed statistics for each provider:

- Request counts and success rates
- Response times and error rates
- Model usage patterns
- Health status history

### Performance Metrics

Monitor system-wide performance:

- Overall request throughput
- Average response times
- Error rates by provider and model
- Resource utilization

### Alerting

Set up alerts for:

- Provider failures or degraded performance
- High error rates
- Unusual traffic patterns
- Resource exhaustion

## Migration from Single Provider

The multi-provider system is fully backward compatible:

1. **Existing Configurations**: All existing environment variables continue to work
2. **API Compatibility**: All original endpoints work unchanged
3. **Gradual Migration**: Enable new providers incrementally
4. **Feature Flags**: Disable multi-provider features if needed

### Migration Steps

1. **Update Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Add New Environment Variables**:

   ```bash
   # Add provider configurations to .env
   OPENAI_ENABLED=true
   OPENAI_API_KEY=your_key
   ```

3. **Start Multi-Provider Mode**:

   ```bash
   python -m src.main
   ```

4. **Monitor and Adjust**:
   - Check health endpoints
   - Monitor provider statistics
   - Adjust routing strategies as needed

## Troubleshooting

### Common Issues

#### Provider Not Available

If a provider is not available:

1. Check the provider's API key and configuration
2. Verify network connectivity to the provider
3. Check the provider's health status
4. Review circuit breaker status

#### Routing Issues

If requests are not being routed correctly:

1. Verify routing strategy configuration
2. Check provider priorities and capabilities
3. Review load balancing settings
4. Monitor provider health status

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=debug python -m src.main
```

### Health Check Debugging

```bash
# Check overall system health
curl http://localhost:11434/health/detailed

# Check specific provider
curl http://localhost:11434/health/providers/openai

# Get provider statistics
curl http://localhost:11434/api/providers
```

## Best Practices

1. **API Key Management**: Store API keys securely and rotate them regularly
2. **Provider Selection**: Choose providers based on your specific use cases and requirements
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Testing**: Test failover scenarios and provider switching
5. **Performance**: Monitor and optimize routing strategies for your workload
6. **Security**: Follow security best practices for all enabled providers
