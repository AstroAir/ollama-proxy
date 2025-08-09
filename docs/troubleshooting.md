# Troubleshooting Guide

This guide provides solutions to common problems you might encounter while using the Ollama Proxy. It includes diagnostic steps and solutions for various issues.

## Connection Issues

### `Connection refused` error

If you get a `Connection refused` error when trying to connect to the proxy, it could be due to several reasons:

**Diagnostic Steps:**

1. **Check if the proxy is running**:

   ```bash
   # If running with Docker
   docker ps | grep ollama-proxy
   
   # Check Docker logs
   docker logs ollama-proxy-container
   
   # If running directly, check process
   ps aux | grep ollama-proxy
   ```

2. **Verify the port is listening**:

   ```bash
   # Check if the port is open
   netstat -tlnp | grep :11434
   
   # Or using lsof (if available)
   lsof -i :11434
   
   # Or using ss (if netstat is not available)
   ss -tlnp | grep :11434
   ```

3. **Test local connectivity**:

   ```bash
   # Test basic connectivity
   curl -v http://localhost:11434/
   
   # Test API endpoint
   curl -v http://localhost:11434/api/version
   ```

**Solutions:**

- **The proxy is not running**: Make sure you have started the proxy server. You can check the logs to confirm it is running.
- **Incorrect host or port**: Ensure that your client is configured to connect to the correct host and port. The default is `http://localhost:11434`.
- **Firewall issues**: A firewall on your system or network might be blocking the connection. Check your firewall settings to ensure that the port is open.
- **Docker networking issues**: If using Docker, ensure the container is properly exposing the port:

    ```bash
    # Check port mapping
    docker port ollama-proxy-container
    
    # Run with explicit port mapping
    docker run -d -p 11434:11434 \
      -e OPENROUTER_API_KEY="your_api_key" \
      --name ollama-proxy-container \
      ollama-proxy
    ```

### `Connection timeout` error

This error typically indicates network connectivity issues.

**Diagnostic Steps:**

1. **Test internet connectivity**:

   ```bash
   ping openrouter.ai
   ```

2. **Test DNS resolution**:

   ```bash
   nslookup openrouter.ai
   dig openrouter.ai
   ```

3. **Test direct OpenRouter connectivity**:

   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
   ```

**Solutions:**

- **Network connectivity**: Ensure your system has internet access.
- **DNS issues**: Try using a different DNS server (e.g., Google's 8.8.8.8).
- **Proxy configuration**: If you're behind a corporate proxy, configure the proxy settings appropriately.

## Model Errors

### `Model not found` error

If you receive a `Model not found` error, it could be because:

**Diagnostic Steps:**

1. **List available models**:

   ```bash
   # Using curl
   curl http://localhost:11434/api/tags
   
   # Using Ollama CLI
   ollama list
   ```

2. **Check OpenRouter directly**:

   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
   ```

3. **Verify model filter configuration**:

   ```bash
   # Check if filter file exists
   cat models-filter.txt
   
   # Check proxy logs for filter information
   docker logs ollama-proxy-container | grep -i filter
   ```

**Solutions:**

- **The model does not exist on OpenRouter**: Double-check the model name to ensure it is correct. You can find a list of available models on the [OpenRouter website](https://openrouter.ai/models).
- **The model is not spelled correctly**: Model names are case-sensitive. Use the exact model name as shown in the `/api/tags` response.
- **The model is filtered out**: If you are using a model filter file, make sure the model you are trying to use is included in the file. See the [Configuration Guide](configuration.md#model-filtering) for more details.

### `Model is not allowed` error

This error means that the model you are trying to use has been blocked by the model filter.

**Diagnostic Steps:**

1. **Check your filter file**:

   ```bash
   cat models-filter.txt
   ```

2. **Check proxy logs**:

   ```bash
   docker logs ollama-proxy-container | grep -i "not allowed"
   ```

**Solutions:**

- **Add the model to your filter**: Add the model to your `models-filter.txt` file.
- **Temporarily disable filtering**: Start the proxy with an empty filter to test:

   ```bash
   ollama-proxy --models-filter ""
   ```

- **Check filter syntax**: Ensure your filter file syntax is correct with no typos.

## API Key Errors

### `401 Unauthorized` error

A `401 Unauthorized` error from the proxy indicates a problem with your OpenRouter API key.

**Diagnostic Steps:**

1. **Verify environment variable**:

   ```bash
   echo $OPENROUTER_API_KEY
   ```

2. **Check .env file**:

   ```bash
   cat .env
   ```

3. **Test API key directly**:

   ```bash
   curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models
   ```

**Solutions:**

- **The API key is missing**: Make sure you have set the `OPENROUTER_API_KEY` environment variable or added it to your `.env` file.
- **The API key is invalid**: Verify that your API key is correct and has not expired. You can check your API key on the [OpenRouter website](https://openrouter.ai/keys).
- **Key format issues**: Ensure your API key doesn't have extra spaces or quotes:

   ```env
   # Correct
   OPENROUTER_API_KEY=sk-youractualkeyhere
   
   # Incorrect - avoid these:
   # OPENROUTER_API_KEY="sk-youractualkeyhere"  # No quotes needed
   # OPENROUTER_API_KEY= sk-youractualkeyhere   # No leading spaces
   ```

### `403 Forbidden` error

This error usually indicates that your API key doesn't have access to a specific model.

**Solutions:**

- **Check model availability**: Some models on OpenRouter require special access or have usage limits.
- **Verify billing**: Ensure your OpenRouter account has valid billing information if required for the model.
- **Try a different model**: Test with a different model to see if the issue is model-specific.

## Performance Issues

### Slow responses

If you are experiencing slow responses, it could be due to:

**Diagnostic Steps:**

1. **Check proxy logs for timing information**:

   ```bash
   docker logs ollama-proxy-container | grep -i "completed\|duration"
   ```

2. **Test direct OpenRouter response time**:

   ```bash
   time curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "google/gemini-pro:latest", "messages": [{"role": "user", "content": "Hello"}]}' \
     https://openrouter.ai/api/v1/chat/completions
   ```

3. **Check system resources**:

   ```bash
   # Check CPU and memory usage
   top
   
   # Check Docker resource usage
   docker stats ollama-proxy-container
   ```

**Solutions:**

- **High latency to OpenRouter**: The proxy needs to make requests to the OpenRouter API, so your connection to OpenRouter can affect performance.
- **Large models**: Larger models can take longer to generate responses. Consider using smaller models for faster responses.
- **High server load**: If the proxy is handling a large number of concurrent requests, it may slow down. Consider:
- Increasing `MAX_CONCURRENT_REQUESTS` if your system can handle more
- Adding more proxy instances behind a load balancer
- Upgrading your system resources

### High memory usage

**Diagnostic Steps:**

1. **Check Docker container memory usage**:

   ```bash
   docker stats ollama-proxy-container
   ```

2. **Check system memory**:

   ```bash
   free -h
   ```

**Solutions:**

- **Model filtering**: Use model filtering to reduce the number of models loaded
- **System resources**: Ensure your system has adequate memory
- **Docker limits**: Set memory limits for Docker containers:

   ```bash
   docker run -d -p 11434:11434 \
     -e OPENROUTER_API_KEY="your_api_key" \
     --memory=512m \
     --name ollama-proxy-container \
     ollama-proxy
   ```

## Logging and Debugging

### Enabling Debug Logging

For more detailed information, enable debug logging:

```bash
# Using environment variable
export LOG_LEVEL=DEBUG
ollama-proxy

# Using command line
ollama-proxy --log-level DEBUG

# In Docker
docker run -d -p 11434:11434 \
  -e OPENROUTER_API_KEY="your_api_key" \
  -e LOG_LEVEL=DEBUG \
  --name ollama-proxy-container \
  ollama-proxy
```

### Checking Health Status

Use the health endpoint to get detailed status information:

```bash
curl http://localhost:11434/health
```

### Monitoring Metrics

Check the metrics endpoint for performance data:

```bash
curl http://localhost:11434/metrics
```

## Docker-Specific Issues

### Container won't start

**Diagnostic Steps:**

1. **Check container status**:

   ```bash
   docker ps -a
   ```

2. **Check container logs**:

   ```bash
   docker logs ollama-proxy-container
   ```

**Solutions:**

- **Configuration errors**: Check logs for configuration validation errors
- **Port conflicts**: Ensure the port is not already in use:

   ```bash
   docker run -d -p 8080:11434 \  # Use different host port
     -e OPENROUTER_API_KEY="your_api_key" \
     --name ollama-proxy-container \
     ollama-proxy
   ```

### Volume mounting issues

When using volume mounts, ensure proper permissions:

```bash
# Create directories with correct permissions
mkdir -p /host/path/to/config
chmod 755 /host/path/to/config

# Run with volume mount
docker run -d -p 11434:11434 \
  -e OPENROUTER_API_KEY="your_api_key" \
  -v /host/path/to/config:/app/config \
  --name ollama-proxy-container \
  ollama-proxy
```

## Advanced Debugging

### Network debugging

If you suspect network issues between the proxy and OpenRouter:

1. **Test connectivity from within the container**:

   ```bash
   docker exec -it ollama-proxy-container sh
   ping openrouter.ai
   curl https://openrouter.ai/api/v1/models
   ```

2. **Check DNS resolution**:

   ```bash
   docker exec -it ollama-proxy-container cat /etc/resolv.conf
   ```

### Testing with different clients

Try different clients to isolate issues:

1. **Direct curl requests**:

   ```bash
   curl http://localhost:11434/api/chat \
     -H "Content-Type: application/json" \
     -d '{
       "model": "google/gemini-pro:latest",
       "messages": [
         {"role": "user", "content": "Hello"}
       ]
     }'
   ```

2. **Ollama CLI**:

   ```bash
   ollama run gemini-pro:latest "Hello"
   ```

## Getting Help

If you are still having trouble after consulting this guide:

1. **Gather diagnostic information**:
   - Proxy logs
   - Output of `docker ps` and `docker logs`
   - Your configuration files (with sensitive information redacted)
   - Steps to reproduce the issue

2. **Open an issue**: Please [open an issue](https://github.com/your-username/ollama-proxy/issues) on GitHub with a detailed description of the problem, including:
   - Your environment (OS, Docker version, etc.)
   - Steps to reproduce
   - Expected vs. actual behavior
   - Any relevant logs or error messages

3. **Community support**: Check if similar issues have been discussed in existing issues or community forums.
