# Deployment Guide

This guide provides instructions for deploying the Ollama Proxy in a production environment. The recommended deployment method is using Docker, but we also cover other deployment options.

## Docker Deployment (Recommended)

Using Docker is the most reliable way to run the Ollama Proxy, as it encapsulates the application and its dependencies in a container.

### Prerequisites

-   [Docker](https://docs.docker.com/get-docker/) installed on your server.
-   [Docker Compose](https://docs.docker.com/compose/install/) (optional, but recommended for easier management).

### Method 1: Using `docker-compose.yml`

The project includes a `docker-compose.yml` file to simplify deployment.

1.  **Create an environment file**:
    Create a file named `.env.production` in the project root and add your production-ready settings:

    ```env
    # .env.production
    OPENROUTER_API_KEY="your_production_api_key"
    LOG_LEVEL=INFO
    MAX_CONCURRENT_REQUESTS=200
    ```

2.  **Start the server**:
    Run the following command to build and start the container in detached mode:

    ```bash
    docker-compose up --build -d
    ```

3.  **Verify the deployment**:
    Check the logs to ensure the server started correctly:

    ```bash
    docker-compose logs -f
    ```

    You should see output indicating that the Uvicorn server is running.

### Method 2: Using `Dockerfile`

If you prefer not to use Docker Compose, you can build and run the Docker image manually.

1.  **Build the Docker image**:
    From the project root, run:

    ```bash
    docker build -t ollama-proxy .
    ```

2.  **Run the Docker container**:
    Be sure to pass in your OpenRouter API key and expose the port.

    ```bash
    docker run -d -p 11434:11434 \
      -e OPENROUTER_API_KEY="your_production_api_key" \
      --name ollama-proxy-container \
      ollama-proxy
    ```

### Advanced Docker Configuration

For production deployments, you might want to use volume mounts for configuration files:

```bash
docker run -d -p 11434:11434 \
  -e OPENROUTER_API_KEY="your_production_api_key" \
  -v /path/to/your/models-filter.txt:/app/models-filter.txt \
  -v /path/to/your/logs:/app/logs \
  --name ollama-proxy-container \
  ollama-proxy
```

## Systemd Deployment (Linux)

For direct deployment on a Linux system, you can use systemd to manage the service.

1.  **Create a service file**:
    Create `/etc/systemd/system/ollama-proxy.service`:

    ```ini
    [Unit]
    Description=Ollama Proxy Service
    After=network.target

    [Service]
    Type=simple
    User=ollama-proxy
    WorkingDirectory=/opt/ollama-proxy
    Environment=OPENROUTER_API_KEY=your_production_api_key
    Environment=LOG_LEVEL=INFO
    ExecStart=/opt/ollama-proxy/.venv/bin/ollama-proxy
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    ```

2.  **Create a dedicated user**:
    ```bash
    sudo useradd -r -s /bin/false ollama-proxy
    ```

3.  **Set up the application directory**:
    ```bash
    sudo mkdir -p /opt/ollama-proxy
    sudo chown ollama-proxy:ollama-proxy /opt/ollama-proxy
    ```

4.  **Install dependencies and set up the virtual environment**:
    ```bash
    sudo -u ollama-proxy bash -c "
      cd /opt/ollama-proxy
      python -m venv .venv
      source .venv/bin/activate
      pip install -e .
    "
    ```

5.  **Enable and start the service**:
    ```bash
    sudo systemctl enable ollama-proxy
    sudo systemctl start ollama-proxy
    ```

6.  **Check the service status**:
    ```bash
    sudo systemctl status ollama-proxy
    ```

## Kubernetes Deployment

For Kubernetes deployments, you can use a deployment manifest like this:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-proxy
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ollama-proxy
  template:
    metadata:
      labels:
        app: ollama-proxy
    spec:
      containers:
      - name: ollama-proxy
        image: your-registry/ollama-proxy:latest
        ports:
        - containerPort: 11434
        env:
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: ollama-proxy-secrets
              key: api-key
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: ollama-proxy-service
spec:
  selector:
    app: ollama-proxy
  ports:
  - protocol: TCP
    port: 11434
    targetPort: 11434
  type: LoadBalancer
```

And create the secret for your API key:

```bash
kubectl create secret generic ollama-proxy-secrets \
  --from-literal=api-key='your_production_api_key'
```

## Production Best Practices

### Security

-   **Secure your API Key**: Never hardcode your OpenRouter API key in scripts or configuration files that might be committed to version control. Use environment variables, secrets management tools, or Kubernetes secrets.
-   **Network Security**: Restrict access to the proxy using firewall rules or network policies. Only expose the necessary ports.
-   **Use HTTPS**: In production, always use HTTPS. You can terminate TLS at a reverse proxy like nginx or use a service that provides automatic TLS termination.

### Performance

-   **Resource Limits**: Set appropriate CPU and memory limits for your containers or processes.
-   **Concurrency**: Adjust `MAX_CONCURRENT_REQUESTS` based on your expected load and available resources.
-   **Model Filtering**: Use model filtering to limit the number of models available, which can reduce memory usage and improve startup time.

### Monitoring and Observability

-   **Health Checks**: Use the `/health` endpoint for load balancer health checks.
-   **Metrics**: The proxy includes a `/metrics` endpoint that can be integrated with monitoring systems like Prometheus to track performance and errors. For more details, see the [Architecture Guide](ARCHITECTURE.md#monitoring-and-observability).
-   **Logging**: In a production environment, configure the `LOG_LEVEL` to `INFO` or `WARNING` and ship your logs to a centralized logging platform like ELK stack, Datadog, or CloudWatch for analysis and alerting.
-   **Alerting**: Set up alerts for high error rates, slow response times, or service downtime.

### Backup and Recovery

-   **Configuration Backups**: Keep backups of your configuration files and model filter files.
-   **Regular Updates**: Regularly update the proxy to get the latest features and security fixes.
-   **Rolling Updates**: When using orchestration platforms like Kubernetes, use rolling updates to minimize downtime during deployments.

### Scaling

-   **Horizontal Scaling**: The proxy is stateless, so you can run multiple instances behind a load balancer to handle more traffic.
-   **Load Balancing**: Use a load balancer to distribute traffic across multiple proxy instances.
-   **Auto-scaling**: Configure auto-scaling policies based on CPU or memory usage, or request rate.
