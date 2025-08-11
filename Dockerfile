# Multi-stage build for ollama-proxy
# Stage 1: Build stage with uv for dependency management
FROM python:3.12-slim as builder

# Build arguments for metadata
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.title="Ollama Proxy" \
      org.opencontainers.image.description="A proxy server that translates Ollama API calls to OpenRouter API calls" \
      org.opencontainers.image.source="https://github.com/your-org/ollama-proxy" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy project metadata and sources needed for editable install
# Include README.md because pyproject references it for build metadata
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies and the project itself (editable) using uv
RUN uv sync --frozen --no-dev --no-cache

# Stage 2: Runtime stage
FROM python:3.12-slim as runtime

# Build arguments for runtime
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata labels
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.title="Ollama Proxy" \
      org.opencontainers.image.description="A proxy server that translates Ollama API calls to OpenRouter API calls" \
      org.opencontainers.image.source="https://github.com/your-org/ollama-proxy" \
      org.opencontainers.image.licenses="MIT"

# Install runtime system dependencies and security updates
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    tini \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r -g 1000 ollama && \
    useradd -r -u 1000 -g ollama -s /bin/false -d /app ollama

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=ollama:ollama /app/.venv /app/.venv

# Copy application source code
COPY --chown=ollama:ollama src/ ./src/
COPY --chown=ollama:ollama pyproject.toml ./

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R ollama:ollama /app

# Switch to non-root user
USER ollama

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default environment variables (can be overridden)
ENV HOST=0.0.0.0
ENV PORT=11434
ENV LOG_LEVEL=INFO
ENV ENVIRONMENT=production

# Health check with improved reliability
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import httpx; import sys; \
    try: \
        resp = httpx.get('http://localhost:${PORT}/health', timeout=5); \
        sys.exit(0 if resp.status_code == 200 else 1) \
    except: \
        sys.exit(1)" || exit 1

# Expose port
EXPOSE 11434

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "-m", "src.main"]
