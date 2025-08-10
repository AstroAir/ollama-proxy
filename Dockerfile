# Multi-stage build for ollama-proxy
# Stage 1: Build stage with uv for dependency management
FROM python:3.12-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project metadata and sources needed for editable install
# Include README.md because pyproject references it for build metadata
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies and the project itself (editable) using uv
RUN uv sync --frozen --no-dev

# Stage 2: Runtime stage
FROM python:3.12-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r ollama && useradd -r -g ollama -s /bin/false ollama

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY src/ ./src/
COPY pyproject.toml ./

# Create directory for model filter file
RUN mkdir -p /app/data

# Set ownership to non-root user
RUN chown -R ollama:ollama /app

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

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health', timeout=5)" || exit 1

# Expose port
EXPOSE 11434

# Default command
CMD ["python", "-m", "src.main"]
