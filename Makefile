# Makefile for ollama-proxy development and deployment

.PHONY: help install install-dev test test-cov lint format type-check clean build docker-build docker-run dev run docs-check docs-view docs-build docs-serve

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black, isort)"
	@echo "  format-check - Check code formatting"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  dev          - Run development server"
	@echo "  run          - Run production server"
	@echo "  check-all    - Run all checks (lint, format-check, type-check, test)"
	@echo "  docs-check   - Check documentation files"
	@echo "  docs-view    - View documentation in browser"
	@echo "  docs-build   - Build documentation site"
	@echo "  docs-serve   - Serve documentation locally"

# Installation
install:
	uv sync --no-dev

install-dev:
	uv sync --all-extras --dev
	uv add --dev mkdocs mkdocs-material

# Testing
test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	uv run flake8 src tests

format:
	uv run black src tests
	uv run isort src tests

format-check:
	uv run black --check src tests
	uv run isort --check-only src tests

type-check:
	uv run mypy src

# Combined checks
check-all: lint format-check type-check test

# Documentation
docs-check:
	@echo "Checking documentation files..."
	@test -f docs/index.md && echo "✓ Main documentation exists"
	@test -f docs/configuration.md && echo "✓ Configuration guide exists"
	@test -f docs/api-reference.md && echo "✓ API reference exists"
	@test -f docs/usage-examples.md && echo "✓ Usage examples exist"
	@test -f docs/deployment.md && echo "✓ Deployment guide exists"
	@test -f docs/architecture.md && echo "✓ Architecture guide exists"
	@test -f docs/troubleshooting.md && echo "✓ Troubleshooting guide exists"
	@test -f docs/contributing.md && echo "✓ Contributing guide exists"
	@test -f mkdocs.yml && echo "✓ MkDocs configuration exists"
	@echo "All documentation files are present."

docs-view:
	@echo "Starting documentation server..."
	@echo "Open http://localhost:8000 in your browser"
	@python -m http.server 8000 -d docs

docs-build:
	@echo "Building documentation site..."
	@uv run mkdocs build
	@echo "Documentation built in 'site' directory"

docs-serve:
	@echo "Starting MkDocs development server..."
	@echo "Open http://localhost:8000 in your browser"
	@uv run mkdocs serve

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	uv build

# Docker
docker-build:
	docker build -t ollama-proxy:latest .

docker-run:
	docker run -p 11434:11434 -e OPENROUTER_API_KEY=${OPENROUTER_API_KEY} ollama-proxy:latest

# Development
dev:
	uv run python -m src.main --reload --log-level DEBUG

run:
	uv run python -m src.main

# Pre-commit setup
setup-pre-commit:
	uv run pre-commit install

# Environment setup
setup: install-dev setup-pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make dev' to start the development server"

# Health check
health:
	@curl -f http://localhost:11434/health || echo "Server not running or unhealthy"

# Show project info
info:
	@echo "Project: ollama-proxy"
	@echo "Python version: $(shell python --version)"
	@echo "UV version: $(shell uv --version)"
	@echo "Dependencies:"
	@uv tree --depth 1
