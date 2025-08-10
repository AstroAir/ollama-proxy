# Makefile for ollama-proxy development and deployment

.PHONY: help install install-dev test test-cov lint format type-check clean build docker-build docker-run dev run docs-check docs-view docs-build docs-serve setup-dev start-dev start-daemon start-admin health-check config-show benchmark test-watch test-unit test-integration lint-fix format-check security-scan update-deps cleanup monitor maintenance

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "🏗️  Setup and Installation:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup-dev    - Complete development environment setup"
	@echo ""
	@echo "🚀 Server Management:"
	@echo "  run          - Run production server"
	@echo "  dev          - Run development server"
	@echo "  start-dev    - Start development server (alternative entry point)"
	@echo "  start-daemon - Start server in daemon mode"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-watch   - Run tests in watch mode"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  lint         - Run linting (flake8)"
	@echo "  lint-fix     - Run linting with auto-fix"
	@echo "  format       - Format code (black, isort)"
	@echo "  format-check - Check code formatting"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  check-all    - Run all checks (lint, format-check, type-check, test)"
	@echo ""
	@echo "🏗️  Build and Deploy:"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs-check   - Check documentation files"
	@echo "  docs-view    - View documentation in browser"
	@echo "  docs-build   - Build documentation site"
	@echo "  docs-serve   - Serve documentation locally"
	@echo ""
	@echo "🔧 Administration and Maintenance:"
	@echo "  health-check - Check server health"
	@echo "  config-show  - Show current configuration"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  security-scan - Run security scans"
	@echo "  update-deps  - Update dependencies"
	@echo "  cleanup      - Clean temporary files and caches"
	@echo "  monitor      - Monitor server in real-time"
	@echo "  maintenance  - Run maintenance tasks"

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
	@echo "Open http://localhost:8001 in your browser"
	@uv run mkdocs serve --dev-addr=127.0.0.1:8001

docs-stop:
	@echo "Stopping MkDocs development server..."
	@pkill -f "mkdocs serve" || echo "No MkDocs server running"

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

# ============================================================================
# New Enhanced Targets
# ============================================================================

# Development environment setup
setup-dev:
	@echo "Setting up development environment..."
	@if [ -x "./scripts/dev-setup.sh" ]; then \
		./scripts/dev-setup.sh; \
	else \
		echo "Development setup script not found. Running basic setup..."; \
		$(MAKE) install-dev; \
		$(MAKE) setup-pre-commit; \
	fi

# Alternative server entry points
start-dev:
	uv run ollama-proxy-dev

start-daemon:
	uv run ollama-proxy-daemon

# Administrative commands
start-admin:
	uv run ollama-proxy-admin

health-check:
	@echo "Checking server health..."
	@uv run ollama-proxy-health || echo "Health check failed or server not running"

config-show:
	@echo "Current configuration:"
	@uv run ollama-proxy-config --show

benchmark:
	@echo "Running performance benchmarks..."
	@uv run ollama-proxy-benchmark

# Enhanced testing targets
test-unit:
	uv run pytest -m "unit"

test-integration:
	uv run pytest -m "integration"

test-watch:
	@echo "Starting test watch mode..."
	@if [ -x "./scripts/test-runner.sh" ]; then \
		./scripts/test-runner.sh --watch; \
	else \
		echo "Test runner script not found. Install watchdog and use: uv run pytest-watch"; \
	fi

# Enhanced linting and formatting
lint-fix:
	@echo "Running linting with auto-fix..."
	@if [ -x "./scripts/launcher.py" ]; then \
		uv run ollama-proxy-lint --fix; \
	else \
		$(MAKE) format; \
		$(MAKE) lint; \
	fi

# Security and maintenance
security-scan:
	@echo "Running security scans..."
	@if [ -x "./scripts/maintenance.sh" ]; then \
		./scripts/maintenance.sh security-scan; \
	else \
		uv run bandit -r src/ -f json -o bandit-report.json || echo "bandit not available"; \
	fi

update-deps:
	@echo "Updating dependencies..."
	@if [ -x "./scripts/maintenance.sh" ]; then \
		./scripts/maintenance.sh update-deps; \
	else \
		uv sync --upgrade; \
		uv sync --upgrade --dev; \
	fi

cleanup:
	@echo "Cleaning up temporary files..."
	@if [ -x "./scripts/maintenance.sh" ]; then \
		./scripts/maintenance.sh cleanup; \
	else \
		$(MAKE) clean; \
		find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; \
		find . -type f -name "*.pyc" -delete; \
	fi

monitor:
	@echo "Starting server monitoring..."
	@if [ -x "./scripts/maintenance.sh" ]; then \
		./scripts/maintenance.sh monitor; \
	else \
		echo "Monitoring script not available. Use: make health-check"; \
	fi

maintenance:
	@echo "Available maintenance commands:"
	@if [ -x "./scripts/maintenance.sh" ]; then \
		./scripts/maintenance.sh help; \
	else \
		echo "  make health-check  - Check server health"; \
		echo "  make cleanup       - Clean temporary files"; \
		echo "  make update-deps   - Update dependencies"; \
		echo "  make security-scan - Run security scans"; \
	fi

# ============================================================================
# Cross-Platform Compatibility Targets
# ============================================================================

# Cross-platform server start (tries different methods)
start:
	@echo "Starting ollama-proxy server..."
	@if [ -x "./scripts/launcher.py" ]; then \
		python scripts/launcher.py; \
	elif [ -x "./scripts/start-server.sh" ]; then \
		./scripts/start-server.sh; \
	else \
		$(MAKE) run; \
	fi

# Cross-platform development start
start-dev-cross:
	@echo "Starting development server..."
	@if [ -x "./scripts/launcher.py" ]; then \
		python scripts/launcher.py --dev; \
	elif [ -x "./scripts/start-server.sh" ]; then \
		./scripts/start-server.sh --dev; \
	else \
		$(MAKE) dev; \
	fi

# Check system and dependencies
check-system:
	@echo "System Information:"
	@echo "=================="
	@echo "OS: $$(uname -s 2>/dev/null || echo 'Windows')"
	@echo "Architecture: $$(uname -m 2>/dev/null || echo 'Unknown')"
	@echo "Python: $$(python --version 2>/dev/null || python3 --version 2>/dev/null || echo 'Not found')"
	@echo "UV: $$(uv --version 2>/dev/null || echo 'Not found')"
	@echo "Make: $$(make --version 2>/dev/null | head -1 || echo 'Not found')"
	@echo "Git: $$(git --version 2>/dev/null || echo 'Not found')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "Project Status:"
	@echo "==============="
	@echo "Virtual Environment: $$(if [ -d .venv ]; then echo 'Present'; else echo 'Missing'; fi)"
	@echo "Dependencies: $$(if [ -f uv.lock ]; then echo 'Locked'; else echo 'Not locked'; fi)"
	@echo "Configuration: $$(if [ -f .env ]; then echo 'Present'; else echo 'Missing (.env file)'; fi)"

# Install system dependencies (placeholder)
install-system:
	@echo "System dependency installation:"
	@echo "This target provides guidance for installing system dependencies."
	@echo ""
	@echo "For Ubuntu/Debian:"
	@echo "  sudo apt update && sudo apt install python3 python3-pip curl git"
	@echo ""
	@echo "For macOS (with Homebrew):"
	@echo "  brew install python@3.12 curl git"
	@echo ""
	@echo "For Windows:"
	@echo "  Install Python from https://python.org"
	@echo "  Install Git from https://git-scm.com"
	@echo ""
	@echo "Then install uv:"
	@echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"

# Quick start for new users
quickstart:
	@echo "🚀 Ollama Proxy Quick Start"
	@echo "=========================="
	@echo ""
	@echo "1. Checking system..."
	@$(MAKE) check-system
	@echo ""
	@echo "2. Setting up development environment..."
	@$(MAKE) setup-dev
	@echo ""
	@echo "3. Next steps:"
	@echo "   - Edit .env file and set your OPENROUTER_API_KEY"
	@echo "   - Run 'make start-dev-cross' to start development server"
	@echo "   - Run 'make test' to run tests"
	@echo "   - Run 'make help' to see all available commands"

# All-in-one development setup
dev-all: setup-dev
	@echo "Running comprehensive development checks..."
	@$(MAKE) check-all
	@echo ""
	@echo "✅ Development environment is ready!"
	@echo "Run 'make start-dev-cross' to start the development server."
