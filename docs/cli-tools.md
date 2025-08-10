# CLI Tools and Administrative Interface

The Ollama Proxy provides a comprehensive set of command-line tools and administrative interfaces for managing, monitoring, and maintaining your proxy deployment.

## Overview

The proxy includes multiple entry points and tools:

- **Main Entry Points**: Standard server, development, and daemon modes
- **Administrative Tools**: Health checks, configuration management, benchmarking
- **Development Tools**: Testing, linting, code formatting
- **Unified CLI Interface**: Single command with multiple subcommands
- **Cross-Platform Scripts**: Shell scripts and batch files for all platforms

## Entry Points

### Main Entry Points

#### `ollama-proxy`
Standard server mode - the primary way to run the proxy in production.

```bash
ollama-proxy
ollama-proxy --host 0.0.0.0 --port 8080
ollama-proxy --api-key sk-or-your-key-here
```

#### `ollama-proxy-server`
Alias for the standard server mode.

```bash
ollama-proxy-server --host 127.0.0.1 --port 11434
```

#### `ollama-proxy-dev`
Development mode with auto-reload and debug logging.

```bash
ollama-proxy-dev
ollama-proxy-dev --host localhost --port 8080
```

#### `ollama-proxy-daemon`
Daemon mode for background services and production deployments.

```bash
ollama-proxy-daemon
ollama-proxy-daemon --host 0.0.0.0 --port 11434
```

### Administrative Tools

#### `ollama-proxy-admin`
Administrative interface for managing the proxy.

```bash
# Show server status
ollama-proxy-admin status

# Show configuration
ollama-proxy-admin config --format json
ollama-proxy-admin config --format yaml

# List available models
ollama-proxy-admin models
```

#### `ollama-proxy-health`
Health check utility for monitoring proxy status.

```bash
# Basic health check
ollama-proxy-health

# Health check with JSON output
ollama-proxy-health --json

# Health check for remote server
ollama-proxy-health --host production.example.com --port 11434
```

#### `ollama-proxy-config`
Configuration management tool.

```bash
# Show current configuration
ollama-proxy-config --show

# Show configuration in different formats
ollama-proxy-config --show --format yaml
ollama-proxy-config --show --format json

# Validate configuration
ollama-proxy-config --validate

# Show configuration help
ollama-proxy-config --help
```

#### `ollama-proxy-benchmark`
Performance benchmarking tool.

```bash
# Basic benchmark
ollama-proxy-benchmark

# Custom benchmark parameters
ollama-proxy-benchmark --requests 1000 --concurrency 50

# Benchmark specific endpoint
ollama-proxy-benchmark --endpoint /api/chat --model gpt-4

# Save benchmark results
ollama-proxy-benchmark --output benchmark-results.json
```

### Development Tools

#### `ollama-proxy-test`
Test runner with various options.

```bash
# Run all tests
ollama-proxy-test

# Run tests with coverage
ollama-proxy-test --coverage

# Run specific test types
ollama-proxy-test --unit
ollama-proxy-test --integration

# Run tests in watch mode
ollama-proxy-test --watch

# Generate HTML coverage report
ollama-proxy-test --coverage --html-report
```

#### `ollama-proxy-lint`
Code linting tool.

```bash
# Run linting
ollama-proxy-lint

# Run linting with auto-fix
ollama-proxy-lint --fix

# Lint specific files
ollama-proxy-lint src/api.py src/config.py

# Show linting statistics
ollama-proxy-lint --stats
```

#### `ollama-proxy-format`
Code formatting tool.

```bash
# Format code
ollama-proxy-format

# Check formatting without making changes
ollama-proxy-format --check

# Format specific files
ollama-proxy-format src/api.py

# Show formatting diff
ollama-proxy-format --diff
```

## Unified CLI Interface

### `ollama-proxy-cli`
Unified command-line interface with subcommands for all functionality.

#### Server Management

```bash
# Start server (default subcommand)
ollama-proxy-cli server
ollama-proxy-cli server --host 0.0.0.0 --port 8080

# Development mode
ollama-proxy-cli dev
ollama-proxy-cli dev --reload --log-level DEBUG

# Daemon mode
ollama-proxy-cli daemon
ollama-proxy-cli daemon --host 0.0.0.0
```

#### Administrative Tasks

```bash
# Server status
ollama-proxy-cli admin status
ollama-proxy-cli admin status --host production.com

# Configuration management
ollama-proxy-cli admin config --format json
ollama-proxy-cli admin config --show --format yaml

# Model management
ollama-proxy-cli admin models
ollama-proxy-cli admin models --provider openai
```

#### Health Monitoring

```bash
# Basic health check
ollama-proxy-cli health

# Detailed health check with JSON output
ollama-proxy-cli health --json --detailed

# Health check for remote server
ollama-proxy-cli health --host production.com --port 11434

# Provider-specific health check
ollama-proxy-cli health --provider openai
```

#### Configuration Management

```bash
# Show configuration
ollama-proxy-cli config --show
ollama-proxy-cli config --show --format yaml

# Validate configuration
ollama-proxy-cli config --validate

# Test configuration
ollama-proxy-cli config --test
```

#### Performance Testing

```bash
# Basic benchmark
ollama-proxy-cli benchmark

# Custom benchmark
ollama-proxy-cli benchmark --requests 1000 --concurrency 50

# Benchmark specific model
ollama-proxy-cli benchmark --model gpt-4 --endpoint /api/chat

# Load testing
ollama-proxy-cli benchmark --duration 300 --ramp-up 60
```

#### Development Tasks

```bash
# Run tests
ollama-proxy-cli test
ollama-proxy-cli test --coverage --html-report

# Code quality checks
ollama-proxy-cli lint
ollama-proxy-cli lint --fix

# Code formatting
ollama-proxy-cli format
ollama-proxy-cli format --check
```

## Cross-Platform Scripts

### Unix/Linux/macOS Scripts

#### `scripts/dev-setup.sh`
Complete development environment setup.

```bash
./scripts/dev-setup.sh
./scripts/dev-setup.sh --python-version 3.12
./scripts/dev-setup.sh --skip-deps
```

#### `scripts/start-server.sh`
Cross-platform server startup script.

```bash
./scripts/start-server.sh
./scripts/start-server.sh --dev
./scripts/start-server.sh --host 0.0.0.0 --port 8080
./scripts/start-server.sh --daemon
```

#### `scripts/test-runner.sh`
Advanced test runner with multiple options.

```bash
./scripts/test-runner.sh
./scripts/test-runner.sh --coverage --html-report
./scripts/test-runner.sh --unit --fast
./scripts/test-runner.sh --integration --verbose
./scripts/test-runner.sh --watch
```

#### `scripts/maintenance.sh`
Server maintenance and monitoring script.

```bash
./scripts/maintenance.sh health
./scripts/maintenance.sh monitor
./scripts/maintenance.sh cleanup
./scripts/maintenance.sh backup
```

#### `scripts/build-deploy.sh`
Build and deployment automation.

```bash
./scripts/build-deploy.sh check
./scripts/build-deploy.sh build
./scripts/build-deploy.sh release --version 1.0.0
./scripts/build-deploy.sh deploy --environment production
```

### Windows Scripts

#### `scripts/dev-setup.bat`
Windows development environment setup.

```cmd
scripts\dev-setup.bat
scripts\dev-setup.bat --python-version 3.12
```

#### `scripts/start-server.bat`
Windows server startup script.

```cmd
scripts\start-server.bat
scripts\start-server.bat --dev
scripts\start-server.bat --host 0.0.0.0 --port 8080
```

#### `scripts/test-runner.bat`
Windows test runner.

```cmd
scripts\test-runner.bat
scripts\test-runner.bat --coverage --html-report
scripts\test-runner.bat --unit
```

#### `scripts/maintenance.bat`
Windows maintenance script.

```cmd
scripts\maintenance.bat health
scripts\maintenance.bat monitor
scripts\maintenance.bat cleanup
```

### PowerShell Scripts

#### `scripts/start-server.ps1`
PowerShell server startup script.

```powershell
.\scripts\start-server.ps1
.\scripts\start-server.ps1 -Dev
.\scripts\start-server.ps1 -Host "0.0.0.0" -Port 8080
```

### Universal Python Launcher

#### `scripts/launcher.py`
Cross-platform Python launcher that works on all platforms.

```bash
# Basic usage
python scripts/launcher.py

# Development mode
python scripts/launcher.py --dev

# Custom configuration
python scripts/launcher.py --host 0.0.0.0 --port 8080

# Daemon mode
python scripts/launcher.py --daemon

# Check dependencies
python scripts/launcher.py --check-deps

# Dry run (show what would be executed)
python scripts/launcher.py --dry-run
```

## Make/Batch Commands

### Unix/Linux/macOS with Make

```bash
# Quick start
make quickstart          # Complete setup and start guide
make dev                 # Start development server
make start               # Start production server
make start-daemon        # Start daemon server

# Testing
make test                # Run all tests
make test-cov           # Run tests with coverage
make test-unit          # Run unit tests only
make test-integration   # Run integration tests only
make test-watch         # Run tests in watch mode

# Code quality
make lint               # Run linting
make lint-fix          # Run linting with auto-fix
make format            # Format code
make format-check      # Check code formatting
make type-check        # Run type checking
make check-all         # Run all quality checks

# Maintenance
make health-check      # Check server health
make config-show       # Show current configuration
make benchmark         # Run performance benchmarks
make cleanup           # Clean temporary files

# Build and deploy
make build             # Build package
make docker-build      # Build Docker image
make docker-run        # Run Docker container
```

### Windows with make.bat

```cmd
REM Quick start
make.bat quickstart     # Complete setup
make.bat dev           # Start development server
make.bat start         # Start production server

REM Testing
make.bat test          # Run all tests
make.bat test-cov      # Run tests with coverage

REM Code quality
make.bat lint          # Run linting
make.bat format        # Format code
make.bat check-all     # Run all checks

REM Maintenance
make.bat health-check  # Check server health
make.bat benchmark     # Run benchmarks
make.bat cleanup       # Clean temporary files
```

## Common Usage Patterns

### Development Workflow

```bash
# 1. Set up development environment
./scripts/dev-setup.sh

# 2. Start development server
make dev
# or
ollama-proxy-dev

# 3. Run tests during development
make test-watch
# or
ollama-proxy-test --watch

# 4. Check code quality
make check-all
# or
ollama-proxy-cli lint && ollama-proxy-cli format --check
```

### Production Deployment

```bash
# 1. Health check before deployment
ollama-proxy-health --host production.com

# 2. Deploy new version
./scripts/build-deploy.sh deploy --environment production

# 3. Verify deployment
ollama-proxy-cli health --host production.com --json

# 4. Monitor performance
ollama-proxy-cli benchmark --host production.com
```

### Monitoring and Maintenance

```bash
# Regular health checks
ollama-proxy-health --json

# Performance monitoring
ollama-proxy-benchmark --requests 100

# Configuration validation
ollama-proxy-config --validate

# System cleanup
make cleanup
```

## Advanced Features

### Configuration Profiles

Use different configuration profiles for different environments:

```bash
# Development profile
ollama-proxy-cli server --config-profile development

# Production profile
ollama-proxy-cli server --config-profile production

# Custom profile
ollama-proxy-cli server --config-profile custom --config-file custom.env
```

### Batch Operations

Perform batch operations across multiple servers:

```bash
# Health check multiple servers
for server in server1 server2 server3; do
  ollama-proxy-health --host $server --json
done

# Deploy to multiple environments
./scripts/build-deploy.sh deploy --environments staging,production
```

### Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Run tests
  run: ollama-proxy-test --coverage

- name: Check code quality
  run: |
    ollama-proxy-lint
    ollama-proxy-format --check

- name: Build and deploy
  run: ./scripts/build-deploy.sh deploy --environment production
```

## Troubleshooting CLI Tools

### Common Issues

1. **Command not found**: Ensure the package is installed and virtual environment is activated
2. **Permission denied**: Check file permissions for script files
3. **Configuration errors**: Use `ollama-proxy-config --validate` to check configuration
4. **Network issues**: Use `ollama-proxy-health` to diagnose connectivity problems

### Debug Mode

Enable debug mode for any CLI tool:

```bash
# Add --debug flag to any command
ollama-proxy-cli server --debug
ollama-proxy-health --debug
ollama-proxy-benchmark --debug
```

### Getting Help

All CLI tools support help options:

```bash
ollama-proxy --help
ollama-proxy-cli --help
ollama-proxy-cli server --help
ollama-proxy-admin --help
```
