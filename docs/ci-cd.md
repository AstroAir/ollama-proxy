# CI/CD Quick Reference Guide

## 🚀 Running Tests Locally

### Basic Test Commands

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m security      # Security tests only
uv run pytest -m performance   # Performance tests only

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_api_comprehensive.py -v
```

### Performance Testing

```bash
# Run performance benchmarks
uv run pytest tests/test_performance.py -m performance

# Run load tests
python tests/load_test.py
# or
locust --headless --users 10 --spawn-rate 2 --host http://localhost:11434 --run-time 60s --locustfile tests/load_test.py
```

## 🔒 Security Scanning

### Manual Security Scans

```bash
# Run comprehensive security scan
python scripts/security-scan.py

# Individual security tools
uv run bandit -r src/
uv run safety check
uv run pip-audit
detect-secrets scan --baseline .secrets.baseline
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 🐳 Docker Operations

### Building Images

```bash
# Build standard image
docker build -t ollama-proxy:latest .

# Build multi-architecture image
docker buildx build --platform linux/amd64,linux/arm64 -t ollama-proxy:multiarch .

# Build with build script
python scripts/build-deploy.py --docker-tag ollama-proxy:test
```

### Running Containers

```bash
# Development environment
docker-compose up -d

# Staging environment
docker-compose -f docker-compose.staging.yml up -d

# Production environment
docker-compose -f docker-compose.production.yml up -d

# With monitoring
docker-compose --profile monitoring up -d
```

## 📊 Monitoring Setup

### Start Monitoring Stack

```bash
# Setup monitoring
python scripts/monitoring-setup.py --setup --environment development

# Run health checks
python scripts/monitoring-setup.py --health-check

# Generate health report
python scripts/monitoring-setup.py --generate-report
```

### Access Monitoring Services

- **Prometheus**: <http://localhost:9090>
- **Grafana**: <http://localhost:3000> (admin/admin)
- **Alertmanager**: <http://localhost:9093>

## 🔧 Build and Deployment

### Automated Build

```bash
# Full build pipeline
python scripts/build-deploy.py

# Build with specific options
python scripts/build-deploy.py \
  --docker-tag ollama-proxy:v1.0.0 \
  --multiarch \
  --push \
  --create-artifacts
```

### Environment Deployment

```bash
# Deploy to staging
python scripts/build-deploy.py --deploy staging

# Deploy to production
python scripts/build-deploy.py --deploy production
```

## 📋 CI/CD Pipeline Triggers

### Automatic Triggers

- **Push to main/develop**: Full CI pipeline
- **Pull Request**: Validation pipeline
- **Nightly**: Comprehensive testing (3 AM UTC)
- **Release Tag**: Release pipeline

### Manual Triggers

- **Workflow Dispatch**: Manual pipeline execution
- **Re-run Jobs**: Re-run failed jobs
- **Skip CI**: Add `[skip ci]` to commit message

## 🧪 Test Categories and Markers

### Available Test Markers

```bash
# Test categories
-m unit           # Fast unit tests
-m integration    # Integration tests
-m e2e           # End-to-end tests
-m performance   # Performance tests
-m security      # Security tests
-m contract      # API contract tests
-m slow          # Slow-running tests

# Exclude slow tests
-m "not slow"

# Combine markers
-m "unit or integration"
```

## 🔍 Debugging CI Issues

### Common Issues and Solutions

#### Test Failures

```bash
# Run tests with verbose output
uv run pytest -v --tb=long

# Run specific failing test
uv run pytest tests/test_file.py::test_function -v -s
```

#### Coverage Issues

```bash
# Check coverage details
uv run pytest --cov=src --cov-report=term-missing --cov-report=html

# View HTML coverage report
open htmlcov/index.html
```

#### Docker Build Issues

```bash
# Build with verbose output
docker build --progress=plain -t ollama-proxy:debug .

# Check container logs
docker logs container_name
```

## 📁 Important Files and Directories

### CI/CD Configuration

- `.github/workflows/` - GitHub Actions workflows
- `docker-compose*.yml` - Environment configurations
- `scripts/` - Automation scripts

### Testing

- `tests/` - All test files
- `tests/pytest.ini` - Pytest configuration
- `.coverage` - Coverage data

### Security

- `.secrets.baseline` - Secret detection baseline
- `.pre-commit-config.yaml` - Pre-commit hooks
- `security-report.json` - Security scan results

### Monitoring

- `monitoring/` - Monitoring configurations
- `health-report.json` - Health check results

## 🚨 Emergency Procedures

### Rollback Deployment

```bash
# Rollback to previous version
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d --scale ollama-proxy=0
# Deploy previous version
```

### Disable Monitoring Alerts

```bash
# Silence alerts in Alertmanager
curl -X POST http://localhost:9093/api/v1/silences \
  -H "Content-Type: application/json" \
  -d '{"matchers":[{"name":"alertname","value":".*","isRegex":true}],"startsAt":"2024-01-01T00:00:00Z","endsAt":"2024-01-01T01:00:00Z","comment":"Emergency maintenance"}'
```

### Emergency Health Check

```bash
# Quick health check
curl http://localhost:11434/health

# Detailed health check
python scripts/monitoring-setup.py --health-check --generate-report
```

## 📞 Support and Troubleshooting

### Log Locations

- **Application Logs**: `logs/` directory
- **Docker Logs**: `docker logs container_name`
- **CI Logs**: GitHub Actions interface

### Useful Commands

```bash
# Check service status
docker-compose ps

# View real-time logs
docker-compose logs -f ollama-proxy

# Check resource usage
docker stats

# Cleanup
docker system prune -a
```

---

**Need Help?** Check the full implementation summary in `CI_CD_IMPLEMENTATION_SUMMARY.md` for detailed information.
