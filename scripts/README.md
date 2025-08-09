# Scripts Directory

This directory contains various utility scripts for the ollama-proxy project. These scripts provide cross-platform support for development, testing, building, deployment, and maintenance tasks.

## Available Scripts

### 🚀 Startup Scripts

#### `start-server.sh` (Unix/Linux/macOS)

Cross-platform startup script for Unix-like systems.

```bash
# Basic usage
./scripts/start-server.sh

# Development mode
./scripts/start-server.sh --dev

# Custom configuration
./scripts/start-server.sh --host 127.0.0.1 --port 8080 --api-key sk-or-...

# Daemon mode
./scripts/start-server.sh --daemon

# Check dependencies
./scripts/start-server.sh --check-deps
```

#### `start-server.bat` (Windows)

Windows batch file for starting the server.

```cmd
REM Basic usage
scripts\start-server.bat

REM Development mode
scripts\start-server.bat --dev

REM Custom configuration
scripts\start-server.bat --host 127.0.0.1 --port 8080
```

#### `start-server.ps1` (PowerShell)

PowerShell script with better Windows support.

```powershell
# Basic usage
.\scripts\start-server.ps1

# Development mode
.\scripts\start-server.ps1 -Dev

# Custom configuration
.\scripts\start-server.ps1 -Host 127.0.0.1 -Port 8080 -ApiKey "sk-or-..."
```

#### `launcher.py` (Cross-platform Python)

Universal Python launcher that works on all platforms.

```bash
# Basic usage
python scripts/launcher.py

# Development mode
python scripts/launcher.py --dev

# Check dependencies
python scripts/launcher.py --check-deps
```

### 🛠️ Development Scripts

#### `dev-setup.sh`

Complete development environment setup script.

```bash
# Full setup
./scripts/dev-setup.sh

# Skip specific components
./scripts/dev-setup.sh --skip-uv --skip-pre-commit

# Force reinstall
./scripts/dev-setup.sh --force
```

Features:

- Installs uv package manager
- Sets up Python dependencies
- Configures pre-commit hooks
- Sets up documentation tools
- Creates .env file from template
- Runs initial code quality checks

#### `test-runner.sh`

Advanced test runner with comprehensive options.

```bash
# Run all tests
./scripts/test-runner.sh

# Run with coverage
./scripts/test-runner.sh --coverage --html-report

# Run specific test types
./scripts/test-runner.sh --unit --fast
./scripts/test-runner.sh --integration --verbose

# Watch mode for development
./scripts/test-runner.sh --watch

# Rerun failed tests
./scripts/test-runner.sh --rerun-failed
```

Features:

- Multiple test selection options (unit, integration, performance)
- Coverage reporting (HTML, XML, terminal)
- Parallel test execution
- Watch mode for continuous testing
- Benchmark support
- JSON reporting

### 🏗️ Build and Deployment Scripts

#### `build-deploy.sh`

Comprehensive build and deployment automation.

```bash
# Build package
./scripts/build-deploy.sh build

# Build Docker image
./scripts/build-deploy.sh docker --tag v1.0.0

# Create release
./scripts/build-deploy.sh release --version 1.0.0

# Clean build artifacts
./scripts/build-deploy.sh clean

# Run pre-build checks
./scripts/build-deploy.sh check
```

Features:

- Package building with uv
- Docker image building and pushing
- Release management with git tagging
- Pre-build quality checks
- Artifact cleanup
- Dry-run mode

### 🔧 Maintenance Scripts

#### `maintenance.sh`

Server maintenance and monitoring utilities.

```bash
# Check server health
./scripts/maintenance.sh health

# Monitor in real-time
./scripts/maintenance.sh monitor

# Show performance metrics
./scripts/maintenance.sh metrics --format json

# Clean up temporary files
./scripts/maintenance.sh cleanup

# Update dependencies
./scripts/maintenance.sh update-deps

# Run security scan
./scripts/maintenance.sh security-scan

# Run benchmarks
./scripts/maintenance.sh benchmark
```

Features:

- Health monitoring
- Performance metrics
- Log analysis
- Cleanup utilities
- Dependency updates
- Security scanning
- Benchmarking

## Usage Patterns

### Development Workflow

1. **Initial Setup**

   ```bash
   ./scripts/dev-setup.sh
   ```

2. **Start Development Server**

   ```bash
   ./scripts/start-server.sh --dev
   # or
   python scripts/launcher.py --dev
   ```

3. **Run Tests During Development**

   ```bash
   ./scripts/test-runner.sh --watch --unit --fast
   ```

4. **Check Code Quality**

   ```bash
   ./scripts/build-deploy.sh check
   ```

### Production Deployment

1. **Build and Test**

   ```bash
   ./scripts/build-deploy.sh check
   ./scripts/build-deploy.sh build
   ```

2. **Create Release**

   ```bash
   ./scripts/build-deploy.sh release --version 1.0.0
   ```

3. **Build Docker Image**

   ```bash
   ./scripts/build-deploy.sh docker --tag v1.0.0 --push
   ```

4. **Monitor Production**

   ```bash
   ./scripts/maintenance.sh health --host production.com
   ./scripts/maintenance.sh monitor --host production.com
   ```

### Maintenance Tasks

1. **Regular Cleanup**

   ```bash
   ./scripts/maintenance.sh cleanup
   ```

2. **Update Dependencies**

   ```bash
   ./scripts/maintenance.sh update-deps
   ./scripts/test-runner.sh  # Verify everything still works
   ```

3. **Security Audit**

   ```bash
   ./scripts/maintenance.sh security-scan
   ```

4. **Performance Monitoring**

   ```bash
   ./scripts/maintenance.sh benchmark
   ./scripts/maintenance.sh metrics
   ```

## Cross-Platform Compatibility

All scripts are designed to work across different platforms:

- **Unix/Linux/macOS**: Use `.sh` scripts
- **Windows**: Use `.bat` or `.ps1` scripts
- **Any Platform**: Use `launcher.py` Python script

### Platform-Specific Notes

#### Windows

- Batch files (`.bat`) have limited functionality
- PowerShell scripts (`.ps1`) provide better features
- May need to set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### macOS

- All shell scripts should work out of the box
- May need to install additional tools via Homebrew

#### Linux

- All shell scripts should work on most distributions
- Some distributions may need additional packages

## Environment Variables

Scripts respect these environment variables:

- `OPENROUTER_API_KEY`: OpenRouter API key (required)
- `HOST`: Default host to bind to
- `PORT`: Default port to listen on
- `LOG_LEVEL`: Default logging level
- `MODELS_FILTER_PATH`: Path to model filter file

## Contributing

When adding new scripts:

1. Follow the existing naming convention
2. Include comprehensive help text
3. Support dry-run mode where applicable
4. Add cross-platform compatibility
5. Update this README
6. Make scripts executable: `chmod +x script-name.sh`

## Troubleshooting

### Common Issues

1. **Permission Denied**

   ```bash
   chmod +x scripts/*.sh
   ```

2. **Command Not Found (uv)**

   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Python Not Found**
   - Ensure Python 3.12+ is installed
   - Check PATH configuration

4. **Docker Issues**
   - Ensure Docker is installed and running
   - Check Docker permissions

### Getting Help

Each script includes a help option:

```bash
./scripts/script-name.sh --help
```

For project-specific help, see the main README.md file.
