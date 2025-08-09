@echo off
REM Windows alternative to Makefile for ollama-proxy
REM Provides common development commands for Windows users

setlocal enabledelayedexpansion

REM Simple prefixes for output
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM Get the command
set "COMMAND=%~1"

if "%COMMAND%"=="" (
    set "COMMAND=help"
)

REM Execute the command
if "%COMMAND%"=="help" goto :help
if "%COMMAND%"=="install" goto :install
if "%COMMAND%"=="install-dev" goto :install_dev
if "%COMMAND%"=="setup-dev" goto :setup_dev
if "%COMMAND%"=="test" goto :test
if "%COMMAND%"=="test-cov" goto :test_cov
if "%COMMAND%"=="test-unit" goto :test_unit
if "%COMMAND%"=="test-integration" goto :test_integration
if "%COMMAND%"=="lint" goto :lint
if "%COMMAND%"=="format" goto :format
if "%COMMAND%"=="format-check" goto :format_check
if "%COMMAND%"=="type-check" goto :type_check
if "%COMMAND%"=="check-all" goto :check_all
if "%COMMAND%"=="clean" goto :clean
if "%COMMAND%"=="build" goto :build
if "%COMMAND%"=="run" goto :run
if "%COMMAND%"=="dev" goto :dev
if "%COMMAND%"=="start-dev" goto :start_dev
if "%COMMAND%"=="start-daemon" goto :start_daemon
if "%COMMAND%"=="health-check" goto :health_check
if "%COMMAND%"=="config-show" goto :config_show
if "%COMMAND%"=="benchmark" goto :benchmark
if "%COMMAND%"=="cleanup" goto :cleanup
if "%COMMAND%"=="update-deps" goto :update_deps
if "%COMMAND%"=="security-scan" goto :security_scan
if "%COMMAND%"=="quickstart" goto :quickstart

echo %ERROR% Unknown command: %COMMAND%
goto :help

:help
echo Available commands for ollama-proxy (Windows):
echo.
echo 🏗️  Setup and Installation:
echo   install         - Install production dependencies
echo   install-dev     - Install development dependencies
echo   setup-dev       - Complete development environment setup
echo.
echo 🚀 Server Management:
echo   run             - Run production server
echo   dev             - Run development server
echo   start-dev       - Start development server (alternative)
echo   start-daemon    - Start server in daemon mode
echo.
echo 🧪 Testing:
echo   test            - Run tests
echo   test-cov        - Run tests with coverage
echo   test-unit       - Run unit tests only
echo   test-integration - Run integration tests only
echo.
echo 🔍 Code Quality:
echo   lint            - Run linting (flake8)
echo   format          - Format code (black, isort)
echo   format-check    - Check code formatting
echo   type-check      - Run type checking (mypy)
echo   check-all       - Run all checks
echo.
echo 🏗️  Build:
echo   clean           - Clean build artifacts
echo   build           - Build package
echo.
echo 🔧 Maintenance:
echo   health-check    - Check server health
echo   config-show     - Show current configuration
echo   benchmark       - Run performance benchmarks
echo   cleanup         - Clean temporary files
echo   update-deps     - Update dependencies
echo   security-scan   - Run security scans
echo.
echo 🚀 Quick Start:
echo   quickstart      - Complete setup and start guide
echo.
echo Examples:
echo   %~nx0 setup-dev     # Set up development environment
echo   %~nx0 dev           # Start development server
echo   %~nx0 test          # Run tests
echo   %~nx0 check-all     # Run all quality checks
echo.
goto :eof

:install
echo %INFO% Installing production dependencies...
uv sync --no-dev
goto :eof

:install_dev
echo %INFO% Installing development dependencies...
uv sync --all-extras --dev
goto :eof

:setup_dev
echo %INFO% Setting up development environment...
if exist "scripts\dev-setup.bat" (
    call scripts\dev-setup.bat
) else (
    call :install_dev
    echo %SUCCESS% Basic development setup complete
)
goto :eof

:test
echo %INFO% Running tests...
uv run pytest
goto :eof

:test_cov
echo %INFO% Running tests with coverage...
uv run pytest --cov=src --cov-report=html --cov-report=term-missing
goto :eof

:test_unit
echo %INFO% Running unit tests...
uv run pytest -m "unit"
goto :eof

:test_integration
echo %INFO% Running integration tests...
uv run pytest -m "integration"
goto :eof

:lint
echo %INFO% Running linting...
uv run flake8 src tests
goto :eof

:format
echo %INFO% Formatting code...
uv run black src tests
uv run isort src tests
goto :eof

:format_check
echo %INFO% Checking code formatting...
uv run black --check src tests
uv run isort --check-only src tests
goto :eof

:type_check
echo %INFO% Running type checking...
uv run mypy src
goto :eof

:check_all
echo %INFO% Running all checks...
call :lint
if errorlevel 1 exit /b 1
call :format_check
if errorlevel 1 exit /b 1
call :type_check
if errorlevel 1 exit /b 1
call :test
if errorlevel 1 exit /b 1
echo %SUCCESS% All checks passed!
goto :eof

:clean
echo %INFO% Cleaning build artifacts...
if exist "build" rd /s /q "build" 2>nul
if exist "dist" rd /s /q "dist" 2>nul
for /d %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d" 2>nul
if exist ".pytest_cache" rd /s /q ".pytest_cache" 2>nul
if exist ".mypy_cache" rd /s /q ".mypy_cache" 2>nul
if exist "htmlcov" rd /s /q "htmlcov" 2>nul
if exist ".coverage" del /q ".coverage" 2>nul
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
echo %SUCCESS% Build artifacts cleaned
goto :eof

:build
echo %INFO% Building package...
call :clean
uv build
goto :eof

:run
echo %INFO% Running production server...
uv run python -m src.main
goto :eof

:dev
echo %INFO% Running development server...
uv run python -m src.main --reload --log-level DEBUG
goto :eof

:start_dev
echo %INFO% Starting development server...
uv run ollama-proxy-dev
goto :eof

:start_daemon
echo %INFO% Starting daemon server...
uv run ollama-proxy-daemon
goto :eof

:health_check
echo %INFO% Checking server health...
uv run ollama-proxy-health
goto :eof

:config_show
echo %INFO% Showing configuration...
uv run ollama-proxy-config --show
goto :eof

:benchmark
echo %INFO% Running benchmarks...
uv run ollama-proxy-benchmark
goto :eof

:cleanup
echo %INFO% Running cleanup...
if exist "scripts\maintenance.bat" (
    call scripts\maintenance.bat cleanup
) else (
    call :clean
)
goto :eof

:update_deps
echo %INFO% Updating dependencies...
if exist "scripts\maintenance.bat" (
    call scripts\maintenance.bat update-deps
) else (
    uv sync --upgrade
    uv sync --upgrade --dev
)
goto :eof

:security_scan
echo %INFO% Running security scan...
if exist "scripts\maintenance.bat" (
    call scripts\maintenance.bat security-scan
) else (
    uv run bandit -r src/ -f json -o bandit-report.json
)
goto :eof

:quickstart
echo 🚀 Ollama Proxy Quick Start (Windows)
echo ====================================
echo.
echo 1. Setting up development environment...
call :setup_dev
echo.
echo 2. Next steps:
echo    - Edit .env file and set your OPENROUTER_API_KEY
echo    - Run '%~nx0 dev' to start development server
echo    - Run '%~nx0 test' to run tests
echo    - Run '%~nx0 help' to see all available commands
echo.
echo ✅ Setup complete! Happy coding! 🚀
goto :eof
