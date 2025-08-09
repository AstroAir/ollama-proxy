@echo off
REM Maintenance and utility script for Windows
REM Handles various maintenance tasks like health monitoring, cleanup, etc.

setlocal enabledelayedexpansion

REM Simple prefixes for output
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM Script directory and project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Default values
set "COMMAND="
set "HOST=localhost"
set "PORT=11434"
set "TIMEOUT=10"
set "FORMAT=table"

REM Function to show usage
:show_usage
echo Usage: %~nx0 [COMMAND] [OPTIONS]
echo.
echo Maintenance and utility script for ollama-proxy.
echo.
echo COMMANDS:
echo     health          Check server health and status
echo     metrics         Show performance metrics
echo     cleanup         Clean up temporary files and caches
echo     update-deps     Update dependencies
echo     security-scan   Run security scans
echo     benchmark       Run performance benchmarks
echo     help            Show this help message
echo.
echo OPTIONS:
echo     --host HOST     Server host (default: localhost)
echo     --port PORT     Server port (default: 11434)
echo     --timeout SEC   Request timeout (default: 10)
echo     --format FORMAT Output format (json, table)
echo.
echo EXAMPLES:
echo     %~nx0 health                    # Check server health
echo     %~nx0 metrics --format json     # Show metrics in JSON
echo     %~nx0 cleanup                   # Clean temporary files
echo     %~nx0 benchmark --host prod.com # Benchmark production
echo.
goto :eof

REM Parse command line arguments
if "%~1"=="" (
    call :show_usage
    exit /b 1
)

set "COMMAND=%~1"
shift

:parse_args
if "%~1"=="" goto :args_done

if "%~1"=="--host" (
    set "HOST=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--timeout" (
    set "TIMEOUT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="--format" (
    set "FORMAT=%~2"
    shift
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help

echo %ERROR% Unknown option: %~1
call :show_usage
exit /b 1

:help
call :show_usage
exit /b 0

:args_done

echo %INFO% Maintenance script for ollama-proxy
echo %INFO% Project root: %PROJECT_ROOT%

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Execute command
if "%COMMAND%"=="health" goto :check_health
if "%COMMAND%"=="metrics" goto :show_metrics
if "%COMMAND%"=="cleanup" goto :cleanup
if "%COMMAND%"=="update-deps" goto :update_deps
if "%COMMAND%"=="security-scan" goto :security_scan
if "%COMMAND%"=="benchmark" goto :run_benchmark
if "%COMMAND%"=="help" goto :help

echo %ERROR% Unknown command: %COMMAND%
call :show_usage
exit /b 1

REM Function to check server health
:check_health
echo %INFO% Checking server health at %HOST%:%PORT%...

REM Use PowerShell to make HTTP request (more reliable than curl on Windows)
powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://%HOST%:%PORT%/health' -TimeoutSec %TIMEOUT%; $status = $response.status; switch ($status) { 'healthy' { Write-Host '[SUCCESS] Server is healthy' -ForegroundColor Green } 'degraded' { Write-Host '[WARNING] Server is degraded' -ForegroundColor Yellow } 'unhealthy' { Write-Host '[ERROR] Server is unhealthy' -ForegroundColor Red } 'critical' { Write-Host '[ERROR] Server is critical' -ForegroundColor Red } default { Write-Host '[WARNING] Server status unknown' -ForegroundColor Yellow } }; if ('%FORMAT%' -eq 'json') { $response | ConvertTo-Json } else { Write-Host 'Health Status Report'; Write-Host '===================='; $response.PSObject.Properties | ForEach-Object { Write-Host ('{0,-20}: {1}' -f $_.Name, $_.Value) } } } catch { Write-Host '[ERROR] Failed to connect to server at %HOST%:%PORT%' -ForegroundColor Red; exit 1 }"

goto :eof

REM Function to show metrics
:show_metrics
echo %INFO% Fetching performance metrics...

powershell -Command "try { $response = Invoke-RestMethod -Uri 'http://%HOST%:%PORT%/health' -TimeoutSec %TIMEOUT%; if ('%FORMAT%' -eq 'json') { $response | ConvertTo-Json } else { Write-Host 'Performance Metrics'; Write-Host '==================='; $metrics = @{ 'Status' = $response.status; 'Uptime (s)' = $response.uptime_seconds; 'Total Requests' = $response.total_requests; 'Success Rate' = '{0:P2}' -f $response.success_rate; 'Avg Duration (ms)' = $response.average_duration_ms; 'Active Endpoints' = $response.active_endpoints }; $metrics.GetEnumerator() | ForEach-Object { Write-Host ('{0,-20}: {1}' -f $_.Key, $_.Value) } } } catch { Write-Host '[ERROR] Failed to fetch metrics from server' -ForegroundColor Red; exit 1 }"

goto :eof

REM Function to cleanup temporary files
:cleanup
echo %INFO% Cleaning up temporary files and caches...

REM Clean Python cache files
echo %INFO% Cleaning Python cache files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul

REM Clean test artifacts
echo %INFO% Cleaning test artifacts...
if exist ".pytest_cache" rd /s /q ".pytest_cache" 2>nul
if exist ".mypy_cache" rd /s /q ".mypy_cache" 2>nul
if exist "htmlcov" rd /s /q "htmlcov" 2>nul
if exist ".coverage" del /q ".coverage" 2>nul

REM Clean build artifacts
echo %INFO% Cleaning build artifacts...
if exist "build" rd /s /q "build" 2>nul
if exist "dist" rd /s /q "dist" 2>nul
for /d %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d" 2>nul

REM Clean temporary files
echo %INFO% Cleaning temporary files...
del /s /q *.tmp 2>nul

echo %SUCCESS% Cleanup completed

goto :eof

REM Function to update dependencies
:update_deps
echo %INFO% Updating dependencies...

REM Update uv itself
echo %INFO% Updating uv...
uv self update
if errorlevel 1 (
    echo %WARNING% Failed to update uv
)

REM Update project dependencies
echo %INFO% Updating project dependencies...
uv sync --upgrade
if errorlevel 1 (
    echo %ERROR% Failed to update dependencies
    exit /b 1
)

REM Update dev dependencies
echo %INFO% Updating dev dependencies...
uv sync --upgrade --dev
if errorlevel 1 (
    echo %ERROR% Failed to update dev dependencies
    exit /b 1
)

echo %SUCCESS% Dependencies updated
echo %INFO% Run tests to ensure everything still works: uv run pytest

goto :eof

REM Function to run security scan
:security_scan
echo %INFO% Running security scans...

REM Run bandit security scan
uv run bandit --version >nul 2>&1
if not errorlevel 1 (
    echo %INFO% Running bandit security scan...
    uv run bandit -r src/ -f json -o bandit-report.json
    if not errorlevel 1 (
        echo %INFO% Bandit report saved to bandit-report.json
    ) else (
        echo %WARNING% Bandit scan completed with issues
    )
) else (
    echo %WARNING% bandit not available, skipping security scan
)

REM Check for known vulnerabilities in dependencies
echo %INFO% Checking for vulnerable dependencies...
uv pip check
if errorlevel 1 (
    echo %WARNING% Some dependency issues found
)

echo %SUCCESS% Security scan completed

goto :eof

REM Function to run benchmarks
:run_benchmark
echo %INFO% Running performance benchmarks...

REM Use the benchmark entry point we created
uv run ollama-proxy-benchmark --host %HOST% --port %PORT% --requests 100 --concurrency 10

goto :eof

REM Main execution starts here
call :parse_args %*
