@echo off
REM Cross-platform startup script for ollama-proxy server (Windows)
REM Batch file version for Windows systems

setlocal enabledelayedexpansion

REM Script directory and project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Default values
set "DEFAULT_HOST=0.0.0.0"
set "DEFAULT_PORT=11434"
set "DEFAULT_LOG_LEVEL=INFO"

REM Initialize variables
set "HOST=%DEFAULT_HOST%"
set "PORT=%DEFAULT_PORT%"
set "LOG_LEVEL=%DEFAULT_LOG_LEVEL%"
set "API_KEY="
set "MODELS_FILTER="
set "DEV_MODE=false"
set "DAEMON_MODE=false"
set "ENV_FILE="
set "CHECK_DEPS=false"
set "DRY_RUN=false"

REM Function to show usage
:show_usage
echo Usage: %~nx0 [OPTIONS]
echo.
echo Start the ollama-proxy server with various configuration options.
echo.
echo OPTIONS:
echo     -h, --help              Show this help message
echo     -H, --host HOST         Host to bind to (default: %DEFAULT_HOST%)
echo     -p, --port PORT         Port to listen on (default: %DEFAULT_PORT%)
echo     -k, --api-key KEY       OpenRouter API key
echo     -l, --log-level LEVEL   Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
echo     -m, --models-filter FILE Path to model filter file
echo     -d, --dev               Run in development mode with auto-reload
echo     -D, --daemon            Run in daemon mode (background service)
echo     --env-file FILE         Load environment from specific file
echo     --check-deps            Check dependencies before starting
echo     --dry-run               Show what would be executed without running
echo.
echo ENVIRONMENT VARIABLES:
echo     OPENROUTER_API_KEY      OpenRouter API key (required)
echo     HOST                    Host to bind to
echo     PORT                    Port to listen on
echo     LOG_LEVEL               Logging level
echo     MODELS_FILTER_PATH      Path to model filter file
echo.
echo EXAMPLES:
echo     %~nx0                                          # Start with defaults
echo     %~nx0 --dev                                    # Start in development mode
echo     %~nx0 --host 127.0.0.1 --port 8080           # Custom host and port
echo     %~nx0 --api-key sk-or-... --daemon            # Start as daemon
echo     %~nx0 --check-deps                            # Check dependencies only
echo.
goto :eof

REM Function to print colored output (Windows doesn't support colors easily in batch)
:print_info
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM Function to check if command exists
:command_exists
where "%~1" >nul 2>&1
goto :eof

REM Function to check dependencies
:check_dependencies
call :print_info "Checking dependencies..."

set "missing_deps="

call :command_exists python
if errorlevel 1 (
    call :command_exists python3
    if errorlevel 1 (
        set "missing_deps=!missing_deps! python"
    )
)

call :command_exists uv
if errorlevel 1 (
    set "missing_deps=!missing_deps! uv"
)

if not "!missing_deps!"=="" (
    call :print_error "Missing dependencies:!missing_deps!"
    call :print_info "Please install the missing dependencies:"
    echo   - Python 3.12+: https://www.python.org/downloads/
    echo   - uv: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

call :print_success "All dependencies are available"
goto :eof

REM Function to validate environment
:validate_environment
call :print_info "Validating environment..."

REM Check if we're in the project root
if not exist "%PROJECT_ROOT%\pyproject.toml" (
    call :print_error "Not in ollama-proxy project root. Expected pyproject.toml file."
    exit /b 1
)

REM Check if API key is set (if not provided via command line)
if "%OPENROUTER_API_KEY%"=="" if "%API_KEY%"=="" (
    call :print_warning "OPENROUTER_API_KEY not set. Make sure to provide it via --api-key or environment variable."
)

call :print_success "Environment validation passed"
goto :eof

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done

if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="-H" goto :set_host
if "%~1"=="--host" goto :set_host
if "%~1"=="-p" goto :set_port
if "%~1"=="--port" goto :set_port
if "%~1"=="-k" goto :set_api_key
if "%~1"=="--api-key" goto :set_api_key
if "%~1"=="-l" goto :set_log_level
if "%~1"=="--log-level" goto :set_log_level
if "%~1"=="-m" goto :set_models_filter
if "%~1"=="--models-filter" goto :set_models_filter
if "%~1"=="-d" goto :set_dev_mode
if "%~1"=="--dev" goto :set_dev_mode
if "%~1"=="-D" goto :set_daemon_mode
if "%~1"=="--daemon" goto :set_daemon_mode
if "%~1"=="--env-file" goto :set_env_file
if "%~1"=="--check-deps" goto :set_check_deps
if "%~1"=="--dry-run" goto :set_dry_run

call :print_error "Unknown option: %~1"
call :show_usage
exit /b 1

:help
call :show_usage
exit /b 0

:set_host
set "HOST=%~2"
shift
shift
goto :parse_args

:set_port
set "PORT=%~2"
shift
shift
goto :parse_args

:set_api_key
set "API_KEY=%~2"
shift
shift
goto :parse_args

:set_log_level
set "LOG_LEVEL=%~2"
shift
shift
goto :parse_args

:set_models_filter
set "MODELS_FILTER=%~2"
shift
shift
goto :parse_args

:set_dev_mode
set "DEV_MODE=true"
shift
goto :parse_args

:set_daemon_mode
set "DAEMON_MODE=true"
shift
goto :parse_args

:set_env_file
set "ENV_FILE=%~2"
shift
shift
goto :parse_args

:set_check_deps
set "CHECK_DEPS=true"
shift
goto :parse_args

:set_dry_run
set "DRY_RUN=true"
shift
goto :parse_args

:args_done

REM Main execution starts here
call :parse_args %*

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Load environment file if specified
if not "%ENV_FILE%"=="" (
    if exist "%ENV_FILE%" (
        call :print_info "Loading environment from %ENV_FILE%"
        REM Note: Windows batch doesn't have a direct equivalent to source
        REM Users should manually set environment variables or use PowerShell
        call :print_warning "Environment file loading not fully supported in batch. Consider using PowerShell script."
    ) else (
        call :print_error "Environment file not found: %ENV_FILE%"
        exit /b 1
    )
)

REM Check dependencies if requested or always in dry-run mode
if "%CHECK_DEPS%"=="true" (
    call :check_dependencies
    if errorlevel 1 exit /b 1
    if "%DRY_RUN%"=="false" exit /b 0
)

if "%DRY_RUN%"=="true" (
    call :check_dependencies
    if errorlevel 1 exit /b 1
)

REM Validate environment
call :validate_environment
if errorlevel 1 exit /b 1

REM Build command arguments
set "CMD_ARGS="

if not "%API_KEY%"=="" (
    set "CMD_ARGS=!CMD_ARGS! --api-key "%API_KEY%""
)

if not "%HOST%"=="%DEFAULT_HOST%" (
    set "CMD_ARGS=!CMD_ARGS! --host "%HOST%""
)

if not "%PORT%"=="%DEFAULT_PORT%" (
    set "CMD_ARGS=!CMD_ARGS! --port "%PORT%""
)

if not "%LOG_LEVEL%"=="%DEFAULT_LOG_LEVEL%" (
    set "CMD_ARGS=!CMD_ARGS! --log-level "%LOG_LEVEL%""
)

if not "%MODELS_FILTER%"=="" (
    set "CMD_ARGS=!CMD_ARGS! --models-filter "%MODELS_FILTER%""
)

REM Determine which entry point to use
if "%DEV_MODE%"=="true" (
    set "ENTRY_POINT=ollama-proxy-dev"
    call :print_info "Starting in development mode..."
) else if "%DAEMON_MODE%"=="true" (
    set "ENTRY_POINT=ollama-proxy-daemon"
    call :print_info "Starting in daemon mode..."
) else (
    set "ENTRY_POINT=ollama-proxy"
    call :print_info "Starting server..."
)

REM Build final command
set "FINAL_CMD=uv run %ENTRY_POINT%!CMD_ARGS!"

REM Show what would be executed
call :print_info "Command: !FINAL_CMD!"
call :print_info "Host: %HOST%"
call :print_info "Port: %PORT%"
call :print_info "Log Level: %LOG_LEVEL%"

if "%DRY_RUN%"=="true" (
    call :print_info "Dry run mode - not executing command"
    exit /b 0
)

REM Execute the command
call :print_success "Starting ollama-proxy..."
%FINAL_CMD%
