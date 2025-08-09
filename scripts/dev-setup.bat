@echo off
REM Development environment setup script for Windows
REM Sets up the complete development environment with all dependencies

setlocal enabledelayedexpansion

REM Colors are limited in batch, so we'll use simple prefixes
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM Script directory and project root
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Default values
set "SKIP_UV=false"
set "SKIP_DEPS=false"
set "SKIP_PRE_COMMIT=false"
set "FORCE=false"

REM Function to show usage
:show_usage
echo Usage: %~nx0 [OPTIONS]
echo.
echo Set up the development environment for ollama-proxy.
echo.
echo OPTIONS:
echo     --skip-uv           Skip uv installation
echo     --skip-deps         Skip dependency installation  
echo     --skip-pre-commit   Skip pre-commit setup
echo     --force             Force reinstallation of components
echo     -h, --help          Show this help message
echo.
echo EXAMPLES:
echo     %~nx0                  # Full setup
echo     %~nx0 --skip-uv        # Skip uv installation
echo     %~nx0 --force          # Force reinstall everything
echo.
goto :eof

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done

if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="--skip-uv" (
    set "SKIP_UV=true"
    shift
    goto :parse_args
)
if "%~1"=="--skip-deps" (
    set "SKIP_DEPS=true"
    shift
    goto :parse_args
)
if "%~1"=="--skip-pre-commit" (
    set "SKIP_PRE_COMMIT=true"
    shift
    goto :parse_args
)
if "%~1"=="--force" (
    set "FORCE=true"
    shift
    goto :parse_args
)

echo %ERROR% Unknown option: %~1
call :show_usage
exit /b 1

:help
call :show_usage
exit /b 0

:args_done

echo %INFO% Setting up development environment for ollama-proxy
echo %INFO% Project root: %PROJECT_ROOT%

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo %ERROR% Not in ollama-proxy project root. Expected pyproject.toml file.
    exit /b 1
)

REM Step 1: Check for Python
echo %INFO% Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        echo %ERROR% Python not found. Please install Python 3.12+ from https://python.org
        exit /b 1
    ) else (
        echo %SUCCESS% Python3 found
        set "PYTHON_CMD=python3"
    )
) else (
    echo %SUCCESS% Python found
    set "PYTHON_CMD=python"
)

REM Step 2: Install uv if not present
if "%SKIP_UV%"=="false" (
    echo %INFO% Checking for uv...
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo %INFO% Installing uv...
        REM For Windows, we need to use PowerShell to install uv
        powershell -Command "& {Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression}"
        if errorlevel 1 (
            echo %ERROR% Failed to install uv. Please install manually from https://docs.astral.sh/uv/getting-started/installation/
            exit /b 1
        )
        REM Refresh PATH
        call refreshenv >nul 2>&1 || echo %WARNING% Could not refresh environment. You may need to restart your terminal.
    ) else (
        echo %SUCCESS% uv is already installed
    )
) else (
    echo %INFO% Skipping uv installation
)

REM Step 3: Install Python dependencies
if "%SKIP_DEPS%"=="false" (
    echo %INFO% Installing Python dependencies...
    
    if "%FORCE%"=="true" (
        echo %INFO% Force mode: removing existing virtual environment
        rmdir /s /q .venv 2>nul
    )
    
    REM Install all dependencies including dev dependencies
    uv sync --all-extras --dev
    if errorlevel 1 (
        echo %ERROR% Failed to install dependencies
        exit /b 1
    )
    
    echo %SUCCESS% Dependencies installed successfully
) else (
    echo %INFO% Skipping dependency installation
)

REM Step 4: Set up pre-commit hooks
if "%SKIP_PRE_COMMIT%"=="false" (
    echo %INFO% Setting up pre-commit hooks...
    
    uv run pre-commit --version >nul 2>&1
    if not errorlevel 1 (
        uv run pre-commit install
        if not errorlevel 1 (
            echo %SUCCESS% Pre-commit hooks installed
        ) else (
            echo %WARNING% Failed to install pre-commit hooks
        )
    ) else (
        echo %WARNING% pre-commit not available, skipping hook installation
    )
) else (
    echo %INFO% Skipping pre-commit setup
)

REM Step 5: Create .env file if it doesn't exist
if not exist ".env" (
    echo %INFO% Creating .env file from template...
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo %WARNING% Please edit .env file and set your OPENROUTER_API_KEY
    ) else (
        (
            echo # Ollama Proxy Configuration
            echo OPENROUTER_API_KEY=your_api_key_here
            echo HOST=0.0.0.0
            echo PORT=11434
            echo LOG_LEVEL=INFO
            echo ENVIRONMENT=development
        ) > .env
        echo %WARNING% Created basic .env file. Please set your OPENROUTER_API_KEY
    )
) else (
    echo %INFO% .env file already exists
)

REM Step 6: Run initial checks
echo %INFO% Running initial checks...

REM Check code formatting
echo %INFO% Checking code formatting...
uv run black --check src tests >nul 2>&1
if not errorlevel 1 (
    echo %SUCCESS% Code formatting is correct
) else (
    echo %WARNING% Code formatting issues found. Run 'uv run black src tests' to fix.
)

REM Check imports
echo %INFO% Checking import sorting...
uv run isort --check-only src tests >nul 2>&1
if not errorlevel 1 (
    echo %SUCCESS% Import sorting is correct
) else (
    echo %WARNING% Import sorting issues found. Run 'uv run isort src tests' to fix.
)

REM Check types
echo %INFO% Checking types...
uv run mypy src >nul 2>&1
if not errorlevel 1 (
    echo %SUCCESS% Type checking passed
) else (
    echo %WARNING% Type checking issues found. Run 'uv run mypy src' for details.
)

REM Step 7: Show next steps
echo.
echo %SUCCESS% Development environment setup complete!
echo.
echo %INFO% Next steps:
echo   1. Edit .env file and set your OPENROUTER_API_KEY
echo   2. Run 'scripts\start-server.bat --dev' to start the development server
echo   3. Run 'uv run pytest' to run the test suite
echo   4. Run 'uv run ollama-proxy-lint' to check code quality
echo.
echo %SUCCESS% Happy coding! 🚀

REM Main execution starts here
call :parse_args %*
