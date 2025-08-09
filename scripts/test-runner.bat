@echo off
REM Test runner script for Windows
REM Provides comprehensive testing options with reporting and coverage

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
set "VERBOSE=false"
set "QUIET=false"
set "COVERAGE=false"
set "FAST=false"
set "SLOW_ONLY=false"
set "UNIT_ONLY=false"
set "INTEGRATION_ONLY=false"
set "STOP_ON_FAIL=false"
set "RERUN_FAILED=false"
set "HTML_REPORT=false"
set "XML_REPORT=false"
set "TEST_PATHS="

REM Function to show usage
:show_usage
echo Usage: %~nx0 [OPTIONS] [TEST_PATHS...]
echo.
echo Test runner for ollama-proxy with comprehensive options.
echo.
echo OPTIONS:
echo     -h, --help              Show this help message
echo     -v, --verbose           Verbose test output
echo     -q, --quiet             Quiet test output
echo     -c, --coverage          Run with coverage reporting
echo     -f, --fast              Skip slow tests
echo     -s, --slow              Run only slow tests
echo     -u, --unit              Run only unit tests
echo     -i, --integration       Run only integration tests
echo     -x, --stop-on-fail      Stop on first failure
echo     -r, --rerun-failed      Rerun only failed tests from last run
echo     --html-report           Generate HTML coverage report
echo     --xml-report            Generate XML coverage report
echo.
echo EXAMPLES:
echo     %~nx0                           # Run all tests
echo     %~nx0 --coverage --html-report  # Run with HTML coverage report
echo     %~nx0 --unit --fast             # Run fast unit tests only
echo     %~nx0 --integration --verbose   # Run integration tests with verbose output
echo     %~nx0 tests\test_api.py         # Run specific test file
echo.
goto :eof

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :args_done

if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="-v" goto :set_verbose
if "%~1"=="--verbose" goto :set_verbose
if "%~1"=="-q" goto :set_quiet
if "%~1"=="--quiet" goto :set_quiet
if "%~1"=="-c" goto :set_coverage
if "%~1"=="--coverage" goto :set_coverage
if "%~1"=="-f" goto :set_fast
if "%~1"=="--fast" goto :set_fast
if "%~1"=="-s" goto :set_slow
if "%~1"=="--slow" goto :set_slow
if "%~1"=="-u" goto :set_unit
if "%~1"=="--unit" goto :set_unit
if "%~1"=="-i" goto :set_integration
if "%~1"=="--integration" goto :set_integration
if "%~1"=="-x" goto :set_stop_on_fail
if "%~1"=="--stop-on-fail" goto :set_stop_on_fail
if "%~1"=="-r" goto :set_rerun_failed
if "%~1"=="--rerun-failed" goto :set_rerun_failed
if "%~1"=="--html-report" goto :set_html_report
if "%~1"=="--xml-report" goto :set_xml_report

REM If it doesn't match any option, treat as test path
if "%TEST_PATHS%"=="" (
    set "TEST_PATHS=%~1"
) else (
    set "TEST_PATHS=%TEST_PATHS% %~1"
)
shift
goto :parse_args

:help
call :show_usage
exit /b 0

:set_verbose
set "VERBOSE=true"
shift
goto :parse_args

:set_quiet
set "QUIET=true"
shift
goto :parse_args

:set_coverage
set "COVERAGE=true"
shift
goto :parse_args

:set_fast
set "FAST=true"
shift
goto :parse_args

:set_slow
set "SLOW_ONLY=true"
shift
goto :parse_args

:set_unit
set "UNIT_ONLY=true"
shift
goto :parse_args

:set_integration
set "INTEGRATION_ONLY=true"
shift
goto :parse_args

:set_stop_on_fail
set "STOP_ON_FAIL=true"
shift
goto :parse_args

:set_rerun_failed
set "RERUN_FAILED=true"
shift
goto :parse_args

:set_html_report
set "HTML_REPORT=true"
set "COVERAGE=true"
shift
goto :parse_args

:set_xml_report
set "XML_REPORT=true"
set "COVERAGE=true"
shift
goto :parse_args

:args_done

echo %INFO% Running tests for ollama-proxy
echo %INFO% Project root: %PROJECT_ROOT%

REM Change to project root
cd /d "%PROJECT_ROOT%"

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo %ERROR% Not in ollama-proxy project root. Expected pyproject.toml file.
    exit /b 1
)

REM Build pytest command
set "PYTEST_CMD=uv run pytest"

REM Add verbosity options
if "%VERBOSE%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% -v"
) else if "%QUIET%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% -q"
)

REM Add coverage options
if "%COVERAGE%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% --cov=src --cov-report=term-missing"
    
    if "%HTML_REPORT%"=="true" (
        set "PYTEST_CMD=%PYTEST_CMD% --cov-report=html"
    )
    
    if "%XML_REPORT%"=="true" (
        set "PYTEST_CMD=%PYTEST_CMD% --cov-report=xml"
    )
)

REM Add test selection markers
if "%FAST%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% -m \"not slow\""
) else if "%SLOW_ONLY%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% -m slow"
) else if "%UNIT_ONLY%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% -m unit"
) else if "%INTEGRATION_ONLY%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% -m integration"
)

REM Add other options
if "%STOP_ON_FAIL%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% -x"
)

if "%RERUN_FAILED%"=="true" (
    set "PYTEST_CMD=%PYTEST_CMD% --lf"
)

REM Add test paths
if not "%TEST_PATHS%"=="" (
    set "PYTEST_CMD=%PYTEST_CMD% %TEST_PATHS%"
)

REM Run the tests
echo %INFO% Running command: %PYTEST_CMD%

%PYTEST_CMD%
set "TEST_RESULT=%ERRORLEVEL%"

if "%TEST_RESULT%"=="0" (
    echo.
    echo %SUCCESS% Tests passed!
    
    REM Show coverage report location if generated
    if "%HTML_REPORT%"=="true" (
        echo %INFO% HTML coverage report: htmlcov\index.html
    )
    
    if "%XML_REPORT%"=="true" (
        echo %INFO% XML coverage report: coverage.xml
    )
) else (
    echo.
    echo %ERROR% Tests failed!
)

exit /b %TEST_RESULT%

REM Main execution starts here
call :parse_args %*
