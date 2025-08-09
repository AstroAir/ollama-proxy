#!/usr/bin/env bash
# Advanced test runner script for ollama-proxy
# Provides comprehensive testing options with reporting and coverage

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [TEST_PATHS...]

Advanced test runner for ollama-proxy with comprehensive options.

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Verbose test output
    -q, --quiet             Quiet test output
    -c, --coverage          Run with coverage reporting
    -f, --fast              Skip slow tests
    -s, --slow              Run only slow tests
    -u, --unit              Run only unit tests
    -i, --integration       Run only integration tests
    -p, --performance       Run only performance tests
    -x, --stop-on-fail      Stop on first failure
    -r, --rerun-failed      Rerun only failed tests from last run
    --parallel N            Run tests in parallel (N workers)
    --html-report           Generate HTML coverage report
    --xml-report            Generate XML coverage report
    --json-report           Generate JSON test report
    --benchmark             Run benchmarks
    --profile               Profile test execution
    --watch                 Watch for changes and rerun tests
    --clean                 Clean test artifacts before running

TEST SELECTION:
    TEST_PATHS              Specific test files or directories to run

EXAMPLES:
    $0                                  # Run all tests
    $0 --coverage --html-report         # Run with HTML coverage report
    $0 --unit --fast                    # Run fast unit tests only
    $0 --integration --verbose          # Run integration tests with verbose output
    $0 tests/test_api.py                # Run specific test file
    $0 --rerun-failed                   # Rerun only failed tests
    $0 --watch                          # Watch mode for development

EOF
}

# Default values
VERBOSE=false
QUIET=false
COVERAGE=false
FAST=false
SLOW_ONLY=false
UNIT_ONLY=false
INTEGRATION_ONLY=false
PERFORMANCE_ONLY=false
STOP_ON_FAIL=false
RERUN_FAILED=false
PARALLEL=""
HTML_REPORT=false
XML_REPORT=false
JSON_REPORT=false
BENCHMARK=false
PROFILE=false
WATCH=false
CLEAN=false
TEST_PATHS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -f|--fast)
            FAST=true
            shift
            ;;
        -s|--slow)
            SLOW_ONLY=true
            shift
            ;;
        -u|--unit)
            UNIT_ONLY=true
            shift
            ;;
        -i|--integration)
            INTEGRATION_ONLY=true
            shift
            ;;
        -p|--performance)
            PERFORMANCE_ONLY=true
            shift
            ;;
        -x|--stop-on-fail)
            STOP_ON_FAIL=true
            shift
            ;;
        -r|--rerun-failed)
            RERUN_FAILED=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --html-report)
            HTML_REPORT=true
            COVERAGE=true
            shift
            ;;
        --xml-report)
            XML_REPORT=true
            COVERAGE=true
            shift
            ;;
        --json-report)
            JSON_REPORT=true
            shift
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --profile)
            PROFILE=true
            shift
            ;;
        --watch)
            WATCH=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            TEST_PATHS+=("$1")
            shift
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info "Running tests for ollama-proxy"
print_info "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in ollama-proxy project root. Expected pyproject.toml file."
    exit 1
fi

# Clean test artifacts if requested
if [ "$CLEAN" = true ]; then
    print_info "Cleaning test artifacts..."
    rm -rf .pytest_cache/
    rm -rf htmlcov/
    rm -f .coverage
    rm -f coverage.xml
    rm -f test-results.json
    print_success "Test artifacts cleaned"
fi

# Build pytest command
PYTEST_CMD=("uv" "run" "pytest")

# Add verbosity options
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD+=("-v")
elif [ "$QUIET" = true ]; then
    PYTEST_CMD+=("-q")
fi

# Add coverage options
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD+=("--cov=src")
    PYTEST_CMD+=("--cov-report=term-missing")
    
    if [ "$HTML_REPORT" = true ]; then
        PYTEST_CMD+=("--cov-report=html")
    fi
    
    if [ "$XML_REPORT" = true ]; then
        PYTEST_CMD+=("--cov-report=xml")
    fi
fi

# Add test selection markers
if [ "$FAST" = true ]; then
    PYTEST_CMD+=("-m" "not slow")
elif [ "$SLOW_ONLY" = true ]; then
    PYTEST_CMD+=("-m" "slow")
elif [ "$UNIT_ONLY" = true ]; then
    PYTEST_CMD+=("-m" "unit")
elif [ "$INTEGRATION_ONLY" = true ]; then
    PYTEST_CMD+=("-m" "integration")
elif [ "$PERFORMANCE_ONLY" = true ]; then
    PYTEST_CMD+=("-m" "performance")
fi

# Add other options
if [ "$STOP_ON_FAIL" = true ]; then
    PYTEST_CMD+=("-x")
fi

if [ "$RERUN_FAILED" = true ]; then
    PYTEST_CMD+=("--lf")
fi

if [ -n "$PARALLEL" ]; then
    PYTEST_CMD+=("-n" "$PARALLEL")
fi

if [ "$JSON_REPORT" = true ]; then
    PYTEST_CMD+=("--json-report" "--json-report-file=test-results.json")
fi

if [ "$BENCHMARK" = true ]; then
    PYTEST_CMD+=("--benchmark-only")
fi

if [ "$PROFILE" = true ]; then
    PYTEST_CMD+=("--profile")
fi

# Add test paths
if [ ${#TEST_PATHS[@]} -gt 0 ]; then
    PYTEST_CMD+=("${TEST_PATHS[@]}")
fi

# Function to run tests
run_tests() {
    print_info "Running command: ${PYTEST_CMD[*]}"
    
    # Run the tests
    if "${PYTEST_CMD[@]}"; then
        print_success "Tests passed!"
        
        # Show coverage report location if generated
        if [ "$HTML_REPORT" = true ]; then
            print_info "HTML coverage report: htmlcov/index.html"
        fi
        
        if [ "$XML_REPORT" = true ]; then
            print_info "XML coverage report: coverage.xml"
        fi
        
        if [ "$JSON_REPORT" = true ]; then
            print_info "JSON test report: test-results.json"
        fi
        
        return 0
    else
        print_error "Tests failed!"
        return 1
    fi
}

# Watch mode
if [ "$WATCH" = true ]; then
    print_info "Starting watch mode..."
    print_info "Press Ctrl+C to stop"
    
    # Check if watchdog is available
    if ! uv run python -c "import watchdog" >/dev/null 2>&1; then
        print_warning "watchdog not installed. Installing..."
        uv add --dev watchdog
    fi
    
    # Simple watch implementation using find and sleep
    LAST_CHANGE=0
    
    while true; do
        # Find the most recent modification time
        CURRENT_CHANGE=$(find src tests -name "*.py" -type f -exec stat -c %Y {} \; 2>/dev/null | sort -n | tail -1)
        
        if [ "$CURRENT_CHANGE" -gt "$LAST_CHANGE" ]; then
            print_info "Changes detected, running tests..."
            run_tests
            LAST_CHANGE=$CURRENT_CHANGE
        fi
        
        sleep 2
    done
else
    # Run tests once
    run_tests
fi
