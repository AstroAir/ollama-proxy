#!/usr/bin/env bash
# Cross-platform startup script for ollama-proxy server
# Works on Linux, macOS, and other Unix-like systems

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="11434"
DEFAULT_LOG_LEVEL="INFO"

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
Usage: $0 [OPTIONS]

Start the ollama-proxy server with various configuration options.

OPTIONS:
    -h, --help              Show this help message
    -H, --host HOST         Host to bind to (default: $DEFAULT_HOST)
    -p, --port PORT         Port to listen on (default: $DEFAULT_PORT)
    -k, --api-key KEY       OpenRouter API key
    -l, --log-level LEVEL   Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    -m, --models-filter FILE Path to model filter file
    -d, --dev               Run in development mode with auto-reload
    -D, --daemon            Run in daemon mode (background service)
    --env-file FILE         Load environment from specific file
    --check-deps            Check dependencies before starting
    --dry-run               Show what would be executed without running

ENVIRONMENT VARIABLES:
    OPENROUTER_API_KEY      OpenRouter API key (required)
    HOST                    Host to bind to
    PORT                    Port to listen on
    LOG_LEVEL               Logging level
    MODELS_FILTER_PATH      Path to model filter file

EXAMPLES:
    $0                                          # Start with defaults
    $0 --dev                                    # Start in development mode
    $0 --host 127.0.0.1 --port 8080           # Custom host and port
    $0 --api-key sk-or-... --daemon            # Start as daemon
    $0 --check-deps                            # Check dependencies only

EOF
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists uv; then
        missing_deps+=("uv")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_info "Please install the missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                python3)
                    echo "  - Python 3.12+: https://www.python.org/downloads/"
                    ;;
                uv)
                    echo "  - uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
                    ;;
            esac
        done
        return 1
    fi
    
    print_success "All dependencies are available"
    return 0
}

# Function to validate environment
validate_environment() {
    print_info "Validating environment..."
    
    # Check if we're in the project root
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        print_error "Not in ollama-proxy project root. Expected pyproject.toml file."
        return 1
    fi
    
    # Check if API key is set (if not provided via command line)
    if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${API_KEY:-}" ]; then
        print_warning "OPENROUTER_API_KEY not set. Make sure to provide it via --api-key or environment variable."
    fi
    
    print_success "Environment validation passed"
    return 0
}

# Parse command line arguments
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
LOG_LEVEL="$DEFAULT_LOG_LEVEL"
API_KEY=""
MODELS_FILTER=""
DEV_MODE=false
DAEMON_MODE=false
ENV_FILE=""
CHECK_DEPS=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -m|--models-filter)
            MODELS_FILTER="$2"
            shift 2
            ;;
        -d|--dev)
            DEV_MODE=true
            shift
            ;;
        -D|--daemon)
            DAEMON_MODE=true
            shift
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --check-deps)
            CHECK_DEPS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Load environment file if specified
if [ -n "$ENV_FILE" ]; then
    if [ -f "$ENV_FILE" ]; then
        print_info "Loading environment from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        print_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
fi

# Check dependencies if requested or always in dry-run mode
if [ "$CHECK_DEPS" = true ] || [ "$DRY_RUN" = true ]; then
    if ! check_dependencies; then
        exit 1
    fi
    
    if [ "$CHECK_DEPS" = true ] && [ "$DRY_RUN" = false ]; then
        exit 0
    fi
fi

# Validate environment
if ! validate_environment; then
    exit 1
fi

# Build command arguments
CMD_ARGS=()

if [ -n "$API_KEY" ]; then
    CMD_ARGS+=("--api-key" "$API_KEY")
fi

if [ "$HOST" != "$DEFAULT_HOST" ]; then
    CMD_ARGS+=("--host" "$HOST")
fi

if [ "$PORT" != "$DEFAULT_PORT" ]; then
    CMD_ARGS+=("--port" "$PORT")
fi

if [ "$LOG_LEVEL" != "$DEFAULT_LOG_LEVEL" ]; then
    CMD_ARGS+=("--log-level" "$LOG_LEVEL")
fi

if [ -n "$MODELS_FILTER" ]; then
    CMD_ARGS+=("--models-filter" "$MODELS_FILTER")
fi

# Determine which entry point to use
if [ "$DEV_MODE" = true ]; then
    ENTRY_POINT="ollama-proxy-dev"
    print_info "Starting in development mode..."
elif [ "$DAEMON_MODE" = true ]; then
    ENTRY_POINT="ollama-proxy-daemon"
    print_info "Starting in daemon mode..."
else
    ENTRY_POINT="ollama-proxy"
    print_info "Starting server..."
fi

# Build final command
FINAL_CMD=("uv" "run" "$ENTRY_POINT")
if [ ${#CMD_ARGS[@]} -gt 0 ]; then
    FINAL_CMD+=("${CMD_ARGS[@]}")
fi

# Show what would be executed
print_info "Command: ${FINAL_CMD[*]}"
print_info "Host: $HOST"
print_info "Port: $PORT"
print_info "Log Level: $LOG_LEVEL"

if [ "$DRY_RUN" = true ]; then
    print_info "Dry run mode - not executing command"
    exit 0
fi

# Execute the command
print_success "Starting ollama-proxy..."
exec "${FINAL_CMD[@]}"
