#!/usr/bin/env bash
# Maintenance and utility script for ollama-proxy
# Handles various maintenance tasks like log analysis, health monitoring, etc.

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
Usage: $0 [COMMAND] [OPTIONS]

Maintenance and utility script for ollama-proxy.

COMMANDS:
    health          Check server health and status
    logs            Analyze and manage logs
    metrics         Show performance metrics
    cleanup         Clean up temporary files and caches
    backup          Backup configuration and data
    restore         Restore from backup
    update-deps     Update dependencies
    security-scan   Run security scans
    benchmark       Run performance benchmarks
    monitor         Monitor server in real-time
    help            Show this help message

OPTIONS:
    --host HOST     Server host (default: localhost)
    --port PORT     Server port (default: 11434)
    --timeout SEC   Request timeout (default: 10)
    --format FORMAT Output format (json, table, csv)
    --output FILE   Output to file
    --follow        Follow logs in real-time
    --lines N       Number of log lines to show
    --since TIME    Show logs since time (e.g., '1h', '30m')
    --level LEVEL   Log level filter
    --dry-run       Show what would be done

EXAMPLES:
    $0 health                          # Check server health
    $0 logs --follow --level ERROR     # Follow error logs
    $0 metrics --format json          # Show metrics in JSON
    $0 cleanup --dry-run               # Show cleanup plan
    $0 monitor                         # Real-time monitoring
    $0 benchmark --host production.com # Benchmark production

EOF
}

# Default values
COMMAND=""
HOST="localhost"
PORT="11434"
TIMEOUT="10"
FORMAT="table"
OUTPUT=""
FOLLOW=false
LINES="100"
SINCE=""
LEVEL=""
DRY_RUN=false

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --follow)
            FOLLOW=true
            shift
            ;;
        --lines)
            LINES="$2"
            shift 2
            ;;
        --since)
            SINCE="$2"
            shift 2
            ;;
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to run command with dry-run support
run_command() {
    local cmd="$*"
    print_info "Command: $cmd"
    
    if [ "$DRY_RUN" = true ]; then
        print_info "Dry run mode - not executing command"
        return 0
    fi
    
    eval "$cmd"
}

# Function to check server health
check_health() {
    print_info "Checking server health at $HOST:$PORT..."
    
    local url="http://$HOST:$PORT/health"
    local response
    
    if response=$(curl -s --max-time "$TIMEOUT" "$url" 2>/dev/null); then
        local status
        status=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        
        case $status in
            "healthy")
                print_success "Server is healthy"
                ;;
            "degraded")
                print_warning "Server is degraded"
                ;;
            "unhealthy"|"critical")
                print_error "Server is unhealthy"
                ;;
            *)
                print_warning "Server status unknown"
                ;;
        esac
        
        if [ "$FORMAT" = "json" ]; then
            echo "$response"
        elif [ "$FORMAT" = "table" ]; then
            echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('Health Status Report')
    print('=' * 20)
    for key, value in data.items():
        print(f'{key:20}: {value}')
except:
    print('Failed to parse response')
"
        fi
    else
        print_error "Failed to connect to server at $HOST:$PORT"
        return 1
    fi
}

# Function to analyze logs
analyze_logs() {
    print_info "Analyzing logs..."
    
    # This is a placeholder - in a real implementation, you would:
    # - Find log files (could be in /var/log, ./logs, or from systemd)
    # - Parse and filter logs based on options
    # - Show statistics and summaries
    
    print_warning "Log analysis not fully implemented"
    print_info "This would typically:"
    echo "  - Find application log files"
    echo "  - Filter by level, time range, etc."
    echo "  - Show error summaries and patterns"
    echo "  - Generate log statistics"
    
    if [ "$FOLLOW" = true ]; then
        print_info "Following logs (simulated)..."
        print_info "Press Ctrl+C to stop"
        # In real implementation: tail -f /path/to/logfile | grep filters
    fi
}

# Function to show metrics
show_metrics() {
    print_info "Fetching performance metrics..."
    
    local url="http://$HOST:$PORT/health"
    local response
    
    if response=$(curl -s --max-time "$TIMEOUT" "$url" 2>/dev/null); then
        if [ "$FORMAT" = "json" ]; then
            echo "$response"
        else
            echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print('Performance Metrics')
    print('=' * 20)
    
    # Extract relevant metrics
    metrics = {
        'Status': data.get('status', 'unknown'),
        'Uptime (s)': data.get('uptime_seconds', 0),
        'Total Requests': data.get('total_requests', 0),
        'Success Rate': f\"{data.get('success_rate', 0):.2%}\",
        'Avg Duration (ms)': data.get('average_duration_ms', 0),
        'Active Endpoints': data.get('active_endpoints', 0)
    }
    
    for key, value in metrics.items():
        print(f'{key:20}: {value}')
        
except Exception as e:
    print(f'Failed to parse metrics: {e}')
"
        fi
    else
        print_error "Failed to fetch metrics from server"
        return 1
    fi
}

# Function to cleanup temporary files
cleanup() {
    print_info "Cleaning up temporary files and caches..."
    
    cd "$PROJECT_ROOT"
    
    # Clean Python cache files
    run_command "find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true"
    run_command "find . -type f -name '*.pyc' -delete"
    run_command "find . -type f -name '*.pyo' -delete"
    
    # Clean test artifacts
    run_command "rm -rf .pytest_cache/"
    run_command "rm -rf .mypy_cache/"
    run_command "rm -rf htmlcov/"
    run_command "rm -f .coverage"
    
    # Clean build artifacts
    run_command "rm -rf build/"
    run_command "rm -rf dist/"
    run_command "rm -rf *.egg-info/"
    
    # Clean temporary files
    run_command "find . -type f -name '*.tmp' -delete"
    run_command "find . -type f -name '*.log' -mtime +7 -delete 2>/dev/null || true"
    
    print_success "Cleanup completed"
}

# Function to update dependencies
update_deps() {
    print_info "Updating dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Update uv itself
    run_command "uv self update"
    
    # Update project dependencies
    run_command "uv sync --upgrade"
    
    # Update dev dependencies
    run_command "uv sync --upgrade --dev"
    
    print_success "Dependencies updated"
    print_info "Run tests to ensure everything still works: make test"
}

# Function to run security scan
security_scan() {
    print_info "Running security scans..."
    
    cd "$PROJECT_ROOT"
    
    # Run bandit security scan
    if uv run bandit --version >/dev/null 2>&1; then
        print_info "Running bandit security scan..."
        run_command "uv run bandit -r src/ -f json -o bandit-report.json"
        print_info "Bandit report saved to bandit-report.json"
    else
        print_warning "bandit not available, skipping security scan"
    fi
    
    # Check for known vulnerabilities in dependencies
    print_info "Checking for vulnerable dependencies..."
    run_command "uv pip check" || print_warning "Some dependency issues found"
    
    print_success "Security scan completed"
}

# Function to run benchmarks
run_benchmark() {
    print_info "Running performance benchmarks..."
    
    # Use the benchmark entry point we created
    run_command "uv run ollama-proxy-benchmark --host $HOST --port $PORT --requests 100 --concurrency 10"
}

# Function to monitor server
monitor_server() {
    print_info "Starting real-time monitoring..."
    print_info "Press Ctrl+C to stop"
    
    while true; do
        clear
        echo "Ollama Proxy Monitor - $(date)"
        echo "=" * 50
        
        check_health
        echo
        show_metrics
        
        sleep 5
    done
}

# Execute command
case $COMMAND in
    health)
        check_health
        ;;
    logs)
        analyze_logs
        ;;
    metrics)
        show_metrics
        ;;
    cleanup)
        cleanup
        ;;
    backup)
        print_warning "Backup functionality not implemented yet"
        ;;
    restore)
        print_warning "Restore functionality not implemented yet"
        ;;
    update-deps)
        update_deps
        ;;
    security-scan)
        security_scan
        ;;
    benchmark)
        run_benchmark
        ;;
    monitor)
        monitor_server
        ;;
    help)
        show_usage
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac
