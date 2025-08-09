#!/usr/bin/env bash
# Development environment setup script for ollama-proxy
# Sets up the complete development environment with all dependencies

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Set up the development environment for ollama-proxy.

OPTIONS:
    -h, --help          Show this help message
    --skip-uv           Skip uv installation
    --skip-deps         Skip dependency installation
    --skip-pre-commit   Skip pre-commit setup
    --skip-docs         Skip documentation setup
    --force             Force reinstallation of components

EXAMPLES:
    $0                  # Full setup
    $0 --skip-uv        # Skip uv installation
    $0 --force          # Force reinstall everything

EOF
}

# Parse command line arguments
SKIP_UV=false
SKIP_DEPS=false
SKIP_PRE_COMMIT=false
SKIP_DOCS=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        --skip-uv)
            SKIP_UV=true
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-pre-commit)
            SKIP_PRE_COMMIT=true
            shift
            ;;
        --skip-docs)
            SKIP_DOCS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
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

print_info "Setting up development environment for ollama-proxy"
print_info "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in ollama-proxy project root. Expected pyproject.toml file."
    exit 1
fi

# Step 1: Install uv if not present
if [ "$SKIP_UV" = false ]; then
    print_info "Checking for uv..."
    if ! command_exists uv || [ "$FORCE" = true ]; then
        print_info "Installing uv..."
        if command_exists curl; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
            # Source the shell configuration to make uv available
            if [ -f "$HOME/.cargo/env" ]; then
                source "$HOME/.cargo/env"
            fi
        else
            print_error "curl not found. Please install curl or uv manually."
            print_info "Visit: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        fi
    else
        print_success "uv is already installed"
    fi
else
    print_info "Skipping uv installation"
fi

# Step 2: Install Python dependencies
if [ "$SKIP_DEPS" = false ]; then
    print_info "Installing Python dependencies..."
    
    if [ "$FORCE" = true ]; then
        print_info "Force mode: removing existing virtual environment"
        rm -rf .venv
    fi
    
    # Install all dependencies including dev dependencies
    uv sync --all-extras --dev
    
    print_success "Dependencies installed successfully"
else
    print_info "Skipping dependency installation"
fi

# Step 3: Set up pre-commit hooks
if [ "$SKIP_PRE_COMMIT" = false ]; then
    print_info "Setting up pre-commit hooks..."
    
    if uv run pre-commit --version >/dev/null 2>&1; then
        uv run pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not available, skipping hook installation"
    fi
else
    print_info "Skipping pre-commit setup"
fi

# Step 4: Set up documentation
if [ "$SKIP_DOCS" = false ]; then
    print_info "Setting up documentation..."
    
    # Install documentation dependencies
    uv add --dev mkdocs mkdocs-material
    
    print_success "Documentation setup complete"
else
    print_info "Skipping documentation setup"
fi

# Step 5: Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_warning "Please edit .env file and set your OPENROUTER_API_KEY"
    else
        cat > .env << EOF
# Ollama Proxy Configuration
OPENROUTER_API_KEY=your_api_key_here
HOST=0.0.0.0
PORT=11434
LOG_LEVEL=INFO
ENVIRONMENT=development
EOF
        print_warning "Created basic .env file. Please set your OPENROUTER_API_KEY"
    fi
else
    print_info ".env file already exists"
fi

# Step 6: Run initial checks
print_info "Running initial checks..."

# Check code formatting
print_info "Checking code formatting..."
if uv run black --check src tests >/dev/null 2>&1; then
    print_success "Code formatting is correct"
else
    print_warning "Code formatting issues found. Run 'make format' to fix."
fi

# Check imports
print_info "Checking import sorting..."
if uv run isort --check-only src tests >/dev/null 2>&1; then
    print_success "Import sorting is correct"
else
    print_warning "Import sorting issues found. Run 'make format' to fix."
fi

# Check types
print_info "Checking types..."
if uv run mypy src >/dev/null 2>&1; then
    print_success "Type checking passed"
else
    print_warning "Type checking issues found. Run 'make type-check' for details."
fi

# Step 7: Show next steps
print_success "Development environment setup complete!"
echo
print_info "Next steps:"
echo "  1. Edit .env file and set your OPENROUTER_API_KEY"
echo "  2. Run 'make dev' to start the development server"
echo "  3. Run 'make test' to run the test suite"
echo "  4. Run 'make check-all' to run all quality checks"
echo "  5. Run 'make docs-serve' to view documentation"
echo
print_info "Available make targets:"
make help

print_success "Happy coding! 🚀"
