#!/usr/bin/env bash
# Build and deployment script for ollama-proxy
# Handles building, packaging, and deployment tasks

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

Build and deployment script for ollama-proxy.

COMMANDS:
    build           Build the package
    docker          Build Docker image
    docker-push     Build and push Docker image
    release         Create a release (build + tag)
    deploy          Deploy to production
    clean           Clean build artifacts
    check           Run pre-build checks
    help            Show this help message

OPTIONS:
    --version VERSION   Specify version for release
    --tag TAG          Docker tag (default: latest)
    --registry URL     Docker registry URL
    --push             Push Docker image after building
    --no-cache         Build Docker image without cache
    --platform ARCH    Target platform for Docker build
    --dry-run          Show what would be done without executing
    --force            Force operations (skip confirmations)

EXAMPLES:
    $0 build                           # Build the package
    $0 docker --tag v1.0.0            # Build Docker image with tag
    $0 docker-push --registry myregistry.com
    $0 release --version 1.0.0        # Create release
    $0 deploy --dry-run                # Show deployment plan
    $0 clean                           # Clean build artifacts

EOF
}

# Default values
COMMAND=""
VERSION=""
DOCKER_TAG="latest"
REGISTRY=""
PUSH=false
NO_CACHE=false
PLATFORM=""
DRY_RUN=false
FORCE=false

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            VERSION="$2"
            shift 2
            ;;
        --tag)
            DOCKER_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
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

print_info "Build and deployment script for ollama-proxy"
print_info "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in ollama-proxy project root. Expected pyproject.toml file."
    exit 1
fi

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

# Function to get current version from pyproject.toml
get_current_version() {
    grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'
}

# Function to run pre-build checks
run_checks() {
    print_info "Running pre-build checks..."
    
    # Check code formatting
    if ! uv run black --check src tests >/dev/null 2>&1; then
        print_error "Code formatting issues found. Run 'make format' to fix."
        return 1
    fi
    
    # Check imports
    if ! uv run isort --check-only src tests >/dev/null 2>&1; then
        print_error "Import sorting issues found. Run 'make format' to fix."
        return 1
    fi
    
    # Check linting
    if ! uv run flake8 src tests >/dev/null 2>&1; then
        print_error "Linting issues found. Run 'make lint' for details."
        return 1
    fi
    
    # Check types
    if ! uv run mypy src >/dev/null 2>&1; then
        print_error "Type checking issues found. Run 'make type-check' for details."
        return 1
    fi
    
    # Run tests
    if ! uv run pytest >/dev/null 2>&1; then
        print_error "Tests failed. Run 'make test' for details."
        return 1
    fi
    
    print_success "All pre-build checks passed"
    return 0
}

# Function to build package
build_package() {
    print_info "Building package..."
    
    # Clean previous builds
    run_command "rm -rf dist/ build/ *.egg-info/"
    
    # Build the package
    run_command "uv build"
    
    if [ "$DRY_RUN" = false ]; then
        print_success "Package built successfully"
        print_info "Built files:"
        ls -la dist/
    fi
}

# Function to build Docker image
build_docker() {
    print_info "Building Docker image..."
    
    local image_name="ollama-proxy"
    local full_tag="$image_name:$DOCKER_TAG"
    
    if [ -n "$REGISTRY" ]; then
        full_tag="$REGISTRY/$full_tag"
    fi
    
    local docker_cmd="docker build -t $full_tag"
    
    if [ "$NO_CACHE" = true ]; then
        docker_cmd="$docker_cmd --no-cache"
    fi
    
    if [ -n "$PLATFORM" ]; then
        docker_cmd="$docker_cmd --platform $PLATFORM"
    fi
    
    docker_cmd="$docker_cmd ."
    
    run_command "$docker_cmd"
    
    if [ "$DRY_RUN" = false ]; then
        print_success "Docker image built: $full_tag"
        
        if [ "$PUSH" = true ]; then
            print_info "Pushing Docker image..."
            run_command "docker push $full_tag"
            print_success "Docker image pushed: $full_tag"
        fi
    fi
}

# Function to create release
create_release() {
    if [ -z "$VERSION" ]; then
        print_error "Version is required for release. Use --version option."
        exit 1
    fi
    
    print_info "Creating release version $VERSION..."
    
    # Check if version already exists
    if git tag | grep -q "^v$VERSION$"; then
        if [ "$FORCE" = false ]; then
            print_error "Version $VERSION already exists. Use --force to override."
            exit 1
        else
            print_warning "Version $VERSION already exists. Forcing update."
        fi
    fi
    
    # Update version in pyproject.toml
    print_info "Updating version in pyproject.toml..."
    run_command "sed -i 's/^version = .*/version = \"$VERSION\"/' pyproject.toml"
    
    # Run checks
    if ! run_checks; then
        print_error "Pre-release checks failed"
        exit 1
    fi
    
    # Build package
    build_package
    
    # Commit version change
    run_command "git add pyproject.toml"
    run_command "git commit -m 'Bump version to $VERSION'"
    
    # Create tag
    run_command "git tag -a v$VERSION -m 'Release version $VERSION'"
    
    print_success "Release $VERSION created successfully"
    print_info "To push the release, run: git push origin main --tags"
}

# Function to clean build artifacts
clean_artifacts() {
    print_info "Cleaning build artifacts..."
    
    run_command "rm -rf build/"
    run_command "rm -rf dist/"
    run_command "rm -rf *.egg-info/"
    run_command "rm -rf .pytest_cache/"
    run_command "rm -rf .mypy_cache/"
    run_command "rm -rf htmlcov/"
    run_command "rm -f .coverage"
    run_command "find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true"
    run_command "find . -type f -name '*.pyc' -delete"
    
    print_success "Build artifacts cleaned"
}

# Function to deploy (placeholder)
deploy() {
    print_info "Deployment functionality..."
    print_warning "Deployment is not implemented yet."
    print_info "This would typically:"
    echo "  - Build and push Docker image"
    echo "  - Update deployment configuration"
    echo "  - Deploy to target environment"
    echo "  - Run health checks"
    echo "  - Rollback on failure"
}

# Execute command
case $COMMAND in
    build)
        if ! run_checks; then
            exit 1
        fi
        build_package
        ;;
    docker)
        build_docker
        ;;
    docker-push)
        PUSH=true
        build_docker
        ;;
    release)
        create_release
        ;;
    deploy)
        deploy
        ;;
    clean)
        clean_artifacts
        ;;
    check)
        run_checks
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
