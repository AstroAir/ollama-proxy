"""Main entry point for the ollama-proxy application."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

import httpx
import uvicorn

from .app import create_app
from .config import get_settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the ollama-proxy application.

    This function sets up and parses command line arguments that can override
    environment variables and configuration file settings. Arguments include
    server configuration, API credentials, and operational settings.

    Returns:
        argparse.Namespace: Parsed command line arguments.

    Example:
        >>> args = parse_args()
        >>> if args.host:
        >>>     print(f"Host override: {args.host}")
    """
    parser = argparse.ArgumentParser(
        description="Ollama-to-OpenRouter Proxy - Translate Ollama API calls to OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --host 0.0.0.0 --port 8080
  %(prog)s --api-key sk-or-... --log-level DEBUG
  %(prog)s --models-filter models.txt --reload
        """
    )

    # Server options
    parser.add_argument("--host", type=str,
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int,
                        help="Port to listen on (default: 11434)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")

    # Configuration options
    parser.add_argument("--api-key", type=str, help="OpenRouter API key")
    parser.add_argument("--models-filter", type=str,
                        help="Path to model filter file")
    parser.add_argument("--log-level", type=str,
                        choices=["DEBUG", "INFO",
                                 "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")

    return parser.parse_args()


def main() -> None:
    """Main entry point for the ollama-proxy application.

    This function orchestrates the complete application startup process:
    1. Parses command line arguments
    2. Loads configuration from environment variables and .env files
    3. Applies command line overrides to configuration
    4. Validates required settings (especially API key)
    5. Creates and starts the FastAPI application server

    The function handles configuration errors gracefully and provides
    clear error messages for missing required settings.

    Raises:
        SystemExit: If configuration is invalid or required settings are missing.

    Example:
        This function is typically called from the command line:
        $ ollama-proxy --host 0.0.0.0 --port 8080 --api-key sk-or-...
    """
    args = parse_args()

    # Load configuration from environment and .env file
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply command line argument overrides to settings
    if args.api_key:
        settings.openrouter_api_key = args.api_key
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.models_filter:
        settings.models_filter_path = args.models_filter
    if args.log_level:
        settings.log_level = args.log_level.upper()
    if args.reload:
        settings.reload = True

    # Validate that required settings are present
    if not settings.openrouter_api_key:
        print(
            "Error: OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the FastAPI application instance
    app = create_app()

    # Start the ASGI server with the configured settings
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


def dev_main() -> None:
    """Development mode entry point with auto-reload and debug settings."""
    # Override sys.argv to add development flags
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], "--reload", "--log-level", "DEBUG"]

    try:
        main()
    finally:
        sys.argv = original_argv


def daemon_main() -> None:
    """Daemon mode entry point for background service operation."""
    args = parse_args()

    # Load configuration
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply command line overrides
    if args.api_key:
        settings.openrouter_api_key = args.api_key
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.models_filter:
        settings.models_filter_path = args.models_filter
    if args.log_level:
        settings.log_level = args.log_level.upper()

    # Validate required settings
    if not settings.openrouter_api_key:
        print("Error: OpenRouter API key is required", file=sys.stderr)
        sys.exit(1)

    # Create app and run in daemon mode (no reload, production settings)
    app = create_app()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
        access_log=False,  # Reduce logging for daemon mode
    )


def admin_main() -> None:
    """Administrative interface entry point."""
    parser = argparse.ArgumentParser(
        description="Ollama Proxy Administration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show server status")
    status_parser.add_argument(
        "--host", default="localhost", help="Server host")
    status_parser.add_argument(
        "--port", type=int, default=11434, help="Server port")

    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument("--format", choices=["json", "yaml", "table"],
                               default="table", help="Output format")

    # Models command
    models_parser = subparsers.add_parser(
        "models", help="List available models")
    models_parser.add_argument(
        "--host", default="localhost", help="Server host")
    models_parser.add_argument(
        "--port", type=int, default=11434, help="Server port")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "status":
        _admin_status(args.host, args.port)
    elif args.command == "config":
        _admin_config(args.format)
    elif args.command == "models":
        _admin_models(args.host, args.port)


def _admin_status(host: str, port: int) -> None:
    """Show server status."""
    try:
        response = httpx.get(f"http://{host}:{port}/health", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy")
            print(f"Status: {data.get('status', 'unknown')}")
            print(f"Uptime: {data.get('uptime_seconds', 0):.1f} seconds")
            print(f"Requests: {data.get('request_count', 0)}")
            print(f"Errors: {data.get('error_count', 0)}")
        else:
            print(f"❌ Server returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")


def _admin_config(format_type: str) -> None:
    """Show configuration."""
    try:
        settings = get_settings()
        config_dict = {
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "environment": settings.environment.value,
            "openrouter_base_url": settings.openrouter_base_url,
            "models_filter_path": settings.models_filter_path,
            "debug": settings.debug,
            "reload": settings.reload,
        }

        if format_type == "json":
            print(json.dumps(config_dict, indent=2))
        elif format_type == "yaml":
            for key, value in config_dict.items():
                print(f"{key}: {value}")
        else:  # table
            print("Configuration:")
            print("-" * 40)
            for key, value in config_dict.items():
                print(f"{key:20}: {value}")
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)


def _admin_models(host: str, port: int) -> None:
    """List available models."""
    try:
        response = httpx.get(f"http://{host}:{port}/api/tags", timeout=30.0)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"Available models ({len(models)}):")
            print("-" * 40)
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0)
                modified = model.get("modified_at", "unknown")
                print(f"{name:30} {size:>10} bytes  {modified}")
        else:
            print(f"❌ Server returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")


def health_check() -> None:
    """Standalone health check entry point."""
    parser = argparse.ArgumentParser(
        description="Health check for ollama-proxy")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=11434, help="Server port")
    parser.add_argument("--timeout", type=float,
                        default=10.0, help="Request timeout")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON format")

    args = parser.parse_args()

    try:
        response = httpx.get(
            f"http://{args.host}:{args.port}/health",
            timeout=args.timeout
        )

        if args.json:
            print(response.text)
        else:
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                print(f"Status: {status}")
                if status == "healthy":
                    sys.exit(0)
                else:
                    sys.exit(1)
            else:
                print(f"Health check failed: HTTP {response.status_code}")
                sys.exit(1)

    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "error": str(e)}))
        else:
            print(f"Health check failed: {e}")
        sys.exit(1)


def config_main() -> None:
    """Configuration management entry point."""
    parser = argparse.ArgumentParser(
        description="Configuration management for ollama-proxy")
    parser.add_argument("--validate", action="store_true",
                        help="Validate configuration")
    parser.add_argument("--show", action="store_true",
                        help="Show current configuration")
    parser.add_argument("--format", choices=["json", "yaml", "env"],
                        default="yaml", help="Output format")

    args = parser.parse_args()

    try:
        settings = get_settings()

        if args.validate:
            print("✅ Configuration is valid")
            return

        if args.show:
            config_dict = {
                "host": settings.host,
                "port": settings.port,
                "log_level": settings.log_level,
                "environment": settings.environment.value,
                "openrouter_base_url": settings.openrouter_base_url,
                "models_filter_path": settings.models_filter_path,
                "debug": settings.debug,
                "reload": settings.reload,
                "max_concurrent_requests": settings.max_concurrent_requests,
                "openrouter_timeout": settings.openrouter_timeout,
            }

            if args.format == "json":
                print(json.dumps(config_dict, indent=2))
            elif args.format == "env":
                for key, value in config_dict.items():
                    env_key = key.upper().replace("OPENROUTER_", "OPENROUTER_")
                    print(f"{env_key}={value}")
            else:  # yaml
                for key, value in config_dict.items():
                    print(f"{key}: {value}")
        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ Configuration error: {e}", file=sys.stderr)
        sys.exit(1)


def benchmark_main() -> None:
    """Benchmarking entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark ollama-proxy performance")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=11434, help="Server port")
    parser.add_argument("--requests", type=int, default=100,
                        help="Number of requests")
    parser.add_argument("--concurrency", type=int,
                        default=10, help="Concurrent requests")
    parser.add_argument("--model", default="llama2", help="Model to test")

    args = parser.parse_args()

    print(f"Benchmarking {args.host}:{args.port}")
    print(f"Requests: {args.requests}, Concurrency: {args.concurrency}")
    print(f"Model: {args.model}")
    print("-" * 50)

    asyncio.run(_run_benchmark(args))


async def _run_benchmark(args) -> None:
    """Run the actual benchmark."""
    base_url = f"http://{args.host}:{args.port}"

    # Test data
    test_payload = {
        "model": args.model,
        "prompt": "Hello, how are you?",
        "stream": False
    }

    start_time = time.time()
    successful_requests = 0
    failed_requests = 0
    response_times = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def make_request():
            nonlocal successful_requests, failed_requests
            async with semaphore:
                request_start = time.time()
                try:
                    response = await client.post(
                        f"{base_url}/api/generate",
                        json=test_payload
                    )
                    request_time = time.time() - request_start
                    response_times.append(request_time)

                    if response.status_code == 200:
                        successful_requests += 1
                    else:
                        failed_requests += 1

                except Exception:
                    failed_requests += 1

        # Run all requests
        tasks = [make_request() for _ in range(args.requests)]
        await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # Calculate statistics
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        requests_per_second = successful_requests / total_time
    else:
        avg_response_time = min_response_time = max_response_time = 0
        requests_per_second = 0

    # Print results
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Requests per second: {requests_per_second:.2f}")
    print(f"Average response time: {avg_response_time:.3f}s")
    print(f"Min response time: {min_response_time:.3f}s")
    print(f"Max response time: {max_response_time:.3f}s")


def test_main() -> None:
    """Testing entry point with various test options."""
    parser = argparse.ArgumentParser(description="Run tests for ollama-proxy")
    parser.add_argument("--coverage", action="store_true",
                        help="Run with coverage")
    parser.add_argument("--verbose", "-v",
                        action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--integration", action="store_true",
                        help="Run integration tests only")
    parser.add_argument("--unit", action="store_true",
                        help="Run unit tests only")
    parser.add_argument("path", nargs="*", help="Specific test paths")

    args = parser.parse_args()

    # Build pytest command
    cmd = ["uv", "run", "pytest"]

    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html",
                   "--cov-report=term-missing"])

    if args.verbose:
        cmd.append("-v")

    if args.fast:
        cmd.extend(["-m", "not slow"])

    if args.integration:
        cmd.extend(["-m", "integration"])
    elif args.unit:
        cmd.extend(["-m", "unit"])

    if args.path:
        cmd.extend(args.path)

    # Run the command
    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print("Error: uv not found. Please install uv first.", file=sys.stderr)
        sys.exit(1)


def lint_main() -> None:
    """Linting entry point."""
    parser = argparse.ArgumentParser(
        description="Run linting for ollama-proxy")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix issues where possible")
    parser.add_argument("--check-only", action="store_true",
                        help="Check only, don't fix")

    args = parser.parse_args()

    commands = []

    if args.fix and not args.check_only:
        # Run formatters first
        commands.extend([
            ["uv", "run", "black", "src", "tests"],
            ["uv", "run", "isort", "src", "tests"],
        ])

    # Always run linters
    commands.extend([
        ["uv", "run", "flake8", "src", "tests"],
        ["uv", "run", "mypy", "src"],
    ])

    if args.check_only or not args.fix:
        commands.extend([
            ["uv", "run", "black", "--check", "src", "tests"],
            ["uv", "run", "isort", "--check-only", "src", "tests"],
        ])

    # Run all commands
    failed = False
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                failed = True
                print(f"❌ Command failed: {' '.join(cmd)}")
            else:
                print(f"✅ Command succeeded: {' '.join(cmd)}")
        except FileNotFoundError:
            print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
            failed = True

    if failed:
        print("\n❌ Some linting checks failed")
        sys.exit(1)
    else:
        print("\n✅ All linting checks passed")


def format_main() -> None:
    """Code formatting entry point."""
    parser = argparse.ArgumentParser(
        description="Format code for ollama-proxy")
    parser.add_argument("--check", action="store_true",
                        help="Check formatting only")
    parser.add_argument("path", nargs="*",
                        default=["src", "tests"], help="Paths to format")

    args = parser.parse_args()

    commands = []

    if args.check:
        commands.extend([
            ["uv", "run", "black", "--check"] + args.path,
            ["uv", "run", "isort", "--check-only"] + args.path,
        ])
    else:
        commands.extend([
            ["uv", "run", "black"] + args.path,
            ["uv", "run", "isort"] + args.path,
        ])

    # Run all commands
    failed = False
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                failed = True
                print(f"❌ Command failed: {' '.join(cmd)}")
            else:
                print(f"✅ Command succeeded: {' '.join(cmd)}")
        except FileNotFoundError:
            print(f"Error: Command not found: {cmd[0]}", file=sys.stderr)
            failed = True

    if failed:
        sys.exit(1)


def cli_main() -> None:
    """Unified CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Ollama Proxy - Unified CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available subcommands:
  server      Start the proxy server (default)
  dev         Start in development mode
  daemon      Start in daemon mode
  admin       Administrative interface
  health      Health check
  config      Configuration management
  benchmark   Performance benchmarking
  test        Run tests
  lint        Code linting
  format      Code formatting

Examples:
  ollama-proxy-cli server --host 0.0.0.0 --port 8080
  ollama-proxy-cli dev
  ollama-proxy-cli admin status
  ollama-proxy-cli health --json
  ollama-proxy-cli config --show
        """
    )

    subparsers = parser.add_subparsers(
        dest="subcommand", help="Available subcommands")

    # Server subcommand (default behavior)
    server_parser = subparsers.add_parser(
        "server", help="Start the proxy server")
    server_parser.add_argument("--host", type=str, help="Host to bind to")
    server_parser.add_argument("--port", type=int, help="Port to listen on")
    server_parser.add_argument(
        "--api-key", type=str, help="OpenRouter API key")
    server_parser.add_argument(
        "--models-filter", type=str, help="Path to model filter file")
    server_parser.add_argument("--log-level", type=str,
                               choices=["DEBUG", "INFO",
                                        "WARNING", "ERROR", "CRITICAL"],
                               help="Logging level")
    server_parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload")

    # Dev subcommand
    dev_parser = subparsers.add_parser("dev", help="Start in development mode")
    dev_parser.add_argument("--host", type=str, help="Host to bind to")
    dev_parser.add_argument("--port", type=int, help="Port to listen on")
    dev_parser.add_argument("--api-key", type=str, help="OpenRouter API key")

    # Daemon subcommand
    daemon_parser = subparsers.add_parser(
        "daemon", help="Start in daemon mode")
    daemon_parser.add_argument("--host", type=str, help="Host to bind to")
    daemon_parser.add_argument("--port", type=int, help="Port to listen on")
    daemon_parser.add_argument(
        "--api-key", type=str, help="OpenRouter API key")

    # Admin subcommand
    admin_parser = subparsers.add_parser(
        "admin", help="Administrative interface")
    admin_subparsers = admin_parser.add_subparsers(
        dest="admin_command", help="Admin commands")

    status_parser = admin_subparsers.add_parser(
        "status", help="Show server status")
    status_parser.add_argument(
        "--host", default="localhost", help="Server host")
    status_parser.add_argument(
        "--port", type=int, default=11434, help="Server port")

    config_admin_parser = admin_subparsers.add_parser(
        "config", help="Show configuration")
    config_admin_parser.add_argument("--format", choices=["json", "yaml", "table"],
                                     default="table", help="Output format")

    models_parser = admin_subparsers.add_parser(
        "models", help="List available models")
    models_parser.add_argument(
        "--host", default="localhost", help="Server host")
    models_parser.add_argument(
        "--port", type=int, default=11434, help="Server port")

    # Health subcommand
    health_parser = subparsers.add_parser("health", help="Health check")
    health_parser.add_argument(
        "--host", default="localhost", help="Server host")
    health_parser.add_argument(
        "--port", type=int, default=11434, help="Server port")
    health_parser.add_argument(
        "--timeout", type=float, default=10.0, help="Request timeout")
    health_parser.add_argument(
        "--json", action="store_true", help="Output JSON format")

    # Config subcommand
    config_parser = subparsers.add_parser(
        "config", help="Configuration management")
    config_parser.add_argument(
        "--validate", action="store_true", help="Validate configuration")
    config_parser.add_argument(
        "--show", action="store_true", help="Show current configuration")
    config_parser.add_argument("--format", choices=["json", "yaml", "env"],
                               default="yaml", help="Output format")

    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Performance benchmarking")
    benchmark_parser.add_argument(
        "--host", default="localhost", help="Server host")
    benchmark_parser.add_argument(
        "--port", type=int, default=11434, help="Server port")
    benchmark_parser.add_argument(
        "--requests", type=int, default=100, help="Number of requests")
    benchmark_parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrent requests")
    benchmark_parser.add_argument(
        "--model", default="llama2", help="Model to test")

    # Test subcommand
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage")
    test_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output")
    test_parser.add_argument(
        "--fast", action="store_true", help="Skip slow tests")
    test_parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only")
    test_parser.add_argument(
        "--unit", action="store_true", help="Run unit tests only")
    test_parser.add_argument("path", nargs="*", help="Specific test paths")

    # Lint subcommand
    lint_parser = subparsers.add_parser("lint", help="Code linting")
    lint_parser.add_argument("--fix", action="store_true",
                             help="Auto-fix issues where possible")
    lint_parser.add_argument(
        "--check-only", action="store_true", help="Check only, don't fix")

    # Format subcommand
    format_parser = subparsers.add_parser("format", help="Code formatting")
    format_parser.add_argument(
        "--check", action="store_true", help="Check formatting only")
    format_parser.add_argument(
        "path", nargs="*", default=["src", "tests"], help="Paths to format")

    args = parser.parse_args()

    # If no subcommand is provided, default to server
    if not args.subcommand:
        args.subcommand = "server"

    # Route to appropriate function based on subcommand
    if args.subcommand == "server":
        # Override sys.argv for the main function
        sys.argv = ["ollama-proxy"]
        if args.host:
            sys.argv.extend(["--host", args.host])
        if args.port:
            sys.argv.extend(["--port", str(args.port)])
        if args.api_key:
            sys.argv.extend(["--api-key", args.api_key])
        if args.models_filter:
            sys.argv.extend(["--models-filter", args.models_filter])
        if args.log_level:
            sys.argv.extend(["--log-level", args.log_level])
        if args.reload:
            sys.argv.append("--reload")
        main()
    elif args.subcommand == "dev":
        # Override sys.argv for dev_main
        sys.argv = ["ollama-proxy-dev"]
        if args.host:
            sys.argv.extend(["--host", args.host])
        if args.port:
            sys.argv.extend(["--port", str(args.port)])
        if args.api_key:
            sys.argv.extend(["--api-key", args.api_key])
        dev_main()
    elif args.subcommand == "daemon":
        # Override sys.argv for daemon_main
        sys.argv = ["ollama-proxy-daemon"]
        if args.host:
            sys.argv.extend(["--host", args.host])
        if args.port:
            sys.argv.extend(["--port", str(args.port)])
        if args.api_key:
            sys.argv.extend(["--api-key", args.api_key])
        daemon_main()
    elif args.subcommand == "admin":
        # Handle admin subcommands
        if args.admin_command == "status":
            _admin_status(args.host, args.port)
        elif args.admin_command == "config":
            _admin_config(args.format)
        elif args.admin_command == "models":
            _admin_models(args.host, args.port)
        else:
            admin_parser.print_help()
    elif args.subcommand == "health":
        # Override sys.argv for health_check
        sys.argv = ["ollama-proxy-health", "--host", args.host, "--port", str(args.port),
                    "--timeout", str(args.timeout)]
        if args.json:
            sys.argv.append("--json")
        health_check()
    elif args.subcommand == "config":
        # Override sys.argv for config_main
        sys.argv = ["ollama-proxy-config"]
        if args.validate:
            sys.argv.append("--validate")
        if args.show:
            sys.argv.append("--show")
        sys.argv.extend(["--format", args.format])
        config_main()
    elif args.subcommand == "benchmark":
        # Override sys.argv for benchmark_main
        sys.argv = ["ollama-proxy-benchmark", "--host", args.host, "--port", str(args.port),
                    "--requests", str(args.requests), "--concurrency", str(args.concurrency),
                    "--model", args.model]
        benchmark_main()
    elif args.subcommand == "test":
        # Override sys.argv for test_main
        sys.argv = ["ollama-proxy-test"]
        if args.coverage:
            sys.argv.append("--coverage")
        if args.verbose:
            sys.argv.append("--verbose")
        if args.fast:
            sys.argv.append("--fast")
        if args.integration:
            sys.argv.append("--integration")
        if args.unit:
            sys.argv.append("--unit")
        if args.path:
            sys.argv.extend(args.path)
        test_main()
    elif args.subcommand == "lint":
        # Override sys.argv for lint_main
        sys.argv = ["ollama-proxy-lint"]
        if args.fix:
            sys.argv.append("--fix")
        if args.check_only:
            sys.argv.append("--check-only")
        lint_main()
    elif args.subcommand == "format":
        # Override sys.argv for format_main
        sys.argv = ["ollama-proxy-format"]
        if args.check:
            sys.argv.append("--check")
        sys.argv.extend(args.path)
        format_main()


if __name__ == "__main__":
    main()
