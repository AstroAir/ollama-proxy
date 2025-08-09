"""Main entry point for the ollama-proxy application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--host", type=str, help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="Port to listen on (default: 11434)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")

    # Configuration options
    parser.add_argument("--api-key", type=str, help="OpenRouter API key")
    parser.add_argument("--models-filter", type=str,
                        help="Path to model filter file")
    parser.add_argument("--log-level", type=str,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
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


if __name__ == "__main__":
    main()
