"""Main entry point for the ollama-proxy application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

from .app import create_app
from .config import get_settings


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ollama-to-OpenRouter Proxy")

    # Server options
    parser.add_argument("--host", type=str, help="Host to bind to")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload")

    # Configuration options
    parser.add_argument("--api-key", type=str, help="OpenRouter API key")
    parser.add_argument("--models-filter", type=str,
                        help="Model filter file path")
    parser.add_argument("--log-level", type=str, help="Logging level")

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Get settings (will load from environment and .env file)
    try:
        settings = get_settings()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Override settings with command line arguments
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

    # Validate required settings
    if not settings.openrouter_api_key:
        print(
            "Error: OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the FastAPI app
    app = create_app()

    # Run the server
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
