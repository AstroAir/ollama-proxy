#!/usr/bin/env python3
"""
Cross-platform launcher script for ollama-proxy.
This script works on all platforms and provides a unified interface.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check for Python
    try:
        result = subprocess.run([sys.executable, "--version"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            missing_deps.append("python")
    except FileNotFoundError:
        missing_deps.append("python")
    
    # Check for uv
    try:
        result = subprocess.run(["uv", "--version"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            missing_deps.append("uv")
    except FileNotFoundError:
        missing_deps.append("uv")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install the missing dependencies:")
        for dep in missing_deps:
            if dep == "python":
                print("  - Python 3.12+: https://www.python.org/downloads/")
            elif dep == "uv":
                print("  - uv: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    print("✅ All dependencies are available")
    return True


def validate_environment(project_root: Path) -> bool:
    """Validate the environment."""
    print("🔍 Validating environment...")
    
    # Check if we're in the project root
    if not (project_root / "pyproject.toml").exists():
        print("❌ Not in ollama-proxy project root. Expected pyproject.toml file.")
        return False
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not set. Make sure to provide it via --api-key or environment variable.")
    
    print("✅ Environment validation passed")
    return True


def load_env_file(env_file: Path) -> bool:
    """Load environment variables from a file."""
    if not env_file.exists():
        print(f"❌ Environment file not found: {env_file}")
        return False
    
    print(f"📁 Loading environment from {env_file}")
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    
    return True


def build_command(args) -> List[str]:
    """Build the command to execute."""
    cmd = ["uv", "run"]
    
    # Determine entry point
    if args.dev:
        cmd.append("ollama-proxy-dev")
        print("🚀 Starting in development mode...")
    elif args.daemon:
        cmd.append("ollama-proxy-daemon")
        print("🚀 Starting in daemon mode...")
    else:
        cmd.append("ollama-proxy")
        print("🚀 Starting server...")
    
    # Add arguments
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    
    if args.host != "0.0.0.0":
        cmd.extend(["--host", args.host])
    
    if args.port != 11434:
        cmd.extend(["--port", str(args.port)])
    
    if args.log_level != "INFO":
        cmd.extend(["--log-level", args.log_level])
    
    if args.models_filter:
        cmd.extend(["--models-filter", args.models_filter])
    
    return cmd


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-platform launcher for ollama-proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                                    # Start with defaults
  python launcher.py --dev                              # Start in development mode
  python launcher.py --host 127.0.0.1 --port 8080     # Custom host and port
  python launcher.py --api-key sk-or-... --daemon      # Start as daemon
  python launcher.py --check-deps                       # Check dependencies only

Environment Variables:
  OPENROUTER_API_KEY      OpenRouter API key (required)
  HOST                    Host to bind to
  PORT                    Port to listen on
  LOG_LEVEL               Logging level
  MODELS_FILTER_PATH      Path to model filter file
        """
    )
    
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=11434,
                       help="Port to listen on (default: 11434)")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Log level")
    parser.add_argument("--models-filter", help="Path to model filter file")
    parser.add_argument("--dev", action="store_true",
                       help="Run in development mode with auto-reload")
    parser.add_argument("--daemon", action="store_true",
                       help="Run in daemon mode (background service)")
    parser.add_argument("--env-file", type=Path,
                       help="Load environment from specific file")
    parser.add_argument("--check-deps", action="store_true",
                       help="Check dependencies before starting")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    # Get project root and change to it
    project_root = get_project_root()
    os.chdir(project_root)
    
    print(f"🏠 Project root: {project_root}")
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Load environment file if specified
    if args.env_file:
        if not load_env_file(args.env_file):
            sys.exit(1)
    
    # Check dependencies if requested or always in dry-run mode
    if args.check_deps or args.dry_run:
        if not check_dependencies():
            sys.exit(1)
        
        if args.check_deps and not args.dry_run:
            sys.exit(0)
    
    # Validate environment
    if not validate_environment(project_root):
        sys.exit(1)
    
    # Build command
    cmd = build_command(args)
    
    # Show what would be executed
    print(f"📋 Command: {' '.join(cmd)}")
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"📊 Log Level: {args.log_level}")
    
    if args.dry_run:
        print("🔍 Dry run mode - not executing command")
        return
    
    # Execute the command
    print("✅ Starting ollama-proxy...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        print("Make sure uv is installed and in your PATH")
        sys.exit(1)


if __name__ == "__main__":
    main()
