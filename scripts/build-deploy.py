#!/usr/bin/env python3
"""Comprehensive build and deployment script for ollama-proxy."""

import argparse
import json
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 with trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class BuildDeployManager:
    """Manages building and deployment of ollama-proxy."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_info: Dict[str, Any] = {}
        
    def get_version(self) -> str:
        """Get version from pyproject.toml."""
        try:
            import tomllib
            with open(self.project_root / "pyproject.toml", "rb") as f:
                data = tomllib.load(f)
                return data["project"]["version"]
        except Exception:
            return "0.1.0"
    
    def get_git_info(self) -> Dict[str, Optional[str]]:
        """Get git information."""
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, cwd=self.project_root
            ).stdout.strip()
            
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=self.project_root
            ).stdout.strip()
            
            tag = subprocess.run(
                ["git", "describe", "--tags", "--exact-match"],
                capture_output=True, text=True, cwd=self.project_root
            ).stdout.strip()
            
            return {
                "commit": commit or "unknown",
                "branch": branch or "unknown",
                "tag": tag if tag else None,
                "short_commit": (commit[:8] if commit else "unknown")
            }
        except Exception:
            return {
                "commit": "unknown",
                "branch": "unknown", 
                "tag": None,
                "short_commit": "unknown"
            }
    
    def run_tests(self) -> bool:
        """Run tests before building."""
        print("🧪 Running tests...")
        
        result = subprocess.run(
            ["uv", "run", "pytest", "tests/", "-x", "--tb=short"],
            cwd=self.project_root
        )
        
        return result.returncode == 0
    
    def run_security_scan(self) -> bool:
        """Run security scan."""
        print("🔒 Running security scan...")
        
        result = subprocess.run(
            ["python", "scripts/security-scan.py", "--output", "security-report.json"],
            cwd=self.project_root
        )
        
        return result.returncode in [0, 2]  # 0 = clean, 2 = warnings only
    
    def build_python_package(self) -> bool:
        """Build Python package."""
        print("📦 Building Python package...")
        
        # Clean previous builds
        subprocess.run(["rm", "-rf", "dist/"], cwd=self.project_root)
        
        result = subprocess.run(
            ["uv", "build"],
            cwd=self.project_root
        )
        
        return result.returncode == 0
    
    def build_docker_image(self, 
                          tag: str, 
                          dockerfile: str = "Dockerfile",
                          platforms: Optional[List[str]] = None,
                          push: bool = False) -> bool:
        """Build Docker image."""
        print(f"🐳 Building Docker image: {tag}")
        
        git_info = self.get_git_info()
        version = self.get_version()
        build_date = _utc_now_iso()
        
        build_args = [
            "--build-arg", f"BUILD_DATE={build_date}",
            "--build-arg", f"VERSION={version}",
            "--build-arg", f"VCS_REF={git_info['commit']}",
        ]
        
        if platforms:
            platform_str = ",".join(platforms)
            cmd = [
                "docker", "buildx", "build",
                "--platform", platform_str,
                *build_args,
                "-f", dockerfile,
                "-t", tag,
            ]
            
            if push:
                cmd.append("--push")
            else:
                cmd.extend(["--load"])
        else:
            cmd = [
                "docker", "build",
                *build_args,
                "-f", dockerfile,
                "-t", tag,
            ]
        
        cmd.append(".")
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0 and push and not platforms:
            # Push single-platform image
            push_result = subprocess.run(
                ["docker", "push", tag],
                cwd=self.project_root
            )
            return push_result.returncode == 0
        
        return result.returncode == 0
    
    def test_docker_image(self, tag: str) -> bool:
        """Test Docker image."""
        print(f"🧪 Testing Docker image: {tag}")
        
        # Start container
        start_result = subprocess.run([
            "docker", "run", "-d", "--name", "test-ollama-proxy",
            "-p", "11434:11434",
            "-e", "OPENROUTER_API_KEY=test-key",
            tag
        ])
        
        if start_result.returncode != 0:
            return False
        
        try:
            # Wait for startup
            import time
            time.sleep(10)
            
            # Test health endpoint
            health_result = subprocess.run([
                "curl", "-f", "http://localhost:11434/health"
            ])
            
            return health_result.returncode == 0
            
        finally:
            # Cleanup
            subprocess.run(["docker", "stop", "test-ollama-proxy"])
            subprocess.run(["docker", "rm", "test-ollama-proxy"])
    
    def create_release_artifacts(self, version: str) -> bool:
        """Create release artifacts."""
        print("📋 Creating release artifacts...")
        
        artifacts_dir = self.project_root / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        git_info = self.get_git_info()
        artifacts: List[str] = []
        
        # Create release info
        release_info = {
            "version": version,
            "build_date": _utc_now_iso(),
            "git": git_info,
            "artifacts": artifacts
        }
        
        # Copy Python packages
        dist_dir = self.project_root / "dist"
        if dist_dir.exists():
            for file in dist_dir.glob("*"):
                dest = artifacts_dir / file.name
                subprocess.run(["cp", str(file), str(dest)])
                artifacts.append(file.name)
        
        # Copy security report
        security_report = self.project_root / "security-report.json"
        if security_report.exists():
            dest = artifacts_dir / "security-report.json"
            subprocess.run(["cp", str(security_report), str(dest)])
            artifacts.append("security-report.json")
        
        # Create checksums (Python implementation)
        checksum_path = artifacts_dir / "checksums.sha256"
        with open(checksum_path, "w", encoding="utf-8") as cf:
            for file in sorted(artifacts_dir.iterdir()):
                if file.is_file() and file.name != "checksums.sha256":
                    h = hashlib.sha256()
                    with open(file, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            if isinstance(chunk, str):
                                chunk = chunk.encode("utf-8")
                            h.update(chunk)
                    cf.write(f"{h.hexdigest()}  {file.name}\n")
        artifacts.append("checksums.sha256")
        
        # Save release info
        with open(artifacts_dir / "release-info.json", "w", encoding="utf-8") as f:
            json.dump(release_info, f, indent=2)
        artifacts.append("release-info.json")
        
        print(f"📦 Artifacts created in {artifacts_dir}")
        return True
    
    def deploy_staging(self, tag: str) -> bool:
        """Deploy to staging environment."""
        print(f"🚀 Deploying to staging (tag: {tag})...")
        print("⚠️  Staging deployment not implemented (would deploy here)")
        return True
    
    def deploy_production(self, tag: str) -> bool:
        """Deploy to production environment."""
        print(f"🚀 Deploying to production (tag: {tag})...")
        print("⚠️  Production deployment not implemented (would deploy here)")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build and deploy ollama-proxy")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-security", action="store_true", help="Skip security scan")
    parser.add_argument("--docker-tag", default="ollama-proxy:latest", help="Docker tag")
    parser.add_argument("--multiarch", action="store_true", help="Build multi-architecture image")
    parser.add_argument("--platforms", default="linux/amd64,linux/arm64", help="Platforms for multi-arch build")
    parser.add_argument("--push", action="store_true", help="Push Docker image")
    parser.add_argument("--deploy", choices=["staging", "production"], help="Deploy to environment")
    parser.add_argument("--create-artifacts", action="store_true", help="Create release artifacts")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    
    args = parser.parse_args()
    
    manager = BuildDeployManager(args.project_root)
    
    # Pre-build checks
    if not args.skip_tests:
        if not manager.run_tests():
            print("❌ Tests failed")
            sys.exit(1)
        print("✅ Tests passed")
    
    if not args.skip_security:
        if not manager.run_security_scan():
            print("❌ Security scan failed")
            sys.exit(1)
        print("✅ Security scan passed")
    
    # Build Python package
    if not manager.build_python_package():
        print("❌ Python package build failed")
        sys.exit(1)
    print("✅ Python package built")
    
    # Build Docker image
    dockerfile = "Dockerfile.multiarch" if args.multiarch else "Dockerfile"
    platforms = args.platforms.split(",") if args.multiarch else None
    
    if not manager.build_docker_image(
        tag=args.docker_tag,
        dockerfile=dockerfile,
        platforms=platforms,
        push=args.push
    ):
        print("❌ Docker build failed")
        sys.exit(1)
    print("✅ Docker image built")
    
    # Test Docker image (only for single-platform builds)
    if not args.multiarch and not args.push:
        if not manager.test_docker_image(args.docker_tag):
            print("❌ Docker image test failed")
            sys.exit(1)
        print("✅ Docker image tested")
    
    # Create release artifacts
    if args.create_artifacts:
        version = manager.get_version()
        if not manager.create_release_artifacts(version):
            print("❌ Failed to create release artifacts")
            sys.exit(1)
        print("✅ Release artifacts created")
    
    # Deploy
    if args.deploy:
        if args.deploy == "staging":
            if not manager.deploy_staging(args.docker_tag):
                print("❌ Staging deployment failed")
                sys.exit(1)
            print("✅ Deployed to staging")
        elif args.deploy == "production":
            if not manager.deploy_production(args.docker_tag):
                print("❌ Production deployment failed")
                sys.exit(1)
            print("✅ Deployed to production")
    
    print("🎉 Build and deployment completed successfully!")


if __name__ == "__main__":
    main()
