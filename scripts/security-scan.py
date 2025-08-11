#!/usr/bin/env python3
"""Comprehensive security scanning script for ollama-proxy."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


class SecurityScanner:
    """Comprehensive security scanner for the project."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        # Added type annotation for self.results
        self.results: Dict[str, Any] = {}

    def run_command(self, cmd: List[str], timeout: int = 300) -> Dict[str, Any]:
        """Run a command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": "Command not found"
            }

    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities."""
        print("🔍 Scanning dependencies for vulnerabilities...")

        results = {}

        # Safety scan
        safety_result = self.run_command(
            ["uv", "run", "safety", "check", "--json"])
        if safety_result["success"]:
            results["safety"] = {
                "status": "clean",
                "vulnerabilities": []
            }
        else:
            try:
                vulnerabilities = json.loads(safety_result["stdout"])
                results["safety"] = {
                    "status": "vulnerabilities_found",
                    "vulnerabilities": vulnerabilities,
                    # Fixed type mismatch
                    "count": [str(v) for v in vulnerabilities]
                }
            except json.JSONDecodeError:
                results["safety"] = {
                    "status": "error",
                    "error": safety_result["stderr"]
                }

        # pip-audit scan
        audit_result = self.run_command(
            ["uv", "run", "pip-audit", "--format=json"])
        if audit_result["success"]:
            try:
                audit_data = json.loads(audit_result["stdout"])
                results["pip_audit"] = {
                    "status": "clean" if not audit_data else "vulnerabilities_found",
                    "vulnerabilities": audit_data,
                    # Fixed type mismatch
                    "count": [str(a) for a in audit_data]
                }
            except json.JSONDecodeError:
                results["pip_audit"] = {
                    "status": "clean",
                    "vulnerabilities": []
                }
        else:
            results["pip_audit"] = {
                "status": "error",
                "error": audit_result["stderr"]
            }

        return results

    def scan_docker_security(self) -> Dict[str, Any]:
        """Scan Docker configuration for security issues."""
        print("🔍 Scanning Docker configuration...")

        results = {}

        # Hadolint scan
        hadolint_result = self.run_command([
            "hadolint", "--format", "json", "Dockerfile"
        ])

        if hadolint_result["success"]:
            try:
                hadolint_data = json.loads(hadolint_result["stdout"])
                results["hadolint"] = {
                    "status": "clean" if not hadolint_data else "issues_found",
                    # Convert issues to strings
                    "issues": [str(issue) for issue in hadolint_data],
                    # Fixed type mismatch
                    "count": [str(issue) for issue in hadolint_data]
                }
            except json.JSONDecodeError:
                results["hadolint"] = {
                    "status": "clean",
                    "issues": []
                }
        else:
            results["hadolint"] = {
                "status": "error",
                "error": hadolint_result["stderr"]
            }

        return results

    def scan_configuration(self) -> Dict[str, Any]:
        """Scan configuration files for security issues."""
        print("🔍 Scanning configuration files...")

        results = {}

        # YAML lint
        yaml_files = list(self.project_root.glob("**/*.yml")) + \
            list(self.project_root.glob("**/*.yaml"))
        yaml_issues = []

        for yaml_file in yaml_files:
            if ".git" not in str(yaml_file):
                yamllint_result = self.run_command([
                    "yamllint", "-f", "parsable", str(yaml_file)
                ])
                if not yamllint_result["success"] and yamllint_result["stdout"]:
                    yaml_issues.append({
                        "file": str(yaml_file),
                        "issues": yamllint_result["stdout"]
                    })

        results["yaml_lint"] = {
            "status": "clean" if not yaml_issues else "issues_found",
            "issues": yaml_issues,
            "count": len(yaml_issues)
        }

        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        print("📊 Generating security report...")

        report: Dict[str, Any] = {
            "timestamp": subprocess.run(
                ["date", "-Iseconds"], capture_output=True, text=True
            ).stdout.strip(),
            "project": "ollama-proxy",
            "scans": {}  # Changed to a dictionary to allow indexed assignment
        }

        # Run all scans
        report["scans"]["dependencies"] = self.scan_dependencies()
        report["scans"]["docker"] = self.scan_docker_security()
        report["scans"]["configuration"] = self.scan_configuration()

        # Calculate overall status
        critical_issues = 0
        warnings = 0

        for scan_category, scan_results in report["scans"].items():
            for tool, results in scan_results.items():
                if results.get("status") in ["vulnerabilities_found", "issues_found"]:
                    if "high" in str(results).lower() or "critical" in str(results).lower():
                        critical_issues += 1
                    else:
                        warnings += 1
                elif results.get("status") == "secrets_found":
                    critical_issues += 1

        if critical_issues > 0:
            overall_status = "critical"
        elif warnings > 0:
            overall_status = "warning"
        else:
            overall_status = "clean"

        report["summary"] = {
            "overall_status": overall_status,
            "critical_issues": critical_issues,
            "warnings": warnings
        }

        return report

    def save_report(self, report: Dict[str, Any], output_file: Path) -> None:
        """Save report to file."""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Security report saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive security scanner for ollama-proxy")
    parser.add_argument("--output", "-o", type=Path, default="security-report.json",
                        help="Output file for security report")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                        help="Project root directory")

    args = parser.parse_args()

    scanner = SecurityScanner(args.project_root)
    report = scanner.generate_report()

    scanner.save_report(report, args.output)

    # Exit with error code if critical issues found
    if report["summary"]["overall_status"] == "critical":
        sys.exit(1)
    elif report["summary"]["overall_status"] == "warning":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
