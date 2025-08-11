#!/usr/bin/env python3
"""Monitoring setup and health check script for ollama-proxy."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import requests


class MonitoringManager:
    """Manages monitoring setup and health checks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.monitoring_dir = project_root / "monitoring"
        
    def setup_monitoring_stack(self, environment: str = "development") -> bool:
        """Set up monitoring stack for the specified environment."""
        print(f"🔧 Setting up monitoring stack for {environment}...")
        
        compose_file = f"docker-compose.{environment}.yml" if environment != "development" else "docker-compose.yml"
        
        # Start monitoring services
        cmd = [
            "docker-compose",
            "-f", compose_file,
            "--profile", "monitoring",
            "up", "-d"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ Monitoring stack started successfully")
            return True
        else:
            print("❌ Failed to start monitoring stack")
            return False
    
    def wait_for_services(self, services: List[str], timeout: int = 300) -> bool:
        """Wait for services to become healthy."""
        print("⏳ Waiting for services to become healthy...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_healthy = True
            
            for service in services:
                if not self.check_service_health(service):
                    all_healthy = False
                    break
            
            if all_healthy:
                print("✅ All services are healthy")
                return True
            
            time.sleep(10)
        
        print("❌ Timeout waiting for services to become healthy")
        return False
    
    def check_service_health(self, service: str) -> bool:
        """Check if a service is healthy."""
        health_endpoints = {
            "ollama-proxy": "http://localhost:11434/health",
            "prometheus": "http://localhost:9090/-/healthy",
            "grafana": "http://localhost:3000/api/health",
            "loki": "http://localhost:3100/ready",
            "redis": "redis://localhost:6379"
        }
        
        endpoint = health_endpoints.get(service)
        if not endpoint:
            return False
        
        try:
            if endpoint.startswith("redis://"):
                # Check Redis with ping
                result = subprocess.run(
                    ["redis-cli", "ping"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0 and "PONG" in result.stdout.decode()
            else:
                # HTTP health check
                response = requests.get(endpoint, timeout=5)
                return response.status_code == 200
        except Exception:
            return False
    
    def run_health_checks(self) -> Dict[str, bool]:
        """Run comprehensive health checks."""
        print("🏥 Running health checks...")
        
        services = ["ollama-proxy", "prometheus", "grafana", "loki", "redis"]
        results = {}
        
        for service in services:
            results[service] = self.check_service_health(service)
            status = "✅" if results[service] else "❌"
            print(f"  {status} {service}")
        
        return results
    
    def collect_metrics(self) -> Dict[str, Any]:  # Fixed `any` to `Any`
        """Collect metrics from various sources."""
        print("📊 Collecting metrics...")
        
        metrics = {}
        
        # Ollama Proxy metrics
        try:
            response = requests.get("http://localhost:11434/metrics", timeout=10)
            if response.status_code == 200:
                metrics["ollama_proxy"] = response.json()
        except Exception as e:
            metrics["ollama_proxy"] = {"error": str(e)}
        
        # Prometheus metrics
        try:
            # Query some basic metrics
            queries = [
                "up",
                "http_requests_total",
                "http_request_duration_seconds",
                "process_resident_memory_bytes",
                "process_cpu_seconds_total"
            ]
            
            prom_metrics = {}
            for query in queries:
                response = requests.get(
                    f"http://localhost:9090/api/v1/query",
                    params={"query": query},
                    timeout=10
                )
                if response.status_code == 200:
                    prom_metrics[query] = response.json()
            
            metrics["prometheus"] = prom_metrics
        except Exception as e:
            metrics["prometheus"] = {"error": str(e)}
        
        return metrics
    
    def generate_health_report(self) -> Dict[str, Any]:  # Fixed `any` to `Any`
        """Generate comprehensive health report."""
        print("📋 Generating health report...")
        
        health_checks = self.run_health_checks()
        metrics = self.collect_metrics()
        
        # Calculate overall health
        healthy_services = sum(1 for healthy in health_checks.values() if healthy)
        total_services = len(health_checks)
        health_percentage = (healthy_services / total_services) * 100
        
        overall_status = "healthy" if health_percentage == 100 else \
                        "degraded" if health_percentage >= 50 else "unhealthy"
        
        report = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "health_percentage": health_percentage,
            "service_health": health_checks,
            "metrics": metrics,
            "summary": {
                "healthy_services": healthy_services,
                "total_services": total_services,
                "unhealthy_services": [
                    service for service, healthy in health_checks.items() 
                    if not healthy
                ]
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: Path) -> None:  # Fixed `any` to `Any`
        """Save health report to file."""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Health report saved to {output_file}")
    
    def setup_alerts(self) -> bool:
        """Set up alerting rules."""
        print("🚨 Setting up alerting rules...")
        
        # Reload Prometheus configuration
        try:
            response = requests.post("http://localhost:9090/-/reload", timeout=10)
            if response.status_code == 200:
                print("✅ Prometheus configuration reloaded")
                return True
        except Exception as e:
            print(f"❌ Failed to reload Prometheus: {e}")
        
        return False  # Ensure a return value for all code paths
    
    def test_alerts(self) -> bool:
        """Test alerting system."""
        print("🧪 Testing alerting system...")
        
        # This would typically involve:
        # - Triggering test alerts
        # - Verifying alert delivery
        # - Testing escalation paths
        
        print("⚠️  Alert testing not implemented (would test alerts here)")
        return True
    
    def cleanup_monitoring(self) -> bool:
        """Clean up monitoring resources."""
        print("🧹 Cleaning up monitoring resources...")
        
        cmd = [
            "docker-compose",
            "--profile", "monitoring",
            "down", "-v"
        ]
        
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitoring setup and health checks")
    parser.add_argument("--setup", action="store_true", help="Set up monitoring stack")
    parser.add_argument("--environment", default="development", help="Environment (development, staging, production)")
    parser.add_argument("--health-check", action="store_true", help="Run health checks")
    parser.add_argument("--collect-metrics", action="store_true", help="Collect metrics")
    parser.add_argument("--generate-report", action="store_true", help="Generate health report")
    parser.add_argument("--output", type=Path, default="health-report.json", help="Output file for reports")
    parser.add_argument("--setup-alerts", action="store_true", help="Set up alerting")
    parser.add_argument("--test-alerts", action="store_true", help="Test alerting system")
    parser.add_argument("--cleanup", action="store_true", help="Clean up monitoring resources")
    parser.add_argument("--wait-timeout", type=int, default=300, help="Timeout for waiting for services")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    
    args = parser.parse_args()
    
    manager = MonitoringManager(args.project_root)
    
    if args.setup:
        if not manager.setup_monitoring_stack(args.environment):
            sys.exit(1)
        
        # Wait for services to become healthy
        services = ["ollama-proxy", "prometheus", "grafana"]
        if not manager.wait_for_services(services, args.wait_timeout):
            sys.exit(1)
    
    if args.health_check:
        health_results = manager.run_health_checks()
        if not all(health_results.values()):
            print("❌ Some services are unhealthy")
            sys.exit(1)
    
    if args.collect_metrics:
        metrics = manager.collect_metrics()
        print(f"📊 Collected metrics from {len(metrics)} sources")
    
    if args.generate_report:
        report = manager.generate_health_report()
        manager.save_report(report, args.output)
        
        if report["overall_status"] == "unhealthy":
            print("❌ System is unhealthy")
            sys.exit(1)
        elif report["overall_status"] == "degraded":
            print("⚠️  System is degraded")
            sys.exit(2)
    
    if args.setup_alerts:
        if not manager.setup_alerts():
            sys.exit(1)
    
    if args.test_alerts:
        if not manager.test_alerts():
            sys.exit(1)
    
    if args.cleanup:
        if not manager.cleanup_monitoring():
            sys.exit(1)
    
    print("✅ Monitoring operations completed successfully")


if __name__ == "__main__":
    main()
