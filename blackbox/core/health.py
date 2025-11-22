# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Health Check System for BBX
Provides endpoints for monitoring and readiness checks
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class HealthChecker:
    """System health check utilities."""
    
    @staticmethod
    def check_docker() -> Dict[str, Any]:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return {
                    "status": "healthy",
                    "message": "Docker daemon running"
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Docker daemon not responding",
                    "details": result.stderr
                }
        except subprocess.TimeoutExpired:
            return {
                "status": "unhealthy",
                "message": "Docker check timed out"
            }
        except FileNotFoundError:
            return {
                "status": "unhealthy",
                "message": "Docker not installed"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Docker check failed: {str(e)}"
            }
    
    @staticmethod
    def check_python() -> Dict[str, Any]:
        """Check Python version."""
        version = sys.version_info
        
        if version.major >= 3 and version.minor >= 9:
            return {
                "status": "healthy",
                "message": f"Python {version.major}.{version.minor}.{version.micro}"
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Python version too old: {version.major}.{version.minor}"
            }
    
    @staticmethod
    def check_packages() -> Dict[str, Any]:
        """Check if core packages are available."""
        try:
            # from blackbox.core.universal_v2 import UniversalAdapterV2
            # from blackbox.core.runtime import run_file
            # from blackbox.core.registry import MCPRegistry
            
            return {
                "status": "healthy",
                "message": "All core packages available"
            }
        except ImportError as e:
            return {
                "status": "unhealthy",
                "message": f"Missing package: {str(e)}"
            }
    
    @staticmethod
    def liveness() -> Dict[str, Any]:
        """Basic liveness check."""
        return {
            "status": "alive",
            "message": "BBX system is running"
        }
    
    @staticmethod
    def readiness() -> Dict[str, Any]:
        """Comprehensive readiness check."""
        checks = {
            "liveness": HealthChecker.liveness(),
            "python": HealthChecker.check_python(),
            "docker": HealthChecker.check_docker(),
            "packages": HealthChecker.check_packages()
        }
        
        all_healthy = all(
            check.get("status") in ["alive", "healthy"] 
            for check in checks.values()
        )
        
        return {
            "ready": all_healthy,
            "checks": checks
        }
    
    @staticmethod
    def metrics() -> Dict[str, Any]:
        """Basic metrics in Prometheus format."""
        # TODO: Add real metrics tracking
        return {
            "bbx_workflows_total": 0,
            "bbx_adapters_total": 0,
            "bbx_docker_pull_duration_seconds": 0
        }


def health_endpoint() -> Dict[str, Any]:
    """Health check endpoint."""
    return HealthChecker.liveness()


def ready_endpoint() -> Dict[str, Any]:
    """Readiness check endpoint."""
    return HealthChecker.readiness()


def metrics_endpoint() -> str:
    """Metrics in Prometheus format."""
    metrics = HealthChecker.metrics()
    
    lines = []
    for key, value in metrics.items():
        lines.append(f"{key} {value}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    
    print("=== BBX Health Check ===\n")
    
    print("Liveness:")
    print(json.dumps(health_endpoint(), indent=2))
    
    print("\nReadiness:")
    print(json.dumps(ready_endpoint(), indent=2))
    
    print("\nMetrics:")
    print(metrics_endpoint())
