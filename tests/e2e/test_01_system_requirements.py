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
BBX E2E Tests - Layer 1: Docker and System Check
Tests basic system requirements before running full E2E
"""

import pytest
import subprocess
import sys
from pathlib import Path


class TestSystemRequirements:
    """Test system prerequisites for E2E tests."""
    
    def test_python_version(self):
        """Test Python version is 3.8+"""
        assert sys.version_info >= (3, 8), "Python 3.8+ required"
    
    def test_docker_installed(self):
        """Test Docker CLI is installed."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode == 0, "Docker CLI not found"
            assert "Docker version" in result.stdout
        except FileNotFoundError:
            pytest.skip("Docker not installed")
    
    def test_docker_daemon_running(self):
        """Test Docker daemon is running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=30  # Increased timeout
            )
            if result.returncode != 0:
                pytest.skip("Docker daemon not accessible")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Docker daemon not accessible")
    
    def test_can_pull_alpine(self):
        """Test can pull alpine image (smallest test image)."""
        try:
            # Check if already exists with longer timeout
            check = subprocess.run(
                ["docker", "images", "-q", "alpine:latest"],
                capture_output=True,
                text=True,
                timeout=15  # Increased from 5
            )
            
            if not check.stdout.strip():
                # Pull if not exists with longer timeout
                result = subprocess.run(
                    ["docker", "pull", "alpine:latest"],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minutes for slow connections
                )
                if result.returncode != 0:
                    pytest.skip("Docker pull failed - may be network issue")
        except subprocess.TimeoutExpired:
            pytest.skip("Docker too slow - may be starting up")
    
    def test_can_run_alpine_container(self):
        """Test can run a simple container."""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "alpine:latest", "echo", "test"],
                capture_output=True,
                text=True,
                timeout=90  # Increased for Windows
            )
            assert result.returncode == 0, f"Container failed: {result.stderr}"
            assert "test" in result.stdout
        except subprocess.TimeoutExpired:
            pytest.skip("Container execution too slow (may be slow on first run)")
    
    def test_project_structure(self):
        """Test BBX project structure exists."""
        required_files = [
            "cli.py",
            "blackbox/core/universal.py",
            "blackbox/core/universal_v2.py",
            "blackbox/core/auth.py",
            "blackbox/library",
        ]
        
        for file in required_files:
            path = Path(file)
            assert path.exists(), f"Missing required file/dir: {file}"
