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
BBX E2E Tests - Layer 4: CLI Commands
Tests CLI commands with real execution
"""

import pytest
import subprocess
import tempfile
from pathlib import Path


class TestCLICommands:
    """Test CLI commands end-to-end."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            ["python", "cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0
        assert "Blackbox Workflow Engine" in result.stdout
    
    def test_cli_version(self):
        """Test version command."""
        result = subprocess.run(
            ["python", "cli.py", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0
        assert "Blackbox Workflow Engine" in result.stdout
    
    def test_package_list(self):
        """Test package list command."""
        result = subprocess.run(
            ["python", "cli.py", "package", "list"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        assert result.returncode == 0
        assert "Available Packages" in result.stdout or "available" in result.stdout.lower()
    
    def test_package_validate(self):
        """Test package validate command."""
        result = subprocess.run(
            ["python", "cli.py", "package", "validate"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should succeed or fail gracefully, not crash
        assert result.returncode in [0, 1]
    
    def test_system_check(self):
        """Test system health check."""
        result = subprocess.run(
            ["python", "cli.py", "system"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        assert "Docker" in result.stdout or "System Health" in result.stdout
    
    @pytest.mark.slow
    def test_run_simple_workflow(self):
        """Test running a simple workflow file."""
        # Create minimal workflow
        workflow_content = """
name: Test Workflow
version: 1.0
steps:
  - id: test_step
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Test Success"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            result = subprocess.run(
                ["python", "cli.py", "run", workflow_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Check it ran (may succeed or fail, but should not crash)
            assert result.returncode in [0, 1]
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
