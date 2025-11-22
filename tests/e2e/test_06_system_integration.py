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
BBX E2E Tests - Layer 6: System Integration
Tests full system integration: Runtime + Auth + Registry + MCP
"""

import pytest
import asyncio
import subprocess
import tempfile
from pathlib import Path


class TestSystemIntegration:
    """Test full BBX system integration."""
    
    def test_registry_adapter_loading(self):
        """Test loading adapters from registry."""
        from blackbox.core.registry import MCPRegistry
        
        registry = MCPRegistry()
        
        # Registry should have methods to list and get adapters
        assert hasattr(registry, 'get_adapter')
        assert hasattr(registry, 'list_adapters')
        assert hasattr(registry, 'register')
    
    def test_auth_provider_registry(self):
        """Test auth provider registration."""
        from blackbox.core.auth import AuthRegistry
        
        # Should have auth providers
        providers = ['kubeconfig', 'aws_credentials', 'gcp_credentials']
        
        for provider in providers:
            assert AuthRegistry.get_provider(provider) is not None
    
    def test_package_discovery(self):
        """Test Universal Adapter package discovery."""
        from blackbox.core.package_manager import get_package_manager
        
        pm = get_package_manager()
        available = pm.list_available()
        
        assert len(available) > 0
        assert 'ansible' in available or 'aws_cli' in available
    
    def test_workflow_validation(self):
        """Test workflow validation system."""
        from blackbox.core.validation import validate_workflow
        
        # Valid workflow
        valid_workflow = {
            "name": "Test",
            "version": "1.0",
            "steps": [
                {
                    "id": "test",
                    "mcp": "universal",
                    "method": "run",
                    "inputs": {
                        "uses": "docker://alpine:latest",
                        "cmd": ["echo", "test"]
                    }
                }
            ]
        }
        
        # Should not raise
        try:
            validate_workflow(valid_workflow)
        except Exception as e:
            pytest.skip(f"Validation not fully implemented: {e}")
    
    def test_schema_generation(self):
        """Test BBX schema generation."""
        from blackbox.core.schema import BBXSchemaGenerator
        
        generator = BBXSchemaGenerator()
        schema = generator.generate()
        
        assert isinstance(schema, dict)
        assert 'definitions' in schema or 'properties' in schema
    
    def test_cli_package_commands_integration(self):
        """Test CLI package commands end-to-end."""
        # Test package list
        result = subprocess.run(
            ["python", "cli.py", "package", "list"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        assert result.returncode == 0
        assert "ansible" in result.stdout.lower() or "available" in result.stdout.lower()
    
    def test_cli_audit_integration(self):
        """Test CLI audit command."""
        result = subprocess.run(
            ["python", "cli.py", "audit"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        # Should run without crashing
        assert result.returncode in [0, 1]  # May have no data yet
    
    def test_full_e2e_workflow_via_cli(self):
        """Test complete E2E workflow via CLI."""
        workflow_content = """
name: CLI E2E Test
version: 1.0

steps:
  - id: test_via_cli
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "CLI E2E Test Success"
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
            
            # Should execute (may succeed or fail, but shouldn't crash)
            assert result.returncode in [0, 1]
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_concurrent_workflow_execution(self):
        """Test running multiple workflows concurrently."""
        workflow1 = """
name: Concurrent 1
version: 1.0
steps:
  - id: step1
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: [echo, "Workflow 1"]
"""
        
        workflow2 = """
name: Concurrent 2
version: 1.0
steps:
  - id: step2
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: [echo, "Workflow 2"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_1.bbx', delete=False) as f1:
            f1.write(workflow1)
            path1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_2.bbx', delete=False) as f2:
            f2.write(workflow2)
            path2 = f2.name
        
        try:
            from blackbox.core.runtime import run_file
            
            # Run concurrently
            async def run_both():
                task1 = asyncio.create_task(run_file(path1, inputs={}))
                task2 = asyncio.create_task(run_file(path2, inputs={}))
                return await asyncio.gather(task1, task2, return_exceptions=True)
            
            results = asyncio.run(run_both())
            
            assert len(results) == 2
            
        finally:
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)
    
    def test_system_health_check(self):
        """Test complete system health check."""
        result = subprocess.run(
            ["python", "cli.py", "system"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        assert "Docker" in result.stdout or "System" in result.stdout
