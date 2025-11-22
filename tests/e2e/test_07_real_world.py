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
BBX E2E Tests - Layer 7: Real-World Scenarios
Tests complex multi-adapter workflows and production-like scenarios
"""

import asyncio
import tempfile
from pathlib import Path


class TestRealWorldScenarios:
    """Test real-world production scenarios."""
    
    def test_multi_adapter_orchestration(self):
        """Test orchestrating multiple adapters in one workflow."""
        workflow_content = """
name: Multi-Adapter Workflow
version: 1.0

steps:
  - id: terraform_check
    mcp: universal
    method: run
    inputs:
      uses: docker://hashicorp/terraform:light
      cmd: ["terraform", "version"]
  
  - id: aws_check
    depends_on: [terraform_check]
    mcp: universal
    method: run
    inputs:
      uses: docker://amazon/aws-cli:latest
      cmd: ["aws", "--version"]
  
  - id: kubectl_check
    depends_on: [aws_check]
    mcp: universal
    method: run
    inputs:
      uses: docker://bitnami/kubectl:latest
      cmd: ["kubectl", "version", "--client"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            assert 'terraform_check' in results
            assert 'aws_check' in results
            assert 'kubectl_check' in results
            
            # All should execute successfully
            assert results['terraform_check'].get('status') == 'success'
            assert results['aws_check'].get('status') == 'success'
            assert results['kubectl_check'].get('status') == 'success'
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_long_running_workflow(self):
        """Test workflow that takes 30+ seconds."""
        workflow_content = """
name: Long Running Workflow
version: 1.0

steps:
  - id: phase1
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["sh", "-c", "echo 'Phase 1' && sleep 10 && echo 'Phase 1 Done'"]
  
  - id: phase2
    depends_on: [phase1]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["sh", "-c", "echo 'Phase 2' && sleep 10 && echo 'Phase 2 Done'"]
  
  - id: phase3
    depends_on: [phase2]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["sh", "-c", "echo 'Phase 3' && sleep 10 && echo 'Phase 3 Done'"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            import time
            
            start = time.time()
            results = asyncio.run(run_file(workflow_path, inputs={}))
            duration = time.time() - start
            
            assert isinstance(results, dict)
            assert len(results) == 3
            # Should take at least 30 seconds
            assert duration >= 30
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_error_recovery_chain(self):
        """Test comprehensive error recovery with fallbacks."""
        workflow_content = """
name: Error Recovery Chain
version: 1.0

steps:
  - id: attempt_risky_operation
    retry: 2
    retry_delay: 1000
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["sh", "-c", "exit 1"]
  
  - id: fallback_operation
    condition: "{{ steps.attempt_risky_operation.status == 'error' }}"
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["echo", "Using fallback strategy"]
  
  - id: notify_failure
    depends_on: [fallback_operation]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["echo", "Recovery complete via fallback"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            # Fallback and notify should execute
            assert 'fallback_operation' in results or 'notify_failure' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up after workflow."""
        workflow_content = """
name: Resource Cleanup Test
version: 1.0

steps:
  - id: create_temp_data
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["sh", "-c", "echo 'temp data' > /tmp/test.txt && cat /tmp/test.txt"]
  
  - id: verify_cleanup
    depends_on: [create_temp_data]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["echo", "Cleanup verification"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            import subprocess
            
            # Count containers before
            before = subprocess.run(
                ["docker", "ps", "-a", "-q"], 
                capture_output=True, 
                text=True
            ).stdout.count('\n')
            
            asyncio.run(run_file(workflow_path, inputs={}))
            
            # Count containers after
            import time
            time.sleep(2)  # Give Docker time to cleanup
            after = subprocess.run(
                ["docker", "ps", "-a", "-q"], 
                capture_output=True, 
                text=True
            ).stdout.count('\n')
            
            # Should not have created permanent containers (--rm flag)
            assert after <= before + 1  # Allow for some leeway
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
