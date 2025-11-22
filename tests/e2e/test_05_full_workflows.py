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
BBX E2E Tests - Layer 5: Full Workflow Execution
Tests complete workflows with runtime, auth, and multi-step execution
"""

import asyncio
import tempfile
from pathlib import Path


class TestFullWorkflowExecution:
    """Test complete workflow execution end-to-end."""
    
    def test_simple_workflow_execution(self):
        """Test executing a simple workflow file."""
        workflow_content = """
name: Simple Test Workflow
version: 1.0

steps:
  - id: echo_test
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Workflow Test"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict), f"Expected dict, got {type(results)}"
            assert len(results) > 0, "No results returned from workflow"
            assert 'echo_test' in results, f"Step 'echo_test' not in results: {list(results.keys())}"
            
            step_result = results['echo_test']
            # Runtime returns {'status': 'success', 'output': {...}}
            assert step_result.get('status') == 'success', f"Step failed: {step_result}"
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_workflow_with_inputs(self):
        """Test workflow with input parameters."""
        workflow_content = """
name: Input Test Workflow
version: 1.0

inputs:
  message:
    type: string
    default: "Default Message"

steps:
  - id: echo_message
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "{{ inputs.message }}"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={"message": "Custom Input"}))
            
            assert isinstance(results, dict)
            assert 'echo_message' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_multi_step_workflow(self):
        """Test workflow with multiple sequential steps."""
        workflow_content = """
name: Multi-Step Workflow
version: 1.0

steps:
  - id: step1
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Step 1"
  
  - id: step2
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Step 2"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            assert 'step1' in results
            assert 'step2' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_workflow_with_environment_vars(self):
        """Test workflow with environment variables."""
        workflow_content = """
name: Env Var Workflow
version: 1.0

steps:
  - id: test_env
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - sh
        - -c
        - "echo $TEST_VAR"
      env:
        TEST_VAR: "Hello from ENV"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            assert 'test_env' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_workflow_error_handling(self):
        """Test workflow handles errors gracefully."""
        workflow_content = """
name: Error Test Workflow
version: 1.0

steps:
  - id: failing_step
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - sh
        - -c
        - "exit 1"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            assert 'failing_step' in results
            # Should have executed but failed
            step_result = results['failing_step']
            # Runtime wraps adapter response in {'status': 'success', 'output': {...}}
            # Check if the output indicates failure
            output = step_result.get('output', {})
            assert not output.get('success', True), f"Step should have failed but got: {step_result}"
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
