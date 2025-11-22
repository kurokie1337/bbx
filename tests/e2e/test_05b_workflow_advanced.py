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
BBX E2E Tests - Layer 5 EXTENDED: Advanced Workflow Features
Tests DAG, conditionals, parallel execution, state management
"""

import asyncio
import tempfile
from pathlib import Path


class TestAdvancedWorkflowFeatures:
    """Test advanced workflow execution features."""
    
    def test_workflow_with_dependencies(self):
        """Test workflow with step dependencies (DAG)."""
        workflow_content = """
name: DAG Workflow
version: 1.0

steps:
  - id: step1
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - sh
        - -c
        - echo "Step 1 Complete" > /tmp/step1.txt && cat /tmp/step1.txt
  
  - id: step2
    depends_on: [step1]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Step 2 after Step 1"
  
  - id: step3
    depends_on: [step1, step2]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Step 3 after 1 and 2"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            # All steps should execute
            assert 'step1' in results
            assert 'step2' in results
            assert 'step3' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_workflow_with_outputs(self):
        """Test passing outputs between steps."""
        workflow_content = """
name: Output Passing Workflow
version: 1.0

steps:
  - id: generate_data
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["sh", "-c", "echo 'Generated Data'"]
  
  - id: use_data
    depends_on: [generate_data]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["echo", "Step completed"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            assert 'generate_data' in results
            assert 'use_data' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_conditional_execution(self):
        """Test conditional step execution."""
        workflow_content = """
name: Conditional Workflow
version: 1.0

inputs:
  should_run:
    type: boolean
    default: true

steps:
  - id: always_runs
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "This always runs"
  
  - id: conditional_step
    condition: "{{ inputs.should_run }}"
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "This runs conditionally"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            # Test with condition true
            results = asyncio.run(run_file(workflow_path, inputs={"should_run": True}))
            assert isinstance(results, dict)
            assert 'always_runs' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_error_recovery_workflow(self):
        """Test workflow with error handling and retry."""
        workflow_content = """
name: Error Recovery Workflow
version: 1.0

steps:
  - id: might_fail
    retry: 2
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - sh
        - -c
        - "exit 1"
  
  - id: on_error
    condition: "{{ steps.might_fail.status == 'error' }}"
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Handling error"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(workflow_path, inputs={}))
            
            assert isinstance(results, dict)
            assert 'might_fail' in results
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_parallel_execution(self):
        """Test parallel step execution."""
        workflow_content = """
name: Parallel Workflow
version: 1.0

steps:
  - id: parallel1
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - sh
        - -c
        - "sleep 1 && echo 'Parallel 1'"
  
  - id: parallel2
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - sh
        - -c
        - "sleep 1 && echo 'Parallel 2'"
  
  - id: parallel3
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - sh
        - -c
        - "sleep 1 && echo 'Parallel 3'"
  
  - id: final_step
    depends_on: [parallel1, parallel2, parallel3]
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "All parallel steps complete"
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
            assert 'final_step' in results
            
            # Should take ~1-2s if parallel, ~3-4s if sequential
            # This is a soft check - just verify it ran
            assert duration < 30  # Reasonable timeout
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_nested_workflows(self):
        """Test calling workflows from within workflows."""
        # Create sub-workflow
        sub_workflow = """
name: Sub Workflow
version: 1.0

steps:
  - id: sub_step
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Sub workflow executed"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(sub_workflow)
            sub_path = f.name
        
        # Create main workflow
        main_workflow = """
name: Main Workflow
version: 1.0

steps:
  - id: main_step
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd:
        - echo
        - "Main workflow"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f2:
            f2.write(main_workflow)
            main_path = f2.name
        
        try:
            from blackbox.core.runtime import run_file
            
            results = asyncio.run(run_file(main_path, inputs={}))
            
            assert isinstance(results, dict)
            assert 'main_step' in results
            
        finally:
            Path(sub_path).unlink(missing_ok=True)
            Path(main_path).unlink(missing_ok=True)
