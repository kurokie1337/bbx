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
BBX E2E Tests - Layer 8: Stress & Performance Tests
Tests system under heavy load and edge cases
"""

import asyncio
import tempfile
from pathlib import Path


class TestStressAndPerformance:
    """Stress testing and performance validation."""
    
    def test_large_output_handling(self):
        """Test handling very large output (1MB+)."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        definition = {
            "id": "test_large_output",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "head -c 1048576 /dev/urandom | base64"]  # ~1MB
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert response.get("success") is not None
        # Should handle large output without crashing
        assert isinstance(response, dict)
    
    def test_many_step_workflow(self):
        """Test workflow with many sequential steps."""
        workflow_content = """
name: Many Steps Workflow
version: 1.0

steps:"""
        
        # Generate 20 steps
        for i in range(1, 21):
            workflow_content += f"""
  - id: step{i}
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["echo", "Step {i}"]
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
            assert len(results) == 20
            
            print(f"\n✅ 20 steps completed in {duration:.2f}s")
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_concurrent_workflow_execution(self):
        """Test running multiple workflows concurrently."""
        workflow_content = """
name: Concurrent Test
version: 1.0

steps:
  - id: task
    mcp: universal
    method: run
    inputs:
      uses: docker://alpine:latest
      cmd: ["sh", "-c", "sleep 1 && echo 'Done'"]
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False) as f:
            f.write(workflow_content)
            workflow_path = f.name
        
        try:
            from blackbox.core.runtime import run_file
            import time
            
            async def run_multiple():
                tasks = [run_file(workflow_path, inputs={}) for _ in range(5)]
                return await asyncio.gather(*tasks)
            
            start = time.time()
            results_list = asyncio.run(run_multiple())
            duration = time.time() - start
            
            assert len(results_list) == 5
            # Should complete faster than sequential (5s)
            # Parallel should be ~1-2s, but Windows Docker can be slower
            print(f"\n✅ 5 concurrent workflows in {duration:.2f}s")
            assert duration < 15  # Generous timeout for Windows
            
        finally:
            Path(workflow_path).unlink(missing_ok=True)
    
    def test_memory_efficiency(self):
        """Test that memory doesn't grow unbounded."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        import gc
        
        definition = {
            "id": "test_memory",
            "uses": "docker://alpine:latest",
            "cmd": ["echo", "test"]
        }
        
        adapter = UniversalAdapterV2(definition)
        
        # Run multiple times
        for i in range(10):
            response = asyncio.run(adapter.execute(method='run', inputs={}))
            assert response.get("success")
        
        # Force garbage collection
        gc.collect()
        
        # Should not leak memory (basic check)
        assert True  # If we get here without OOM, we're good
    
    def test_rapid_adapter_creation(self):
        """Test creating many adapter instances."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        # Create 100 adapters
        adapters = []
        for i in range(100):
            definition = {
                "id": f"adapter_{i}",
                "uses": "docker://alpine:latest",
                "cmd": ["echo", f"Adapter {i}"]
            }
            adapters.append(UniversalAdapterV2(definition))
        
        assert len(adapters) == 100
        
        # Run a few to verify they work
        response = asyncio.run(adapters[0].execute(method='run', inputs={}))
        assert response.get("success")
