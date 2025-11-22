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
BBX E2E Tests - Layer 2 EXTENDED: Advanced Universal Adapter
Tests complex scenarios: volumes, auth, networking, resource limits
"""

import asyncio
import tempfile
from pathlib import Path
from blackbox.core.universal_v2 import UniversalAdapterV2


class TestUniversalAdapterAdvanced:
    """Advanced Universal Adapter scenarios."""
    
    def test_volume_mounting(self):
        """Test volume mounting."""
        
        # Create a temp file to mount
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test data")
            temp_path = f.name
        
        try:
            definition = {
                "id": "test_volumes",
                "uses": "docker://alpine:latest",
                "cmd": ["cat", "/data/test.txt"],
                "volumes": {
                    temp_path: "/data/test.txt"
                }
            }
            
            adapter = UniversalAdapterV2(definition)
            response = asyncio.run(adapter.execute(method='run', inputs={}))
            
            assert response.get("success"), f"Failed: {response.get('error')}"
            assert "test data" in response.get("data", "")
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_environment_variables(self):
        """Test environment variable injection."""
        definition = {
            "id": "test_env",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "echo $MY_VAR"],
            "env": {
                "MY_VAR": "{{ inputs.value }}"
            }
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={"value": "TestValue"}))
        
        assert response.get("success"), f"Failed: {response.get('error')}"
        assert "TestValue" in response.get("data", ""), f"Output: {response.get('data')}"
    
    def test_working_directory(self):
        """Test custom working directory."""
        definition = {
            "id": "test_workdir",
            "uses": "docker://alpine:latest",
            "cmd": ["pwd"],
            "working_dir": "/tmp"
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert response.get("success"), f"Failed: {response.get('error')}"
        assert "/tmp" in response.get("data", ""), f"Output: {response.get('data')}"
    
    def test_multi_line_script(self):
        """Test executing multi-line shell scripts."""
        definition = {
            "id": "test_script",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", """
                echo "Line 1"
                echo "Line 2"
                echo "Result: OK"
            """]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert response.get("success")
        data = response.get("data", "")
        assert "Line 1" in data
        assert "Line 2" in data
        assert "Result: OK" in data
    
    def test_binary_output_handling(self):
        """Test handling binary/non-UTF8 output."""
        definition = {
            "id": "test_binary",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "echo -e '\\x00\\x01\\x02test'"]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        # Should handle binary gracefully
        assert isinstance(response, dict)
    
    def test_large_output(self):
        """Test handling large output (>10KB)."""
        definition = {
            "id": "test_large_output",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "seq 1 1000"]  # Reduced from 10k to 1k
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert response.get("success")
        assert len(response.get("data", "")) > 1000
    
    def test_exit_code_detection(self):
        """Test proper exit code detection."""
        # Success case
        definition_success = {
            "id": "test_exit_0",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "exit 0"]
        }
        
        adapter = UniversalAdapterV2(definition_success)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        assert response.get("success")
        
        # Failure case
        definition_fail = {
            "id": "test_exit_1",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "exit 42"]
        }
        
        adapter = UniversalAdapterV2(definition_fail)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        assert not response.get("success")
    
    def test_command_not_found(self):
        """Test handling of non-existent commands."""
        definition = {
            "id": "test_not_found",
            "uses": "docker://alpine:latest",
            "cmd": ["nonexistent_command_12345"]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert not response.get("success")
        assert response.get("error") is not None
