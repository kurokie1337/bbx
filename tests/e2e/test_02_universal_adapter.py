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
BBX E2E Tests - Layer 2: Universal Adapter Basic Execution
Tests UniversalAdapter with real Docker execution
"""

import asyncio
from blackbox.core.universal_v2 import UniversalAdapterV2


class TestUniversalAdapterExecution:
    """Test UniversalAdapter with real containers."""
    
    def test_simple_echo_command(self):
        """Test simple echo command in Alpine."""
        definition = {
            "id": "test_echo",
            "uses": "docker://alpine:latest",
            "cmd": ["echo", "Hello BBX"]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert response.get("success")
        # Output is in 'data' field
        assert "Hello BBX" in response.get("data", "")
    
    def test_command_with_inputs(self):
        """Test command with Jinja2 template inputs."""
        definition = {
            "id": "test_template",
            "uses": "docker://alpine:latest",
            # Use 'message' directly in template - it's available from inputs context
            "cmd": ["echo", "{{ inputs.message }}"]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={"message": "Test Message"}))
        
        assert response.get("success"), f"Failed: {response.get('error')}"
        assert "Test Message" in response.get("data", "")
    
    def test_command_failure_handling(self):
        """Test adapter handles command failures."""
        definition = {
            "id": "test_fail",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "exit 1"]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert not response.get("success")
        assert response.get("error") is not None
    
    def test_json_output_parsing(self):
        """Test JSON output parser."""
        definition = {
            "id": "test_json",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", 'echo \'{"status": "ok", "value": 42}\''],
            "output_parser": {
                "type": "json"
            }
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        assert response.get("success")
    
    def test_timeout_enforcement(self):
        """Test that commands timeout properly."""
        definition = {
            "id": "test_timeout",
            "uses": "docker://alpine:latest",
            "cmd": ["sleep", "10"],
            "timeout": 2  # 2 second timeout
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        # Should fail due to timeout
        assert not response.get("success")
        error_msg = response.get("error", "").lower()
        assert "timed out" in error_msg or "timeout" in error_msg
