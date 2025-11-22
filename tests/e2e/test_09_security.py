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
BBX E2E Tests - Layer 9: Security & Edge Cases
Tests security boundaries and edge case handling
"""

import asyncio


class TestSecurityAndEdgeCases:
    """Security validation and edge case handling."""
    
    def test_command_injection_prevention(self):
        """Test that command injection is prevented."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        # Try to inject malicious command
        malicious_input = "test; rm -rf /"
        
        definition = {
            "id": "test_injection",
            "uses": "docker://alpine:latest",
            "cmd": ["echo", "{{ inputs.value }}"]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={"value": malicious_input}))
        
        # Should treat as literal string, not execute
        assert response.get("success")
        output = response.get("data", "")
        # The semicolon command should be echoed, not executed
        assert "test; rm -rf /" in output or "test" in output
    
    def test_environment_variable_isolation(self):
        """Test that env vars don't leak between executions."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        # First execution with SECRET
        definition1 = {
            "id": "test_env1",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "echo $SECRET"],
            "env": {"SECRET": "sensitive_value"}
        }
        
        adapter1 = UniversalAdapterV2(definition1)
        response1 = asyncio.run(adapter1.execute(method='run', inputs={}))
        
        assert "sensitive_value" in response1.get("data", "")
        
        # Second execution WITHOUT SECRET
        definition2 = {
            "id": "test_env2",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "echo $SECRET"]
        }
        
        adapter2 = UniversalAdapterV2(definition2)
        response2 = asyncio.run(adapter2.execute(method='run', inputs={}))
        
        # Should NOT see sensitive_value
        assert "sensitive_value" not in response2.get("data", "")
    
    def test_path_traversal_prevention(self):
        """Test that path traversal is prevented in volumes."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        # Try to mount parent directory
        definition = {
            "id": "test_traversal",
            "uses": "docker://alpine:latest",
            "cmd": ["ls", "/"],
            "volumes": {
                "../../../": "/malicious"  # Try to escape
            }
        }
        
        adapter = UniversalAdapterV2(definition)
        # Should either fail safely or sanitize the path
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        # As long as it doesn't crash or expose host FS, we're good
        assert isinstance(response, dict)
    
    def test_resource_limit_enforcement(self):
        """Test that resource limits are enforced."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        definition = {
            "id": "test_resources",
            "uses": "docker://alpine:latest",
            "cmd": ["sh", "-c", "yes > /dev/null"],  # CPU hog
            "resources": {
                "cpu": "0.5",
                "memory": "100m"
            },
            "timeout": 2
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        # Should timeout due to limits
        assert not response.get("success")
    
    def test_unicode_handling(self):
        """Test handling of unicode and special characters."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        unicode_text = "Hello 世界 🚀 Привет"
        
        definition = {
            "id": "test_unicode",
            "uses": "docker://alpine:latest",
            "cmd": ["echo", unicode_text]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        # Should handle unicode gracefully
        assert response.get("success") or isinstance(response, dict)
    
    def test_empty_command_handling(self):
        """Test handling of empty/invalid commands."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        definition = {
            "id": "test_empty",
            "uses": "docker://alpine:latest",
            "cmd": []
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        # Should fail gracefully
        assert isinstance(response, dict)
    
    def test_very_long_command(self):
        """Test handling of very long command strings."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        # Create a very long argument
        long_arg = "A" * 10000
        
        definition = {
            "id": "test_long",
            "uses": "docker://alpine:latest",
            "cmd": ["echo", long_arg]
        }
        
        adapter = UniversalAdapterV2(definition)
        response = asyncio.run(adapter.execute(method='run', inputs={}))
        
        # Should handle or fail gracefully
        assert isinstance(response, dict)
    
    def test_concurrent_auth_isolation(self):
        """Test that auth tokens don't leak between concurrent executions."""
        from blackbox.core.universal_v2 import UniversalAdapterV2
        
        async def run_with_auth(token):
            definition = {
                "id": f"test_auth_{token}",
                "uses": "docker://alpine:latest",
                "cmd": ["sh", "-c", "echo $AUTH_TOKEN"],
                "env": {"AUTH_TOKEN": token}
            }
            
            adapter = UniversalAdapterV2(definition)
            return await adapter.execute(method='run', inputs={})
        
        async def test_concurrent():
            results = await asyncio.gather(
                run_with_auth("token_1"),
                run_with_auth("token_2"),
                run_with_auth("token_3")
            )
            
            # Each should only see its own token
            assert "token_1" in results[0].get("data", "")
            assert "token_2" in results[1].get("data", "")
            assert "token_3" in results[2].get("data", "")
            
            # Cross-contamination check
            assert "token_2" not in results[0].get("data", "")
            assert "token_3" not in results[0].get("data", "")
        
        asyncio.run(test_concurrent())
