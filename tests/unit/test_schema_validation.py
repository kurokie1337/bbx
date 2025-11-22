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
BBX Core - Unit Tests for Schema Validation
Bottom-up testing: Layer 1
"""

from blackbox.core.universal_schema import validate_definition


class TestSchemaValidation:
    """Test YAML definition schema validation."""
    
    def test_valid_minimal_definition(self):
        """Test minimal valid definition."""
        definition = {
            "id": "test_adapter",
            "uses": "docker://alpine:latest",
            "cmd": ["echo", "hello"]
        }
        
        is_valid, errors = validate_definition(definition)
        assert is_valid
        assert len(errors) == 0
    
    def test_missing_id(self):
        """Test validation fails when ID is missing."""
        definition = {
            "uses": "alpine:latest",
            "cmd": ["echo", "test"]
        }
        
        is_valid, errors = validate_definition(definition)
        assert not is_valid
        assert any("id" in str(e).lower() for e in errors)
    
    def test_missing_uses(self):
        """Test validation fails when uses is missing."""
        definition = {
            "id": "test",
            "cmd": ["echo", "test"]
        }
        
        is_valid, errors = validate_definition(definition)
        assert not is_valid
    
    def test_valid_with_auth(self):
        """Test valid definition with auth."""
        definition = {
            "id": "kubectl",
            "uses": "bitnami/kubectl:latest",
            "cmd": ["kubectl", "version"],
            "auth": {"type": "kubeconfig"}
        }
        
        is_valid, errors = validate_definition(definition)
        assert is_valid
