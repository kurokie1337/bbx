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
BBX Core - Unit Tests for Package Manager
Bottom-up testing: Layer 2
"""

import pytest
from blackbox.core.package_manager import AdapterPackageManager


@pytest.fixture
def temp_library(tmp_path):
    """Create temporary library with test definitions."""
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    
    # Valid definition
    (library_dir / "test_valid.yaml").write_text("""
id: test_valid
uses: docker://alpine:latest
cmd:
  - echo
  - hello
""")
    
    return library_dir


class TestPackageManager:
    """Test package manager."""
    
    def test_list_available(self, temp_library):
        """Test listing available packages."""
        pm = AdapterPackageManager(library_dir=temp_library)
        available = pm.list_available()
        
        assert "test_valid" in available
    
    def test_install_valid_package(self, temp_library):
        """Test installing valid package."""
        pm = AdapterPackageManager(library_dir=temp_library)
        success = pm.install("test_valid")
        
        assert success
        assert "test_valid" in pm.list_installed()
    
    def test_get_package(self, temp_library):
        """Test getting package."""
        pm = AdapterPackageManager(library_dir=temp_library)
        definition = pm.get("test_valid")
        
        assert definition is not None
        assert definition["id"] == "test_valid"
