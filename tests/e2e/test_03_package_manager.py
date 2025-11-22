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
BBX E2E Tests - Layer 3: Package Manager Integration
Tests package manager with real YAML definitions
"""

import pytest
from pathlib import Path
from blackbox.core.package_manager import AdapterPackageManager


class TestPackageManagerIntegration:
    """Test package manager with real library."""
    
    @pytest.fixture
    def package_manager(self):
        """Get package manager instance."""
        lib_dir = Path("blackbox/library")
        if not lib_dir.exists():
            pytest.skip("Library directory not found")
        return AdapterPackageManager(library_dir=lib_dir)
    
    def test_list_available_packages(self, package_manager):
        """Test listing available packages from library."""
        available = package_manager.list_available()
        
        assert len(available) > 0, "No packages found in library"
        assert isinstance(available, list)
    
    def test_load_and_validate_packages(self, package_manager):
        """Test all library packages are valid."""
        results = package_manager.validate_all()
        
        invalid = {pkg: errors for pkg, (valid, errors) in results.items() if not valid}
        
        assert len(invalid) == 0, f"Invalid packages found: {list(invalid.keys())}"
    
    def test_install_package(self, package_manager):
        """Test installing a package."""
        available = package_manager.list_available()
        if not available:
            pytest.skip("No packages available")
        
        pkg_name = available[0]
        success = package_manager.install(pkg_name)
        
        assert success, f"Failed to install {pkg_name}"
        assert pkg_name in package_manager.list_installed()
    
    def test_get_package_definition(self, package_manager):
        """Test getting package definition."""
        available = package_manager.list_available()
        if not available:
            pytest.skip("No packages available")
        
        pkg_name = available[0]
        definition = package_manager.get(pkg_name)
        
        assert definition is not None
        assert "id" in definition
        
        # Template packages may not have 'uses' - they have 'template': True
        if definition.get("template"):
            pytest.skip(f"{pkg_name} is a template package")
        else:
            assert "uses" in definition
        assert "cmd" in definition or "steps" in definition
    
    def test_reload_package(self, package_manager):
        """Test reloading a package."""
        available = package_manager.list_available()
        if not available:
            pytest.skip("No packages available")
        
        pkg_name = available[0]
        package_manager.install(pkg_name)
        
        success = package_manager.reload(pkg_name)
        assert success
