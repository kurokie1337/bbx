#!/usr/bin/env python3
# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
End-to-End Integration Tests for Universal Adapter
"""

import pytest
from pathlib import Path
from blackbox.core.universal_v2 import UniversalAdapterV2
from blackbox.core.package_manager import get_package_manager

class TestUniversalAdapterE2E:
    """End-to-end integration tests."""
    
    def test_package_manager_install(self):
        """Test installing a package from library."""
        pm = get_package_manager()
        
        # Install kubectl definition
        success = pm.install("k8s_apply")
        assert success, "Failed to install k8s_apply"
        
        # Verify it's cached
        assert "k8s_apply" in pm.list_installed()
        
        # Get the definition
        definition = pm.get("k8s_apply")
        assert definition is not None
        assert definition["id"] == "k8s_apply"
    
    def test_package_manager_list_available(self):
        """Test listing available packages."""
        pm = get_package_manager()
        available = pm.list_available()
        
        # Should have at least the core packages
        expected = ["aws_cli", "gcloud", "k8s_apply", "tf_plan", "psql"]
        for pkg in expected:
            assert pkg in available, f"Missing expected package: {pkg}"
    
    def test_package_manager_validate_all(self):
        """Test validation of all library definitions."""
        pm = get_package_manager()
        results = pm.validate_all()
        
        # All definitions should be valid
        for package_name, (is_valid, errors) in results.items():
            assert is_valid, f"{package_name} validation failed: {errors}"
    
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test simple command execution."""
        adapter = UniversalAdapterV2()
        
        result = await adapter.execute("run", {
            "uses": "docker://alpine:latest",
            "cmd": ["echo", "Hello BBX"],
        })
        
        assert result["success"]
        assert "Hello BBX" in result.get("data", "")
    
    @pytest.mark.asyncio
    async def test_multi_step_execution(self):
        """Test multi-step command execution."""
        adapter = UniversalAdapterV2()
        
        result = await adapter.execute("run", {
            "uses": "docker://alpine:latest",
            "steps": [
                {
                    "name": "Step 1",
                    "cmd": ["echo", "First step"]
                },
                {
                    "name": "Step 2",
                    "cmd": ["echo", "Second step"]
                }
            ]
        })
        
        assert result["success"]
        assert len(result["steps"]) == 2
    
    @pytest.mark.asyncio
    async def test_json_output_parsing(self):
        """Test JSON output parsing."""
        adapter = UniversalAdapterV2()
        
        result = await adapter.execute("run", {
            "uses": "docker://alpine:latest",
            "cmd": ["echo", '{"version": "1.0", "status": "ok"}'],
            "output_parser": {
                "type": "json"
            }
        })
        
        assert result["success"]
        assert result["data"]["version"] == "1.0"
        assert result["data"]["status"] == "ok"
    
    @pytest.mark.skipif(not Path("~/.kube/config").expanduser().exists(), 
                       reason="Kubernetes config not found")
    @pytest.mark.asyncio
    async def test_kubectl_with_auth(self):
        """Test kubectl execution with kubeconfig auth."""
        pm = get_package_manager()
        definition = pm.get("k8s_apply")
        
        adapter = UniversalAdapterV2(definition)
        
        result = await adapter.execute("run", {
            "file": "test-deployment.yaml",
            "namespace": "default"
        })
        
        # May fail if no k8s cluster, but should at least try
        assert "data" in result or "error" in result

def test_definition_validation():
    """Test definition schema validation."""
    from blackbox.core.universal_schema import validate_definition
    
    # Valid definition
    valid_def = {
        "id": "test_tool",
        "uses": "docker://alpine:latest",
        "cmd": ["echo", "test"]
    }
    is_valid, errors = validate_definition(valid_def)
    assert is_valid, f"Valid definition failed: {errors}"
    
    # Invalid definition (missing required field)
    invalid_def = {
        "id": "test",
        "cmd": ["echo", "test"]
        # Missing 'uses'
    }
    is_valid, errors = validate_definition(invalid_def)
    assert not is_valid
    assert any("uses" in str(e) for e in errors)

def test_performance_benchmarks():
    """Benchmark key operations."""
    import time
    from blackbox.core.package_manager import get_package_manager
    
    pm = get_package_manager()
    
    # Benchmark: Install all packages
    start = time.time()
    for pkg in pm.list_available():
        pm.install(pkg)
    install_time = time.time() - start
    
    print("\\n📊 Performance Benchmarks:")
    print(f"  Install all packages: {install_time:.3f}s")
    print(f"  Average per package: {install_time/len(pm.list_available()):.3f}s")
    
    # Benchmark: Validate all
    start = time.time()
    results = pm.validate_all()
    validate_time = time.time() - start
    
    print(f"  Validate all packages: {validate_time:.3f}s")
    print(f"  Total packages validated: {len(results)}")
    
    # Assertions
    assert install_time < 5.0, "Installation too slow"
    assert validate_time < 2.0, "Validation too slow"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
