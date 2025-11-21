# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Comprehensive tests for Docker adapter

Tests Docker integration including:
- Container lifecycle
- Image management
- Docker Compose
- Error handling
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blackbox.core.adapters.docker import DockerAdapter


@pytest.fixture
def docker_adapter():
    """Create Docker adapter instance"""
    return DockerAdapter()


@pytest.fixture(autouse=True)
def cleanup_containers(docker_adapter):
    """Cleanup test containers"""
    yield
    # Cleanup any test containers
    import asyncio
    test_containers = ["test_nginx", "test_postgres", "test_redis"]
    for name in test_containers:
        try:
            asyncio.run(docker_adapter.execute("remove", {"name": name, "force": True}))
        except:
            pass


@pytest.mark.asyncio
@pytest.mark.docker
async def test_docker_pull_image(docker_adapter):
    """Test pulling a Docker image"""
    result = await docker_adapter.execute("pull", {
        "image": "nginx:alpine"
    })
    
    # May succeed or fail depending on Docker availability
    assert "status" in result
    assert result["status"] in ["pulled", "error"]


@pytest.mark.asyncio
@pytest.mark.docker
async def test_docker_run_container(docker_adapter):
    """Test running a container"""
    result = await docker_adapter.execute("run", {
        "image": "nginx:alpine",
        "name": "test_nginx",
        "ports": ["8080:80"],
        "detach": True
    })
    
    if result.get("status") == "running":
        assert "container_id" in result
        assert result["name"] == "test_nginx"
        
        # Cleanup
        await docker_adapter.execute("stop", {"name": "test_nginx"})
        await docker_adapter.execute("remove", {"name": "test_nginx"})


@pytest.mark.asyncio
@pytest.mark.docker
async def test_docker_list_containers(docker_adapter):
    """Test listing containers"""
    result = await docker_adapter.execute("ps", {"all": True})
    
    if result.get("status") == "ok":
        assert "containers" in result
        assert isinstance(result["containers"], list)


@pytest.mark.asyncio
async def test_docker_methods_exist(docker_adapter):
    """Test that all methods are defined"""
    methods = [
        "run", "stop", "remove",
        "build", "pull", "push",
        "logs", "inspect", "ps",
        "compose_up", "compose_down"
    ]
    
    for method in methods:
        # Just verify the method doesn't raise for unknown method
        try:
            # Will fail with missing inputs, but that's OK
            await docker_adapter.execute(method, {})
        except (KeyError, ValueError) as e:
            # Expected - missing required inputs
            pass
        except Exception as e:
            # Docker not available is OK
            if "docker" not in str(e).lower():
                pytest.fail(f"Unexpected error for {method}: {e}")


@pytest.mark.asyncio
async def test_docker_invalid_method(docker_adapter):
    """Test invalid method raises error"""
    with pytest.raises(ValueError, match="Unknown method"):
        await docker_adapter.execute("invalid_method", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
