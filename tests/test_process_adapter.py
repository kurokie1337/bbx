# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Comprehensive tests for process management adapter

Tests:
- Process start/stop/restart
- Status monitoring
- Health checks
- Auto-restart
- Error handling
"""

import pytest
import time
import sys
from pathlib import Path

# Add blackbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from blackbox.core.adapters.process import ProcessAdapter, _process_manager


@pytest.fixture
def process_adapter():
    """Create process adapter instance"""
    return ProcessAdapter()


@pytest.fixture(autouse=True)
def cleanup_processes():
    """Cleanup all test processes after each test"""
    yield
    # Stop all running test processes
    for name in list(_process_manager.processes.keys()):
        if name.startswith("test_"):
            _process_manager.stop(name, force=True)


@pytest.mark.asyncio
async def test_start_process(process_adapter):
    """Test starting a process"""
    result = await process_adapter.execute("start", {
        "name": "test_echo",
        "command": "python -c \"import time; time.sleep(5)\""
    })
    
    assert result["status"] == "started"
    assert result["name"] == "test_echo"
    assert "pid" in result
    assert result["pid"] > 0


@pytest.mark.asyncio
async def test_stop_process(process_adapter):
    """Test stopping a process"""
    # Start
    start_result = await process_adapter.execute("start", {
        "name": "test_stop",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    assert start_result["status"] == "started"
    pid = start_result["pid"]
    
    # Wait a bit
    time.sleep(0.5)
    
    # Stop
    stop_result = await process_adapter.execute("stop", {
        "name": "test_stop"
    })
    
    assert stop_result["status"] in ["stopped", "killed"]
    assert stop_result["name"] == "test_stop"
    assert stop_result["pid"] == pid


@pytest.mark.asyncio
async def test_process_status(process_adapter):
    """Test getting process status"""
    # Start process
    await process_adapter.execute("start", {
        "name": "test_status",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    time.sleep(0.5)
    
    # Get status
    status = await process_adapter.execute("status", {
        "name": "test_status"
    })
    
    assert status["status"] == "running"
    assert status["name"] == "test_status"
    assert "pid" in status
    assert "cpu_percent" in status
    assert "memory_mb" in status
    assert "uptime_seconds" in status


@pytest.mark.asyncio
async def test_process_restart(process_adapter):
    """Test restarting a process"""
    # Start
    start_result = await process_adapter.execute("start", {
        "name": "test_restart",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    original_pid = start_result["pid"]
    time.sleep(0.5)
    
    # Restart
    restart_result = await process_adapter.execute("restart", {
        "name": "test_restart"
    })
    
    assert restart_result["status"] == "restarted"
    assert restart_result["start"]["pid"] != original_pid


@pytest.mark.asyncio
async def test_process_not_found(process_adapter):
    """Test operations on non-existent process"""
    result = await process_adapter.execute("stop", {
        "name": "nonexistent_process"
    })
    
    assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_already_running(process_adapter):
    """Test starting already running process"""
    # Start first time
    await process_adapter.execute("start", {
        "name": "test_duplicate",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    time.sleep(0.5)
    
    # Try to start again
    result = await process_adapter.execute("start", {
        "name": "test_duplicate",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    assert result["status"] == "already_running"


@pytest.mark.asyncio
async def test_health_check(process_adapter):
    """Test health check functionality"""
    # Start process
    await process_adapter.execute("start", {
        "name": "test_health",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    time.sleep(0.5)
    
    # Health check
    health = await process_adapter.execute("health_check", {
        "name": "test_health"
    })
    
    assert health["healthy"] is True
    assert health["status"] in ["healthy", "running"]  # Both are valid
    assert "pid" in health


@pytest.mark.asyncio
async def test_status_all_processes(process_adapter):
    """Test getting status of all processes"""
    # Start multiple processes
    await process_adapter.execute("start", {
        "name": "test_all_1",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    await process_adapter.execute("start", {
        "name": "test_all_2",
        "command": "python -c \"import time; time.sleep(10)\""
    })
    
    time.sleep(0.5)
    
    # Get all status
    result = await process_adapter.execute("status", {})
    
    assert result["status"] == "ok"
    assert "processes" in result
    assert len(result["processes"]) >= 2


@pytest.mark.asyncio
async def test_force_kill(process_adapter):
    """Test force killing a process"""
    # Start long-running process
    await process_adapter.execute("start", {
        "name": "test_force",
        "command": "python -c \"import time; time.sleep(100)\""
    })
    
    time.sleep(0.5)
    
    # Force kill
    result = await process_adapter.execute("stop", {
        "name": "test_force",
        "force": True,
        "timeout": 1
    })
    
    assert result["status"] in ["stopped", "killed"]


@pytest.mark.asyncio
async def test_process_with_cwd(process_adapter):
    """Test starting process with custom working directory"""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")
        
        # Start process in that directory
        result = await process_adapter.execute("start", {
            "name": "test_cwd",
            "command": "python -c \"import os; print(os.getcwd())\"",
            "cwd": tmpdir
        })
        
        assert result["status"] == "started"
        
        time.sleep(0.5)
        
        # Cleanup
        await process_adapter.execute("stop", {
            "name": "test_cwd",
            "force": True
        })


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
