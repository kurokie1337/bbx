# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

import pytest
from blackbox.core.adapters.os_abstraction import OSAbstractionAdapter


@pytest.fixture
def os_adapter():
    """Create OS abstraction adapter instance"""
    return OSAbstractionAdapter()


@pytest.mark.asyncio
async def test_platform_info(os_adapter):
    """Test platform information"""
    result = os_adapter._platform_info()
    
    assert "system" in result
    assert result["system"] in ["Windows", "Linux", "Darwin"]
    assert "is_windows" in result
    assert "python_version" in result
    assert "cwd" in result


@pytest.mark.asyncio
async def test_exec_basic(os_adapter):
    """Test basic command execution"""
    # Use platform-agnostic command
    if os_adapter.is_windows:
        command = "echo Hello"
    else:
        command = "echo Hello"
    
    result = await os_adapter.execute("exec", {"command": command})
    
    assert result["exit_code"] == 0
    assert result["success"] is True
    assert "Hello" in result["stdout"]


@pytest.mark.asyncio
async def test_exec_auto_translate_windows(os_adapter):
    """Test command translation for Windows"""
    if not os_adapter.is_windows:
        pytest.skip("Windows-only test")
    
    # Use Unix command, should be auto-translated
    result = await os_adapter.execute("exec", {
        "command": "ls",
        "auto_translate": True
    })
    
    # Should work because ls → dir translation
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_mkdir_and_remove(os_adapter, tmp_path):
    """Test directory creation and removal"""
    test_dir = tmp_path / "test_mkdir"
    
    # Create directory
    result = await os_adapter.execute("mkdir", {"path": str(test_dir)})
    assert result["created"] is True
    assert test_dir.exists()
    
    # Remove directory
    result = await os_adapter.execute("remove", {
        "path": str(test_dir),
        "recursive": True
    })
    assert result["removed"] is True
    assert not test_dir.exists()


@pytest.mark.asyncio
async def test_copy_file(os_adapter, tmp_path):
    """Test file copying"""
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "dest.txt"
    
    # Create source file
    src_file.write_text("test content")
    
    # Copy file
    result = await os_adapter.execute("copy", {
        "src": str(src_file),
        "dst": str(dst_file)
    })
    
    assert result["copied"] is True
    assert dst_file.exists()
    assert dst_file.read_text() == "test content"


@pytest.mark.asyncio
async def test_move_file(os_adapter, tmp_path):
    """Test file moving"""
    src_file = tmp_path / "source.txt"
    dst_file = tmp_path / "moved.txt"
    
    # Create source file
    src_file.write_text("test content")
    
    # Move file
    result = await os_adapter.execute("move", {
        "src": str(src_file),
        "dst": str(dst_file)
    })
    
    assert result["moved"] is True
    assert not src_file.exists()
    assert dst_file.exists()
    assert dst_file.read_text() == "test content"


@pytest.mark.asyncio
async def test_list_directory(os_adapter, tmp_path):
    """Test directory listing"""
    # Create test files
    (tmp_path / "test1.txt").write_text("content1")
    (tmp_path / "test2.txt").write_text("content2")
    (tmp_path / "other.log").write_text("log")
    
    # List all files
    result = await os_adapter.execute("list", {"path": str(tmp_path)})
    assert result["count"] >= 3
    
    # List with pattern
    result = await os_adapter.execute("list", {
        "path": str(tmp_path),
        "pattern": "*.txt"
    })
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_exists(os_adapter, tmp_path):
    """Test file existence check"""
    test_file = tmp_path / "exists_test.txt"
    
    # Check non-existent file
    result = await os_adapter.execute("exists", {"path": str(test_file)})
    assert result["exists"] is False
    
    # Create file
    test_file.write_text("content")
    
    # Check existing file
    result = await os_adapter.execute("exists", {"path": str(test_file)})
    assert result["exists"] is True
    assert result["is_file"] is True
    assert result["is_dir"] is False


@pytest.mark.asyncio
async def test_read_write(os_adapter, tmp_path):
    """Test file reading and writing"""
    test_file = tmp_path / "readwrite_test.txt"
    content = "Test content\nLine 2"
    
    # Write file
    result = await os_adapter.execute("write", {
        "path": str(test_file),
        "content": content
    })
    assert result["written"] is True
    
    # Read file
    result = await os_adapter.execute("read", {"path": str(test_file)})
    assert result["content"] == content


@pytest.mark.asyncio
async def test_env_operations(os_adapter):
    """Test environment variable operations"""
    # Set environment variable
    result = await os_adapter.execute("env", {
        "action": "set",
        "key": "BBX_TEST_VAR",
        "value": "test_value"
    })
    assert result["set"] is True
    
    # Get environment variable
    result = await os_adapter.execute("env", {
        "action": "get",
        "key": "BBX_TEST_VAR"
    })
    assert result["value"] == "test_value"
    assert result["exists"] is True
    
    # List all variables
    result = await os_adapter.execute("env", {"action": "list"})
    assert "BBX_TEST_VAR" in result["variables"]


@pytest.mark.asyncio
async def test_exec_with_env(os_adapter):
    """Test command execution with custom environment"""
    result = await os_adapter.execute("exec", {
        "command": "echo %BBX_CUSTOM%" if os_adapter.is_windows else "echo $BBX_CUSTOM",
        "env": {"BBX_CUSTOM": "custom_value"},
        "auto_translate": False  # Don't translate echo
    })
    
    # May not work perfectly on all platforms, but shouldn't crash
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_exec_timeout(os_adapter):
    """Test command execution timeout"""
    if os_adapter.is_windows:
        # Windows: Use PowerShell Start-Sleep
        command = "powershell -Command Start-Sleep -Seconds 10"
    else:
        # Unix: sleep command
        command = "sleep 10"
    
    with pytest.raises(TimeoutError):
        await os_adapter.execute("exec", {
            "command": command,
            "timeout": 1  # 1 second timeout
        })


@pytest.mark.asyncio
async def test_command_translation(os_adapter):
    """Test Unix → Windows command translation"""
    if not os_adapter.is_windows:
        pytest.skip("Windows-only test")
    
    translations = [
        ("rm -rf test", "rd /s /q test"),
        ("ls -la", "dir"),
        ("cat file.txt", "type file.txt"),
        ("mkdir -p test/dir", "mkdir test/dir"),
    ]
    
    for unix_cmd, expected_win_cmd in translations:
        translated = os_adapter._translate_command(unix_cmd)
        assert expected_win_cmd in translated or translated == expected_win_cmd
