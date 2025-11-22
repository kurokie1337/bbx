"""Tests for workflow versioning."""
import pytest
from pathlib import Path
from blackbox.core.versioning.manager import VersionManager
from blackbox.core.versioning.models import WorkflowVersion

@pytest.fixture
def version_manager(tmp_path):
    return VersionManager(tmp_path)

def test_create_version(version_manager):
    """Test creating a workflow version."""
    content = {"workflow": {"id": "test", "steps": []}}

    version = version_manager.create_version(
        workflow_id="test",
        content=content,
        version="1.0.0",
        created_by="test_user",
        description="Initial version"
    )

    assert version.version == "1.0.0"
    assert version.workflow_id == "test"
    assert version.created_by == "test_user"

def test_list_versions(version_manager):
    """Test listing workflow versions."""
    content = {"workflow": {"id": "test", "steps": []}}

    version_manager.create_version("test", content, "1.0.0", "user1")
    version_manager.create_version("test", content, "1.0.1", "user1")
    version_manager.create_version("test", content, "1.1.0", "user1")

    versions = version_manager.list_versions("test")
    assert len(versions) == 3

def test_rollback(version_manager):
    """Test rolling back to a previous version."""
    content1 = {"workflow": {"id": "test", "steps": [{"id": "step1"}]}}
    content2 = {"workflow": {"id": "test", "steps": [{"id": "step2"}]}}

    version_manager.create_version("test", content1, "1.0.0", "user1")
    version_manager.create_version("test", content2, "1.1.0", "user1")

    rollback = version_manager.rollback("test", "1.0.0")
    assert rollback.parent_version == "1.0.0"
    assert rollback.content == content1

def test_diff(version_manager):
    """Test diff between versions."""
    content1 = {"workflow": {"id": "test", "steps": [{"id": "step1"}]}}
    content2 = {"workflow": {"id": "test", "steps": [{"id": "step2"}]}}

    version_manager.create_version("test", content1, "1.0.0", "user1")
    version_manager.create_version("test", content2, "1.1.0", "user1")

    diff = version_manager.diff("test", "1.0.0", "1.1.0")
    assert "step1" in diff.removed_steps
    assert "step2" in diff.added_steps
