# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""Workflow version management."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .models import VersionDiff, WorkflowVersion


class VersionManager:
    """Manages workflow versions and history."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        workflow_id: str,
        content: dict,
        version: str,
        created_by: str,
        description: Optional[str] = None,
    ) -> WorkflowVersion:
        """Create a new workflow version."""
        version_obj = WorkflowVersion(
            workflow_id=workflow_id,
            version=version,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            content=content,
        )

        # Save to storage
        self._save_version(version_obj)
        return version_obj

    def get_version(self, workflow_id: str, version: str) -> Optional[WorkflowVersion]:
        """Retrieve specific version."""
        version_file = self.storage_path / workflow_id / f"{version}.json"
        if not version_file.exists():
            return None

        with open(version_file) as f:
            data = json.load(f)
        return WorkflowVersion(**data)

    def list_versions(self, workflow_id: str) -> List[WorkflowVersion]:
        """List all versions of a workflow."""
        workflow_dir = self.storage_path / workflow_id
        if not workflow_dir.exists():
            return []

        versions = []
        for version_file in workflow_dir.glob("*.json"):
            with open(version_file) as f:
                data = json.load(f)
            versions.append(WorkflowVersion(**data))

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def rollback(self, workflow_id: str, target_version: str) -> WorkflowVersion:
        """Rollback to a previous version."""
        target = self.get_version(workflow_id, target_version)
        if not target:
            raise ValueError(f"Version {target_version} not found")

        # Create new version based on target
        current_versions = self.list_versions(workflow_id)
        next_version = self._increment_version(current_versions[0].version)

        rollback_version = self.create_version(
            workflow_id=workflow_id,
            content=target.content,
            version=next_version,
            created_by="system",
            description=f"Rollback to version {target_version}",
        )
        rollback_version.parent_version = target_version
        self._save_version(rollback_version)

        return rollback_version

    def diff(self, workflow_id: str, from_version: str, to_version: str) -> VersionDiff:
        """Generate diff between two versions."""
        v1 = self.get_version(workflow_id, from_version)
        v2 = self.get_version(workflow_id, to_version)

        if not v1 or not v2:
            raise ValueError("One or both versions not found")

        steps1 = {s["id"]: s for s in v1.content.get("workflow", {}).get("steps", [])}
        steps2 = {s["id"]: s for s in v2.content.get("workflow", {}).get("steps", [])}

        added = list(set(steps2.keys()) - set(steps1.keys()))
        removed = list(set(steps1.keys()) - set(steps2.keys()))
        modified = [
            sid
            for sid in set(steps1.keys()) & set(steps2.keys())
            if steps1[sid] != steps2[sid]
        ]

        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            added_steps=added,
            removed_steps=removed,
            modified_steps=modified,
            changes=self._detailed_diff(v1.content, v2.content),
        )

    def _save_version(self, version: WorkflowVersion):
        """Save version to disk."""
        workflow_dir = self.storage_path / version.workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=True)

        version_file = workflow_dir / f"{version.version}.json"
        with open(version_file, "w") as f:
            json.dump(version.dict(), f, indent=2, default=str)

    def _increment_version(self, version: str) -> str:
        """Increment semantic version."""
        major, minor, patch = map(int, version.split("."))
        return f"{major}.{minor}.{patch + 1}"

    def _detailed_diff(self, d1: dict, d2: dict) -> dict:
        """Generate detailed diff between two dicts."""
        # Simplified implementation
        return {"summary": "Changes detected"}
