# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""Workflow versioning data models."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowVersion(BaseModel):
    """Represents a single version of a workflow."""

    workflow_id: str
    version: str  # Semantic version: 1.0.0
    created_at: datetime
    created_by: str
    description: Optional[str] = None
    content: Dict[str, Any]  # Full BBX YAML content
    parent_version: Optional[str] = None
    is_published: bool = False
    tags: List[str] = Field(default_factory=list)


class VersionHistory(BaseModel):
    """Version history for a workflow."""

    workflow_id: str
    versions: List[WorkflowVersion]
    current_version: str


class VersionDiff(BaseModel):
    """Diff between two workflow versions."""

    from_version: str
    to_version: str
    added_steps: List[str]
    removed_steps: List[str]
    modified_steps: List[str]
    changes: Dict[str, Any]
