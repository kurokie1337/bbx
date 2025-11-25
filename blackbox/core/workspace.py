# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Workspace System

Workspaces provide isolated environments for AI agents, similar to
/home/user directories in Linux. Each workspace contains:
- main.bbx: Entry point workflow (created empty)
- config.yaml: Local configuration
- state/: Persistent state storage
- logs/: Execution logs and history
- workflows/: Sub-workflows
- data/: Project data

This enables agents to:
- Work in isolated environments
- Maintain persistent state between sessions
- Track execution history
- Organize complex multi-workflow projects
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbx.workspace")

# Default main.bbx template (empty workflow for agent to fill)
DEFAULT_MAIN_BBX = """# BBX Workspace Entry Point
# This is your main workflow. Edit this file to define your automation.

id: main
name: Main Workflow
version: "1.0.0"
description: "Workspace entry point - customize this workflow"

# Define inputs for your workflow
inputs: {}

# Define your workflow steps
steps:
  start:
    use: logger.info
    args:
      message: "Workflow started - add your steps here"
"""

# Default workspace config template
DEFAULT_WORKSPACE_CONFIG = """# BBX Workspace Configuration

workspace:
  name: "{name}"
  description: "{description}"
  created_at: "{created_at}"

runtime:
  default_timeout_ms: 30000
  max_parallel_steps: 10
  enable_caching: true

observability:
  log_level: INFO
  logs_retention: 100  # max run logs to keep

security:
  sandbox_mode: false
"""


@dataclass
class WorkspacePaths:
    """Paths within a workspace"""
    root: Path
    main_bbx: Path
    config: Path
    state_dir: Path
    logs_dir: Path
    runs_dir: Path
    workflows_dir: Path
    data_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "WorkspacePaths":
        """Create WorkspacePaths from root directory"""
        return cls(
            root=root,
            main_bbx=root / "main.bbx",
            config=root / "config.yaml",
            state_dir=root / "state",
            logs_dir=root / "logs",
            runs_dir=root / "logs" / "runs",
            workflows_dir=root / "workflows",
            data_dir=root / "data",
        )


@dataclass
class ExecutionRecord:
    """Record of a single workflow execution"""
    run_id: str
    workflow_file: str
    status: str  # pending, running, completed, failed, cancelled
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "run_id": self.run_id,
            "workflow_file": self.workflow_file,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "step_results": self.step_results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionRecord":
        """Create from dictionary"""
        return cls(
            run_id=data["run_id"],
            workflow_file=data["workflow_file"],
            status=data["status"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            duration_ms=data.get("duration_ms"),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            error=data.get("error"),
            step_results=data.get("step_results", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionRecord":
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))


@dataclass
class WorkspaceMetadata:
    """Workspace metadata stored in .workspace.json"""
    workspace_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceMetadata":
        return cls(
            workspace_id=data["workspace_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", "1.0.0"),
        )


class Workspace:
    """
    Represents a BBX workspace - an isolated environment for AI agents.

    A workspace is like a Linux home directory for an AI agent:
    - Contains all project files
    - Maintains persistent state
    - Tracks execution history
    - Provides isolation from other projects

    Usage:
        # Create new workspace
        ws = Workspace.create("/path/to/my-project", name="My Project")

        # Load existing workspace
        ws = Workspace.load("/path/to/my-project")

        # Run main workflow
        result = await ws.run()

        # Get execution history
        history = ws.get_execution_history(limit=10)
    """

    def __init__(self, root: Path, metadata: WorkspaceMetadata):
        self.root = root
        self.metadata = metadata
        self.paths = WorkspacePaths.from_root(root)
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all workspace directories exist"""
        directories = [
            self.paths.state_dir,
            self.paths.logs_dir,
            self.paths.runs_dir,
            self.paths.workflows_dir,
            self.paths.data_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        root: Path,
        name: str,
        description: str = "",
        template: Optional[str] = None,
    ) -> "Workspace":
        """
        Create a new workspace.

        Args:
            root: Root directory for the workspace
            name: Human-readable name
            description: Optional description
            template: Optional template name (future: agent, pipeline, etc.)

        Returns:
            New Workspace instance
        """
        root = Path(root)

        if root.exists() and any(root.iterdir()):
            raise ValueError(f"Directory {root} already exists and is not empty")

        # Create root directory
        root.mkdir(parents=True, exist_ok=True)

        # Create metadata
        now = datetime.now()
        metadata = WorkspaceMetadata(
            workspace_id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )

        # Create workspace instance
        workspace = cls(root, metadata)

        # Write metadata file
        metadata_file = root / ".workspace.json"
        metadata_file.write_text(json.dumps(metadata.to_dict(), indent=2))

        # Write main.bbx (empty template for agent)
        workspace.paths.main_bbx.write_text(DEFAULT_MAIN_BBX)

        # Write config.yaml
        config_content = DEFAULT_WORKSPACE_CONFIG.format(
            name=name,
            description=description,
            created_at=now.isoformat(),
        )
        workspace.paths.config.write_text(config_content)

        # Create .gitignore
        gitignore = root / ".gitignore"
        gitignore.write_text(
            "# BBX Workspace\n"
            "state/\n"
            "logs/\n"
            "data/cache/\n"
            "*.pyc\n"
            "__pycache__/\n"
            ".env\n"
        )

        # Create empty state files
        (workspace.paths.state_dir / "vars.json").write_text("{}")

        logger.info(f"Created workspace: {name} at {root}")
        return workspace

    @classmethod
    def load(cls, root: Path) -> "Workspace":
        """
        Load an existing workspace.

        Args:
            root: Root directory of the workspace

        Returns:
            Workspace instance

        Raises:
            ValueError: If directory is not a valid workspace
        """
        root = Path(root)

        if not root.exists():
            raise ValueError(f"Workspace directory does not exist: {root}")

        metadata_file = root / ".workspace.json"
        if not metadata_file.exists():
            raise ValueError(f"Not a valid workspace (missing .workspace.json): {root}")

        try:
            metadata_data = json.loads(metadata_file.read_text())
            metadata = WorkspaceMetadata.from_dict(metadata_data)
        except Exception as e:
            raise ValueError(f"Failed to load workspace metadata: {e}")

        return cls(root, metadata)

    @classmethod
    def is_workspace(cls, path: Path) -> bool:
        """Check if a directory is a valid workspace"""
        return (Path(path) / ".workspace.json").exists()

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate workspace integrity.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if not self.paths.main_bbx.exists():
            issues.append("Missing main.bbx")

        if not self.paths.config.exists():
            issues.append("Missing config.yaml")

        if not self.paths.state_dir.exists():
            issues.append("Missing state directory")

        return (len(issues) == 0, issues)

    # ==========================================================================
    # State Management
    # ==========================================================================

    def get_state_file(self) -> Path:
        """Get path to state vars file"""
        return self.paths.state_dir / "vars.json"

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state variable"""
        state_file = self.get_state_file()
        if not state_file.exists():
            return default

        try:
            state = json.loads(state_file.read_text())
            return state.get(key, default)
        except Exception:
            return default

    def set_state(self, key: str, value: Any) -> None:
        """Set a state variable"""
        state_file = self.get_state_file()

        if state_file.exists():
            state = json.loads(state_file.read_text())
        else:
            state = {}

        state[key] = value
        state_file.write_text(json.dumps(state, indent=2, default=str))

    def get_all_state(self) -> Dict[str, Any]:
        """Get all state variables"""
        state_file = self.get_state_file()
        if not state_file.exists():
            return {}

        try:
            return json.loads(state_file.read_text())
        except Exception:
            return {}

    def clear_state(self) -> None:
        """Clear all state variables"""
        state_file = self.get_state_file()
        state_file.write_text("{}")

    # ==========================================================================
    # Execution History
    # ==========================================================================

    def save_execution(self, record: ExecutionRecord) -> None:
        """Save an execution record"""
        run_file = self.paths.runs_dir / f"{record.run_id}.json"
        run_file.write_text(record.to_json())

        # Update metadata
        self.metadata.updated_at = datetime.now()
        metadata_file = self.root / ".workspace.json"
        metadata_file.write_text(json.dumps(self.metadata.to_dict(), indent=2))

    def get_execution(self, run_id: str) -> Optional[ExecutionRecord]:
        """Get a specific execution record"""
        run_file = self.paths.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            return None

        try:
            return ExecutionRecord.from_json(run_file.read_text())
        except Exception as e:
            logger.error(f"Failed to load execution {run_id}: {e}")
            return None

    def get_execution_history(self, limit: int = 10) -> List[ExecutionRecord]:
        """Get recent execution history"""
        runs = []

        if not self.paths.runs_dir.exists():
            return runs

        # Get all run files sorted by modification time (newest first)
        run_files = sorted(
            self.paths.runs_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        for run_file in run_files[:limit]:
            try:
                record = ExecutionRecord.from_json(run_file.read_text())
                runs.append(record)
            except Exception as e:
                logger.warning(f"Failed to load run {run_file}: {e}")

        return runs

    def get_last_execution(self) -> Optional[ExecutionRecord]:
        """Get the most recent execution"""
        history = self.get_execution_history(limit=1)
        return history[0] if history else None

    def cleanup_old_runs(self, keep: int = 100) -> int:
        """
        Clean up old execution records.

        Args:
            keep: Number of recent runs to keep

        Returns:
            Number of runs deleted
        """
        if not self.paths.runs_dir.exists():
            return 0

        run_files = sorted(
            self.paths.runs_dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        deleted = 0
        for run_file in run_files[keep:]:
            try:
                run_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {run_file}: {e}")

        return deleted

    # ==========================================================================
    # Workflow Discovery
    # ==========================================================================

    def list_workflows(self) -> List[Path]:
        """List all workflow files in the workspace"""
        workflows = [self.paths.main_bbx]

        if self.paths.workflows_dir.exists():
            workflows.extend(self.paths.workflows_dir.rglob("*.bbx"))

        return [w for w in workflows if w.exists()]

    def get_workflow_path(self, name: str) -> Path:
        """
        Get path to a workflow by name.

        Args:
            name: Workflow name (without .bbx extension)
                  Use "main" for main.bbx
                  Use "workflows/subname" for nested workflows

        Returns:
            Path to the workflow file
        """
        if name == "main":
            return self.paths.main_bbx

        # Check if it's a relative path
        if "/" in name or "\\" in name:
            path = self.root / name
            if not path.suffix:
                path = path.with_suffix(".bbx")
            return path

        # Check in workflows directory
        path = self.paths.workflows_dir / f"{name}.bbx"
        return path

    # ==========================================================================
    # Info & Display
    # ==========================================================================

    def info(self) -> Dict[str, Any]:
        """Get workspace information"""
        valid, issues = self.validate()

        return {
            "workspace_id": self.metadata.workspace_id,
            "name": self.metadata.name,
            "description": self.metadata.description,
            "root": str(self.root),
            "created_at": self.metadata.created_at.isoformat(),
            "updated_at": self.metadata.updated_at.isoformat(),
            "valid": valid,
            "issues": issues,
            "workflows_count": len(self.list_workflows()),
            "runs_count": len(list(self.paths.runs_dir.glob("*.json"))) if self.paths.runs_dir.exists() else 0,
            "state_keys": list(self.get_all_state().keys()),
        }

    def __repr__(self) -> str:
        return f"Workspace(name={self.metadata.name!r}, root={self.root})"
