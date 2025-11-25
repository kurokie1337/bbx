# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Workspace Manager

Manages workspace lifecycle:
- Create new workspaces
- Load existing workspaces
- List available workspaces
- Set/get current workspace context
- Workspace templates

Think of this as the "user management" system in Linux,
but for AI agent workspaces.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .workspace import Workspace, WorkspaceMetadata

logger = logging.getLogger("bbx.workspace_manager")


class WorkspaceManager:
    """
    Manages BBX workspaces.

    Workspaces are stored in:
    - ~/.bbx/workspaces/ (default)
    - Or any custom directory

    Current workspace context is stored in:
    - ~/.bbx/current_workspace.json

    Usage:
        manager = WorkspaceManager()

        # Create new workspace
        ws = manager.create("my-project", description="My AI project")

        # List all workspaces
        for ws_info in manager.list():
            print(ws_info["name"])

        # Set current workspace
        manager.set_current("/path/to/my-project")

        # Get current workspace
        ws = manager.get_current()
    """

    def __init__(self, workspaces_root: Optional[Path] = None):
        """
        Initialize WorkspaceManager.

        Args:
            workspaces_root: Root directory for workspaces.
                           Defaults to ~/.bbx/workspaces/
        """
        config = get_config()

        if workspaces_root:
            self.workspaces_root = Path(workspaces_root)
        else:
            self.workspaces_root = config.paths.bbx_home / "workspaces"

        self.workspaces_root.mkdir(parents=True, exist_ok=True)

        # Current workspace state file
        self._current_workspace_file = config.paths.bbx_home / "current_workspace.json"

    # ==========================================================================
    # Workspace Creation
    # ==========================================================================

    def create(
        self,
        name: str,
        description: str = "",
        path: Optional[Path] = None,
        template: Optional[str] = None,
    ) -> Workspace:
        """
        Create a new workspace.

        Args:
            name: Workspace name (used as directory name if path not specified)
            description: Optional description
            path: Custom path for workspace (default: workspaces_root/name)
            template: Template to use (future: "agent", "pipeline", "basic")

        Returns:
            New Workspace instance
        """
        # Determine workspace path
        if path:
            workspace_path = Path(path)
        else:
            # Sanitize name for directory
            safe_name = self._sanitize_name(name)
            workspace_path = self.workspaces_root / safe_name

        # Check if already exists
        if workspace_path.exists() and any(workspace_path.iterdir()):
            raise ValueError(f"Workspace already exists at {workspace_path}")

        # Create workspace
        workspace = Workspace.create(
            root=workspace_path,
            name=name,
            description=description,
            template=template,
        )

        logger.info(f"Created workspace '{name}' at {workspace_path}")
        return workspace

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as directory name"""
        # Replace spaces and special chars with underscores
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return safe.lower()

    # ==========================================================================
    # Workspace Loading
    # ==========================================================================

    def load(self, path: Path) -> Workspace:
        """
        Load a workspace from path.

        Args:
            path: Path to workspace root

        Returns:
            Workspace instance
        """
        return Workspace.load(Path(path))

    def load_by_name(self, name: str) -> Optional[Workspace]:
        """
        Load a workspace by name from the default workspaces directory.

        Args:
            name: Workspace name

        Returns:
            Workspace if found, None otherwise
        """
        safe_name = self._sanitize_name(name)
        path = self.workspaces_root / safe_name

        if not path.exists():
            # Try exact name
            path = self.workspaces_root / name

        if not path.exists():
            return None

        try:
            return Workspace.load(path)
        except ValueError:
            return None

    # ==========================================================================
    # Workspace Discovery
    # ==========================================================================

    def list(self, directory: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        List all workspaces.

        Args:
            directory: Directory to search in (default: workspaces_root)

        Returns:
            List of workspace info dictionaries
        """
        search_dir = Path(directory) if directory else self.workspaces_root
        workspaces = []

        if not search_dir.exists():
            return workspaces

        for item in search_dir.iterdir():
            if item.is_dir() and Workspace.is_workspace(item):
                try:
                    ws = Workspace.load(item)
                    workspaces.append({
                        "name": ws.metadata.name,
                        "path": str(item),
                        "workspace_id": ws.metadata.workspace_id,
                        "description": ws.metadata.description,
                        "created_at": ws.metadata.created_at.isoformat(),
                        "updated_at": ws.metadata.updated_at.isoformat(),
                    })
                except Exception as e:
                    logger.warning(f"Failed to load workspace at {item}: {e}")

        # Sort by updated_at (most recent first)
        workspaces.sort(key=lambda x: x["updated_at"], reverse=True)
        return workspaces

    def find(self, name_or_id: str) -> Optional[Workspace]:
        """
        Find a workspace by name or ID.

        Args:
            name_or_id: Workspace name or workspace_id

        Returns:
            Workspace if found, None otherwise
        """
        for ws_info in self.list():
            if ws_info["name"] == name_or_id or ws_info["workspace_id"] == name_or_id:
                return Workspace.load(Path(ws_info["path"]))
        return None

    def exists(self, name: str) -> bool:
        """Check if a workspace with given name exists"""
        safe_name = self._sanitize_name(name)
        return (self.workspaces_root / safe_name).exists()

    # ==========================================================================
    # Current Workspace Context
    # ==========================================================================

    def set_current(self, path: Path) -> Workspace:
        """
        Set the current workspace.

        This stores the workspace path so that subsequent commands
        can operate in the context of this workspace.

        Args:
            path: Path to workspace

        Returns:
            The workspace that was set as current
        """
        path = Path(path).resolve()

        if not Workspace.is_workspace(path):
            raise ValueError(f"Not a valid workspace: {path}")

        workspace = Workspace.load(path)

        # Save current workspace reference
        self._current_workspace_file.write_text(json.dumps({
            "path": str(path),
            "workspace_id": workspace.metadata.workspace_id,
            "name": workspace.metadata.name,
        }, indent=2))

        logger.info(f"Set current workspace to: {workspace.metadata.name}")
        return workspace

    def get_current(self) -> Optional[Workspace]:
        """
        Get the current workspace.

        Returns:
            Current Workspace or None if not set
        """
        if not self._current_workspace_file.exists():
            return None

        try:
            data = json.loads(self._current_workspace_file.read_text())
            path = Path(data["path"])

            if not path.exists():
                logger.warning(f"Current workspace path no longer exists: {path}")
                return None

            return Workspace.load(path)
        except Exception as e:
            logger.error(f"Failed to load current workspace: {e}")
            return None

    def get_current_path(self) -> Optional[Path]:
        """Get path to current workspace without loading it"""
        if not self._current_workspace_file.exists():
            return None

        try:
            data = json.loads(self._current_workspace_file.read_text())
            return Path(data["path"])
        except Exception:
            return None

    def clear_current(self) -> None:
        """Clear the current workspace setting"""
        if self._current_workspace_file.exists():
            self._current_workspace_file.unlink()
            logger.info("Cleared current workspace")

    def auto_detect_workspace(self, start_path: Optional[Path] = None) -> Optional[Workspace]:
        """
        Auto-detect workspace by walking up directory tree.

        Looks for .workspace.json starting from start_path (or cwd)
        and walking up to root.

        Args:
            start_path: Starting path (default: current working directory)

        Returns:
            Workspace if found, None otherwise
        """
        path = Path(start_path) if start_path else Path.cwd()

        while path != path.parent:
            if Workspace.is_workspace(path):
                return Workspace.load(path)
            path = path.parent

        # Check root
        if Workspace.is_workspace(path):
            return Workspace.load(path)

        return None

    # ==========================================================================
    # Workspace Deletion
    # ==========================================================================

    def delete(self, path: Path, force: bool = False) -> bool:
        """
        Delete a workspace.

        Args:
            path: Path to workspace
            force: If True, delete even if not empty

        Returns:
            True if deleted, False otherwise
        """
        import shutil

        path = Path(path)

        if not Workspace.is_workspace(path):
            raise ValueError(f"Not a valid workspace: {path}")

        # Check if it's the current workspace
        current_path = self.get_current_path()
        if current_path and current_path.resolve() == path.resolve():
            self.clear_current()

        try:
            if force:
                shutil.rmtree(path)
            else:
                # Only delete if workspace structure files exist
                # but no user data
                ws = Workspace.load(path)
                workflows = ws.list_workflows()

                # Check if only default main.bbx exists
                if len(workflows) > 1:
                    raise ValueError(
                        f"Workspace has {len(workflows)} workflows. "
                        "Use force=True to delete."
                    )

                shutil.rmtree(path)

            logger.info(f"Deleted workspace at {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete workspace: {e}")
            return False

    # ==========================================================================
    # Workspace Info
    # ==========================================================================

    def info(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Get information about a workspace or the current workspace.

        Args:
            path: Path to workspace (default: current workspace)

        Returns:
            Workspace information dictionary
        """
        if path:
            ws = Workspace.load(Path(path))
        else:
            ws = self.get_current()
            if not ws:
                raise ValueError("No current workspace set. Use 'bbx workspace set <path>'")

        return ws.info()


# ==========================================================================
# Global Workspace Manager Instance
# ==========================================================================

_manager: Optional[WorkspaceManager] = None


def get_workspace_manager() -> WorkspaceManager:
    """Get global WorkspaceManager instance"""
    global _manager
    if _manager is None:
        _manager = WorkspaceManager()
    return _manager


def get_current_workspace() -> Optional[Workspace]:
    """Convenience function to get current workspace"""
    return get_workspace_manager().get_current()


def require_workspace() -> Workspace:
    """
    Get current workspace or raise error.

    Use this in code that requires a workspace context.
    """
    ws = get_current_workspace()
    if ws is None:
        raise ValueError(
            "No workspace context. Either:\n"
            "  1. Run 'bbx workspace create <name>' to create a new workspace\n"
            "  2. Run 'bbx workspace set <path>' to set current workspace\n"
            "  3. Navigate to a directory containing a workspace"
        )
    return ws
