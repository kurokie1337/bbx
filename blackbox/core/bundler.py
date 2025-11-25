# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0
# BBX Bundler - Pack and Unpack Workflows

"""
BBX Bundler

Creates self-contained workflow packages (.bbxpkg) that include:
- Workflow files (.bbx)
- Config files (config.yaml)
- State data (optional)
- Dependencies (other workflows)
- Metadata

Usage:
    bbx pack deploy.bbx -o deploy.bbxpkg
    bbx unpack deploy.bbxpkg -d ./output
"""

import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class BBXBundler:
    """
    BBX Workflow Bundler.

    Creates and extracts self-contained workflow packages.
    """

    MANIFEST_FILE = "manifest.json"
    WORKFLOWS_DIR = "workflows"
    CONFIG_DIR = "config"
    STATE_DIR = "state"
    ADAPTERS_DIR = "adapters"

    def __init__(self):
        self._collected_workflows: Set[str] = set()
        self._collected_configs: Set[str] = set()

    def pack(
        self,
        workflow_path: str,
        output_path: Optional[str] = None,
        include_state: bool = False,
        include_deps: bool = True,
        workspace_path: Optional[str] = None,
    ) -> str:
        """
        Pack a workflow into a .bbxpkg file.

        Args:
            workflow_path: Path to main .bbx workflow file
            output_path: Output .bbxpkg path (default: workflow_name.bbxpkg)
            include_state: Include state data
            include_deps: Include dependent workflows
            workspace_path: Workspace root for resolving deps

        Returns:
            Path to created .bbxpkg file
        """
        workflow_path = Path(workflow_path).resolve()

        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_path}")

        # Determine output path
        if not output_path:
            output_path = workflow_path.with_suffix(".bbxpkg")
        output_path = Path(output_path)

        # Parse workflow
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow_data = yaml.safe_load(f)

        if "workflow" in workflow_data:
            workflow = workflow_data["workflow"]
        else:
            workflow = workflow_data

        # Collect all files
        self._collected_workflows = set()
        self._collected_configs = set()

        files_to_pack: Dict[str, Path] = {}

        # Add main workflow
        main_name = workflow_path.name
        files_to_pack[f"{self.WORKFLOWS_DIR}/{main_name}"] = workflow_path
        self._collected_workflows.add(str(workflow_path))

        # Collect dependencies
        if include_deps:
            deps = self._collect_dependencies(workflow, workflow_path.parent, workspace_path)
            for dep_name, dep_path in deps.items():
                files_to_pack[f"{self.WORKFLOWS_DIR}/{dep_name}"] = dep_path

        # Collect config files
        config_files = self._collect_configs(workflow_path.parent, workspace_path)
        for cfg_name, cfg_path in config_files.items():
            files_to_pack[f"{self.CONFIG_DIR}/{cfg_name}"] = cfg_path

        # Collect state (optional)
        state_files = {}
        if include_state:
            state_files = self._collect_state(workflow_path.parent, workspace_path)
            for state_name, state_path in state_files.items():
                files_to_pack[f"{self.STATE_DIR}/{state_name}"] = state_path

        # Create manifest
        manifest = self._create_manifest(
            workflow_path=workflow_path,
            workflow=workflow,
            files=files_to_pack,
            include_state=include_state,
        )

        # Create package
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write manifest
            manifest_path = tmpdir_path / self.MANIFEST_FILE
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)

            # Copy files
            for archive_path, source_path in files_to_pack.items():
                dest = tmpdir_path / archive_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest)

            # Create tarball
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(tmpdir_path, arcname=".")

        return str(output_path)

    def unpack(
        self,
        package_path: str,
        output_dir: Optional[str] = None,
        restore_state: bool = False,
    ) -> Dict[str, Any]:
        """
        Unpack a .bbxpkg file.

        Args:
            package_path: Path to .bbxpkg file
            output_dir: Output directory (default: current dir)
            restore_state: Restore state data

        Returns:
            Dict with unpacking results
        """
        package_path = Path(package_path)

        if not package_path.exists():
            raise FileNotFoundError(f"Package not found: {package_path}")

        # Determine output directory
        if not output_dir:
            output_dir = Path.cwd()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "package": str(package_path),
            "output_dir": str(output_dir),
            "workflows": [],
            "configs": [],
            "state_restored": False,
            "manifest": None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Extract tarball
            with tarfile.open(package_path, "r:gz") as tar:
                tar.extractall(tmpdir_path)

            # Read manifest
            manifest_path = tmpdir_path / self.MANIFEST_FILE
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                result["manifest"] = manifest
            else:
                manifest = {}

            # Copy workflows
            workflows_dir = tmpdir_path / self.WORKFLOWS_DIR
            if workflows_dir.exists():
                for wf_file in workflows_dir.glob("**/*.bbx"):
                    rel_path = wf_file.relative_to(workflows_dir)
                    dest = output_dir / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(wf_file, dest)
                    result["workflows"].append(str(rel_path))

            # Copy configs
            config_dir = tmpdir_path / self.CONFIG_DIR
            if config_dir.exists():
                for cfg_file in config_dir.glob("**/*"):
                    if cfg_file.is_file():
                        rel_path = cfg_file.relative_to(config_dir)
                        dest = output_dir / rel_path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(cfg_file, dest)
                        result["configs"].append(str(rel_path))

            # Restore state (optional)
            if restore_state:
                state_dir = tmpdir_path / self.STATE_DIR
                if state_dir.exists():
                    dest_state = output_dir / "state"
                    dest_state.mkdir(parents=True, exist_ok=True)
                    for state_file in state_dir.glob("**/*"):
                        if state_file.is_file():
                            rel_path = state_file.relative_to(state_dir)
                            dest = dest_state / rel_path
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(state_file, dest)
                    result["state_restored"] = True

        return result

    def info(self, package_path: str) -> Dict[str, Any]:
        """
        Get information about a .bbxpkg file without extracting.

        Args:
            package_path: Path to .bbxpkg file

        Returns:
            Package metadata
        """
        package_path = Path(package_path)

        if not package_path.exists():
            raise FileNotFoundError(f"Package not found: {package_path}")

        result = {
            "package": str(package_path),
            "size": package_path.stat().st_size,
            "size_human": self._human_size(package_path.stat().st_size),
            "manifest": None,
            "files": [],
        }

        with tarfile.open(package_path, "r:gz") as tar:
            result["files"] = [m.name for m in tar.getmembers()]

            # Try to read manifest
            try:
                manifest_member = tar.getmember(f"./{self.MANIFEST_FILE}")
                manifest_file = tar.extractfile(manifest_member)
                if manifest_file:
                    result["manifest"] = json.load(manifest_file)
            except (KeyError, json.JSONDecodeError):
                pass

        return result

    def _collect_dependencies(
        self,
        workflow: Dict,
        base_dir: Path,
        workspace_path: Optional[str],
    ) -> Dict[str, Path]:
        """Collect dependent workflow files."""
        deps = {}
        steps = workflow.get("steps", [])

        if isinstance(steps, dict):
            steps = list(steps.values())

        for step in steps:
            # Check for workflow.run steps
            use = step.get("use", "")
            if use.startswith("workflow."):
                args = step.get("args", {})
                path = args.get("path", "")
                if path:
                    dep_path = self._resolve_path(path, base_dir, workspace_path)
                    if dep_path and dep_path.exists() and str(dep_path) not in self._collected_workflows:
                        self._collected_workflows.add(str(dep_path))
                        deps[dep_path.name] = dep_path

                        # Recursively collect deps
                        with open(dep_path, "r", encoding="utf-8") as f:
                            sub_workflow = yaml.safe_load(f)
                        if "workflow" in sub_workflow:
                            sub_workflow = sub_workflow["workflow"]
                        sub_deps = self._collect_dependencies(sub_workflow, dep_path.parent, workspace_path)
                        deps.update(sub_deps)

        return deps

    def _collect_configs(
        self,
        base_dir: Path,
        workspace_path: Optional[str],
    ) -> Dict[str, Path]:
        """Collect configuration files."""
        configs = {}

        search_dirs = [base_dir]
        if workspace_path:
            search_dirs.append(Path(workspace_path))

        config_patterns = ["config.yaml", "config.yml", "bbx.yaml", "bbx.yml", ".env.example"]

        for search_dir in search_dirs:
            for pattern in config_patterns:
                cfg_path = search_dir / pattern
                if cfg_path.exists() and str(cfg_path) not in self._collected_configs:
                    self._collected_configs.add(str(cfg_path))
                    configs[pattern] = cfg_path

        return configs

    def _collect_state(
        self,
        base_dir: Path,
        workspace_path: Optional[str],
    ) -> Dict[str, Path]:
        """Collect state files."""
        state_files = {}

        search_dirs = [base_dir]
        if workspace_path:
            search_dirs.append(Path(workspace_path))

        for search_dir in search_dirs:
            state_dir = search_dir / "state"
            if state_dir.exists():
                for state_file in state_dir.glob("**/*.json"):
                    rel_path = state_file.relative_to(state_dir)
                    state_files[str(rel_path)] = state_file

        return state_files

    def _resolve_path(
        self,
        path: str,
        base_dir: Path,
        workspace_path: Optional[str],
    ) -> Optional[Path]:
        """Resolve a path relative to base_dir or workspace."""
        # Try relative to base_dir
        resolved = base_dir / path
        if resolved.exists():
            return resolved.resolve()

        # Try relative to workspace
        if workspace_path:
            resolved = Path(workspace_path) / path
            if resolved.exists():
                return resolved.resolve()

        # Try absolute
        resolved = Path(path)
        if resolved.exists():
            return resolved.resolve()

        return None

    def _create_manifest(
        self,
        workflow_path: Path,
        workflow: Dict,
        files: Dict[str, Path],
        include_state: bool,
    ) -> Dict[str, Any]:
        """Create package manifest."""
        return {
            "bbx_version": "1.0.0",
            "package_version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "main_workflow": workflow_path.name,
            "workflow_id": workflow.get("id", workflow_path.stem),
            "workflow_name": workflow.get("name", workflow_path.stem),
            "workflow_version": workflow.get("version", "1.0.0"),
            "description": workflow.get("description", ""),
            "include_state": include_state,
            "files": {
                "count": len(files),
                "list": list(files.keys()),
            },
            "checksums": {},  # TODO: Add file checksums
        }

    @staticmethod
    def _human_size(size: int) -> str:
        """Convert bytes to human readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


# Singleton
_bundler: Optional[BBXBundler] = None


def get_bundler() -> BBXBundler:
    """Get singleton bundler instance."""
    global _bundler
    if _bundler is None:
        _bundler = BBXBundler()
    return _bundler
