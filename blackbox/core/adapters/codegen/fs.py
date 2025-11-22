# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File System Generation Adapter

Provides file and directory generation for code scaffolding.
Supports creating files, directories, and entire project structures.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml as yaml_lib

from blackbox.core.base_adapter import MCPAdapter


class FileSystemGenAdapter(MCPAdapter):
    """
    File system generation adapter.

    Enables creating files, directories, and project structures programmatically.
    Designed for code generation and scaffolding workflows.
    """

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute file system generation method"""

        if method == "create_file":
            return await self._create_file(inputs)
        elif method == "create_files":
            return await self._create_files(inputs)
        elif method == "create_dir":
            return await self._create_dir(inputs)
        elif method == "create_structure":
            return await self._create_structure(inputs)
        elif method == "write_json":
            return await self._write_json(inputs)
        elif method == "write_yaml":
            return await self._write_yaml(inputs)
        elif method == "copy_template":
            return await self._copy_template(inputs)
        else:
            raise ValueError(f"Unknown codegen.fs method: {method}")

    # ========== File Operations ==========

    async def _create_file(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create single file with content.

        Args:
            path: File path
            content: File content
            encoding: Text encoding (default: utf-8)
            create_dirs: Create parent directories (default: True)
            overwrite: Overwrite if exists (default: False)

        Returns:
            File creation info
        """
        path = inputs.get("path")
        content = inputs.get("content", "")
        encoding = inputs.get("encoding", "utf-8")
        create_dirs = inputs.get("create_dirs", True)
        overwrite = inputs.get("overwrite", False)

        if not path:
            raise ValueError("path is required")

        file_path = Path(path)

        # Check if file exists
        if file_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {path} (use overwrite=true to replace)"
            )

        # Create parent directories
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path.write_text(content, encoding=encoding)

        return {
            "created": True,
            "path": str(file_path),
            "size": len(content),
            "encoding": encoding,
        }

    async def _create_files(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create multiple files at once.

        Args:
            files: List of file definitions [{"path": ..., "content": ...}, ...]
            base_dir: Base directory for relative paths (default: current dir)
            create_dirs: Create parent directories (default: True)
            overwrite: Overwrite if exists (default: False)

        Returns:
            List of created files
        """
        files = inputs.get("files", [])
        base_dir = inputs.get("base_dir", ".")
        create_dirs = inputs.get("create_dirs", True)
        overwrite = inputs.get("overwrite", False)

        if not files:
            raise ValueError("files list is required")

        base_path = Path(base_dir)
        created_files = []

        for file_def in files:
            path = file_def.get("path")
            content = file_def.get("content", "")
            encoding = file_def.get("encoding", "utf-8")

            if not path:
                raise ValueError("Each file must have a 'path'")

            # Resolve path relative to base_dir
            if not os.path.isabs(path):
                full_path = base_path / path
            else:
                full_path = Path(path)

            # Create file
            result = await self._create_file(
                {
                    "path": str(full_path),
                    "content": content,
                    "encoding": encoding,
                    "create_dirs": create_dirs,
                    "overwrite": overwrite,
                }
            )

            created_files.append(result)

        return {"created_count": len(created_files), "files": created_files}

    # ========== Directory Operations ==========

    async def _create_dir(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create directory.

        Args:
            path: Directory path
            parents: Create parent directories (default: True)
            exist_ok: Don't error if exists (default: True)

        Returns:
            Directory creation info
        """
        path = inputs.get("path")
        parents = inputs.get("parents", True)
        exist_ok = inputs.get("exist_ok", True)

        if not path:
            raise ValueError("path is required")

        dir_path = Path(path)

        try:
            dir_path.mkdir(parents=parents, exist_ok=exist_ok)
            return {
                "created": True,
                "path": str(dir_path),
                "existed": dir_path.exists(),
            }
        except FileExistsError:
            return {
                "created": False,
                "path": str(dir_path),
                "error": "Directory already exists",
            }

    async def _create_structure(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create directory structure from list or dict.

        Args:
            base_path: Base directory
            structure: List of paths OR nested dict structure
                       Example list: ["src/", "src/components/", "tests/"]
                       Example dict: {"src": {"components": {}, "utils": {}}}

        Returns:
            Created directories
        """
        base_path = inputs.get("base_path") or inputs.get("path")
        structure = inputs.get("structure")

        if not base_path:
            raise ValueError("base_path is required")
        if not structure:
            raise ValueError("structure is required")

        base = Path(base_path)
        created_dirs = []

        if isinstance(structure, list):
            # List format: ["dir1/", "dir2/subdir/"]
            for dir_path in structure:
                full_path = base / dir_path
                result = await self._create_dir(
                    {"path": str(full_path), "parents": True, "exist_ok": True}
                )
                if result["created"]:
                    created_dirs.append(str(full_path))

        elif isinstance(structure, dict):
            # Nested dict format
            created_dirs = self._create_from_dict(base, structure)

        else:
            raise ValueError("structure must be list or dict")

        return {
            "created_count": len(created_dirs),
            "directories": created_dirs,
            "base_path": str(base),
        }

    def _create_from_dict(self, base: Path, struct: dict) -> List[str]:
        """Recursively create directory structure from dict"""
        created = []

        for name, children in struct.items():
            dir_path = base / name
            dir_path.mkdir(parents=True, exist_ok=True)
            created.append(str(dir_path))

            if isinstance(children, dict) and children:
                # Recursively create subdirectories
                created.extend(self._create_from_dict(dir_path, children))

        return created

    # ========== Structured File Writing ==========

    async def _write_json(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write data to JSON file.

        Args:
            path: Output file path
            data: Data to serialize
            indent: JSON indentation (default: 2)
            create_dirs: Create parent directories (default: True)

        Returns:
            File write info
        """
        path = inputs.get("path")
        data = inputs.get("data")
        indent = inputs.get("indent", 2)
        create_dirs = inputs.get("create_dirs", True)

        if not path:
            raise ValueError("path is required")
        if data is None:
            raise ValueError("data is required")

        file_path = Path(path)

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to JSON
        try:
            json_content = json.dumps(data, indent=indent, ensure_ascii=False)
            file_path.write_text(json_content, encoding="utf-8")

            return {
                "written": True,
                "path": str(file_path),
                "size": len(json_content),
                "format": "json",
            }
        except (TypeError, ValueError) as e:
            raise ValueError(f"JSON serialization error: {str(e)}")

    async def _write_yaml(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write data to YAML file.

        Args:
            path: Output file path
            data: Data to serialize
            create_dirs: Create parent directories (default: True)

        Returns:
            File write info
        """
        path = inputs.get("path")
        data = inputs.get("data")
        create_dirs = inputs.get("create_dirs", True)

        if not path:
            raise ValueError("path is required")
        if data is None:
            raise ValueError("data is required")

        file_path = Path(path)

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to YAML
        try:
            yaml_content = yaml_lib.dump(
                data, default_flow_style=False, allow_unicode=True
            )
            file_path.write_text(yaml_content, encoding="utf-8")

            return {
                "written": True,
                "path": str(file_path),
                "size": len(yaml_content),
                "format": "yaml",
            }
        except Exception as e:
            raise ValueError(f"YAML serialization error: {str(e)}")

    # ========== Template Copying ==========

    async def _copy_template(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Copy template directory to destination.

        Args:
            src: Source template directory
            dst: Destination directory
            overwrite: Overwrite existing files (default: False)

        Returns:
            Copy operation info
        """
        src = inputs.get("src") or inputs.get("source")
        dst = inputs.get("dst") or inputs.get("destination")
        overwrite = inputs.get("overwrite", False)

        if not src:
            raise ValueError("src is required")
        if not dst:
            raise ValueError("dst is required")

        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise FileNotFoundError(f"Source template not found: {src}")

        if not src_path.is_dir():
            raise ValueError(f"Source must be a directory: {src}")

        # Check if destination exists when overwrite is False
        if dst_path.exists() and not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst} (use overwrite=true)"
            )

        # Copy directory tree
        try:
            shutil.copytree(src_path, dst_path, dirs_exist_ok=overwrite)

            # Count copied files
            file_count = sum(1 for _ in dst_path.rglob("*") if _.is_file())

            return {
                "copied": True,
                "src": str(src_path),
                "dst": str(dst_path),
                "file_count": file_count,
            }
        except Exception as e:
            # Don't wrap FileExistsError in RuntimeError
            if isinstance(e, FileExistsError):
                raise
            raise RuntimeError(f"Copy operation failed: {str(e)}")
