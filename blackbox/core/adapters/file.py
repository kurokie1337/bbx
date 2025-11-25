# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0
# BBX File Adapter - File Operations

"""
BBX File Adapter

Provides file operations for BBX-only coding.

Methods:
    - read: Read file content
    - write: Write content to file
    - append: Append content to file
    - copy: Copy file or directory
    - move: Move/rename file or directory
    - delete: Delete file or directory
    - exists: Check if path exists
    - list: List directory contents
    - mkdir: Create directory
    - info: Get file/directory info

Usage in .bbx:
    steps:
      read_config:
        use: file.read
        args:
          path: config.yaml

      write_output:
        use: file.write
        args:
          path: output.txt
          content: ${steps.process.output}
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from blackbox.core.base_adapter import MCPAdapter


class FileAdapter(MCPAdapter):
    """
    File operations adapter for BBX workflows.

    Enables BBX-only coding without writing Python.
    """

    def __init__(self):
        super().__init__("file")

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute file adapter method."""
        self.log_execution(method, inputs)

        method_map = {
            "read": self._read,
            "write": self._write,
            "append": self._append,
            "copy": self._copy,
            "move": self._move,
            "delete": self._delete,
            "exists": self._exists,
            "list": self._list,
            "mkdir": self._mkdir,
            "info": self._info,
            "hash": self._hash,
            "find": self._find,
            "glob": self._glob,
        }

        handler = method_map.get(method)
        if not handler:
            raise ValueError(f"Unknown file method: {method}. Available: {list(method_map.keys())}")

        try:
            result = await handler(**inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            raise

    async def _read(
        self,
        path: str,
        encoding: str = "utf-8",
        binary: bool = False,
        json_parse: bool = False,
        yaml_parse: bool = False,
        lines: bool = False,
    ) -> Dict[str, Any]:
        """
        Read file content.

        Args:
            path: File path
            encoding: Text encoding
            binary: Read as binary
            json_parse: Parse as JSON
            yaml_parse: Parse as YAML
            lines: Return list of lines

        Returns:
            File content and metadata
        """
        file_path = Path(path)

        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"status": "error", "error": f"Not a file: {path}"}

        try:
            if binary:
                content = file_path.read_bytes()
                return {
                    "status": "success",
                    "path": str(file_path.resolve()),
                    "content": content.hex(),
                    "size": len(content),
                    "binary": True,
                }

            content = file_path.read_text(encoding=encoding)

            if json_parse:
                content = json.loads(content)
            elif yaml_parse:
                import yaml
                content = yaml.safe_load(content)
            elif lines:
                content = content.splitlines()

            return {
                "status": "success",
                "path": str(file_path.resolve()),
                "content": content,
                "size": file_path.stat().st_size,
                "encoding": encoding,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _write(
        self,
        path: str,
        content: Union[str, Dict, List],
        encoding: str = "utf-8",
        json_format: bool = False,
        yaml_format: bool = False,
        indent: int = 2,
        create_dirs: bool = True,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """
        Write content to file.

        Args:
            path: File path
            content: Content to write
            encoding: Text encoding
            json_format: Format as JSON
            yaml_format: Format as YAML
            indent: Indentation for JSON/YAML
            create_dirs: Create parent directories
            overwrite: Overwrite existing file

        Returns:
            Write result
        """
        file_path = Path(path)

        if file_path.exists() and not overwrite:
            return {"status": "error", "error": f"File exists (overwrite=False): {path}"}

        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            if json_format:
                content = json.dumps(content, indent=indent, ensure_ascii=False, default=str)
            elif yaml_format:
                import yaml
                content = yaml.dump(content, allow_unicode=True, default_flow_style=False)
            elif isinstance(content, (dict, list)):
                content = json.dumps(content, indent=indent, ensure_ascii=False, default=str)

            file_path.write_text(content, encoding=encoding)

            return {
                "status": "success",
                "path": str(file_path.resolve()),
                "size": file_path.stat().st_size,
                "created": not file_path.exists(),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _append(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        newline: bool = True,
        create: bool = True,
    ) -> Dict[str, Any]:
        """
        Append content to file.

        Args:
            path: File path
            content: Content to append
            encoding: Text encoding
            newline: Add newline before content
            create: Create file if not exists

        Returns:
            Append result
        """
        file_path = Path(path)

        if not file_path.exists() and not create:
            return {"status": "error", "error": f"File not found: {path}"}

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "a", encoding=encoding) as f:
                if newline and file_path.stat().st_size > 0:
                    f.write("\n")
                f.write(content)

            return {
                "status": "success",
                "path": str(file_path.resolve()),
                "size": file_path.stat().st_size,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _copy(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Copy file or directory.

        Args:
            source: Source path
            destination: Destination path
            overwrite: Overwrite existing
            recursive: Copy directories recursively

        Returns:
            Copy result
        """
        src = Path(source)
        dst = Path(destination)

        if not src.exists():
            return {"status": "error", "error": f"Source not found: {source}"}

        if dst.exists() and not overwrite:
            return {"status": "error", "error": f"Destination exists: {destination}"}

        try:
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif recursive:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                return {"status": "error", "error": "Cannot copy directory without recursive=True"}

            return {
                "status": "success",
                "source": str(src.resolve()),
                "destination": str(dst.resolve()),
                "is_directory": src.is_dir(),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _move(
        self,
        source: str,
        destination: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Move/rename file or directory.

        Args:
            source: Source path
            destination: Destination path
            overwrite: Overwrite existing

        Returns:
            Move result
        """
        src = Path(source)
        dst = Path(destination)

        if not src.exists():
            return {"status": "error", "error": f"Source not found: {source}"}

        if dst.exists() and not overwrite:
            return {"status": "error", "error": f"Destination exists: {destination}"}

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))

            return {
                "status": "success",
                "source": source,
                "destination": str(dst.resolve()),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _delete(
        self,
        path: str,
        recursive: bool = False,
        missing_ok: bool = True,
    ) -> Dict[str, Any]:
        """
        Delete file or directory.

        Args:
            path: Path to delete
            recursive: Delete directories recursively
            missing_ok: Don't error if not exists

        Returns:
            Delete result
        """
        target = Path(path)

        if not target.exists():
            if missing_ok:
                return {"status": "success", "path": path, "deleted": False}
            return {"status": "error", "error": f"Path not found: {path}"}

        try:
            if target.is_file():
                target.unlink()
            elif recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()  # Only works for empty dirs

            return {
                "status": "success",
                "path": path,
                "deleted": True,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _exists(
        self,
        path: str,
        type: Optional[str] = None,  # "file", "dir", or None for any
    ) -> Dict[str, Any]:
        """
        Check if path exists.

        Args:
            path: Path to check
            type: Expected type ("file" or "dir")

        Returns:
            Existence check result
        """
        target = Path(path)

        exists = target.exists()
        is_file = target.is_file() if exists else False
        is_dir = target.is_dir() if exists else False

        if type == "file":
            result = exists and is_file
        elif type == "dir":
            result = exists and is_dir
        else:
            result = exists

        return {
            "status": "success",
            "path": path,
            "exists": result,
            "is_file": is_file,
            "is_directory": is_dir,
        }

    async def _list(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
        files_only: bool = False,
        dirs_only: bool = False,
    ) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            path: Directory path
            pattern: Glob pattern
            recursive: Search recursively
            files_only: Only return files
            dirs_only: Only return directories

        Returns:
            Directory listing
        """
        dir_path = Path(path)

        if not dir_path.exists():
            return {"status": "error", "error": f"Directory not found: {path}"}

        if not dir_path.is_dir():
            return {"status": "error", "error": f"Not a directory: {path}"}

        try:
            if recursive:
                items = list(dir_path.rglob(pattern))
            else:
                items = list(dir_path.glob(pattern))

            results = []
            for item in items:
                if files_only and not item.is_file():
                    continue
                if dirs_only and not item.is_dir():
                    continue

                results.append({
                    "name": item.name,
                    "path": str(item.relative_to(dir_path)),
                    "is_file": item.is_file(),
                    "is_directory": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                })

            return {
                "status": "success",
                "path": str(dir_path.resolve()),
                "pattern": pattern,
                "recursive": recursive,
                "items": results,
                "count": len(results),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _mkdir(
        self,
        path: str,
        parents: bool = True,
        exist_ok: bool = True,
    ) -> Dict[str, Any]:
        """
        Create directory.

        Args:
            path: Directory path
            parents: Create parent directories
            exist_ok: Don't error if exists

        Returns:
            Create result
        """
        dir_path = Path(path)

        try:
            dir_path.mkdir(parents=parents, exist_ok=exist_ok)
            return {
                "status": "success",
                "path": str(dir_path.resolve()),
                "created": True,
            }
        except FileExistsError:
            return {"status": "error", "error": f"Directory exists: {path}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _info(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """
        Get file/directory information.

        Args:
            path: Path to get info for

        Returns:
            File/directory metadata
        """
        target = Path(path)

        if not target.exists():
            return {"status": "error", "error": f"Path not found: {path}"}

        try:
            stat = target.stat()
            return {
                "status": "success",
                "path": str(target.resolve()),
                "name": target.name,
                "stem": target.stem,
                "suffix": target.suffix,
                "is_file": target.is_file(),
                "is_directory": target.is_dir(),
                "is_symlink": target.is_symlink(),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _hash(
        self,
        path: str,
        algorithm: str = "sha256",
    ) -> Dict[str, Any]:
        """
        Calculate file hash.

        Args:
            path: File path
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)

        Returns:
            File hash
        """
        file_path = Path(path)

        if not file_path.exists():
            return {"status": "error", "error": f"File not found: {path}"}

        if not file_path.is_file():
            return {"status": "error", "error": f"Not a file: {path}"}

        try:
            hash_func = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)

            return {
                "status": "success",
                "path": str(file_path.resolve()),
                "algorithm": algorithm,
                "hash": hash_func.hexdigest(),
            }

        except ValueError:
            return {"status": "error", "error": f"Unknown algorithm: {algorithm}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _find(
        self,
        path: str = ".",
        name: Optional[str] = None,
        pattern: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        extensions: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Find files matching criteria.

        Args:
            path: Search directory
            name: Exact file name
            pattern: Glob pattern
            min_size: Minimum file size
            max_size: Maximum file size
            extensions: List of extensions (e.g., [".py", ".js"])
            max_depth: Maximum directory depth

        Returns:
            Matching files
        """
        dir_path = Path(path)

        if not dir_path.exists():
            return {"status": "error", "error": f"Directory not found: {path}"}

        try:
            results = []

            for item in dir_path.rglob(pattern or "*"):
                if not item.is_file():
                    continue

                # Check depth
                if max_depth is not None:
                    rel_parts = item.relative_to(dir_path).parts
                    if len(rel_parts) > max_depth:
                        continue

                # Check name
                if name and item.name != name:
                    continue

                # Check extension
                if extensions and item.suffix.lower() not in [e.lower() for e in extensions]:
                    continue

                # Check size
                size = item.stat().st_size
                if min_size is not None and size < min_size:
                    continue
                if max_size is not None and size > max_size:
                    continue

                results.append({
                    "name": item.name,
                    "path": str(item.relative_to(dir_path)),
                    "full_path": str(item.resolve()),
                    "size": size,
                })

            return {
                "status": "success",
                "path": str(dir_path.resolve()),
                "files": results,
                "count": len(results),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _glob(
        self,
        pattern: str,
        path: str = ".",
    ) -> Dict[str, Any]:
        """
        Glob pattern matching.

        Args:
            pattern: Glob pattern
            path: Base directory

        Returns:
            Matching paths
        """
        dir_path = Path(path)

        if not dir_path.exists():
            return {"status": "error", "error": f"Directory not found: {path}"}

        try:
            matches = list(dir_path.glob(pattern))
            results = []
            for item in matches:
                results.append({
                    "name": item.name,
                    "path": str(item.relative_to(dir_path)),
                    "is_file": item.is_file(),
                    "is_directory": item.is_dir(),
                })

            return {
                "status": "success",
                "pattern": pattern,
                "path": str(dir_path.resolve()),
                "matches": results,
                "count": len(results),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}
