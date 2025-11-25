# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
OS Abstraction Layer Adapter

Provides cross-platform operations for file system, processes, and commands.
This is the core of BBX's "write once, run anywhere" philosophy.
"""

import asyncio
import os
import platform
import re
import shutil
from pathlib import Path
from typing import Any, Dict

from blackbox.core.base_adapter import MCPAdapter


class OSAbstractionAdapter(MCPAdapter):
    """
    Cross-platform OS operations adapter.

    Replaces system.exec with platform-agnostic methods.
    Automatically translates Unix commands to Windows equivalents.
    """

    def __init__(self):
        self.platform = platform.system()  # Windows, Linux, Darwin (macOS)
        self.is_windows = self.platform == "Windows"
        self.is_linux = self.platform == "Linux"
        self.is_macos = self.platform == "Darwin"

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute OS abstraction method"""

        if method == "exec":
            return await self._exec(inputs)
        elif method == "remove":
            return await self._remove(inputs)
        elif method == "copy":
            return await self._copy(inputs)
        elif method == "move":
            return await self._move(inputs)
        elif method == "list":
            return await self._list(inputs)
        elif method == "mkdir":
            return await self._mkdir(inputs)
        elif method == "exists":
            return await self._exists(inputs)
        elif method == "read":
            return await self._read(inputs)
        elif method == "write":
            return await self._write(inputs)
        elif method == "env":
            return await self._env(inputs)
        elif method == "platform_info":
            return self._platform_info()
        else:
            raise ValueError(f"Unknown bbx.os method: {method}")

    # ========== Command Execution ==========

    async def _exec(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute shell command with cross-platform support.

        Features:
        - Auto-translate Unix commands to Windows if auto_translate=true
        - Custom environment variables
        - Working directory support
        - Timeout support
        """
        command = inputs.get("command")
        auto_translate = inputs.get("auto_translate", True)
        cwd = inputs.get("cwd", os.getcwd())
        env = inputs.get("env", {})
        timeout = inputs.get("timeout", 30)  # seconds
        shell = inputs.get("shell")  # Optional: explicit shell to use

        if not command:
            raise ValueError("command is required")

        # Auto-detect bash for Unix-style commands on Windows
        use_bash = False
        if self.is_windows and not shell:
            # Detect heredocs, pipes with cat, or other Unix constructs
            unix_patterns = ["<<", "cat >", "#!/", "|", "mkdir -p"]
            if any(pattern in command for pattern in unix_patterns):
                # Try to find bash
                bash_path = shutil.which("bash")
                if bash_path:
                    shell = bash_path
                    use_bash = True

        # Auto-translate Unix commands to Windows (only if not using bash)
        if auto_translate and self.is_windows and not use_bash:
            command = self._translate_command(command)

        # Prepare environment
        exec_env = os.environ.copy()
        exec_env.update(env)

        # Execute command
        try:
            if shell:
                # Use explicit shell
                process = await asyncio.create_subprocess_exec(
                    shell,
                    "-c",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=exec_env,
                )
            else:
                # Use default shell
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=exec_env,
                )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode(errors="replace").strip(),
                "stderr": stderr.decode(errors="replace").strip(),
                "success": process.returncode == 0,
            }

        except asyncio.TimeoutError:
            process.kill()
            raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    def _translate_command(self, command: str) -> str:
        """
        Translate Unix commands to Windows equivalents.

        Examples:
        - rm -rf dist/ → del /s /q dist
        - ls -la → dir
        - touch file.txt → type nul > file.txt
        - cat file.txt → type file.txt
        - grep "pattern" → findstr "pattern"
        """
        if not self.is_windows:
            return command

        # Remove dangerous commands
        if re.search(r"\brm\s+-rf\s+/\b", command):
            raise ValueError("Dangerous command detected: rm -rf /")

        translations = [
            # File operations
            (r"\brm\s+-rf\s+", "rd /s /q "),
            (r"\brm\s+-r\s+", "rd /s "),
            (r"\brm\s+", "del "),
            (r"\bcp\s+-r\s+", "xcopy /e /i "),
            (r"\bcp\s+", "copy "),
            (r"\bmv\s+", "move "),
            (r"\btouch\s+", "type nul > "),
            # Directory operations
            (r"\bmkdir\s+-p\s+", "mkdir "),
            (r"\bls\s+-la\b", "dir"),
            (r"\bls\s+-l\b", "dir"),
            (r"\bls\b", "dir"),
            (r"\bpwd\b", "cd"),
            # File viewing
            (r"\bcat\s+", "type "),
            (r"\bhead\s+", "type "),  # Not perfect but works
            (r"\btail\s+", "type "),  # Not perfect
            # Search
            (r"\bgrep\s+", "findstr "),
            # Process
            (r"\bkill\s+-9\s+", "taskkill /F /PID "),
            (r"\bps\s+aux\b", "tasklist"),
            # Network
            (r"\bcurl\s+", "curl "),  # curl is now on Windows
            (r"\bwget\s+", "curl -O "),
        ]

        translated = command
        for pattern, replacement in translations:
            translated = re.sub(pattern, replacement, translated)

        return translated

    # ========== File Operations ==========

    async def _remove(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove file or directory (cross-platform)"""
        path = inputs.get("path")
        recursive = inputs.get("recursive", False)

        if not path:
            raise ValueError("path is required")

        path = Path(path)

        if not path.exists():
            return {"removed": False, "reason": "Path does not exist"}

        try:
            if path.is_file():
                path.unlink()
                return {"removed": True, "type": "file", "path": str(path)}
            elif path.is_dir():
                if recursive:
                    shutil.rmtree(path)
                else:
                    path.rmdir()
                return {"removed": True, "type": "directory", "path": str(path)}
        except Exception as e:
            return {"removed": False, "error": str(e)}

    async def _copy(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Copy file or directory (cross-platform)"""
        src = inputs.get("src") or inputs.get("from")
        dst = inputs.get("dst") or inputs.get("to")
        recursive = inputs.get("recursive", True)

        if not src or not dst:
            raise ValueError("src and dst are required")

        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise FileNotFoundError(f"Source does not exist: {src}")

        try:
            if src_path.is_file():
                # Copy file
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                return {
                    "copied": True,
                    "type": "file",
                    "src": str(src_path),
                    "dst": str(dst_path),
                }
            elif src_path.is_dir():
                # Copy directory
                if recursive:
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    return {
                        "copied": True,
                        "type": "directory",
                        "src": str(src_path),
                        "dst": str(dst_path),
                    }
                else:
                    raise ValueError("Cannot copy directory without recursive=true")
        except Exception as e:
            return {"copied": False, "error": str(e)}

    async def _move(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move file or directory (cross-platform)"""
        src = inputs.get("src") or inputs.get("from")
        dst = inputs.get("dst") or inputs.get("to")

        if not src or not dst:
            raise ValueError("src and dst are required")

        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise FileNotFoundError(f"Source does not exist: {src}")

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            return {"moved": True, "src": str(src_path), "dst": str(dst_path)}
        except Exception as e:
            return {"moved": False, "error": str(e)}

    async def _list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents (cross-platform)"""
        path = inputs.get("path", ".")
        pattern = inputs.get("pattern", "*")
        recursive = inputs.get("recursive", False)

        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not path_obj.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Use glob for pattern matching
        if recursive:
            files = [str(p) for p in path_obj.rglob(pattern)]
        else:
            files = [str(p) for p in path_obj.glob(pattern)]

        return {"files": files, "count": len(files), "path": str(path_obj)}

    async def _mkdir(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create directory (cross-platform)"""
        path = inputs.get("path")
        parents = inputs.get("parents", True)
        exist_ok = inputs.get("exist_ok", True)

        if not path:
            raise ValueError("path is required")

        path_obj = Path(path)

        try:
            path_obj.mkdir(parents=parents, exist_ok=exist_ok)
            return {"created": True, "path": str(path_obj)}
        except Exception as e:
            return {"created": False, "error": str(e)}

    async def _exists(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Check if path exists (cross-platform)"""
        path = inputs.get("path")

        if not path:
            raise ValueError("path is required")

        path_obj = Path(path)

        return {
            "exists": path_obj.exists(),
            "is_file": path_obj.is_file() if path_obj.exists() else False,
            "is_dir": path_obj.is_dir() if path_obj.exists() else False,
            "path": str(path_obj),
        }

    async def _read(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Read file contents (cross-platform)"""
        path = inputs.get("path")
        encoding = inputs.get("encoding", "utf-8")

        if not path:
            raise ValueError("path is required")

        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File does not exist: {path}")

        try:
            content = path_obj.read_text(encoding=encoding)
            return {"content": content, "size": len(content), "path": str(path_obj)}
        except Exception as e:
            return {"error": str(e)}

    async def _write(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Write file contents (cross-platform)"""
        path = inputs.get("path")
        content = inputs.get("content")
        encoding = inputs.get("encoding", "utf-8")
        append = inputs.get("append", False)

        if not path:
            raise ValueError("path is required")
        if content is None:
            raise ValueError("content is required")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            if append:
                with path_obj.open("a", encoding=encoding) as f:
                    f.write(content)
            else:
                path_obj.write_text(content, encoding=encoding)

            return {"written": True, "size": len(content), "path": str(path_obj)}
        except Exception as e:
            return {"written": False, "error": str(e)}

    # ========== Environment & System Info ==========

    async def _env(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get or set environment variables"""
        action = inputs.get("action", "get")  # get, set, list
        key = inputs.get("key")
        value = inputs.get("value")

        if action == "get":
            if not key:
                raise ValueError("key is required for get action")
            return {
                "key": key,
                "value": os.environ.get(key),
                "exists": key in os.environ,
            }
        elif action == "set":
            if not key or value is None:
                raise ValueError("key and value are required for set action")
            os.environ[key] = str(value)
            return {"key": key, "value": value, "set": True}
        elif action == "list":
            return {"variables": dict(os.environ)}
        else:
            raise ValueError(f"Unknown action: {action}")

    def _platform_info(self) -> Dict[str, Any]:
        """Get platform information"""
        return {
            "system": self.platform,
            "is_windows": self.is_windows,
            "is_linux": self.is_linux,
            "is_macos": self.is_macos,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cwd": os.getcwd(),
        }
