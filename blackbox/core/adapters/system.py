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


import logging


import asyncio
import os
import shutil
import glob
from typing import Dict, Any
from blackbox.core.base_adapter import MCPAdapter

logger = logging.getLogger("bbx.system")

class SystemAdapter(MCPAdapter):
    """
    Adapter for system operations: shell commands and file management.
    """

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        if method == "shell":
            return await self._shell(inputs)
        elif method == "fs.delete":
            return await self._fs_delete(inputs)
        elif method == "fs.list":
            return await self._fs_list(inputs)
        else:
            raise ValueError(f"Unknown system method: {method}")

    async def _shell(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command."""
        command = inputs.get("command")
        cwd = inputs.get("cwd", os.getcwd())

        if not command:
            raise ValueError("Command is required")

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )

        stdout, stderr = await process.communicate()

        return {
            "exit_code": process.returncode,
            "stdout": stdout.decode(errors='replace').strip(),
            "stderr": stderr.decode(errors='replace').strip()
        }

    async def _fs_delete(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Delete files or directories matching a pattern."""
        pattern = inputs.get("pattern")
        recursive = inputs.get("recursive", False)

        if not pattern:
            raise ValueError("Pattern is required")

        deleted = []

        # Handle recursive glob if needed
        if recursive:
            files = glob.glob(pattern, recursive=True)
        else:
            files = glob.glob(pattern)

        for path in files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    deleted.append(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
                    deleted.append(path)
            except Exception as e:
                logger.error(f"Failed to delete {path}: {e}")

        return {
            "deleted": deleted,
            "count": len(deleted)
        }

    async def _fs_list(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """List files matching a pattern."""
        pattern = inputs.get("pattern", "*")
        recursive = inputs.get("recursive", False)

        files = glob.glob(pattern, recursive=recursive)
        return {
            "files": files,
            "count": len(files)
        }
