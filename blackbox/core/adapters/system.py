# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX System Adapter

Exposes core system capabilities to workflows, allowing BBX to manage itself.
"""

import os
import sys
import asyncio
from typing import Any, Dict

from ..base_adapter import BaseAdapter, AdapterResponse, AdapterErrorType


class SystemAdapter(BaseAdapter):
    """
    Adapter for system-level operations.
    """

    def __init__(self):
        super().__init__("system")
        self.register_method("restart", self.restart)
        self.register_method("update", self.update)
        self.register_method("install_package", self.install_package)
        self.register_method("get_status", self.get_status)
        self.register_method("exec", self.exec_command)

    async def restart(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Restart the BBX process"""
        delay = inputs.get("delay", 1)
        self.logger.warning(f"🔄 System restart requested in {delay}s")
        
        # Schedule restart
        asyncio.create_task(self._do_restart(delay))
        
        return AdapterResponse.success_response(
            message=f"Restart scheduled in {delay}s"
        ).to_dict()

    async def _do_restart(self, delay: int):
        await asyncio.sleep(delay)
        # Re-execute the current process with same arguments
        os.execv(sys.executable, [sys.executable] + sys.argv)

    async def update(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Update BBX from source or binary"""
        source = inputs.get("source", "git")
        branch = inputs.get("branch", "main")
        
        if source == "git":
            # Git pull
            proc = await asyncio.create_subprocess_shell(
                f"git pull origin {branch}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                return AdapterResponse.success_response(
                    output=stdout.decode()
                ).to_dict()
            else:
                return AdapterResponse.error_response(
                    error=stderr.decode(),
                    error_type=AdapterErrorType.EXECUTION_ERROR
                ).to_dict()
        
        return AdapterResponse.error_response(
            error=f"Unknown update source: {source}"
        ).to_dict()

    async def install_package(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Install a Python package into the current environment"""
        package = inputs.get("package")
        if not package:
            return AdapterResponse.error_response("Package name required").to_dict()
            
        cmd = [sys.executable, "-m", "pip", "install", package]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode == 0:
            return AdapterResponse.success_response(
                output=stdout.decode()
            ).to_dict()
        else:
            return AdapterResponse.error_response(
                error=stderr.decode()
            ).to_dict()

    async def get_status(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get system status info"""
        import platform
        import psutil  # type: ignore
        
        return AdapterResponse.success_response(
            os=platform.system(),
            python=platform.python_version(),
            cpu_percent=psutil.cpu_percent(),
            memory_used=psutil.virtual_memory().percent,
            disk_used=psutil.disk_usage("/").percent
        ).to_dict()

    async def exec_command(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute arbitrary shell command.

        Supports two formats:
        1. Legacy: {"cmd": "python cli.py --version"}
        2. New: {"command": "python", "args": ["cli.py", "--version"]}
        """
        # Check for new format first (command + args)
        command = inputs.get("command")
        args = inputs.get("args", [])

        # Fallback to legacy format (cmd)
        if not command:
            cmd = inputs.get("cmd")
            if not cmd:
                return AdapterResponse.error_response("Command required").to_dict()

            # Security check (basic)
            if "rm -rf /" in cmd:
                return AdapterResponse.error_response("Unsafe command blocked").to_dict()

            # Use shell execution for legacy format
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            return AdapterResponse.success_response(
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                exit_code=proc.returncode
            ).to_dict()

        # New format: command + args (more secure, no shell injection)
        full_cmd = [command] + args

        # Security check
        full_cmd_str = " ".join(full_cmd)
        if "rm -rf /" in full_cmd_str:
            return AdapterResponse.error_response("Unsafe command blocked").to_dict()

        try:
            proc = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            return AdapterResponse.success_response(
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                exit_code=proc.returncode
            ).to_dict()

        except FileNotFoundError:
            return AdapterResponse.error_response(
                error=f"Command not found: {command}",
                error_type=AdapterErrorType.EXECUTION_ERROR
            ).to_dict()
        except Exception as e:
            return AdapterResponse.error_response(
                error=str(e),
                error_type=AdapterErrorType.EXECUTION_ERROR
            ).to_dict()
