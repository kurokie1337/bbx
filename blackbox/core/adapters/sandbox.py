# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
BBX Sandbox Adapter (Phase 7 Foundation)

Provides isolated execution environments:
- Process isolation
- Container sandboxing
- Android emulator support (future)
- iOS simulator support (future)
- WebAssembly runtime (future)

Examples:
    # Run in sandbox
    - id: run_sandboxed
      mcp: bbx.sandbox
      method: run
      inputs:
        command: "python untrusted.py"
        isolation: "container"

    # WebAssembly execution
    - id: run_wasm
      mcp: bbx.sandbox
      method: wasm
      inputs:
        module: "program.wasm"
"""

import os
import subprocess
from typing import Any, Dict

from blackbox.core.base_adapter import MCPAdapter


class SandboxAdapter(MCPAdapter):
    """Sandbox execution adapter"""

    def __init__(self):
        super().__init__("bbx.sandbox")
        self.has_docker = False
        self.has_wasmtime = False
        self._check_sandbox_tools()

    def _check_sandbox_tools(self):
        """Check available sandbox tools"""
        self.has_docker = self._check_command("docker")
        self.has_wasmtime = self._check_command("wasmtime")

    def _check_command(self, cmd: str) -> bool:
        """Check if command is available"""
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute sandbox method"""

        if method == "run":
            return await self._run(inputs)
        elif method == "wasm":
            return await self._wasm(inputs)
        elif method == "container":
            return await self._run_container(inputs)
        elif method == "status":
            return await self._status(inputs)
        else:
            raise ValueError(f"Unknown method: {method}")

    async def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run command in sandbox

        Inputs:
            command: Command to run
            isolation: Isolation type (container/process/none)
            image: Docker image (if isolation=container)
            timeout: Timeout in seconds
            env: Environment variables
        """
        inputs["command"]
        isolation = inputs.get("isolation", "process")

        if isolation == "container" and self.has_docker:
            return await self._run_container(inputs)
        else:
            return await self._run_process(inputs)

    async def _run_process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run in process sandbox (limited isolation)"""
        command = inputs["command"]
        timeout = inputs.get("timeout", 30)
        env = inputs.get("env", {})

        try:
            # On Windows, use shell=True for commands like 'echo'
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, **env},
                shell=True,  # Allow shell commands
            )

            return {
                "status": "completed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "isolation": "process",
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_container(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run in Docker container (strong isolation)"""
        command = inputs["command"]
        image = inputs.get("image", "python:3.11-alpine")
        timeout = inputs.get("timeout", 30)

        if not self.has_docker:
            return {"status": "error", "error": "Docker not available"}

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",  # No network access
            "--memory",
            "512m",  # Memory limit
            "--cpus",
            "1",  # CPU limit
            image,
            "sh",
            "-c",
            command,
        ]

        try:
            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=timeout
            )

            return {
                "status": "completed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "isolation": "container",
                "image": image,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Container timed out after {timeout}s",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _wasm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run WebAssembly module (Phase 7 - Future)

        Inputs:
            module: Path to .wasm file
            function: Function to call
            args: Function arguments
        """
        if not self.has_wasmtime:
            return {
                "status": "error",
                "error": "WebAssembly runtime (wasmtime) not available",
                "note": "Install: curl https://wasmtime.dev/install.sh -sSf | bash",
            }

        module = inputs["module"]
        function = inputs.get("function", "main")
        args = inputs.get("args", [])

        cmd = ["wasmtime", "run", module, "--invoke", function] + args

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            return {
                "status": "completed",
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "runtime": "wasmtime",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _status(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get sandbox capabilities"""
        return {
            "status": "ok",
            "capabilities": {
                "docker": self.has_docker,
                "wasmtime": self.has_wasmtime,
                "isolation_levels": (
                    ["process", "container"] if self.has_docker else ["process"]
                ),
            },
            "note": "Phase 7 features (Android/iOS/full WebAssembly) coming soon!",
        }
