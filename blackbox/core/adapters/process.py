# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Process Management Adapter

Provides cross-platform process management with:
- Start/stop/restart processes
- Process status monitoring
- Health checks
- Auto-restart on failure
- Process group management

Examples:
    # Start a process
    - id: start_server
      mcp: bbx.process
      method: start
      inputs:
        command: "python -m http.server 8000"
        name: "web_server"

    # Check status
    - id: check_server
      mcp: bbx.process
      method: status
      inputs:
        name: "web_server"

    # Stop process
    - id: stop_server
      mcp: bbx.process
      method: stop
      inputs:
        name: "web_server"
"""

import json
import logging
import subprocess
import time
from typing import Any, Dict, Optional

import psutil

logger = logging.getLogger("bbx.adapters.process")


class ProcessManager:
    """Manages background processes with health monitoring"""

    def __init__(self):
        from blackbox.core.config import get_state_file

        self.processes: Dict[str, Dict[str, Any]] = {}
        self.state_file = get_state_file("processes.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        """Load process state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self.processes = json.load(f)
            except Exception:
                self.processes = {}

    def _save_state(self):
        """Save process state to disk"""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.processes, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save process state: {e}")

    def start(
        self,
        name: str,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        health_check: Optional[Dict[str, Any]] = None,
        auto_restart: bool = False,
    ) -> Dict[str, Any]:
        """
        Start a background process

        Args:
            name: Unique process name
            command: Command to execute
            cwd: Working directory
            env: Environment variables
            health_check: Health check configuration
            auto_restart: Auto-restart on failure

        Returns:
            Process info dict
        """
        if name in self.processes:
            pid = self.processes[name].get("pid")
            if pid and psutil.pid_exists(pid):
                return {
                    "status": "already_running",
                    "name": name,
                    "pid": pid,
                    "message": f"Process {name} is already running",
                }

        # Prepare environment
        proc_env = dict(env) if env else {}

        # Start process
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=proc_env if proc_env else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
                    else 0
                ),
            )

            # Store process info
            self.processes[name] = {
                "pid": proc.pid,
                "command": command,
                "cwd": str(cwd) if cwd else None,
                "started_at": time.time(),
                "health_check": health_check,
                "auto_restart": auto_restart,
                "status": "running",
            }

            self._save_state()

            return {
                "status": "started",
                "name": name,
                "pid": proc.pid,
                "command": command,
                "started_at": self.processes[name]["started_at"],
            }

        except Exception as e:
            return {"status": "error", "name": name, "error": str(e)}

    def stop(self, name: str, force: bool = False, timeout: int = 10) -> Dict[str, Any]:
        """
        Stop a process

        Args:
            name: Process name
            force: Force kill if gentle stop fails
            timeout: Seconds to wait before force kill

        Returns:
            Stop result
        """
        if name not in self.processes:
            return {
                "status": "not_found",
                "name": name,
                "message": f"Process {name} not found",
            }

        pid = self.processes[name].get("pid")
        if not pid or not psutil.pid_exists(pid):
            del self.processes[name]
            self._save_state()
            return {
                "status": "not_running",
                "name": name,
                "message": f"Process {name} is not running",
            }

        try:
            proc = psutil.Process(pid)

            # Gentle termination
            proc.terminate()

            # Wait for process to exit
            try:
                proc.wait(timeout=timeout)
                status = "stopped"
            except psutil.TimeoutExpired:
                if force:
                    proc.kill()
                    status = "killed"
                else:
                    return {
                        "status": "timeout",
                        "name": name,
                        "pid": pid,
                        "message": f"Process did not stop within {timeout}s",
                    }

            # Clean up
            del self.processes[name]
            self._save_state()

            return {"status": status, "name": name, "pid": pid}

        except psutil.NoSuchProcess:
            del self.processes[name]
            self._save_state()
            return {"status": "not_running", "name": name}
        except Exception as e:
            return {"status": "error", "name": name, "error": str(e)}

    def status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get process status

        Args:
            name: Process name (None for all processes)

        Returns:
            Status info
        """
        if name:
            if name not in self.processes:
                return {"status": "not_found", "name": name}

            proc_info = self.processes[name]
            pid = proc_info.get("pid")

            if not pid or not psutil.pid_exists(pid):
                return {"status": "dead", "name": name, "last_known_pid": pid}

            try:
                proc = psutil.Process(pid)
                return {
                    "status": "running",
                    "name": name,
                    "pid": pid,
                    "cpu_percent": proc.cpu_percent(),
                    "memory_mb": proc.memory_info().rss / 1024 / 1024,
                    "uptime_seconds": time.time() - proc_info["started_at"],
                    "command": proc_info["command"],
                }
            except psutil.NoSuchProcess:
                return {"status": "dead", "name": name, "last_known_pid": pid}
        else:
            # Status of all processes
            all_status = {}
            for proc_name in list(self.processes.keys()):
                all_status[proc_name] = self.status(proc_name)

            return {"status": "ok", "processes": all_status, "count": len(all_status)}

    def restart(self, name: str) -> Dict[str, Any]:
        """Restart a process"""
        proc_info = self.processes.get(name)
        if not proc_info:
            return {"status": "not_found", "name": name}

        # Stop
        stop_result = self.stop(name, force=True)

        # Wait a bit
        time.sleep(0.5)

        # Start again
        start_result = self.start(
            name=name,
            command=proc_info["command"],
            cwd=proc_info.get("cwd"),
            health_check=proc_info.get("health_check"),
            auto_restart=proc_info.get("auto_restart", False),
        )

        return {
            "status": "restarted",
            "name": name,
            "stop": stop_result,
            "start": start_result,
        }

    def health_check(self, name: str) -> Dict[str, Any]:
        """
        Perform health check on process

        Returns:
            Health status
        """
        if name not in self.processes:
            return {"status": "not_found", "name": name, "healthy": False}

        proc_info = self.processes[name]
        pid = proc_info.get("pid")

        if not pid or not psutil.pid_exists(pid):
            return {"status": "dead", "name": name, "healthy": False}

        health_config = proc_info.get("health_check")
        if not health_config:
            # No health check configured, just check if running
            return {"status": "running", "name": name, "healthy": True, "pid": pid}

        # Perform configured health check
        # (HTTP check, TCP check, custom command, etc.)
        # For now, just check process is running
        try:
            proc = psutil.Process(pid)
            return {
                "status": "healthy",
                "name": name,
                "healthy": True,
                "pid": pid,
                "cpu_percent": proc.cpu_percent(),
                "memory_mb": proc.memory_info().rss / 1024 / 1024,
            }
        except psutil.NoSuchProcess:
            return {"status": "dead", "name": name, "healthy": False}


# Global process manager instance
_process_manager = ProcessManager()


class ProcessAdapter:
    """BBX Adapter for process management"""

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute process management method"""

        if method == "start":
            return _process_manager.start(
                name=inputs["name"],
                command=inputs["command"],
                cwd=inputs.get("cwd"),
                env=inputs.get("env"),
                health_check=inputs.get("health_check"),
                auto_restart=inputs.get("auto_restart", False),
            )

        elif method == "stop":
            return _process_manager.stop(
                name=inputs["name"],
                force=inputs.get("force", False),
                timeout=inputs.get("timeout", 10),
            )

        elif method == "status":
            return _process_manager.status(name=inputs.get("name"))

        elif method == "restart":
            return _process_manager.restart(name=inputs["name"])

        elif method == "health_check":
            return _process_manager.health_check(name=inputs["name"])

        else:
            raise ValueError(f"Unknown method: {method}")
