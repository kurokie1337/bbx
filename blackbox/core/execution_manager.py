# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Execution Manager

Manages workflow execution lifecycle:
- Run workflows in background (like `command &` in Linux)
- List running processes (like `ps`)
- Kill processes (like `kill`)
- Wait for completion (like `wait`)
- Stream logs (like `tail -f`)

This is the "process manager" of BBX - the OS for AI agents.
"""

import asyncio
import logging
import os
import signal
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .events import EventBus, EventType
from .execution_store import (
    Execution,
    ExecutionStatus,
    ExecutionStore,
    get_execution_store,
)

logger = logging.getLogger("bbx.execution_manager")


class ExecutionManager:
    """
    Manages background workflow executions.

    Like a process manager in an OS:
    - `run_background()` → start a process
    - `ps()` → list processes
    - `kill()` → terminate a process
    - `wait()` → wait for process to finish
    - `logs()` → get process output

    Usage:
        manager = ExecutionManager()

        # Start background execution
        exec_id = await manager.run_background("deploy.bbx")
        print(f"Started: {exec_id}")

        # List running
        running = await manager.ps()

        # Get logs
        async for log in manager.logs(exec_id, follow=True):
            print(log)

        # Wait for completion
        result = await manager.wait(exec_id, timeout=300)

        # Or kill it
        await manager.kill(exec_id)
    """

    def __init__(self, store: Optional[ExecutionStore] = None):
        self.store = store or get_execution_store()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._event_buses: Dict[str, EventBus] = {}

    # ==========================================================================
    # Background Execution
    # ==========================================================================

    async def run_background(
        self,
        workflow_path: str,
        inputs: Optional[Dict[str, Any]] = None,
        workspace_path: Optional[str] = None,
    ) -> str:
        """
        Run a workflow in background.

        Like `command &` in Linux.

        Args:
            workflow_path: Path to .bbx workflow file
            inputs: Workflow inputs
            workspace_path: Optional workspace context

        Returns:
            execution_id (like a PID)
        """
        # Generate execution ID
        execution_id = str(uuid.uuid4())

        # Extract workflow ID from file
        workflow_id = Path(workflow_path).stem

        # Create execution record
        execution = self.store.create(
            execution_id=execution_id,
            workflow_path=workflow_path,
            workflow_id=workflow_id,
            inputs=inputs,
            workspace_path=workspace_path,
            background=True,
        )

        # Create event bus for this execution
        event_bus = EventBus()
        self._event_buses[execution_id] = event_bus

        # Subscribe to events for logging
        event_bus.subscribe(
            EventType.STEP_START,
            lambda e: self._log_event(execution_id, "INFO", f"Step started: {e.data.get('step_id')}", e.data.get('step_id'), e.data),
        )
        event_bus.subscribe(
            EventType.STEP_COMPLETE,
            lambda e: self._log_event(execution_id, "INFO", f"Step completed: {e.data.get('step_id')}", e.data.get('step_id'), e.data),
        )
        event_bus.subscribe(
            EventType.STEP_ERROR,
            lambda e: self._log_event(execution_id, "ERROR", f"Step error: {e.data.get('step_id')}: {e.data.get('error')}", e.data.get('step_id'), e.data),
        )

        # Start background task
        task = asyncio.create_task(
            self._run_with_tracking(execution_id, workflow_path, inputs, event_bus)
        )
        self._running_tasks[execution_id] = task

        logger.info(f"Started background execution: {execution_id}")
        return execution_id

    async def _run_with_tracking(
        self,
        execution_id: str,
        workflow_path: str,
        inputs: Optional[Dict[str, Any]],
        event_bus: EventBus,
    ):
        """Internal: Run workflow with status tracking"""
        # Import here to avoid circular imports
        from .runtime import run_file

        # Update status to running
        self.store.update_status(execution_id, ExecutionStatus.RUNNING)
        self._log_event(execution_id, "INFO", f"Workflow started: {workflow_path}")

        try:
            # Run the workflow
            results = await run_file(
                workflow_path,
                event_bus=event_bus,
                inputs=inputs,
            )

            # Update with results
            self.store.update_outputs(execution_id, results)
            self.store.update_status(execution_id, ExecutionStatus.COMPLETED)
            self._log_event(execution_id, "INFO", "Workflow completed successfully")

        except asyncio.CancelledError:
            # Task was cancelled (killed)
            self.store.update_status(
                execution_id,
                ExecutionStatus.CANCELLED,
                error="Execution was cancelled",
            )
            self._log_event(execution_id, "WARNING", "Workflow was cancelled")
            raise

        except Exception as e:
            # Workflow failed
            error_msg = str(e)
            self.store.update_status(
                execution_id,
                ExecutionStatus.FAILED,
                error=error_msg,
            )
            self._log_event(execution_id, "ERROR", f"Workflow failed: {error_msg}")
            logger.error(f"Execution {execution_id} failed: {e}")

        finally:
            # Cleanup
            self._running_tasks.pop(execution_id, None)
            self._event_buses.pop(execution_id, None)

    def _log_event(
        self,
        execution_id: str,
        level: str,
        message: str,
        step_id: Optional[str] = None,
        data: Optional[Dict] = None,
    ):
        """Add log entry for execution"""
        self.store.add_log(execution_id, level, message, step_id, data)

    # ==========================================================================
    # Process Management (ps, kill, wait)
    # ==========================================================================

    async def ps(
        self,
        all: bool = False,
        workspace_path: Optional[str] = None,
    ) -> List[Execution]:
        """
        List executions.

        Like `ps` in Linux.

        Args:
            all: If True, show all (not just running). Like `ps aux`
            workspace_path: Filter by workspace

        Returns:
            List of Execution objects
        """
        if all:
            return self.store.list(workspace_path=workspace_path, limit=50)
        else:
            return self.store.list(
                status=ExecutionStatus.RUNNING,
                workspace_path=workspace_path,
            )

    async def kill(self, execution_id: str, force: bool = False) -> bool:
        """
        Kill a running execution.

        Like `kill` in Linux.

        Args:
            execution_id: Execution to kill
            force: If True, force kill (like kill -9)

        Returns:
            True if killed successfully
        """
        # Check if it's in our running tasks
        if execution_id in self._running_tasks:
            task = self._running_tasks[execution_id]

            if force:
                task.cancel()
            else:
                # Try graceful cancellation first
                task.cancel()

            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

            self.store.update_status(
                execution_id,
                ExecutionStatus.CANCELLED,
                error="Killed by user",
            )

            logger.info(f"Killed execution: {execution_id}")
            return True

        # Check if it's in the database but marked as running
        execution = self.store.get(execution_id)
        if execution and execution.status == ExecutionStatus.RUNNING:
            # Mark as cancelled even if we can't actually stop it
            self.store.update_status(
                execution_id,
                ExecutionStatus.CANCELLED,
                error="Killed by user (process may have already terminated)",
            )
            return True

        return False

    async def wait(
        self,
        execution_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Execution]:
        """
        Wait for execution to complete.

        Like `wait` in Linux.

        Args:
            execution_id: Execution to wait for
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            Execution object with final status, or None if timeout
        """
        # If we have the task, wait for it directly
        if execution_id in self._running_tasks:
            task = self._running_tasks[execution_id]
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                return None
            except asyncio.CancelledError:
                pass

        # Return the final execution state
        return self.store.get(execution_id)

    # ==========================================================================
    # Logs
    # ==========================================================================

    def get_logs(
        self,
        execution_id: str,
        limit: int = 100,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get logs for an execution"""
        return self.store.get_logs(execution_id, limit=limit, level=level)

    async def logs(
        self,
        execution_id: str,
        follow: bool = False,
        poll_interval: float = 0.5,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream logs for an execution.

        Like `tail -f` in Linux.

        Args:
            execution_id: Execution ID
            follow: If True, keep streaming new logs
            poll_interval: How often to check for new logs (seconds)

        Yields:
            Log entries
        """
        last_id = 0

        while True:
            logs, last_id = self.store.get_logs_since(execution_id, last_id)

            for log in logs:
                yield log

            if not follow:
                break

            # Check if execution is still running
            execution = self.store.get(execution_id)
            if execution and execution.is_finished:
                # Get any remaining logs
                remaining_logs, _ = self.store.get_logs_since(execution_id, last_id)
                for log in remaining_logs:
                    yield log
                break

            await asyncio.sleep(poll_interval)

    # ==========================================================================
    # Status & Info
    # ==========================================================================

    async def status(self, execution_id: str) -> Optional[Execution]:
        """Get execution status"""
        return self.store.get(execution_id)

    async def info(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed execution info"""
        execution = self.store.get(execution_id)
        if not execution:
            return None

        info = execution.to_dict()
        info["logs_count"] = len(self.store.get_logs(execution_id))
        info["is_in_memory"] = execution_id in self._running_tasks

        return info

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    async def cleanup_stale(self) -> int:
        """Mark stale running executions as failed"""
        return self.store.mark_stale_as_failed()

    async def cleanup_old(self, keep_days: int = 7) -> int:
        """Clean up old execution records"""
        return self.store.cleanup_old(keep_days)


# ==========================================================================
# Global Manager Instance
# ==========================================================================

_manager: Optional[ExecutionManager] = None


def get_execution_manager() -> ExecutionManager:
    """Get global ExecutionManager instance"""
    global _manager
    if _manager is None:
        _manager = ExecutionManager()
    return _manager


# ==========================================================================
# Convenience Functions
# ==========================================================================

async def run_background(
    workflow_path: str,
    inputs: Optional[Dict[str, Any]] = None,
    workspace_path: Optional[str] = None,
) -> str:
    """Run workflow in background, return execution_id"""
    return await get_execution_manager().run_background(
        workflow_path, inputs, workspace_path
    )


async def ps(all: bool = False) -> List[Execution]:
    """List executions (running by default, all with all=True)"""
    return await get_execution_manager().ps(all=all)


async def kill(execution_id: str, force: bool = False) -> bool:
    """Kill an execution"""
    return await get_execution_manager().kill(execution_id, force)


async def wait(execution_id: str, timeout: Optional[float] = None) -> Optional[Execution]:
    """Wait for execution to complete"""
    return await get_execution_manager().wait(execution_id, timeout)
