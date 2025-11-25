# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Workflow Adapter - Nested Workflow Execution

Enables workflows to spawn other workflows, creating:
- Hierarchical execution (parent → child workflows)
- Self-healing loops (monitor → restart on failure)
- Parallel workflow orchestration
- Background workflow execution

This is like fork() in Linux - workflows can spawn child workflows.

Examples:
    # Run a workflow synchronously (wait for completion)
    - id: deploy
      use: workflow.run
      inputs:
        path: workflows/deploy.bbx
        inputs:
          env: production

    # Run in background (like command &)
    - id: background_deploy
      use: workflow.run
      inputs:
        path: workflows/deploy.bbx
        background: true

    # Wait for background workflow
    - id: wait_deploy
      use: workflow.wait
      inputs:
        execution_id: ${{ steps.background_deploy.outputs.execution_id }}
        timeout: 300

    # Self-healing loop pattern
    - id: monitor
      use: workflow.run
      inputs:
        path: workflows/health_check.bbx
        on_failure:
          retry: 3
          fallback: workflows/recovery.bbx

    # Get status
    - id: check_status
      use: workflow.status
      inputs:
        execution_id: ${{ steps.background_deploy.outputs.execution_id }}

    # Kill a running workflow
    - id: abort
      use: workflow.kill
      inputs:
        execution_id: ${{ steps.background_deploy.outputs.execution_id }}

    # Get logs
    - id: get_logs
      use: workflow.logs
      inputs:
        execution_id: ${{ steps.background_deploy.outputs.execution_id }}
        limit: 50
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbx.adapters.workflow")


class WorkflowAdapter:
    """
    BBX Adapter for nested workflow execution.

    Provides OS-like process spawning for workflows:
    - run: Execute workflow (sync or background)
    - status: Check execution status
    - wait: Wait for completion
    - kill: Terminate execution
    - logs: Get execution logs
    - list: List executions (ps-like)
    """

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute workflow method"""

        if method == "run":
            return await self._run(
                path=inputs["path"],
                workflow_inputs=inputs.get("inputs"),
                background=inputs.get("background", False),
                on_failure=inputs.get("on_failure"),
                timeout=inputs.get("timeout"),
            )

        elif method == "status":
            return await self._status(
                execution_id=inputs["execution_id"],
            )

        elif method == "wait":
            return await self._wait(
                execution_id=inputs["execution_id"],
                timeout=inputs.get("timeout"),
            )

        elif method == "kill":
            return await self._kill(
                execution_id=inputs["execution_id"],
                force=inputs.get("force", False),
            )

        elif method == "logs":
            return await self._logs(
                execution_id=inputs["execution_id"],
                limit=inputs.get("limit", 100),
                level=inputs.get("level"),
            )

        elif method == "list":
            return await self._list(
                all=inputs.get("all", False),
            )

        else:
            raise ValueError(f"Unknown workflow method: {method}")

    async def _run(
        self,
        path: str,
        workflow_inputs: Optional[Dict[str, Any]] = None,
        background: bool = False,
        on_failure: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run a workflow.

        Args:
            path: Path to .bbx workflow file
            workflow_inputs: Inputs to pass to the workflow
            background: If True, run in background and return immediately
            on_failure: Failure handling config (retry, fallback)
            timeout: Timeout in seconds (for sync execution)

        Returns:
            For background: {"execution_id": "...", "status": "started"}
            For sync: {"status": "completed", "outputs": {...}} or error
        """
        # Import here to avoid circular imports
        from blackbox.core.execution_manager import get_execution_manager
        from blackbox.core.runtime import run_file
        from blackbox.core.workspace_manager import get_current_workspace

        # Resolve path
        workflow_path = Path(path)
        if not workflow_path.is_absolute():
            # Try relative to current workspace
            workspace = get_current_workspace()
            if workspace:
                workflow_path = workspace.paths.workflows / path
                if not workflow_path.exists():
                    workflow_path = workspace.root / path

            # Fall back to current directory
            if not workflow_path.exists():
                workflow_path = Path.cwd() / path

        if not workflow_path.exists():
            return {
                "status": "error",
                "error": f"Workflow not found: {path}",
            }

        manager = get_execution_manager()

        if background:
            # Run in background, return immediately
            workspace = get_current_workspace()
            workspace_path = str(workspace.root) if workspace else None

            execution_id = await manager.run_background(
                workflow_path=str(workflow_path),
                inputs=workflow_inputs,
                workspace_path=workspace_path,
            )

            return {
                "status": "started",
                "execution_id": execution_id,
                "background": True,
                "workflow_path": str(workflow_path),
            }

        else:
            # Run synchronously (with optional timeout)
            try:
                if timeout:
                    outputs = await asyncio.wait_for(
                        run_file(str(workflow_path), inputs=workflow_inputs),
                        timeout=timeout,
                    )
                else:
                    outputs = await run_file(str(workflow_path), inputs=workflow_inputs)

                return {
                    "status": "completed",
                    "outputs": outputs,
                    "workflow_path": str(workflow_path),
                }

            except asyncio.TimeoutError:
                return {
                    "status": "timeout",
                    "error": f"Workflow timed out after {timeout}s",
                    "workflow_path": str(workflow_path),
                }

            except Exception as e:
                error_result = {
                    "status": "failed",
                    "error": str(e),
                    "workflow_path": str(workflow_path),
                }

                # Handle on_failure if configured
                if on_failure:
                    return await self._handle_failure(
                        error_result,
                        on_failure,
                        workflow_path,
                        workflow_inputs,
                    )

                return error_result

    async def _handle_failure(
        self,
        error_result: Dict[str, Any],
        on_failure: Dict[str, Any],
        workflow_path: Path,
        workflow_inputs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Handle workflow failure with retry/fallback.

        on_failure config:
            retry: int - Number of retries
            retry_delay: float - Seconds between retries
            fallback: str - Path to fallback workflow
        """
        from blackbox.core.runtime import run_file

        retry_count = on_failure.get("retry", 0)
        retry_delay = on_failure.get("retry_delay", 1.0)
        fallback = on_failure.get("fallback")

        # Try retries
        for attempt in range(retry_count):
            logger.info(f"Retry attempt {attempt + 1}/{retry_count} for {workflow_path}")
            await asyncio.sleep(retry_delay)

            try:
                outputs = await run_file(str(workflow_path), inputs=workflow_inputs)
                return {
                    "status": "completed",
                    "outputs": outputs,
                    "workflow_path": str(workflow_path),
                    "retry_attempt": attempt + 1,
                }
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed: {e}")

        # Try fallback workflow
        if fallback:
            logger.info(f"Running fallback workflow: {fallback}")
            try:
                # Pass the error info to fallback
                fallback_inputs = {
                    **(workflow_inputs or {}),
                    "_error": error_result["error"],
                    "_failed_workflow": str(workflow_path),
                }
                outputs = await run_file(fallback, inputs=fallback_inputs)
                return {
                    "status": "fallback_completed",
                    "outputs": outputs,
                    "original_error": error_result["error"],
                    "fallback_workflow": fallback,
                }
            except Exception as e:
                return {
                    "status": "fallback_failed",
                    "original_error": error_result["error"],
                    "fallback_error": str(e),
                    "fallback_workflow": fallback,
                }

        # All retries failed, no fallback
        return {
            **error_result,
            "retries_attempted": retry_count,
        }

    async def _status(self, execution_id: str) -> Dict[str, Any]:
        """Get execution status"""
        from blackbox.core.execution_manager import get_execution_manager

        manager = get_execution_manager()
        execution = await manager.status(execution_id)

        if execution is None:
            return {
                "status": "not_found",
                "execution_id": execution_id,
            }

        return execution.to_dict()

    async def _wait(
        self,
        execution_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Wait for execution to complete"""
        from blackbox.core.execution_manager import get_execution_manager

        manager = get_execution_manager()
        execution = await manager.wait(execution_id, timeout=timeout)

        if execution is None:
            return {
                "status": "timeout",
                "execution_id": execution_id,
                "message": f"Execution did not complete within {timeout}s",
            }

        return execution.to_dict()

    async def _kill(
        self,
        execution_id: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Kill a running execution"""
        from blackbox.core.execution_manager import get_execution_manager

        manager = get_execution_manager()
        killed = await manager.kill(execution_id, force=force)

        return {
            "killed": killed,
            "execution_id": execution_id,
            "force": force,
        }

    async def _logs(
        self,
        execution_id: str,
        limit: int = 100,
        level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get execution logs"""
        from blackbox.core.execution_manager import get_execution_manager

        manager = get_execution_manager()
        logs = manager.get_logs(execution_id, limit=limit, level=level)

        return {
            "execution_id": execution_id,
            "logs": logs,
            "count": len(logs),
        }

    async def _list(self, all: bool = False) -> Dict[str, Any]:
        """List executions (like ps)"""
        from blackbox.core.execution_manager import get_execution_manager

        manager = get_execution_manager()
        executions = await manager.ps(all=all)

        return {
            "executions": [e.to_dict() for e in executions],
            "count": len(executions),
            "show_all": all,
        }
