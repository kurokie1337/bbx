# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""Event emitter for workflow execution events."""

from datetime import datetime
from typing import Any, Dict, Optional

from .server import manager


class WorkflowEventEmitter:
    """Emits workflow execution events to WebSocket clients."""

    @staticmethod
    async def emit_workflow_start(workflow_id: str, workflow_name: str):
        """Emit workflow start event."""
        await manager.broadcast_to_workflow(
            workflow_id,
            {
                "event": "workflow_start",
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    async def emit_workflow_complete(
        workflow_id: str, status: str, outputs: Optional[Dict[str, Any]] = None
    ):
        """Emit workflow completion event."""
        await manager.broadcast_to_workflow(
            workflow_id,
            {
                "event": "workflow_complete",
                "workflow_id": workflow_id,
                "status": status,
                "outputs": outputs,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    async def emit_step_start(workflow_id: str, step_id: str, step_name: str):
        """Emit step start event."""
        await manager.broadcast_to_workflow(
            workflow_id,
            {
                "event": "step_start",
                "workflow_id": workflow_id,
                "step_id": step_id,
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    async def emit_step_complete(
        workflow_id: str,
        step_id: str,
        status: str,
        output: Optional[Any] = None,
        error: Optional[str] = None,
    ):
        """Emit step completion event."""
        await manager.broadcast_to_workflow(
            workflow_id,
            {
                "event": "step_complete",
                "workflow_id": workflow_id,
                "step_id": step_id,
                "status": status,
                "output": output,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    async def emit_step_progress(
        workflow_id: str, step_id: str, progress: int, message: Optional[str] = None
    ):
        """Emit step progress update."""
        await manager.broadcast_to_workflow(
            workflow_id,
            {
                "event": "step_progress",
                "workflow_id": workflow_id,
                "step_id": step_id,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    async def emit_log(
        workflow_id: str, level: str, message: str, step_id: Optional[str] = None
    ):
        """Emit log message."""
        await manager.broadcast_to_workflow(
            workflow_id,
            {
                "event": "log",
                "workflow_id": workflow_id,
                "step_id": step_id,
                "level": level,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            },
        )

    @staticmethod
    async def emit_metric(
        workflow_id: str,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Emit metric update."""
        await manager.broadcast_to_workflow(
            workflow_id,
            {
                "event": "metric",
                "workflow_id": workflow_id,
                "metric_name": metric_name,
                "value": value,
                "labels": labels or {},
                "timestamp": datetime.now().isoformat(),
            },
        )


emitter = WorkflowEventEmitter()
