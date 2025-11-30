"""
Execution API routes
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.bbx import bbx_bridge
from app.api.schemas.execution import (
    ExecutionState,
    ExecutionListItem,
    ExecutionLogsResponse,
)
from app.api.schemas.common import SuccessResponse

router = APIRouter()


@router.get("/", response_model=List[ExecutionListItem])
async def list_executions(
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List workflow executions"""
    # Get from bridge (in-memory for now)
    executions = []

    for exec_id, exec_data in bbx_bridge._active_executions.items():
        if workflow_id and exec_data.get("workflow_id") != workflow_id:
            continue
        if status and exec_data.get("status") != status:
            continue

        executions.append(ExecutionListItem(
            id=exec_id,
            workflow_id=exec_data.get("workflow_id", "unknown"),
            workflow_name=exec_data.get("workflow_name"),
            status=exec_data.get("status", "unknown"),
            started_at=exec_data.get("started_at"),
            completed_at=exec_data.get("completed_at"),
            duration_ms=exec_data.get("duration_ms"),
            step_count=0,
            steps_completed=0,
        ))

    # Apply pagination
    return executions[offset:offset + limit]


@router.get("/{execution_id}", response_model=ExecutionState)
async def get_execution(execution_id: str):
    """Get execution status"""
    execution = await bbx_bridge.get_execution(execution_id)

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # Convert to response model
    return ExecutionState(
        id=execution_id,
        workflow_id=execution.get("workflow_id", "unknown"),
        workflow_name=execution.get("workflow_name"),
        status=execution.get("status", "unknown"),
        inputs=execution.get("inputs", {}),
        results=execution.get("results", {}),
        error=execution.get("error"),
        started_at=execution.get("started_at"),
        completed_at=execution.get("completed_at"),
        duration_ms=execution.get("duration_ms"),
        steps={},
        current_level=0,
        progress=0,
    )


@router.post("/{execution_id}/cancel", response_model=SuccessResponse)
async def cancel_execution(execution_id: str):
    """Cancel a running execution"""
    success = await bbx_bridge.cancel_execution(execution_id)

    if not success:
        raise HTTPException(status_code=404, detail="Execution not found or already completed")

    return SuccessResponse(success=True, message="Execution cancelled")


@router.get("/{execution_id}/logs", response_model=ExecutionLogsResponse)
async def get_execution_logs(execution_id: str):
    """Get execution logs"""
    execution = await bbx_bridge.get_execution(execution_id)

    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # In real implementation, would fetch from database
    return ExecutionLogsResponse(
        execution_id=execution_id,
        logs=[],
    )
