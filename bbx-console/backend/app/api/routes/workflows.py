"""
Workflow API routes
"""

from typing import List

from fastapi import APIRouter, HTTPException, Depends

from app.bbx import bbx_bridge
from app.api.schemas.workflow import (
    WorkflowListItem,
    WorkflowDetail,
    WorkflowValidationRequest,
    WorkflowValidationResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from app.ws import ws_manager

router = APIRouter()


@router.get("/", response_model=List[WorkflowListItem])
async def list_workflows(directory: str = "."):
    """List all workflows in directory"""
    workflows = await bbx_bridge.list_workflows(directory)
    return [WorkflowListItem(**wf) for wf in workflows]


@router.get("/{workflow_id}", response_model=WorkflowDetail)
async def get_workflow(workflow_id: str, file_path: str = None):
    """Get workflow details"""
    # If file_path provided, use it; otherwise construct from ID
    path = file_path or f"{workflow_id}.bbx"

    workflow = await bbx_bridge.get_workflow(path)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return WorkflowDetail(**workflow)


@router.post("/validate", response_model=WorkflowValidationResponse)
async def validate_workflow(request: WorkflowValidationRequest):
    """Validate workflow YAML content"""
    import yaml

    errors = []
    warnings = []

    try:
        data = yaml.safe_load(request.content)

        # Check for workflow key
        wf = data.get("workflow", data)

        # Check required fields
        if "steps" not in wf:
            errors.append("Missing 'steps' field")
        else:
            steps = wf["steps"]
            step_ids = set()

            for idx, step in enumerate(steps):
                if "id" not in step:
                    errors.append(f"Step {idx}: missing 'id' field")
                else:
                    if step["id"] in step_ids:
                        errors.append(f"Duplicate step ID: {step['id']}")
                    step_ids.add(step["id"])

                if "mcp" not in step:
                    errors.append(f"Step {step.get('id', idx)}: missing 'mcp' field")

                if "method" not in step:
                    errors.append(f"Step {step.get('id', idx)}: missing 'method' field")

                # Check dependencies reference valid steps
                depends_on = step.get("depends_on", [])
                for dep in depends_on:
                    if dep not in step_ids:
                        # Might reference a later step, add as warning
                        warnings.append(f"Step {step.get('id')}: depends on '{dep}' which appears later")

        # Check for cycles (simplified)
        # Full cycle detection done in DAG class

    except yaml.YAMLError as e:
        errors.append(f"YAML syntax error: {str(e)}")

    return WorkflowValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


@router.post("/{workflow_id}/run", response_model=WorkflowRunResponse)
async def run_workflow(workflow_id: str, request: WorkflowRunRequest, file_path: str = None):
    """Run a workflow"""
    path = file_path or f"{workflow_id}.bbx"

    # Check workflow exists
    workflow = await bbx_bridge.get_workflow(path)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    # WebSocket callback for real-time updates
    async def ws_callback(event: str, data: dict):
        await ws_manager.broadcast(
            f"execution:{data.get('execution_id')}",
            event,
            data
        )

    try:
        execution_id = await bbx_bridge.run_workflow(
            path,
            request.inputs,
            ws_callback=ws_callback,
        )

        # Broadcast execution started
        await ws_manager.broadcast(
            "executions",
            "execution:started",
            {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "inputs": request.inputs,
            }
        )

        return WorkflowRunResponse(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status="running",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
