# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX REST API Server

Provides HTTP and WebSocket APIs for workflow execution and management.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from blackbox.core.events import EventBus
from blackbox.core.runtime import run_file
from blackbox.server.websocket.server import manager

app = FastAPI(
    title="BBX API Server",
    description="Blackbox Workflow Engine REST API",
    version="1.0.0"
)
event_bus = EventBus()


class WorkflowRequest(BaseModel):
    workflow_id: str
    inputs: dict = {}


# =============================================================================
# Workflow Execution API
# =============================================================================

@app.post("/api/execute/{workflow_id}")
async def execute_workflow(workflow_id: str, request: WorkflowRequest):
    """Execute a workflow by ID."""
    file_path = f"workflows/{workflow_id}.bbx"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Workflow not found")

    try:
        result = await run_file(file_path, event_bus)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/workflows/execute")
async def remote_execute_workflow(
    workflow: UploadFile = File(...),
    inputs: dict = Body(default={}),
    wait: bool = Body(default=True)
):
    """Execute an uploaded workflow file."""
    execution_id = str(uuid.uuid4())

    # Save workflow temporarily
    workflow_content = await workflow.read()
    temp_dir = Path("/tmp") if os.name != "nt" else Path(os.environ.get("TEMP", "."))
    workflow_path = temp_dir / f"workflow_{execution_id}.bbx"

    with open(workflow_path, "wb") as f:
        f.write(workflow_content)

    # Execute workflow
    if wait:
        result = await run_file(str(workflow_path), event_bus, inputs=inputs)
        return {"execution_id": execution_id, "status": "completed", "output": result}
    else:
        asyncio.create_task(run_file(str(workflow_path), event_bus, inputs=inputs))
        return {"execution_id": execution_id, "status": "running"}


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
def root():
    """API root - returns basic info."""
    return {
        "name": "BBX API Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# =============================================================================
# Versioning API
# =============================================================================

@app.post("/api/workflows/{workflow_id}/versions")
async def create_workflow_version(
    workflow_id: str,
    version: str = Body(...),
    content: dict = Body(...),
    description: Optional[str] = Body(None),
):
    """Create a new workflow version."""
    from blackbox.core.versioning.manager import VersionManager

    mgr = VersionManager(Path("~/.bbx/versions").expanduser())
    version_obj = mgr.create_version(
        workflow_id=workflow_id,
        content=content,
        version=version,
        created_by="api",
        description=description,
    )

    return {"status": "success", "version": version_obj.dict()}


@app.get("/api/workflows/{workflow_id}/versions")
async def list_workflow_versions(workflow_id: str):
    """List all versions of a workflow."""
    from blackbox.core.versioning.manager import VersionManager

    mgr = VersionManager(Path("~/.bbx/versions").expanduser())
    versions = mgr.list_versions(workflow_id)

    return {"versions": [v.dict() for v in versions]}


@app.post("/api/workflows/{workflow_id}/rollback")
async def rollback_workflow(workflow_id: str, target_version: str = Body(...)):
    """Rollback workflow to a previous version."""
    from blackbox.core.versioning.manager import VersionManager

    mgr = VersionManager(Path("~/.bbx/versions").expanduser())
    rollback_version = mgr.rollback(workflow_id, target_version)

    return {"status": "success", "rollback_version": rollback_version.dict()}


@app.get("/api/workflows/{workflow_id}/diff")
async def diff_workflow_versions(workflow_id: str, from_version: str, to_version: str):
    """Get diff between two workflow versions."""
    from blackbox.core.versioning.manager import VersionManager

    mgr = VersionManager(Path("~/.bbx/versions").expanduser())
    diff = mgr.diff(workflow_id, from_version, to_version)

    return {"diff": diff.dict()}


# =============================================================================
# WebSocket API
# =============================================================================

@app.websocket("/ws/workflows/{workflow_id}")
async def websocket_workflow_updates(websocket: WebSocket, workflow_id: str):
    """WebSocket endpoint for workflow execution updates."""
    await manager.connect(websocket, workflow_id)

    try:
        await manager.send_personal_message(
            json.dumps({
                "event": "connected",
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat(),
            }),
            websocket,
        )

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await manager.send_personal_message(
                    json.dumps({"type": "pong"}), websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket, workflow_id)


@app.websocket("/ws/global")
async def websocket_global_updates(websocket: WebSocket):
    """WebSocket endpoint for global system updates."""
    await websocket.accept()

    try:
        while True:
            await asyncio.sleep(5)
            status = {
                "event": "system_status",
                "active_workflows": len(manager.active_connections),
                "timestamp": datetime.now().isoformat(),
            }
            await websocket.send_text(json.dumps(status))

    except WebSocketDisconnect:
        pass
