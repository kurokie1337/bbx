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

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from strawberry.fastapi import GraphQLRouter

from blackbox.core.events import EventBus
from blackbox.core.runtime import run_file
from blackbox.graphql.schema import schema
from blackbox.server.websocket.server import manager

app = FastAPI(title="Blackbox Server")
event_bus = EventBus()


class WorkflowRequest(BaseModel):
    workflow_id: str
    inputs: dict = {}


@app.post("/api/execute/{workflow_id}")
async def execute_workflow(workflow_id: str, request: WorkflowRequest):
    # In a real system, we would look up the file path from DB
    # For MVP, we assume local file
    file_path = f"workflows/{workflow_id}.bbx"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Workflow not found")

    try:
        result = await run_file(file_path, event_bus)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# Versioning API Endpoints
@app.post("/api/workflows/{workflow_id}/versions")
async def create_workflow_version(
    workflow_id: str,
    version: str = Body(...),
    content: dict = Body(...),
    description: Optional[str] = Body(None),
):
    """Create a new workflow version."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    version_obj = manager.create_version(
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

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    versions = manager.list_versions(workflow_id)

    return {"versions": [v.dict() for v in versions]}


@app.post("/api/workflows/{workflow_id}/rollback")
async def rollback_workflow(workflow_id: str, target_version: str = Body(...)):
    """Rollback workflow to a previous version."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    rollback_version = manager.rollback(workflow_id, target_version)

    return {"status": "success", "rollback_version": rollback_version.dict()}


@app.get("/api/workflows/{workflow_id}/diff")
async def diff_workflow_versions(workflow_id: str, from_version: str, to_version: str):
    """Get diff between two workflow versions."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    diff = manager.diff(workflow_id, from_version, to_version)

    return {"diff": diff.dict()}

# WebSocket Endpoints

@app.websocket("/ws/workflows/{workflow_id}")
async def websocket_workflow_updates(websocket: WebSocket, workflow_id: str):
    """WebSocket endpoint for workflow execution updates."""
    await manager.connect(websocket, workflow_id)

    try:
        # Send initial connection message
        await manager.send_personal_message(
            json.dumps(
                {
                    "event": "connected",
                    "workflow_id": workflow_id,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            websocket,
        )

        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle ping/pong for keepalive
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
        # Send system status updates periodically
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


# Remote Execution Endpoints
@app.post("/api/workflows/execute")
async def remote_execute_workflow(
    workflow: UploadFile = File(...), inputs: dict = Body({}), wait: bool = Body(True)
):
    """Execute uploaded workflow."""
    execution_id = str(uuid.uuid4())

    # Save workflow temporarily
    workflow_content = await workflow.read()
    workflow_path = f"/tmp/workflow_{execution_id}.bbx"

    with open(workflow_path, "wb") as f:
        f.write(workflow_content)

    # Execute workflow
    if wait:
        # Execute synchronously
        result = await run_file(workflow_path, event_bus, inputs=inputs)
        return {"execution_id": execution_id, "status": "completed", "output": result}
    else:
        # Execute asynchronously (fire and forget)
        asyncio.create_task(run_file(workflow_path, event_bus, inputs=inputs))
        return {"execution_id": execution_id, "status": "running"}


# GraphQL Integration
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


# Marketplace API
@app.get("/api/marketplace/templates")
async def list_templates(category: Optional[str] = None):
    # Return list of workflow templates
    return {"templates": []}


@app.post("/api/marketplace/install")
async def install_template(template_id: str):
    # Install workflow template
    return {"status": "installed"}


@app.post("/api/marketplace/publish")
async def publish_template(template: dict):
    # Publish workflow to marketplace
    return {"status": "published", "template_id": "xxx"}


# Tenant middleware
@app.middleware("http")
async def tenant_middleware(request: Request, call_next):
    tenant_id = request.headers.get("X-Tenant-ID")
    if tenant_id:
        request.state.tenant_id = tenant_id
    return await call_next(request)


@app.post("/api/tenants")
async def create_tenant(tenant: dict):
    # Create new tenant
    return {"tenant_id": "tenant-123"}


# Scheduler API
@app.post("/api/schedules")
async def create_schedule(schedule: dict):
    """Create workflow schedule."""
    # Add schedule to Celery beat
    return {"schedule_id": "sched-123"}


@app.get("/api/schedules")
async def list_schedules():
    """List all schedules."""
    return {"schedules": []}


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete schedule."""
    return {"status": "deleted"}


# Serve Designer UI
app.mount(
    "/designer", StaticFiles(directory="designer/build", html=True), name="designer"
)
