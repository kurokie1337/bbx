# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
A2A HTTP Server

FastAPI-based server implementing Google Agent2Agent Protocol.
Enables BBX to receive tasks from other A2A agents.

Endpoints:
- GET  /.well-known/agent-card.json  - Agent Card (capability discovery)
- POST /a2a/tasks                     - Create new task
- GET  /a2a/tasks/{id}               - Get task status
- POST /a2a/tasks/{id}/cancel        - Cancel task
- GET  /a2a/tasks/{id}/stream        - SSE stream for task updates
- POST /a2a/rpc                       - JSON-RPC 2.0 endpoint
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import (
    AgentCard,
    A2ATask,
    A2ATaskInput,
    A2ATaskStatus,
    A2AMessage,
    A2AMessageRole,
    JsonRpcRequest,
    JsonRpcResponse,
)
from .agent_card import AgentCardGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# Task Store (in-memory, can be replaced with persistent store)
# =============================================================================

class TaskStore:
    """In-memory task storage with optional persistence."""

    def __init__(self):
        self.tasks: Dict[str, A2ATask] = {}
        self._subscribers: Dict[str, list] = {}  # task_id -> list of queues

    def create(self, task: A2ATask) -> A2ATask:
        """Store a new task."""
        self.tasks[task.id] = task
        self._subscribers[task.id] = []
        return task

    def get(self, task_id: str) -> Optional[A2ATask]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def update(self, task: A2ATask) -> A2ATask:
        """Update task and notify subscribers."""
        self.tasks[task.id] = task
        self._notify_subscribers(task)
        return task

    def list_tasks(self, limit: int = 100) -> list:
        """List recent tasks."""
        tasks = list(self.tasks.values())
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """Subscribe to task updates."""
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[task_id].append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        """Unsubscribe from task updates."""
        if task_id in self._subscribers:
            try:
                self._subscribers[task_id].remove(queue)
            except ValueError:
                pass

    def _notify_subscribers(self, task: A2ATask):
        """Notify all subscribers of task update."""
        if task.id in self._subscribers:
            for queue in self._subscribers[task.id]:
                try:
                    queue.put_nowait(task)
                except asyncio.QueueFull:
                    pass


# Global task store
_task_store = TaskStore()


# =============================================================================
# Task Executor
# =============================================================================

class TaskExecutor:
    """Executes A2A tasks by mapping to BBX operations."""

    def __init__(self, workspace_path: Optional[str] = None):
        self.workspace_path = workspace_path

    async def execute(self, task: A2ATask) -> A2ATask:
        """Execute a task and return updated task."""
        task.status = A2ATaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        task.messages.append(A2AMessage(
            role=A2AMessageRole.AGENT,
            content=f"Starting execution of skill: {task.skill_id}"
        ))
        _task_store.update(task)

        try:
            # Route to appropriate handler based on skill_id
            if task.skill_id.startswith("bbx."):
                result = await self._execute_builtin(task)
            elif task.skill_id.startswith("workflow."):
                result = await self._execute_workflow(task)
            else:
                raise ValueError(f"Unknown skill: {task.skill_id}")

            task.status = A2ATaskStatus.COMPLETED
            task.output = result
            task.completed_at = datetime.utcnow()
            task.messages.append(A2AMessage(
                role=A2AMessageRole.AGENT,
                content="Task completed successfully"
            ))

        except Exception as e:
            task.status = A2ATaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            task.messages.append(A2AMessage(
                role=A2AMessageRole.AGENT,
                content=f"Task failed: {str(e)}"
            ))
            logger.exception(f"Task {task.id} failed")

        _task_store.update(task)
        return task

    async def _execute_builtin(self, task: A2ATask) -> Dict[str, Any]:
        """Execute built-in BBX skill."""
        skill = task.skill_id.replace("bbx.", "")

        if skill == "run_workflow":
            return await self._run_workflow(task.input)
        elif skill == "state_management":
            return await self._manage_state(task.input)
        elif skill == "process_management":
            return await self._manage_process(task.input)
        elif skill == "mcp_bridge":
            return await self._mcp_call(task.input)
        elif skill == "mcp_discover":
            return await self._mcp_discover()
        else:
            raise ValueError(f"Unknown builtin skill: {skill}")

    async def _execute_workflow(self, task: A2ATask) -> Dict[str, Any]:
        """Execute a workflow skill."""
        workflow_id = task.skill_id.replace("workflow.", "")
        return await self._run_workflow({
            "workflow": workflow_id,
            "inputs": task.input
        })

    async def _run_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a BBX workflow."""
        from blackbox.core.runtime import run_file

        workflow = params.get("workflow")
        inputs = params.get("inputs", {})
        background = params.get("background", False)

        if background:
            from blackbox.core.execution_manager import get_execution_manager
            manager = get_execution_manager()
            exec_id = await manager.run_background(workflow, inputs)
            return {
                "execution_id": exec_id,
                "status": "started",
                "background": True
            }
        else:
            results = await run_file(workflow, inputs=inputs)
            return {
                "status": "completed",
                "outputs": results
            }

    async def _manage_state(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage persistent state."""
        from blackbox.core.adapters.state import StateAdapter

        adapter = StateAdapter()
        operation = params.get("operation")
        key = params.get("key")
        value = params.get("value")
        namespace = params.get("namespace")

        if operation == "get":
            result = await adapter._get(key=key, namespace=namespace)
            return {"value": result.get("value"), "found": result.get("found")}
        elif operation == "set":
            result = await adapter._set(key=key, value=value, namespace=namespace)
            return result
        elif operation == "delete":
            result = await adapter._delete(key=key, namespace=namespace)
            return result
        elif operation == "list":
            result = await adapter._keys(pattern=params.get("pattern", "*"), namespace=namespace)
            return result
        elif operation == "increment":
            result = await adapter._increment(key=key, by=params.get("by", 1), namespace=namespace)
            return result
        elif operation == "append":
            result = await adapter._append(key=key, value=value, namespace=namespace)
            return result
        else:
            raise ValueError(f"Unknown state operation: {operation}")

    async def _manage_process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manage workflow processes."""
        from blackbox.core.execution_manager import get_execution_manager

        manager = get_execution_manager()
        operation = params.get("operation")
        exec_id = params.get("execution_id")

        if operation == "ps":
            executions = await manager.ps(all=params.get("all", False))
            return {
                "executions": [
                    {
                        "id": e.execution_id,
                        "status": e.status.value,
                        "workflow": e.workflow_path,
                        "started_at": str(e.started_at) if e.started_at else None
                    }
                    for e in executions
                ]
            }
        elif operation == "kill":
            result = await manager.kill(exec_id, force=params.get("force", False))
            return {"killed": result}
        elif operation == "wait":
            result = await manager.wait(exec_id, timeout=params.get("timeout"))
            return {
                "status": result.status.value if result else "timeout",
                "outputs": result.outputs if result else None
            }
        elif operation == "logs":
            from blackbox.core.execution_store import get_execution_store
            store = get_execution_store()
            logs = store.get_logs(exec_id, limit=params.get("limit", 50))
            return {"logs": logs}
        else:
            raise ValueError(f"Unknown process operation: {operation}")

    async def _mcp_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        from blackbox.mcp.client.manager import get_mcp_manager

        server = params.get("server")
        tool = params.get("tool")
        arguments = params.get("arguments", {})

        manager = get_mcp_manager()
        await manager.load_config()
        result = await manager.call_tool(server, tool, arguments)

        return {"result": result}

    async def _mcp_discover(self) -> Dict[str, Any]:
        """Discover MCP tools."""
        from blackbox.mcp.client.manager import get_mcp_manager

        manager = get_mcp_manager()
        await manager.load_config()
        tools = await manager.list_all_tools()

        return {"tools": tools}


# Global executor
_executor = TaskExecutor()


# =============================================================================
# FastAPI Application
# =============================================================================

def create_a2a_app(
    name: str = "bbx-agent",
    description: str = "BBX Operating System for AI Agents",
    url: str = "http://localhost:8000",
    workspace_path: Optional[str] = None,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create FastAPI application for A2A server.

    Args:
        name: Agent name for Agent Card
        description: Agent description
        url: Base URL of this agent
        workspace_path: Path to BBX workspace
        cors_origins: List of allowed CORS origins

    Returns:
        Configured FastAPI application
    """

    # Generate Agent Card
    card_generator = AgentCardGenerator(
        name=name,
        description=description,
        url=url,
        workspace_path=workspace_path,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan handler."""
        logger.info(f"Starting A2A server: {name}")
        logger.info(f"Agent Card available at: {url}/.well-known/agent-card.json")
        yield
        logger.info("Shutting down A2A server")

    app = FastAPI(
        title=f"{name} - A2A Agent",
        description=description,
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # =========================================================================
    # Agent Card Endpoint
    # =========================================================================

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card() -> dict:
        """
        Agent Card endpoint per A2A spec.

        Returns JSON metadata describing this agent's capabilities.
        """
        card = card_generator.generate()
        return card.model_dump(by_alias=True)

    # =========================================================================
    # Task Endpoints
    # =========================================================================

    @app.post("/a2a/tasks", response_model=dict)
    async def create_task(
        task_input: A2ATaskInput,
        background_tasks: BackgroundTasks
    ) -> dict:
        """
        Create a new task for this agent to execute.

        The task will be queued and executed asynchronously.
        Use the returned task ID to check status or stream updates.
        """
        # Create task
        task = A2ATask(
            skill_id=task_input.skill_id,
            input=task_input.input,
            metadata=task_input.metadata,
        )

        # Add initial message
        task.messages.append(A2AMessage(
            role=A2AMessageRole.USER,
            content=f"Execute skill: {task_input.skill_id}",
            data=task_input.input
        ))

        # Store task
        _task_store.create(task)

        # Execute in background
        background_tasks.add_task(_executor.execute, task)

        logger.info(f"Created task {task.id} for skill {task.skill_id}")

        return task.model_dump(by_alias=True)

    @app.get("/a2a/tasks/{task_id}")
    async def get_task(task_id: str) -> dict:
        """Get task status and results."""
        task = _task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return task.model_dump(by_alias=True)

    @app.post("/a2a/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str) -> dict:
        """Cancel a running task."""
        task = _task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        if task.status in [A2ATaskStatus.COMPLETED, A2ATaskStatus.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task in {task.status} state"
            )

        task.status = A2ATaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        task.messages.append(A2AMessage(
            role=A2AMessageRole.AGENT,
            content="Task cancelled by request"
        ))
        _task_store.update(task)

        # If linked to BBX execution, kill it
        if task.bbx_execution_id:
            try:
                from blackbox.core.execution_manager import get_execution_manager
                manager = get_execution_manager()
                await manager.kill(task.bbx_execution_id)
            except Exception:
                pass

        return task.model_dump(by_alias=True)

    @app.get("/a2a/tasks/{task_id}/stream")
    async def stream_task(task_id: str) -> StreamingResponse:
        """
        SSE stream for task updates.

        Clients can subscribe to real-time updates for a task.
        Events are sent as Server-Sent Events (SSE).
        """
        task = _task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        async def event_generator():
            queue = _task_store.subscribe(task_id)
            try:
                # Send current state
                yield task.to_sse_event("initial")

                # Stream updates
                while True:
                    try:
                        updated_task = await asyncio.wait_for(queue.get(), timeout=30)
                        yield updated_task.to_sse_event("update")

                        # Stop streaming on terminal states
                        if updated_task.status in [
                            A2ATaskStatus.COMPLETED,
                            A2ATaskStatus.FAILED,
                            A2ATaskStatus.CANCELLED
                        ]:
                            yield updated_task.to_sse_event("complete")
                            break
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield ": keepalive\n\n"
            finally:
                _task_store.unsubscribe(task_id, queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    @app.get("/a2a/tasks")
    async def list_tasks(limit: int = 100) -> dict:
        """List recent tasks."""
        tasks = _task_store.list_tasks(limit=limit)
        return {
            "tasks": [t.model_dump(by_alias=True) for t in tasks],
            "count": len(tasks)
        }

    # =========================================================================
    # JSON-RPC Endpoint (A2A standard)
    # =========================================================================

    @app.post("/a2a/rpc")
    async def json_rpc(request: JsonRpcRequest, background_tasks: BackgroundTasks) -> dict:
        """
        JSON-RPC 2.0 endpoint per A2A spec.

        Supports methods:
        - tasks/create: Create new task
        - tasks/get: Get task status
        - tasks/cancel: Cancel task
        - tasks/list: List tasks
        - agent/card: Get agent card
        """
        try:
            method = request.method
            params = request.params or {}

            if method == "tasks/create":
                task_input = A2ATaskInput(**params)
                task = A2ATask(
                    skill_id=task_input.skill_id,
                    input=task_input.input,
                    metadata=task_input.metadata,
                )
                _task_store.create(task)
                background_tasks.add_task(_executor.execute, task)
                result = task.model_dump(by_alias=True)

            elif method == "tasks/get":
                task = _task_store.get(params.get("task_id"))
                if not task:
                    return JsonRpcResponse(
                        id=request.id,
                        error={"code": -32001, "message": "Task not found"}
                    ).model_dump()
                result = task.model_dump(by_alias=True)

            elif method == "tasks/cancel":
                task = _task_store.get(params.get("task_id"))
                if not task:
                    return JsonRpcResponse(
                        id=request.id,
                        error={"code": -32001, "message": "Task not found"}
                    ).model_dump()
                task.status = A2ATaskStatus.CANCELLED
                task.completed_at = datetime.utcnow()
                _task_store.update(task)
                result = task.model_dump(by_alias=True)

            elif method == "tasks/list":
                tasks = _task_store.list_tasks(limit=params.get("limit", 100))
                result = {"tasks": [t.model_dump(by_alias=True) for t in tasks]}

            elif method == "agent/card":
                card = card_generator.generate()
                result = card.model_dump(by_alias=True)

            else:
                return JsonRpcResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Method not found: {method}"}
                ).model_dump()

            return JsonRpcResponse(id=request.id, result=result).model_dump()

        except Exception as e:
            return JsonRpcResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)}
            ).model_dump()

    # =========================================================================
    # Health Check
    # =========================================================================

    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "agent": name,
            "protocol": "a2a",
            "version": "0.3"
        }

    return app


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    name: str = "bbx-agent",
    workspace_path: Optional[str] = None,
):
    """
    Run A2A server.

    Args:
        host: Host to bind to
        port: Port to listen on
        name: Agent name
        workspace_path: Path to BBX workspace
    """
    import uvicorn

    url = f"http://{host}:{port}"
    if host == "0.0.0.0":
        url = f"http://localhost:{port}"

    app = create_a2a_app(
        name=name,
        url=url,
        workspace_path=workspace_path,
        cors_origins=["*"],
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    host = "0.0.0.0"
    port = 8000

    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    run_server(host=host, port=port)
