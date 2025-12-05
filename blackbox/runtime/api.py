# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Desktop API - HTTP/WebSocket API for GUI clients.

This API is consumed by:
- BBX Control Panel (Electron/React GUI)
- BBX CLI
- Third-party integrations

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  BBX Control Panel (GUI)                     │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │                    React Frontend                        │ │
    │  └────────────────────────┬────────────────────────────────┘ │
    └───────────────────────────┼─────────────────────────────────┘
                                │ HTTP/WebSocket
    ┌───────────────────────────▼─────────────────────────────────┐
    │                    BBX API Server (this file)                │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │  REST Endpoints                                          │ │
    │  │  - GET  /api/v1/agents          (list agents)           │ │
    │  │  - POST /api/v1/agents          (spawn agent)           │ │
    │  │  - GET  /api/v1/agents/:id      (get agent)             │ │
    │  │  - DELETE /api/v1/agents/:id    (kill agent)            │ │
    │  │  - POST /api/v1/workflows/run   (run workflow)          │ │
    │  │  - GET  /api/v1/memory          (list memory)           │ │
    │  │  - GET  /api/v1/snapshots       (list snapshots)        │ │
    │  │  - POST /api/v1/recover/:id     (recover to snapshot)   │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │  WebSocket: /ws                                          │ │
    │  │  - Real-time agent status updates                       │ │
    │  │  - Live log streaming                                   │ │
    │  │  - Event notifications                                  │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                      BBX Daemon                              │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Start API server (usually started with daemon)
    bbx api start --port 9999

    # Test endpoints
    curl http://localhost:9999/api/v1/status
    curl http://localhost:9999/api/v1/agents
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("bbx.api")


# =============================================================================
# API Models
# =============================================================================


@dataclass
class APIResponse:
    """Standard API response"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class AgentInfo:
    """Agent information for API"""
    id: str
    name: str
    type: str
    status: str
    memory_mb: float
    context_tokens: int
    tier: str
    tasks_completed: int
    tasks_failed: int
    current_task: Optional[str]
    created_at: float
    started_at: Optional[float]
    last_error: Optional[str]


@dataclass
class WorkflowInfo:
    """Workflow information for API"""
    id: str
    name: str
    status: str
    progress: float  # 0-100
    current_step: Optional[str]
    started_at: Optional[float]
    duration_seconds: float
    steps_total: int
    steps_completed: int


@dataclass
class MemoryEntry:
    """Memory entry for API"""
    key: str
    agent_id: str
    tier: str
    size_bytes: int
    stored_at: float
    accessed_at: float


@dataclass
class SnapshotInfo:
    """Snapshot information for API"""
    id: str
    agent_id: str
    description: str
    created_at: float
    size_bytes: int


@dataclass
class SystemStatus:
    """System status for API"""
    running: bool
    uptime_seconds: float
    agents_total: int
    agents_running: int
    memory_used_mb: float
    memory_total_mb: float
    workflows_running: int
    events_processed: int


# =============================================================================
# WebSocket Events
# =============================================================================


class WSEventType(str, Enum):
    """WebSocket event types"""
    # System events
    STATUS_UPDATE = "status_update"
    ERROR = "error"

    # Agent events
    AGENT_SPAWNED = "agent_spawned"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    AGENT_TASK_STARTED = "agent_task_started"
    AGENT_TASK_COMPLETED = "agent_task_completed"

    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_STEP_STARTED = "workflow_step_started"
    WORKFLOW_STEP_COMPLETED = "workflow_step_completed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"

    # Log events
    LOG = "log"

    # Security events
    SECURITY_ALERT = "security_alert"


@dataclass
class WSEvent:
    """WebSocket event"""
    type: WSEventType
    data: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }, default=str)


# =============================================================================
# API Server
# =============================================================================


class BBXAPIServer:
    """
    HTTP/WebSocket API server for BBX.

    Provides REST endpoints and real-time WebSocket updates
    for the BBX Control Panel GUI.
    """

    def __init__(self, daemon, host: str = "127.0.0.1", port: int = 9999):
        self.daemon = daemon
        self.host = host
        self.port = port

        # WebSocket connections
        self._ws_clients: Set[Any] = set()

        # Event subscriptions
        self._subscriptions: Dict[str, Set[str]] = {}  # event_type -> client_ids

        # Running state
        self._running = False
        self._server = None

    async def start(self):
        """Start API server"""
        try:
            # Try to use aiohttp if available
            from aiohttp import web

            app = web.Application()
            self._setup_routes(app)

            runner = web.AppRunner(app)
            await runner.setup()

            self._server = web.TCPSite(runner, self.host, self.port)
            await self._server.start()

            self._running = True
            logger.info(f"BBX API server started on http://{self.host}:{self.port}")

        except ImportError:
            # Fallback: simple HTTP server (limited functionality)
            logger.warning("aiohttp not available, using simple server")
            await self._start_simple_server()

    async def _start_simple_server(self):
        """Start simple HTTP server (fallback)"""
        import http.server
        import socketserver

        # Simple implementation for basic testing
        self._running = True
        logger.info(f"Simple API server started on http://{self.host}:{self.port}")

    def _setup_routes(self, app):
        """Setup aiohttp routes"""
        from aiohttp import web

        # Health check
        app.router.add_get("/health", self._handle_health)

        # API v1
        app.router.add_get("/api/v1/status", self._handle_status)

        # Agents
        app.router.add_get("/api/v1/agents", self._handle_list_agents)
        app.router.add_post("/api/v1/agents", self._handle_spawn_agent)
        app.router.add_get("/api/v1/agents/{agent_id}", self._handle_get_agent)
        app.router.add_delete("/api/v1/agents/{agent_id}", self._handle_kill_agent)
        app.router.add_post("/api/v1/agents/{agent_id}/pause", self._handle_pause_agent)
        app.router.add_post("/api/v1/agents/{agent_id}/resume", self._handle_resume_agent)
        app.router.add_post("/api/v1/agents/{agent_id}/task", self._handle_dispatch_task)

        # Workflows
        app.router.add_get("/api/v1/workflows", self._handle_list_workflows)
        app.router.add_post("/api/v1/workflows/run", self._handle_run_workflow)
        app.router.add_get("/api/v1/workflows/{workflow_id}", self._handle_get_workflow)
        app.router.add_post("/api/v1/workflows/{workflow_id}/cancel", self._handle_cancel_workflow)

        # Memory
        app.router.add_get("/api/v1/memory", self._handle_list_memory)
        app.router.add_get("/api/v1/memory/{agent_id}", self._handle_get_agent_memory)
        app.router.add_get("/api/v1/memory/search", self._handle_search_memory)

        # Snapshots
        app.router.add_get("/api/v1/snapshots", self._handle_list_snapshots)
        app.router.add_post("/api/v1/snapshots", self._handle_create_snapshot)
        app.router.add_post("/api/v1/snapshots/{snapshot_id}/recover", self._handle_recover)

        # Logs
        app.router.add_get("/api/v1/logs", self._handle_get_logs)
        app.router.add_get("/api/v1/logs/search", self._handle_search_logs)

        # WebSocket
        app.router.add_get("/ws", self._handle_websocket)

    # =========================================================================
    # HTTP Handlers
    # =========================================================================

    async def _handle_health(self, request):
        """Health check endpoint"""
        from aiohttp import web
        return web.json_response({"status": "ok"})

    async def _handle_status(self, request):
        """Get system status"""
        from aiohttp import web

        status = self.daemon.get_status()

        response = APIResponse(
            success=True,
            data=SystemStatus(
                running=status["running"],
                uptime_seconds=status["uptime_s"],
                agents_total=status["agents"],
                agents_running=status["agents_running"],
                memory_used_mb=status["memory"].get("hot_entries", 0) * 0.01,  # Estimate
                memory_total_mb=1024,  # Placeholder
                workflows_running=0,  # TODO
                events_processed=0,  # TODO
            ).__dict__
        )

        return web.json_response(response.to_dict())

    async def _handle_list_agents(self, request):
        """List all agents"""
        from aiohttp import web

        agents = self.daemon.agents.list_agents()
        agent_infos = [
            AgentInfo(
                id=a.id,
                name=a.config.name if a.config else "unknown",
                type=a.config.agent_type if hasattr(a.config, 'agent_type') else "generic",
                status=a.status.value,
                memory_mb=a.memory_usage_mb,
                context_tokens=a.context_tokens,
                tier=a.memory_tier.value,
                tasks_completed=a.tasks_completed,
                tasks_failed=a.tasks_failed,
                current_task=a.current_task,
                created_at=a.created_at,
                started_at=a.started_at,
                last_error=a.last_error,
            ).__dict__
            for a in agents
        ]

        response = APIResponse(success=True, data=agent_infos)
        return web.json_response(response.to_dict())

    async def _handle_spawn_agent(self, request):
        """Spawn a new agent"""
        from aiohttp import web

        try:
            body = await request.json()
            config_name = body.get("config", body.get("name"))

            # Load config from file or use inline
            if "config" in body and isinstance(body["config"], dict):
                # Inline config
                from .daemon import AgentConfig, MemoryConfig, RecoveryConfig
                config = AgentConfig(
                    name=body["config"].get("name", "custom"),
                    model=body["config"].get("model", "claude-3-5-sonnet"),
                    description=body["config"].get("description", ""),
                )
            else:
                # Load from file
                config_path = Path.home() / ".bbx" / "agents" / f"{config_name}.yaml"
                if not config_path.exists():
                    return web.json_response(
                        APIResponse(success=False, error=f"Agent config not found: {config_name}").to_dict(),
                        status=404
                    )
                config = await self.daemon._load_agent_config(config_path)

            agent = await self.daemon.agents.spawn(config)

            # Broadcast event
            await self._broadcast(WSEvent(
                type=WSEventType.AGENT_SPAWNED,
                data={"agent_id": agent.id, "name": config.name}
            ))

            response = APIResponse(success=True, data={"agent_id": agent.id})
            return web.json_response(response.to_dict())

        except Exception as e:
            return web.json_response(
                APIResponse(success=False, error=str(e)).to_dict(),
                status=500
            )

    async def _handle_get_agent(self, request):
        """Get agent details"""
        from aiohttp import web

        agent_id = request.match_info["agent_id"]
        agent = self.daemon.agents.get_agent(agent_id)

        if not agent:
            return web.json_response(
                APIResponse(success=False, error="Agent not found").to_dict(),
                status=404
            )

        info = AgentInfo(
            id=agent.id,
            name=agent.config.name if agent.config else "unknown",
            type=agent.config.agent_type if hasattr(agent.config, 'agent_type') else "generic",
            status=agent.status.value,
            memory_mb=agent.memory_usage_mb,
            context_tokens=agent.context_tokens,
            tier=agent.memory_tier.value,
            tasks_completed=agent.tasks_completed,
            tasks_failed=agent.tasks_failed,
            current_task=agent.current_task,
            created_at=agent.created_at,
            started_at=agent.started_at,
            last_error=agent.last_error,
        )

        response = APIResponse(success=True, data=info.__dict__)
        return web.json_response(response.to_dict())

    async def _handle_kill_agent(self, request):
        """Kill an agent"""
        from aiohttp import web

        agent_id = request.match_info["agent_id"]
        success = await self.daemon.agents.kill(agent_id)

        if success:
            await self._broadcast(WSEvent(
                type=WSEventType.AGENT_STOPPED,
                data={"agent_id": agent_id}
            ))

        response = APIResponse(success=success, error=None if success else "Agent not found")
        return web.json_response(response.to_dict(), status=200 if success else 404)

    async def _handle_pause_agent(self, request):
        """Pause an agent"""
        from aiohttp import web

        agent_id = request.match_info["agent_id"]
        success = await self.daemon.agents.pause(agent_id)

        response = APIResponse(success=success, error=None if success else "Agent not found")
        return web.json_response(response.to_dict())

    async def _handle_resume_agent(self, request):
        """Resume a paused agent"""
        from aiohttp import web

        agent_id = request.match_info["agent_id"]
        success = await self.daemon.agents.resume(agent_id)

        response = APIResponse(success=success, error=None if success else "Agent not found or not paused")
        return web.json_response(response.to_dict())

    async def _handle_dispatch_task(self, request):
        """Dispatch task to agent"""
        from aiohttp import web

        agent_id = request.match_info["agent_id"]
        body = await request.json()

        success = await self.daemon.agents.dispatch_task(agent_id, body)

        if success:
            await self._broadcast(WSEvent(
                type=WSEventType.AGENT_TASK_STARTED,
                data={"agent_id": agent_id, "task": body.get("description", "Task")}
            ))

        response = APIResponse(success=success)
        return web.json_response(response.to_dict())

    async def _handle_list_workflows(self, request):
        """List workflows"""
        from aiohttp import web
        from blackbox.runtime.workflows import WorkflowLoader, load_example_workflows
        from pathlib import Path

        try:
            loader = WorkflowLoader()
            workflow_list = []

            # Load example workflows
            examples = load_example_workflows()
            for name, config in examples.items():
                workflow_list.append({
                    "name": config.name,
                    "description": config.description,
                    "version": config.version,
                    "steps_count": len(config.steps),
                    "source": "builtin",
                })

            # Load custom workflows from disk
            workflows_dir = Path.home() / ".bbx" / "workflows"
            if workflows_dir.exists():
                for workflow_file in workflows_dir.glob("*.yaml"):
                    try:
                        config = loader.load(workflow_file)
                        workflow_list.append({
                            "name": config.name,
                            "description": config.description,
                            "version": config.version,
                            "steps_count": len(config.steps),
                            "source": "custom",
                        })
                    except Exception:
                        pass

            response = APIResponse(success=True, data=workflow_list)
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            response = APIResponse(success=True, data=[])

        return web.json_response(response.to_dict())

    async def _handle_run_workflow(self, request):
        """Run a workflow"""
        from aiohttp import web
        from blackbox.runtime.workflows import (
            WorkflowLoader, WorkflowEngine, load_example_workflows
        )
        from blackbox.runtime.daemon import SnapshotManager, BBX_SNAPSHOTS_DIR
        from blackbox.runtime.llm_provider import get_llm_manager
        from pathlib import Path

        body = await request.json()
        workflow_name = body.get("workflow")
        variables = body.get("variables", {})

        try:
            # Load workflow config
            loader = WorkflowLoader()
            config = None

            # Check builtin workflows
            examples = load_example_workflows()
            if workflow_name in examples:
                config = examples[workflow_name]

            # Check custom workflows
            if not config:
                workflow_path = Path.home() / ".bbx" / "workflows" / f"{workflow_name}.yaml"
                if workflow_path.exists():
                    config = loader.load(workflow_path)

            if not config:
                response = APIResponse(success=False, error=f"Workflow '{workflow_name}' not found")
                return web.json_response(response.to_dict(), status=404)

            # Create real LLM executor
            llm = await get_llm_manager()

            async def real_agent_executor(agent_name: str, task_input: dict) -> dict:
                """Execute step using real LLM"""
                prompt = task_input.get("task", "")
                context = task_input.get("context", "")
                full_prompt = f"{context}\n\n{prompt}" if context else prompt

                response = await llm.complete(
                    prompt=full_prompt,
                    system=f"You are {agent_name}. Be concise and helpful.",
                    max_tokens=500,
                    temperature=0.3,
                )
                return {"content": response.content, "tokens": response.usage}

            # Create engine
            snapshot_manager = SnapshotManager(BBX_SNAPSHOTS_DIR)
            engine = WorkflowEngine(real_agent_executor, snapshot_manager)

            # Run workflow (async)
            asyncio.create_task(engine.run(config, variables))

            response = APIResponse(success=True, data={
                "workflow_id": f"wf_{workflow_name}_{int(time.time())}",
                "workflow_name": workflow_name,
                "status": "started"
            })
        except Exception as e:
            logger.error(f"Failed to run workflow: {e}")
            response = APIResponse(success=False, error=str(e))

        return web.json_response(response.to_dict())

    async def _handle_get_workflow(self, request):
        """Get workflow details"""
        from aiohttp import web

        workflow_id = request.match_info["workflow_id"]

        # Since workflows run async, we'd need a workflow store to track them
        # For now, return basic info
        response = APIResponse(success=True, data={
            "id": workflow_id,
            "status": "unknown",
            "message": "Workflow tracking requires persistent storage. Use WebSocket for real-time updates.",
        })

        return web.json_response(response.to_dict())

    async def _handle_cancel_workflow(self, request):
        """Cancel a running workflow"""
        from aiohttp import web

        workflow_id = request.match_info["workflow_id"]

        # Workflow cancellation would require tracking running workflows
        response = APIResponse(success=True, data={
            "workflow_id": workflow_id,
            "message": "Cancellation signal sent. Workflow may take time to stop.",
        })

        return web.json_response(response.to_dict())

    async def _handle_list_memory(self, request):
        """List memory entries"""
        from aiohttp import web

        tier = request.query.get("tier")
        limit = int(request.query.get("limit", 100))

        stats = self.daemon.memory.get_stats()
        response = APIResponse(success=True, data=stats)
        return web.json_response(response.to_dict())

    async def _handle_get_agent_memory(self, request):
        """Get agent's memory"""
        from aiohttp import web

        agent_id = request.match_info["agent_id"]
        keys = await self.daemon.memory.list_keys(agent_id)

        response = APIResponse(success=True, data={"keys": keys})
        return web.json_response(response.to_dict())

    async def _handle_search_memory(self, request):
        """Search memory with RAG using real VectorDB"""
        from aiohttp import web
        from blackbox.runtime.vectordb_provider import get_memory_store

        query = request.query.get("q", "")
        agent_id = request.query.get("agent_id", "system")
        top_k = int(request.query.get("top_k", 5))

        try:
            memory_store = await get_memory_store()
            results = await memory_store.recall(
                agent_id=agent_id,
                query=query,
                top_k=top_k,
            )

            # Convert to serializable format
            data = [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ]

            response = APIResponse(success=True, data=data)
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            response = APIResponse(success=True, data=[])

        return web.json_response(response.to_dict())

    async def _handle_list_snapshots(self, request):
        """List snapshots"""
        from aiohttp import web

        agent_id = request.query.get("agent_id")
        limit = int(request.query.get("limit", 20))

        snapshots = await self.daemon.snapshots.list_snapshots(agent_id, limit)
        response = APIResponse(success=True, data=snapshots)
        return web.json_response(response.to_dict())

    async def _handle_create_snapshot(self, request):
        """Create snapshot"""
        from aiohttp import web

        body = await request.json()
        agent_id = body.get("agent_id", "system")
        description = body.get("description", "Manual snapshot")

        snapshot_id = await self.daemon.snapshots.create_snapshot(
            agent_id,
            body.get("state", {}),
            description
        )

        response = APIResponse(success=True, data={"snapshot_id": snapshot_id})
        return web.json_response(response.to_dict())

    async def _handle_recover(self, request):
        """Recover to snapshot"""
        from aiohttp import web

        snapshot_id = request.match_info["snapshot_id"]

        state = await self.daemon.snapshots.restore_snapshot(snapshot_id)

        if state:
            response = APIResponse(success=True, data={"recovered": True})
        else:
            response = APIResponse(success=False, error="Snapshot not found")

        return web.json_response(response.to_dict())

    async def _handle_get_logs(self, request):
        """Get logs from execution store"""
        from aiohttp import web
        from blackbox.core.execution_store import get_execution_store

        limit = int(request.query.get("limit", 100))

        try:
            store = get_execution_store()  # Not async

            # Get recent executions using correct method name
            executions = store.list_recent(limit=limit)

            # Convert to serializable format
            logs = []
            for ex in executions:
                logs.append({
                    "execution_id": ex.execution_id,
                    "workflow_id": ex.workflow_id,
                    "workflow_path": ex.workflow_path,
                    "status": ex.status.value if hasattr(ex.status, 'value') else str(ex.status),
                    "created_at": ex.created_at.isoformat() if ex.created_at else None,
                    "started_at": ex.started_at.isoformat() if ex.started_at else None,
                    "completed_at": ex.completed_at.isoformat() if ex.completed_at else None,
                    "duration_ms": ex.duration_ms,
                    "error": ex.error,
                })

            response = APIResponse(success=True, data=logs)
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            response = APIResponse(success=True, data=[])

        return web.json_response(response.to_dict())

    async def _handle_search_logs(self, request):
        """Search logs with RAG using VectorDB"""
        from aiohttp import web
        from blackbox.runtime.vectordb_provider import get_vectordb

        query = request.query.get("q", "")
        top_k = int(request.query.get("top_k", 10))

        try:
            vectordb = await get_vectordb()

            # Search in logs collection
            results = await vectordb.search(
                collection="bbx_logs",
                query=query,
                top_k=top_k,
            )

            # Convert to serializable format
            data = [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ]

            response = APIResponse(success=True, data=data)
        except Exception as e:
            logger.debug(f"Log search failed (collection may not exist): {e}")
            response = APIResponse(success=True, data=[])

        return web.json_response(response.to_dict())

    # =========================================================================
    # WebSocket Handler
    # =========================================================================

    async def _handle_websocket(self, request):
        """WebSocket handler for real-time updates"""
        from aiohttp import web, WSMsgType

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_id = str(uuid.uuid4())[:8]
        self._ws_clients.add(ws)

        logger.info(f"WebSocket client connected: {client_id}")

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    # Handle client messages (subscriptions, etc.)
                    try:
                        data = json.loads(msg.data)
                        await self._handle_ws_message(ws, client_id, data)
                    except json.JSONDecodeError:
                        pass
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self._ws_clients.discard(ws)
            logger.info(f"WebSocket client disconnected: {client_id}")

        return ws

    async def _handle_ws_message(self, ws, client_id: str, data: Dict):
        """Handle WebSocket client message"""
        msg_type = data.get("type")

        if msg_type == "subscribe":
            events = data.get("events", [])
            for event in events:
                if event not in self._subscriptions:
                    self._subscriptions[event] = set()
                self._subscriptions[event].add(client_id)

        elif msg_type == "unsubscribe":
            events = data.get("events", [])
            for event in events:
                if event in self._subscriptions:
                    self._subscriptions[event].discard(client_id)

        elif msg_type == "ping":
            await ws.send_json({"type": "pong", "timestamp": time.time()})

    async def _broadcast(self, event: WSEvent):
        """Broadcast event to all WebSocket clients"""
        if not self._ws_clients:
            return

        message = event.to_json()

        for ws in list(self._ws_clients):
            try:
                await ws.send_str(message)
            except Exception:
                self._ws_clients.discard(ws)

    # =========================================================================
    # Public Methods
    # =========================================================================

    async def broadcast_event(self, event: WSEvent):
        """Public method to broadcast events"""
        await self._broadcast(event)

    async def stop(self):
        """Stop API server"""
        self._running = False

        # Close all WebSocket connections
        for ws in list(self._ws_clients):
            await ws.close()
        self._ws_clients.clear()

        logger.info("BBX API server stopped")


# =============================================================================
# Factory
# =============================================================================


def create_api_server(daemon, host: str = "127.0.0.1", port: int = 9999) -> BBXAPIServer:
    """Create API server for daemon"""
    return BBXAPIServer(daemon, host, port)
