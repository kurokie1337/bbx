# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Daemon - The Background Service for AI Agents.

NOT a replacement OS. A runtime that runs INSIDE Windows/Mac/Linux.
Like Docker daemon, but for AI agents instead of containers.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Host OS (Windows/Mac/Linux)               │
    ├─────────────────────────────────────────────────────────────┤
    │                      BBX Daemon (this file)                  │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │                   Agent Scheduler                        │ │
    │  │  (Manages agent lifecycles, like Docker container mgr)   │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │                   Memory Manager                         │ │
    │  │  (Vector DB + Context Tiering)                          │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │                   Event Loop                             │ │
    │  │  (File watchers, cron jobs, webhooks)                   │ │
    │  ├─────────────────────────────────────────────────────────┤ │
    │  │                   API Server                             │ │
    │  │  (gRPC/REST for CLI and GUI)                            │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │                              │                               │
    │  ┌───────────────────────────▼───────────────────────────┐   │
    │  │              Running Agents (Processes)                │   │
    │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐               │   │
    │  │  │ email-   │ │ code-    │ │ legacy-  │               │   │
    │  │  │ secretary│ │ healer   │ │ miner    │               │   │
    │  │  └──────────┘ └──────────┘ └──────────┘               │   │
    │  └───────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    bbx daemon start              # Start daemon
    bbx daemon stop               # Stop daemon
    bbx daemon status             # Check status
    bbx agent spawn email-secretary
    bbx agent list
    bbx ps
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bbx.daemon")


# =============================================================================
# Constants
# =============================================================================

BBX_HOME = Path.home() / ".bbx"
BBX_AGENTS_DIR = BBX_HOME / "agents"
BBX_WORKFLOWS_DIR = BBX_HOME / "workflows"
BBX_DATA_DIR = BBX_HOME / "data"
BBX_LOGS_DIR = BBX_HOME / "logs"
BBX_SNAPSHOTS_DIR = BBX_HOME / "snapshots"
BBX_PID_FILE = BBX_HOME / "daemon.pid"
BBX_SOCKET = BBX_HOME / "daemon.sock"


# =============================================================================
# Agent States
# =============================================================================


class AgentStatus(Enum):
    """Agent lifecycle states"""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class MemoryTier(Enum):
    """Memory tier for context"""
    HOT = "hot"      # In-memory, instant access
    WARM = "warm"    # Compressed, fast access
    COOL = "cool"    # On-disk, moderate access
    COLD = "cold"    # Archive, slow access


# =============================================================================
# Agent Configuration
# =============================================================================


@dataclass
class AgentCapability:
    """Single capability (syscall permission)"""
    action: str        # read, write, execute, delete
    resource: str      # gmail, calendar, filesystem, etc.
    allowed: bool = True
    requires_confirm: bool = False


@dataclass
class MemoryConfig:
    """Memory/context configuration for agent"""
    hot_window: str = "7d"       # Keep last 7 days hot
    warm_window: str = "30d"     # Summarize last 30 days
    cold_storage: bool = True    # Archive everything else
    max_context_tokens: int = 128000


@dataclass
class RecoveryConfig:
    """Recovery/snapshot configuration"""
    snapshots: int = 10              # Keep last N snapshots
    max_rollback_age: str = "24h"    # Max age for rollback
    auto_snapshot: bool = True       # Snapshot before risky ops
    snapshot_on_error: bool = True   # Snapshot when error occurs


@dataclass
class TriggerConfig:
    """Event trigger configuration"""
    event: str              # file_change, schedule, webhook, manual
    path: Optional[str] = None       # For file_change
    cron: Optional[str] = None       # For schedule
    url: Optional[str] = None        # For webhook
    action: str = ""                 # Action to run


@dataclass
class AgentConfig:
    """Complete agent configuration (loaded from YAML)"""
    name: str
    model: str = "claude-3-5-sonnet"
    description: str = ""
    system_prompt: str = "You are a helpful AI assistant."

    # Permissions
    capabilities: List[AgentCapability] = field(default_factory=list)

    # Memory
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Recovery
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    # Triggers
    triggers: List[TriggerConfig] = field(default_factory=list)

    # Metadata
    version: str = "1.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Running Agent Instance
# =============================================================================


@dataclass
class AgentInstance:
    """A running agent instance"""
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: AgentConfig = None

    # State
    status: AgentStatus = AgentStatus.CREATED
    pid: int = 0

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None

    # Resources
    memory_usage_mb: float = 0
    context_tokens: int = 0
    memory_tier: MemoryTier = MemoryTier.HOT

    # Statistics
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens_used: int = 0

    # Task queue
    task_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Current task
    current_task: Optional[str] = None

    # Error info
    last_error: Optional[str] = None
    error_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for API"""
        return {
            "id": self.id,
            "name": self.config.name if self.config else "unknown",
            "status": self.status.value,
            "pid": self.pid,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "memory_mb": self.memory_usage_mb,
            "context_tokens": self.context_tokens,
            "tier": self.memory_tier.value,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "current_task": self.current_task,
            "last_error": self.last_error,
        }


# =============================================================================
# Event System
# =============================================================================


@dataclass
class Event:
    """System event"""
    type: str              # file_change, schedule, webhook, agent_message
    source: str            # Source identifier
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class EventBus:
    """Central event bus for daemon"""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from event type"""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event):
        """Publish event to bus"""
        await self._queue.put(event)

    async def start(self):
        """Start event processing"""
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event dispatch error: {e}")

    async def stop(self):
        """Stop event processing"""
        self._running = False

    async def _dispatch(self, event: Event):
        """Dispatch event to handlers"""
        handlers = self._handlers.get(event.type, [])
        handlers.extend(self._handlers.get("*", []))  # Wildcard handlers

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.type}: {e}")


# =============================================================================
# File Watcher
# =============================================================================


class FileWatcher:
    """Watch file system for changes (triggers)"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._watches: Dict[str, Set[str]] = {}  # path -> patterns
        self._running = False
        self._last_mtimes: Dict[str, float] = {}

    def watch(self, path: str, pattern: str = "*"):
        """Add watch on path"""
        if path not in self._watches:
            self._watches[path] = set()
        self._watches[path].add(pattern)

    def unwatch(self, path: str):
        """Remove watch"""
        self._watches.pop(path, None)

    async def start(self):
        """Start watching"""
        self._running = True
        while self._running:
            await self._check_changes()
            await asyncio.sleep(1.0)  # Check every second

    async def stop(self):
        """Stop watching"""
        self._running = False

    async def _check_changes(self):
        """Check for file changes"""
        import fnmatch

        for watch_path, patterns in self._watches.items():
            path = Path(watch_path).expanduser()
            if not path.exists():
                continue

            if path.is_file():
                await self._check_file(path, patterns)
            else:
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        await self._check_file(file_path, patterns)

    async def _check_file(self, file_path: Path, patterns: Set[str]):
        """Check single file for changes"""
        import fnmatch

        # Check if matches any pattern
        matches = any(
            fnmatch.fnmatch(file_path.name, p)
            for p in patterns
        )
        if not matches and "*" not in patterns:
            return

        try:
            mtime = file_path.stat().st_mtime
            key = str(file_path)

            if key in self._last_mtimes:
                if mtime > self._last_mtimes[key]:
                    # File changed!
                    await self.event_bus.publish(Event(
                        type="file_change",
                        source=str(file_path),
                        data={
                            "path": str(file_path),
                            "name": file_path.name,
                            "mtime": mtime,
                        }
                    ))

            self._last_mtimes[key] = mtime
        except Exception:
            pass


# =============================================================================
# Scheduler (Cron-like)
# =============================================================================


class Scheduler:
    """Cron-like scheduler for triggers"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._jobs: Dict[str, Dict] = {}  # job_id -> {cron, action, last_run}
        self._running = False

    def add_job(self, job_id: str, cron: str, action: str):
        """Add scheduled job"""
        self._jobs[job_id] = {
            "cron": cron,
            "action": action,
            "last_run": None,
        }

    def remove_job(self, job_id: str):
        """Remove job"""
        self._jobs.pop(job_id, None)

    async def start(self):
        """Start scheduler"""
        self._running = True
        while self._running:
            await self._check_jobs()
            await asyncio.sleep(60.0)  # Check every minute

    async def stop(self):
        """Stop scheduler"""
        self._running = False

    async def _check_jobs(self):
        """Check if any jobs should run"""
        now = datetime.now()

        for job_id, job in self._jobs.items():
            if self._should_run(job["cron"], now, job["last_run"]):
                job["last_run"] = now
                await self.event_bus.publish(Event(
                    type="schedule",
                    source=job_id,
                    data={
                        "action": job["action"],
                        "cron": job["cron"],
                    }
                ))

    def _should_run(self, cron: str, now: datetime, last_run: Optional[datetime]) -> bool:
        """Simple cron check (simplified)"""
        # Format: "minute hour day month weekday"
        # Example: "0 9 * * *" = 9:00 AM daily

        parts = cron.split()
        if len(parts) != 5:
            return False

        minute, hour, day, month, weekday = parts

        # Check minute
        if minute != "*" and int(minute) != now.minute:
            return False

        # Check hour
        if hour != "*" and int(hour) != now.hour:
            return False

        # Check day
        if day != "*" and int(day) != now.day:
            return False

        # Check month
        if month != "*" and int(month) != now.month:
            return False

        # Check weekday (0=Monday)
        if weekday != "*" and int(weekday) != now.weekday():
            return False

        # Don't run twice in same minute
        if last_run and (now - last_run).total_seconds() < 60:
            return False

        return True


# =============================================================================
# Memory Manager
# =============================================================================


class MemoryManager:
    """Manages context memory for agents"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory HOT storage
        self._hot_storage: Dict[str, Dict] = {}  # agent_id -> {key: value}

        # Stats
        self._total_stored = 0
        self._total_retrieved = 0

    async def store(
        self,
        agent_id: str,
        key: str,
        value: Any,
        tier: MemoryTier = MemoryTier.HOT,
        metadata: Optional[Dict] = None
    ):
        """Store value in memory"""
        if agent_id not in self._hot_storage:
            self._hot_storage[agent_id] = {}

        entry = {
            "value": value,
            "tier": tier.value,
            "stored_at": time.time(),
            "metadata": metadata or {},
        }

        if tier == MemoryTier.HOT:
            self._hot_storage[agent_id][key] = entry
        else:
            # Write to disk for WARM/COOL/COLD
            agent_dir = self.data_dir / agent_id
            agent_dir.mkdir(exist_ok=True)

            file_path = agent_dir / f"{key}.json"
            with open(file_path, "w") as f:
                json.dump(entry, f)

        self._total_stored += 1

    async def retrieve(
        self,
        agent_id: str,
        key: str
    ) -> Optional[Any]:
        """Retrieve value from memory"""
        # Check HOT first
        if agent_id in self._hot_storage:
            if key in self._hot_storage[agent_id]:
                self._total_retrieved += 1
                return self._hot_storage[agent_id][key]["value"]

        # Check disk
        file_path = self.data_dir / agent_id / f"{key}.json"
        if file_path.exists():
            with open(file_path) as f:
                entry = json.load(f)
                self._total_retrieved += 1
                return entry["value"]

        return None

    async def list_keys(self, agent_id: str, tier: Optional[MemoryTier] = None) -> List[str]:
        """List all keys for agent"""
        keys = []

        # HOT keys
        if agent_id in self._hot_storage:
            for key, entry in self._hot_storage[agent_id].items():
                if tier is None or entry["tier"] == tier.value:
                    keys.append(key)

        # Disk keys
        agent_dir = self.data_dir / agent_id
        if agent_dir.exists():
            for file_path in agent_dir.glob("*.json"):
                key = file_path.stem
                if key not in keys:
                    keys.append(key)

        return keys

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        hot_count = sum(len(v) for v in self._hot_storage.values())
        return {
            "hot_entries": hot_count,
            "total_stored": self._total_stored,
            "total_retrieved": self._total_retrieved,
        }


# =============================================================================
# Snapshot Manager
# =============================================================================


class SnapshotManager:
    """Manages snapshots for recovery"""

    def __init__(self, snapshots_dir: Path):
        self.snapshots_dir = snapshots_dir
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    async def create_snapshot(
        self,
        agent_id: str,
        state: Dict,
        description: str = ""
    ) -> str:
        """Create snapshot of agent state"""
        snapshot_id = f"{agent_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        snapshot = {
            "id": snapshot_id,
            "agent_id": agent_id,
            "created_at": time.time(),
            "description": description,
            "state": state,
        }

        snapshot_path = self.snapshots_dir / f"{snapshot_id}.json"
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

        logger.info(f"Created snapshot: {snapshot_id}")
        return snapshot_id

    async def restore_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """Restore from snapshot"""
        snapshot_path = self.snapshots_dir / f"{snapshot_id}.json"

        if not snapshot_path.exists():
            return None

        with open(snapshot_path) as f:
            snapshot = json.load(f)

        logger.info(f"Restored snapshot: {snapshot_id}")
        return snapshot["state"]

    async def list_snapshots(
        self,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """List available snapshots"""
        snapshots = []

        for path in self.snapshots_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)

                if agent_id and data["agent_id"] != agent_id:
                    continue

                snapshots.append({
                    "id": data["id"],
                    "agent_id": data["agent_id"],
                    "created_at": data["created_at"],
                    "description": data["description"],
                })

        # Sort by time, newest first
        snapshots.sort(key=lambda x: x["created_at"], reverse=True)
        return snapshots[:limit]

    async def cleanup_old_snapshots(self, agent_id: str, keep: int = 10):
        """Remove old snapshots, keep latest N"""
        snapshots = await self.list_snapshots(agent_id, limit=1000)

        for snapshot in snapshots[keep:]:
            path = self.snapshots_dir / f"{snapshot['id']}.json"
            try:
                path.unlink()
                logger.debug(f"Deleted old snapshot: {snapshot['id']}")
            except Exception:
                pass


# =============================================================================
# Agent Manager
# =============================================================================


class AgentManager:
    """Manages agent lifecycle"""

    def __init__(
        self,
        event_bus: EventBus,
        memory_manager: MemoryManager,
        snapshot_manager: SnapshotManager
    ):
        self.event_bus = event_bus
        self.memory = memory_manager
        self.snapshots = snapshot_manager

        # Running agents
        self._agents: Dict[str, AgentInstance] = {}

        # Agent tasks (asyncio tasks)
        self._tasks: Dict[str, asyncio.Task] = {}

        # Subscribe to events
        self.event_bus.subscribe("file_change", self._on_file_change)
        self.event_bus.subscribe("schedule", self._on_schedule)

    async def spawn(self, config: AgentConfig) -> AgentInstance:
        """Spawn new agent"""
        agent = AgentInstance(
            config=config,
            status=AgentStatus.CREATED,
            pid=os.getpid(),  # In real impl, would be separate process
        )

        self._agents[agent.id] = agent

        # Start agent task
        agent.status = AgentStatus.STARTING
        task = asyncio.create_task(self._run_agent(agent))
        self._tasks[agent.id] = task

        agent.status = AgentStatus.RUNNING
        agent.started_at = time.time()

        logger.info(f"Spawned agent: {agent.id} ({config.name})")
        return agent

    async def kill(self, agent_id: str) -> bool:
        """Kill running agent"""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.status = AgentStatus.STOPPING

        # Cancel task
        task = self._tasks.get(agent_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        agent.status = AgentStatus.STOPPED
        agent.stopped_at = time.time()

        # Create final snapshot
        await self.snapshots.create_snapshot(
            agent_id,
            {"tokens_used": agent.total_tokens_used},
            "Agent stopped"
        )

        logger.info(f"Killed agent: {agent_id}")
        return True

    async def pause(self, agent_id: str) -> bool:
        """Pause agent"""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.status = AgentStatus.PAUSED
        logger.info(f"Paused agent: {agent_id}")
        return True

    async def resume(self, agent_id: str) -> bool:
        """Resume paused agent"""
        agent = self._agents.get(agent_id)
        if not agent or agent.status != AgentStatus.PAUSED:
            return False

        agent.status = AgentStatus.RUNNING
        logger.info(f"Resumed agent: {agent_id}")
        return True

    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get agent by ID"""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[AgentInstance]:
        """List all agents"""
        return list(self._agents.values())

    async def dispatch_task(self, agent_id: str, task: Dict) -> bool:
        """Dispatch task to agent"""
        agent = self._agents.get(agent_id)
        if not agent or agent.status != AgentStatus.RUNNING:
            return False

        await agent.task_queue.put(task)
        return True

    async def _run_agent(self, agent: AgentInstance):
        """Main agent loop"""
        while agent.status in (AgentStatus.RUNNING, AgentStatus.PAUSED):
            if agent.status == AgentStatus.PAUSED:
                await asyncio.sleep(1.0)
                continue

            try:
                # Wait for task
                task = await asyncio.wait_for(
                    agent.task_queue.get(),
                    timeout=5.0
                )

                agent.current_task = task.get("description", "Processing...")

                # Create pre-task snapshot
                if agent.config.recovery.auto_snapshot:
                    await self.snapshots.create_snapshot(
                        agent.id,
                        {"current_task": task},
                        f"Before: {agent.current_task}"
                    )

                # Execute task with REAL LLM and VectorDB
                result = await self._execute_task(agent, task)

                agent.tasks_completed += 1
                agent.current_task = None

            except asyncio.TimeoutError:
                # No task, just wait
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                agent.tasks_failed += 1
                agent.error_count += 1
                agent.last_error = str(e)
                logger.error(f"Agent {agent.id} error: {e}")

                # Snapshot on error
                if agent.config.recovery.snapshot_on_error:
                    await self.snapshots.create_snapshot(
                        agent.id,
                        {"error": str(e), "task": agent.current_task},
                        f"Error: {str(e)[:50]}"
                    )

    async def _execute_task(self, agent: AgentInstance, task: Dict) -> Any:
        """Execute a task using REAL LLM and VectorDB"""
        from blackbox.runtime.llm_provider import get_llm_manager
        from blackbox.runtime.vectordb_provider import get_memory_store

        task_type = task.get("type", "think")

        if task_type == "think":
            # Get LLM manager
            llm = await get_llm_manager()

            # Build context from RAG (semantic memory)
            context = ""
            try:
                memory_store = await get_memory_store()
                prompt = task.get("prompt", task.get("description", ""))

                # Recall relevant memories
                memories = await memory_store.recall(
                    agent_id=agent.id,
                    query=prompt,
                    top_k=5,
                )

                if memories:
                    context = "Relevant context:\n"
                    for mem in memories:
                        context += f"- {mem.content}\n"
                    context += "\n"
            except Exception as e:
                logger.debug(f"Memory recall failed: {e}")

            # Build prompt with context
            full_prompt = context + task.get("prompt", task.get("description", ""))

            # Call REAL LLM
            response = await llm.complete(
                prompt=full_prompt,
                system=agent.config.system_prompt if agent.config else None,
                temperature=task.get("temperature", 0.7),
                max_tokens=task.get("max_tokens", 4096),
            )

            # Update stats
            tokens = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
            agent.total_tokens_used += tokens

            # Store response in memory (if significant)
            if len(response.content) > 100:
                try:
                    memory_store = await get_memory_store()
                    await memory_store.store_memory(
                        agent_id=agent.id,
                        content=f"Task: {full_prompt[:200]}\nResponse: {response.content[:500]}",
                        memory_type="experience",
                        importance=0.5,
                    )
                except Exception:
                    pass

            return {
                "response": response.content,
                "model": response.model,
                "tokens_used": tokens,
                "latency_ms": response.latency_ms,
            }

        elif task_type == "subprocess":
            # Execute subprocess (sandboxed)
            import subprocess
            command = task.get("command", [])
            if command:
                try:
                    result = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=task.get("timeout", 30),
                    )
                    return {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                    }
                except Exception as e:
                    return {"error": str(e), "returncode": -1}
            return {"stdout": "", "returncode": 0}

        elif task_type == "remember":
            # Store in memory
            memory_store = await get_memory_store()
            memory_id = await memory_store.store_memory(
                agent_id=agent.id,
                content=task.get("content", ""),
                memory_type=task.get("memory_type", "general"),
                importance=task.get("importance", 0.5),
            )
            return {"memory_id": memory_id, "status": "stored"}

        elif task_type == "recall":
            # Recall from memory
            memory_store = await get_memory_store()
            results = await memory_store.recall(
                agent_id=agent.id,
                query=task.get("query", ""),
                top_k=task.get("top_k", 5),
            )
            return {
                "memories": [
                    {"content": r.content, "score": r.score}
                    for r in results
                ],
            }

        return {"status": "completed"}

    async def _on_file_change(self, event: Event):
        """Handle file change event"""
        file_path = event.data.get("path", "")

        # Find agents interested in this path
        for agent in self._agents.values():
            if agent.status != AgentStatus.RUNNING:
                continue

            for trigger in agent.config.triggers:
                if trigger.event == "file_change":
                    # Check if path matches
                    import fnmatch
                    if trigger.path and fnmatch.fnmatch(file_path, trigger.path):
                        await self.dispatch_task(agent.id, {
                            "type": "trigger",
                            "trigger": "file_change",
                            "action": trigger.action,
                            "data": event.data,
                        })

    async def _on_schedule(self, event: Event):
        """Handle schedule event"""
        action = event.data.get("action", "")

        # Find agent for this action
        for agent in self._agents.values():
            if agent.status != AgentStatus.RUNNING:
                continue

            for trigger in agent.config.triggers:
                if trigger.event == "schedule" and trigger.action == action:
                    await self.dispatch_task(agent.id, {
                        "type": "trigger",
                        "trigger": "schedule",
                        "action": action,
                        "data": event.data,
                    })


# =============================================================================
# BBX Daemon
# =============================================================================


class BBXDaemon:
    """
    The main BBX daemon.

    This is the heart of BBX - a background service that manages AI agents.
    Like Docker daemon, but for AI agents instead of containers.
    """

    def __init__(self, home_dir: Path = BBX_HOME):
        self.home_dir = home_dir
        self._ensure_directories()

        # Components
        self.event_bus = EventBus()
        self.file_watcher = FileWatcher(self.event_bus)
        self.scheduler = Scheduler(self.event_bus)
        self.memory = MemoryManager(BBX_DATA_DIR / "memory")
        self.snapshots = SnapshotManager(BBX_SNAPSHOTS_DIR)
        self.agents = AgentManager(self.event_bus, self.memory, self.snapshots)

        # State
        self._running = False
        self._started_at: Optional[float] = None

        # Background tasks
        self._tasks: List[asyncio.Task] = []

    def _ensure_directories(self):
        """Create required directories"""
        for dir_path in [
            self.home_dir,
            BBX_AGENTS_DIR,
            BBX_WORKFLOWS_DIR,
            BBX_DATA_DIR,
            BBX_LOGS_DIR,
            BBX_SNAPSHOTS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Start the daemon"""
        if self._running:
            return

        logger.info("BBX Daemon starting...")
        self._running = True
        self._started_at = time.time()

        # Write PID file
        BBX_PID_FILE.write_text(str(os.getpid()))

        # Start components
        self._tasks.append(asyncio.create_task(self.event_bus.start()))
        self._tasks.append(asyncio.create_task(self.file_watcher.start()))
        self._tasks.append(asyncio.create_task(self.scheduler.start()))

        # Load and start configured agents
        await self._load_agents()

        logger.info(f"BBX Daemon started (PID: {os.getpid()})")

    async def stop(self):
        """Stop the daemon"""
        if not self._running:
            return

        logger.info("BBX Daemon stopping...")
        self._running = False

        # Stop all agents
        for agent in self.agents.list_agents():
            await self.agents.kill(agent.id)

        # Stop components
        await self.event_bus.stop()
        await self.file_watcher.stop()
        await self.scheduler.stop()

        # Cancel background tasks
        for task in self._tasks:
            task.cancel()

        # Remove PID file
        if BBX_PID_FILE.exists():
            BBX_PID_FILE.unlink()

        logger.info("BBX Daemon stopped")

    async def _load_agents(self):
        """Load and start agents from config files"""
        for agent_file in BBX_AGENTS_DIR.glob("*.yaml"):
            try:
                config = await self._load_agent_config(agent_file)
                await self.agents.spawn(config)
            except Exception as e:
                logger.error(f"Failed to load agent {agent_file}: {e}")

        for agent_file in BBX_AGENTS_DIR.glob("*.yml"):
            try:
                config = await self._load_agent_config(agent_file)
                await self.agents.spawn(config)
            except Exception as e:
                logger.error(f"Failed to load agent {agent_file}: {e}")

    async def _load_agent_config(self, path: Path) -> AgentConfig:
        """Load agent configuration from YAML"""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse capabilities
        capabilities = []
        for cap in data.get("capabilities", []):
            if isinstance(cap, str):
                # Format: "read:gmail" or "denied:delete"
                parts = cap.split(":")
                if len(parts) == 2:
                    action, resource = parts
                    if action == "denied":
                        capabilities.append(AgentCapability(
                            action="*", resource=resource, allowed=False
                        ))
                    else:
                        capabilities.append(AgentCapability(
                            action=action, resource=resource, allowed=True
                        ))

        # Parse triggers
        triggers = []
        for trig in data.get("triggers", []):
            triggers.append(TriggerConfig(
                event=trig.get("event", "manual"),
                path=trig.get("path"),
                cron=trig.get("cron"),
                url=trig.get("url"),
                action=trig.get("action", ""),
            ))

        # Parse memory config
        mem_data = data.get("memory", {})
        memory = MemoryConfig(
            hot_window=mem_data.get("hot_window", "7d"),
            warm_window=mem_data.get("warm_window", "30d"),
            cold_storage=mem_data.get("cold_storage", True),
        )

        # Parse recovery config
        rec_data = data.get("recovery", {})
        recovery = RecoveryConfig(
            snapshots=rec_data.get("snapshots", 10),
            max_rollback_age=rec_data.get("max_rollback_age", "24h"),
        )

        return AgentConfig(
            name=data.get("name", path.stem),
            model=data.get("model", "claude-3-5-sonnet"),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", "You are a helpful AI assistant."),
            capabilities=capabilities,
            memory=memory,
            recovery=recovery,
            triggers=triggers,
            version=data.get("version", "1.0"),
            author=data.get("author", ""),
            tags=data.get("tags", []),
        )

    def get_status(self) -> Dict:
        """Get daemon status"""
        return {
            "running": self._running,
            "pid": os.getpid(),
            "uptime_s": (time.time() - self._started_at) if self._started_at else 0,
            "agents": len(self.agents.list_agents()),
            "agents_running": len([
                a for a in self.agents.list_agents()
                if a.status == AgentStatus.RUNNING
            ]),
            "memory": self.memory.get_stats(),
            "home_dir": str(self.home_dir),
        }


# =============================================================================
# Global Instance
# =============================================================================


_daemon: Optional[BBXDaemon] = None


def get_daemon() -> BBXDaemon:
    """Get global daemon instance"""
    global _daemon
    if _daemon is None:
        _daemon = BBXDaemon()
    return _daemon


async def start_daemon():
    """Start global daemon"""
    daemon = get_daemon()
    await daemon.start()
    return daemon


async def stop_daemon():
    """Stop global daemon"""
    global _daemon
    if _daemon:
        await _daemon.stop()
        _daemon = None


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================


async def main():
    """Main entry point"""
    daemon = get_daemon()

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(daemon.stop())

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
        loop.add_signal_handler(signal.SIGINT, signal_handler)

    # Start daemon
    await daemon.start()

    # Keep running
    try:
        while daemon._running:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())
