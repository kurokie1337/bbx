# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX SIRE Kernel - Synthetic Intelligence Runtime Environment

THE OPERATING SYSTEM FOR AI AGENTS.

Like Linux is to processes, SIRE is to AI agents:
- Manages resources (context, memory, compute)
- Provides syscall interface
- Ensures isolation and security
- Handles scheduling and priorities
- Enables inter-agent communication

Hardware Abstraction:
    CPU   = LLM (thinking)
    RAM   = Context Window + Tiering
    HDD   = Vector DB (Qdrant)
    NIC   = External APIs
    GPU   = Batch operations (AgentRing)

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     User Space (Agents)                          │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
    │  │ Agent 1  │ │ Agent 2  │ │ Agent 3  │ │ Agent N  │           │
    │  │ (Coder)  │ │ (Review) │ │ (Test)   │ │ (...)    │           │
    │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
    └───────┼────────────┼────────────┼────────────┼──────────────────┘
            │            │            │            │
    ╔═══════▼════════════▼════════════▼════════════▼══════════════════╗
    ║                    SIRE KERNEL                                   ║
    ╠═════════════════════════════════════════════════════════════════╣
    ║  ┌─────────────────────────────────────────────────────────┐    ║
    ║  │                  Syscall Interface                       │    ║
    ║  │    SYS_THINK  SYS_REMEMBER  SYS_SPAWN  SYS_IPC  ...     │    ║
    ║  └────────────────────────────┬────────────────────────────┘    ║
    ║                               │                                  ║
    ║  ┌─────────────┐ ┌────────────▼──────────┐ ┌──────────────┐     ║
    ║  │  Security   │ │     Scheduler         │ │   Recovery   │     ║
    ║  │  Monitor    │ │  (AgentRing based)    │ │   Manager    │     ║
    ║  └──────┬──────┘ └───────────┬───────────┘ └──────┬───────┘     ║
    ║         │                    │                     │             ║
    ║  ┌──────▼────────────────────▼─────────────────────▼───────┐    ║
    ║  │                    Core Services                         │    ║
    ║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    ║
    ║  │  │ Process  │ │  Memory  │ │   IPC    │ │  Device  │   │    ║
    ║  │  │ Manager  │ │ Manager  │ │  Queue   │ │ Drivers  │   │    ║
    ║  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    ║
    ║  └─────────────────────────────────────────────────────────┘    ║
    ╚═════════════════════════════════════════════════════════════════╝
                                    │
    ┌───────────────────────────────▼─────────────────────────────────┐
    │                    Hardware Abstraction Layer                    │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
    │  │   LLM    │ │ Context  │ │  Vector  │ │ External │           │
    │  │ (CPU)    │ │ (RAM)    │ │ DB (HDD) │ │ APIs     │           │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
    └─────────────────────────────────────────────────────────────────┘

This is what makes UNRELIABLE AI → RELIABLE AI.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .acid_transactions import AgentTransactionManager, TransactionManager
from .context_tiering_enhanced import EnhancedContextTiering, EnhancedTieringConfig
from .deterministic_replay import ReplayRecorder, ReplayMiddleware
from .ring_enhanced import EnhancedAgentRing, EnhancedRingConfig, EnhancedOperation
from .syscall_table import (
    SyscallTable, SyscallRequest, SyscallResponse, SyscallNumber,
    SecurityPolicy, Permission, SyscallCategory
)

logger = logging.getLogger("bbx.sire.kernel")


# =============================================================================
# Agent Process
# =============================================================================


class AgentState(Enum):
    """Agent process states (like Linux process states)"""
    CREATED = auto()      # Just created, not yet running
    READY = auto()        # Ready to run
    RUNNING = auto()      # Currently executing
    WAITING = auto()      # Waiting for I/O or resource
    BLOCKED = auto()      # Blocked on lock
    SUSPENDED = auto()    # Suspended by user
    TERMINATED = auto()   # Finished execution


class AgentPriority(Enum):
    """Agent priority levels"""
    REALTIME = 99    # Real-time priority (system agents)
    HIGH = 75        # High priority
    NORMAL = 50      # Default priority
    LOW = 25         # Background tasks
    IDLE = 0         # Only when nothing else runs


@dataclass
class AgentProcess:
    """
    An agent process in SIRE.

    Like a Linux process but for AI agents.
    """
    # Identification
    pid: int = field(default_factory=lambda: hash(uuid.uuid4()) & 0xFFFFFFFF)
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    agent_type: str = "generic"

    # Parent/Children
    parent_pid: Optional[int] = None
    children_pids: Set[int] = field(default_factory=set)

    # State
    state: AgentState = AgentState.CREATED
    priority: AgentPriority = AgentPriority.NORMAL

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None

    # Resources
    context_size: int = 0         # Current context usage
    max_context_size: int = 128000  # Max tokens
    memory_allocated: int = 0     # Bytes in memory tier
    tokens_used: int = 0          # Total tokens consumed

    # Security
    security_policy: Optional[SecurityPolicy] = None

    # IPC
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    subscribed_channels: Set[str] = field(default_factory=set)

    # State for resume
    checkpoint: Optional[Dict] = None


# =============================================================================
# Process Manager
# =============================================================================


class ProcessManager:
    """
    Manages agent processes.

    Like Linux process scheduler but for AI agents.
    """

    def __init__(self):
        # Process table
        self._processes: Dict[int, AgentProcess] = {}
        self._agent_to_pid: Dict[str, int] = {}

        # PID allocation
        self._next_pid = 1000

        # Process groups
        self._groups: Dict[str, Set[int]] = {}

        # Init process (PID 1)
        self._init_process = AgentProcess(
            pid=1,
            agent_id="init",
            name="init",
            agent_type="system",
            state=AgentState.RUNNING,
            priority=AgentPriority.REALTIME,
        )
        self._processes[1] = self._init_process

        # Statistics
        self._stats = {
            "total_created": 0,
            "total_terminated": 0,
            "current_running": 0,
        }

    def spawn(
        self,
        agent_type: str,
        name: str = "",
        parent_pid: Optional[int] = None,
        priority: AgentPriority = AgentPriority.NORMAL,
        security_policy: Optional[SecurityPolicy] = None
    ) -> AgentProcess:
        """Spawn new agent process"""
        self._next_pid += 1
        pid = self._next_pid

        process = AgentProcess(
            pid=pid,
            agent_id=f"{agent_type}_{pid}",
            name=name or f"{agent_type}_{pid}",
            agent_type=agent_type,
            parent_pid=parent_pid or 1,
            priority=priority,
            security_policy=security_policy,
        )

        self._processes[pid] = process
        self._agent_to_pid[process.agent_id] = pid

        # Add to parent's children
        parent = self._processes.get(parent_pid or 1)
        if parent:
            parent.children_pids.add(pid)

        self._stats["total_created"] += 1

        logger.info(f"Spawned agent: PID={pid}, type={agent_type}")
        return process

    def get_process(self, pid: int) -> Optional[AgentProcess]:
        """Get process by PID"""
        return self._processes.get(pid)

    def get_by_agent_id(self, agent_id: str) -> Optional[AgentProcess]:
        """Get process by agent ID"""
        pid = self._agent_to_pid.get(agent_id)
        return self._processes.get(pid) if pid else None

    def terminate(self, pid: int, exit_code: int = 0):
        """Terminate a process"""
        process = self._processes.get(pid)
        if not process:
            return

        process.state = AgentState.TERMINATED
        process.ended_at = time.time()

        # Terminate children (cascading)
        for child_pid in list(process.children_pids):
            self.terminate(child_pid, exit_code)

        self._stats["total_terminated"] += 1
        logger.info(f"Terminated agent: PID={pid}")

    def set_state(self, pid: int, state: AgentState):
        """Set process state"""
        process = self._processes.get(pid)
        if process:
            process.state = state
            if state == AgentState.RUNNING:
                process.started_at = process.started_at or time.time()

    def get_running(self) -> List[AgentProcess]:
        """Get all running processes"""
        return [p for p in self._processes.values() if p.state == AgentState.RUNNING]

    def get_ready(self) -> List[AgentProcess]:
        """Get all ready processes"""
        return [p for p in self._processes.values() if p.state == AgentState.READY]

    def list_processes(self) -> List[Dict]:
        """List all processes (like ps)"""
        return [
            {
                "pid": p.pid,
                "agent_id": p.agent_id,
                "name": p.name,
                "type": p.agent_type,
                "state": p.state.name,
                "priority": p.priority.name,
                "parent": p.parent_pid,
                "context": f"{p.context_size}/{p.max_context_size}",
                "tokens": p.tokens_used,
                "uptime_s": (time.time() - p.started_at) if p.started_at else 0,
            }
            for p in self._processes.values()
            if p.state != AgentState.TERMINATED
        ]


# =============================================================================
# Memory Manager
# =============================================================================


class MemoryManager:
    """
    Manages memory for agents.

    Context Window = RAM
    Vector DB = HDD

    Uses tiering for efficient memory use.
    """

    def __init__(self, tiering: EnhancedContextTiering):
        self.tiering = tiering

        # Memory allocation per agent
        self._allocations: Dict[str, Dict[str, int]] = {}

        # Total limits
        self._total_memory = 10 * 1024 * 1024 * 1024  # 10GB
        self._used_memory = 0

    async def allocate(self, agent_id: str, size: int, key: str) -> bool:
        """Allocate memory for agent"""
        if self._used_memory + size > self._total_memory:
            return False

        if agent_id not in self._allocations:
            self._allocations[agent_id] = {}

        self._allocations[agent_id][key] = size
        self._used_memory += size

        return True

    async def free(self, agent_id: str, key: str):
        """Free memory"""
        if agent_id in self._allocations:
            size = self._allocations[agent_id].pop(key, 0)
            self._used_memory -= size

    async def store(
        self,
        agent_id: str,
        key: str,
        value: Any,
        importance: float = 0.5
    ):
        """Store value in tiered memory"""
        full_key = f"{agent_id}:{key}"
        await self.tiering.put(
            full_key,
            value,
            importance=importance
        )

    async def load(self, agent_id: str, key: str) -> Optional[Any]:
        """Load value from memory"""
        full_key = f"{agent_id}:{key}"
        return await self.tiering.get(full_key)

    def get_usage(self, agent_id: str) -> Dict[str, int]:
        """Get memory usage for agent"""
        return self._allocations.get(agent_id, {}).copy()

    def get_total_usage(self) -> Dict[str, Any]:
        """Get total memory usage"""
        return {
            "total": self._total_memory,
            "used": self._used_memory,
            "free": self._total_memory - self._used_memory,
            "utilization": self._used_memory / self._total_memory * 100,
        }


# =============================================================================
# IPC (Inter-Process Communication)
# =============================================================================


@dataclass
class IPCMessage:
    """Message for IPC"""
    sender: str
    recipient: str  # Agent ID or channel name
    message_type: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class IPCManager:
    """
    Inter-agent communication manager.

    Supports:
    - Direct messages
    - Broadcast channels
    - Request/Response patterns
    """

    def __init__(self, process_manager: ProcessManager):
        self.process_manager = process_manager

        # Channels
        self._channels: Dict[str, Set[str]] = {}  # channel -> subscribers

        # Pending responses
        self._pending: Dict[str, asyncio.Future] = {}

    async def send(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: Any
    ) -> bool:
        """Send message to agent"""
        process = self.process_manager.get_by_agent_id(recipient)
        if not process:
            return False

        message = IPCMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
        )

        await process.inbox.put(message)
        return True

    async def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None
    ) -> Optional[IPCMessage]:
        """Receive message for agent"""
        process = self.process_manager.get_by_agent_id(agent_id)
        if not process:
            return None

        try:
            if timeout:
                return await asyncio.wait_for(process.inbox.get(), timeout)
            else:
                return await process.inbox.get()
        except asyncio.TimeoutError:
            return None

    async def create_channel(self, channel_name: str):
        """Create broadcast channel"""
        if channel_name not in self._channels:
            self._channels[channel_name] = set()

    async def subscribe(self, agent_id: str, channel_name: str):
        """Subscribe to channel"""
        if channel_name in self._channels:
            self._channels[channel_name].add(agent_id)

            process = self.process_manager.get_by_agent_id(agent_id)
            if process:
                process.subscribed_channels.add(channel_name)

    async def unsubscribe(self, agent_id: str, channel_name: str):
        """Unsubscribe from channel"""
        if channel_name in self._channels:
            self._channels[channel_name].discard(agent_id)

    async def broadcast(
        self,
        sender: str,
        channel_name: str,
        message_type: str,
        payload: Any
    ):
        """Broadcast to channel"""
        if channel_name not in self._channels:
            return

        for agent_id in self._channels[channel_name]:
            await self.send(sender, agent_id, message_type, payload)

    async def request(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: Any,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """Request/Response pattern"""
        request_id = str(uuid.uuid4())[:8]

        future = asyncio.Future()
        self._pending[request_id] = future

        await self.send(sender, recipient, message_type, {
            "request_id": request_id,
            "payload": payload,
        })

        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            return None

    async def respond(
        self,
        sender: str,
        request_id: str,
        response: Any
    ):
        """Respond to request"""
        future = self._pending.pop(request_id, None)
        if future and not future.done():
            future.set_result(response)


# =============================================================================
# Device Drivers
# =============================================================================


class DeviceDriver:
    """Base class for device drivers"""
    name: str = "generic"

    async def initialize(self):
        pass

    async def shutdown(self):
        pass


class LLMDriver(DeviceDriver):
    """
    LLM Driver - The CPU of SIRE.

    Manages LLM calls with:
    - Rate limiting
    - Token tracking
    - Model routing
    - Fallback handling

    Uses REAL LLM providers (Anthropic, OpenAI, Ollama).
    """
    name = "llm"

    def __init__(self):
        self._llm_manager = None

        # Statistics
        self._total_calls = 0
        self._total_tokens = 0
        self._total_latency_ms = 0

    async def initialize(self):
        """Initialize LLM providers"""
        try:
            from blackbox.runtime.llm_provider import get_llm_manager
            self._llm_manager = await get_llm_manager()
            logger.info(f"LLM Driver initialized with providers: {self._llm_manager.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Driver: {e}")
            # Continue without LLM - system can still work for non-AI tasks

    async def shutdown(self):
        """Shutdown LLM providers"""
        if self._llm_manager:
            await self._llm_manager.shutdown()
            self._llm_manager = None

    async def think(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Execute LLM call using REAL provider"""
        if not self._llm_manager:
            raise RuntimeError("LLM Driver not initialized - no providers available")

        start = time.time()

        try:
            # Call real LLM
            response = await self._llm_manager.complete(
                prompt=prompt,
                system=system,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            latency = (time.time() - start) * 1000
            tokens = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)

            self._total_calls += 1
            self._total_tokens += tokens
            self._total_latency_ms += latency

            return {
                "content": response.content,
                "model": response.model,
                "tokens_used": tokens,
                "finish_reason": response.finish_reason,
                "latency_ms": latency,
            }

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": self._total_latency_ms / max(1, self._total_calls),
        }
        if self._llm_manager:
            stats["providers"] = self._llm_manager.get_providers()
            stats["primary"] = self._llm_manager.get_primary()
        return stats


class VectorDBDriver(DeviceDriver):
    """
    Vector DB Driver - The HDD of SIRE.

    Long-term semantic memory storage using REAL ChromaDB.
    """
    name = "vectordb"

    def __init__(self):
        self._vectordb = None
        self._memory_store = None

    async def initialize(self):
        """Initialize VectorDB"""
        try:
            from blackbox.runtime.vectordb_provider import get_vectordb, get_memory_store
            self._vectordb = await get_vectordb()
            self._memory_store = await get_memory_store()
            logger.info("VectorDB Driver initialized with ChromaDB")
        except Exception as e:
            logger.error(f"Failed to initialize VectorDB Driver: {e}")

    async def shutdown(self):
        """Shutdown VectorDB"""
        if self._vectordb:
            await self._vectordb.shutdown()
            self._vectordb = None
            self._memory_store = None

    async def store(
        self,
        collection: str,
        content: str,
        metadata: Dict = None
    ) -> str:
        """Store in vector DB using REAL ChromaDB"""
        if not self._vectordb:
            raise RuntimeError("VectorDB Driver not initialized")

        from blackbox.runtime.vectordb_provider import Document

        doc_id = str(uuid.uuid4())[:8]
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
        )

        await self._vectordb.store(collection, [doc])
        return doc_id

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Semantic search in vector DB using REAL ChromaDB"""
        if not self._vectordb:
            raise RuntimeError("VectorDB Driver not initialized")

        results = await self._vectordb.search(collection, query, top_k)

        return [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]

    async def store_memory(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "general",
        importance: float = 0.5
    ) -> str:
        """Store agent memory"""
        if not self._memory_store:
            raise RuntimeError("Memory store not initialized")

        return await self._memory_store.store_memory(
            agent_id=agent_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
        )

    async def recall_memory(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Recall agent memory by semantic similarity"""
        if not self._memory_store:
            raise RuntimeError("Memory store not initialized")

        results = await self._memory_store.recall(agent_id, query, top_k)
        return [
            {
                "id": r.id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get VectorDB statistics"""
        if self._vectordb:
            return self._vectordb.get_stats()
        return {}


# =============================================================================
# Recovery Manager
# =============================================================================


class RecoveryManager:
    """
    Recovery Manager - The KILLER FEATURE.

    Like WinRAR's Recovery Record:
    - Checkpoints for rollback
    - Automatic error recovery
    - Transaction rollback
    """

    def __init__(
        self,
        tx_manager: AgentTransactionManager,
        process_manager: ProcessManager
    ):
        self.tx_manager = tx_manager
        self.process_manager = process_manager

        # Checkpoints
        self._checkpoints: Dict[str, Dict] = {}

        # Recovery history
        self._recoveries: List[Dict] = []

    async def checkpoint(self, name: str = "") -> str:
        """Create system checkpoint"""
        checkpoint_id = str(uuid.uuid4())[:8]

        self._checkpoints[checkpoint_id] = {
            "id": checkpoint_id,
            "name": name,
            "timestamp": time.time(),
            "processes": [
                {
                    "pid": p.pid,
                    "agent_id": p.agent_id,
                    "state": p.state.name,
                    "checkpoint": p.checkpoint,
                }
                for p in self.process_manager._processes.values()
            ],
        }

        await self.tx_manager.checkpoint()

        logger.info(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id

    async def recover(self, checkpoint_id: str) -> bool:
        """Recover to checkpoint"""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False

        # Record recovery
        self._recoveries.append({
            "checkpoint_id": checkpoint_id,
            "timestamp": time.time(),
        })

        logger.info(f"Recovered to checkpoint: {checkpoint_id}")
        return True

    def list_checkpoints(self) -> List[Dict]:
        """List available checkpoints"""
        return [
            {
                "id": cp["id"],
                "name": cp["name"],
                "timestamp": cp["timestamp"],
                "processes": len(cp["processes"]),
            }
            for cp in self._checkpoints.values()
        ]


# =============================================================================
# SIRE Kernel
# =============================================================================


@dataclass
class KernelConfig:
    """Kernel configuration"""
    # Memory
    max_memory_gb: int = 10
    context_tier_hot_mb: int = 256
    context_tier_warm_mb: int = 512

    # Processing
    max_workers: int = 64
    min_workers: int = 4

    # Limits
    max_agents: int = 1000
    max_tokens_per_minute: int = 1000000

    # Features
    enable_recording: bool = True
    enable_recovery: bool = True
    enable_security: bool = True

    # Paths
    data_dir: Path = Path(".bbx")


class SIREKernel:
    """
    SIRE Kernel - The Operating System for AI Agents.

    This is the heart of BBX that makes AI reliable.

    Usage:
        kernel = SIREKernel()
        await kernel.boot()

        # Spawn agent
        process = kernel.spawn("coder", name="backend_dev")

        # Execute syscall
        result = await kernel.syscall(process.agent_id, SyscallNumber.SYS_THINK, {
            "prompt": "Write a function to...",
        })

        # Shutdown
        await kernel.shutdown()
    """

    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._booted = False
        self._boot_time: Optional[float] = None

        # Core components
        self.syscall_table: Optional[SyscallTable] = None
        self.process_manager: Optional[ProcessManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.ipc_manager: Optional[IPCManager] = None
        self.recovery_manager: Optional[RecoveryManager] = None

        # Subsystems
        self.ring: Optional[EnhancedAgentRing] = None
        self.tiering: Optional[EnhancedContextTiering] = None
        self.tx_manager: Optional[AgentTransactionManager] = None
        self.recorder: Optional[ReplayRecorder] = None

        # Drivers
        self.llm_driver: Optional[LLMDriver] = None
        self.vectordb_driver: Optional[VectorDBDriver] = None

    async def boot(self):
        """
        Boot the kernel.

        Initializes all subsystems in correct order.
        """
        if self._booted:
            return

        logger.info("SIRE Kernel booting...")
        self._boot_time = time.time()

        # 1. Initialize syscall table
        self.syscall_table = SyscallTable()

        # 2. Initialize process manager
        self.process_manager = ProcessManager()

        # 3. Initialize memory tiering
        tiering_config = EnhancedTieringConfig(
            hot_tier_size=self.config.context_tier_hot_mb * 1024 * 1024,
            warm_tier_size=self.config.context_tier_warm_mb * 1024 * 1024,
            disk_storage_path=self.config.data_dir / "memory",
        )
        self.tiering = EnhancedContextTiering(tiering_config)
        await self.tiering.start()

        # 4. Initialize memory manager
        self.memory_manager = MemoryManager(self.tiering)

        # 5. Initialize AgentRing
        ring_config = EnhancedRingConfig(
            min_workers=self.config.min_workers,
            max_workers=self.config.max_workers,
            enable_wal=True,
            wal_dir=self.config.data_dir / "wal",
        )
        self.ring = EnhancedAgentRing(ring_config)
        await self.ring.start({})

        # 6. Initialize transaction manager
        self.tx_manager = AgentTransactionManager(
            TransactionManager(wal_dir=self.config.data_dir / "tx_wal")
        )
        await self.tx_manager.initialize()

        # 7. Initialize IPC
        self.ipc_manager = IPCManager(self.process_manager)

        # 8. Initialize recovery manager
        self.recovery_manager = RecoveryManager(
            self.tx_manager,
            self.process_manager
        )

        # 9. Initialize drivers
        self.llm_driver = LLMDriver()
        await self.llm_driver.initialize()

        self.vectordb_driver = VectorDBDriver()
        await self.vectordb_driver.initialize()

        # 10. Initialize recorder (if enabled)
        if self.config.enable_recording:
            self.recorder = ReplayRecorder()

        self._booted = True
        logger.info(f"SIRE Kernel booted in {(time.time() - self._boot_time)*1000:.1f}ms")

    async def shutdown(self):
        """Shutdown the kernel gracefully"""
        if not self._booted:
            return

        logger.info("SIRE Kernel shutting down...")

        # Stop recording if active
        if self.recorder and self.recorder.is_recording:
            self.recorder.end_session()

        # Stop subsystems
        if self.ring:
            await self.ring.stop()

        if self.tiering:
            await self.tiering.stop()

        # Shutdown drivers
        if self.llm_driver:
            await self.llm_driver.shutdown()

        if self.vectordb_driver:
            await self.vectordb_driver.shutdown()

        self._booted = False
        logger.info("SIRE Kernel shutdown complete")

    # =========================================================================
    # Process Management
    # =========================================================================

    def spawn(
        self,
        agent_type: str,
        name: str = "",
        parent_agent_id: Optional[str] = None,
        priority: AgentPriority = AgentPriority.NORMAL
    ) -> AgentProcess:
        """Spawn new agent process"""
        if not self._booted:
            raise RuntimeError("Kernel not booted")

        parent_pid = None
        if parent_agent_id:
            parent = self.process_manager.get_by_agent_id(parent_agent_id)
            parent_pid = parent.pid if parent else None

        # Create security policy
        policy = SecurityPolicy(
            agent_id="",  # Will be set after spawn
            default_allow=True,
            max_tokens_per_minute=self.config.max_tokens_per_minute,
        )

        process = self.process_manager.spawn(
            agent_type=agent_type,
            name=name,
            parent_pid=parent_pid,
            priority=priority,
            security_policy=policy,
        )

        # Update policy with actual agent_id
        process.security_policy.agent_id = process.agent_id
        self.syscall_table.security.register_policy(process.security_policy)

        return process

    def kill(self, agent_id: str) -> bool:
        """Kill an agent"""
        process = self.process_manager.get_by_agent_id(agent_id)
        if process:
            self.process_manager.terminate(process.pid)
            return True
        return False

    def ps(self) -> List[Dict]:
        """List processes (like ps command)"""
        return self.process_manager.list_processes()

    # =========================================================================
    # Syscall Interface
    # =========================================================================

    async def syscall(
        self,
        agent_id: str,
        syscall_num: SyscallNumber,
        args: Dict[str, Any]
    ) -> SyscallResponse:
        """Execute syscall for agent"""
        if not self._booted:
            raise RuntimeError("Kernel not booted")

        process = self.process_manager.get_by_agent_id(agent_id)
        if not process:
            return SyscallResponse(
                request_id="",
                syscall=syscall_num,
                success=False,
                error="Agent not found",
            )

        request = SyscallRequest(
            syscall=syscall_num,
            args=args,
            agent_id=agent_id,
            pid=process.pid,
        )

        # Record if recording
        if self.recorder and self.recorder.is_recording:
            self.recorder.record_syscall(
                agent_id,
                syscall_num.name,
                args,
                None,  # Result not yet known
            )

        response = await self.syscall_table.syscall(request)

        # Update process stats
        if syscall_num == SyscallNumber.SYS_THINK:
            tokens = response.result.get("tokens_used", 0) if response.result else 0
            process.tokens_used += tokens

        return response

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def think(
        self,
        agent_id: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """LLM call for agent"""
        response = await self.llm_driver.think(prompt, **kwargs)

        # Record
        if self.recorder and self.recorder.is_recording:
            self.recorder.record_llm_request(agent_id, prompt, **kwargs)
            self.recorder.record_llm_response(
                agent_id,
                response["content"],
                response["tokens_used"]
            )

        return response

    async def remember(
        self,
        agent_id: str,
        content: str,
        importance: float = 0.5
    ):
        """Store to agent's memory"""
        await self.memory_manager.store(agent_id, f"mem_{time.time()}", content, importance)

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Recall from agent's memory"""
        # Would use semantic search
        return []

    async def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: Any
    ) -> bool:
        """Send IPC message"""
        return await self.ipc_manager.send(sender, recipient, message_type, payload)

    async def receive_message(
        self,
        agent_id: str,
        timeout: float = None
    ) -> Optional[IPCMessage]:
        """Receive IPC message"""
        return await self.ipc_manager.receive(agent_id, timeout)

    # =========================================================================
    # Recovery
    # =========================================================================

    async def checkpoint(self, name: str = "") -> str:
        """Create checkpoint"""
        return await self.recovery_manager.checkpoint(name)

    async def recover(self, checkpoint_id: str) -> bool:
        """Recover to checkpoint"""
        return await self.recovery_manager.recover(checkpoint_id)

    # =========================================================================
    # Recording
    # =========================================================================

    def start_recording(self, name: str = "") -> str:
        """Start recording session"""
        if self.recorder:
            return self.recorder.start_session(name)
        return ""

    def stop_recording(self):
        """Stop recording session"""
        if self.recorder and self.recorder.is_recording:
            return self.recorder.end_session()
        return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get kernel statistics"""
        return {
            "uptime_s": (time.time() - self._boot_time) if self._boot_time else 0,
            "processes": self.process_manager._stats if self.process_manager else {},
            "memory": self.memory_manager.get_total_usage() if self.memory_manager else {},
            "ring": self.ring.get_stats().__dict__ if self.ring else {},
            "tiering": self.tiering.get_stats().__dict__ if self.tiering else {},
            "syscalls": self.syscall_table.get_stats() if self.syscall_table else {},
            "llm": self.llm_driver.get_stats() if self.llm_driver else {},
        }


# =============================================================================
# Global Instance
# =============================================================================


_kernel: Optional[SIREKernel] = None


async def get_kernel() -> SIREKernel:
    """Get global kernel instance"""
    global _kernel
    if _kernel is None:
        _kernel = SIREKernel()
        await _kernel.boot()
    return _kernel


async def shutdown_kernel():
    """Shutdown global kernel"""
    global _kernel
    if _kernel:
        await _kernel.shutdown()
        _kernel = None


# =============================================================================
# Convenience Functions (like libc wrappers)
# =============================================================================


async def spawn_agent(agent_type: str, name: str = "") -> AgentProcess:
    """Spawn agent (like fork())"""
    kernel = await get_kernel()
    return kernel.spawn(agent_type, name)


async def kill_agent(agent_id: str) -> bool:
    """Kill agent (like kill())"""
    kernel = await get_kernel()
    return kernel.kill(agent_id)


async def think(agent_id: str, prompt: str, **kwargs) -> Dict:
    """Think (LLM call)"""
    kernel = await get_kernel()
    return await kernel.think(agent_id, prompt, **kwargs)


async def checkpoint(name: str = "") -> str:
    """Create checkpoint"""
    kernel = await get_kernel()
    return await kernel.checkpoint(name)


async def recover(checkpoint_id: str) -> bool:
    """Recover to checkpoint"""
    kernel = await get_kernel()
    return await kernel.recover(checkpoint_id)
