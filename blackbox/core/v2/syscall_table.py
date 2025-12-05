# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX SIRE Syscall Table - Controlled API for AI Agents.

Like Linux syscalls but for AI:
- Every agent operation goes through syscall table
- Security monitor validates each call
- Audit log for every operation
- Resource limits enforced

Syscall Categories:
    0xx - File Operations (read, write, create, delete)
    1xx - Memory Operations (alloc, free, mmap)
    2xx - Process Operations (spawn, kill, wait)
    3xx - IPC Operations (send, recv, share)
    4xx - Network Operations (fetch, connect)
    5xx - AI Operations (think, remember, learn)

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Agent Application                         │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ syscall(SYS_READ, path, buffer, size)                   │ │
    │  └────────────────────────┬────────────────────────────────┘ │
    └───────────────────────────┼─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                   Syscall Table                              │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
    │  │  0x01   │ │  0x02   │ │  0x03   │ │  ...    │           │
    │  │  READ   │ │  WRITE  │ │  SPAWN  │ │         │           │
    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │
    └───────┼───────────┼───────────┼───────────┼─────────────────┘
            │           │           │           │
    ┌───────▼───────────▼───────────▼───────────▼─────────────────┐
    │                 Security Monitor (eBPF-inspired)             │
    │  ┌──────────────────────────────────────────────────────┐   │
    │  │ - Permission checks                                   │   │
    │  │ - Resource limits                                     │   │
    │  │ - Audit logging                                       │   │
    │  │ - Rate limiting                                       │   │
    │  └──────────────────────────────────────────────────────┘   │
    └───────────────────────────┬─────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────┐
    │                   Kernel Services                            │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
    │  │   VFS    │ │  Memory  │ │ Process  │ │   IPC    │       │
    │  │ Service  │ │ Manager  │ │ Manager  │ │  Queue   │       │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.sire.syscall")


# =============================================================================
# Syscall Numbers (like Linux syscall table)
# =============================================================================


class SyscallNumber(IntEnum):
    """Syscall numbers organized by category"""

    # 0xx - File Operations
    SYS_READ = 0x01
    SYS_WRITE = 0x02
    SYS_CREATE = 0x03
    SYS_DELETE = 0x04
    SYS_STAT = 0x05
    SYS_LIST = 0x06
    SYS_MOVE = 0x07
    SYS_COPY = 0x08
    SYS_WATCH = 0x09

    # 1xx - Memory Operations
    SYS_ALLOC = 0x10
    SYS_FREE = 0x11
    SYS_STORE = 0x12
    SYS_LOAD = 0x13
    SYS_MMAP = 0x14
    SYS_PIN = 0x15
    SYS_UNPIN = 0x16

    # 2xx - Process/Agent Operations
    SYS_SPAWN = 0x20
    SYS_KILL = 0x21
    SYS_WAIT = 0x22
    SYS_SIGNAL = 0x23
    SYS_GETPID = 0x24
    SYS_SETPRIORITY = 0x25
    SYS_YIELD = 0x26

    # 3xx - IPC Operations
    SYS_IPC_SEND = 0x30
    SYS_IPC_RECV = 0x31
    SYS_IPC_SHARE = 0x32
    SYS_IPC_CREATE_CHANNEL = 0x33
    SYS_IPC_CLOSE_CHANNEL = 0x34
    SYS_BROADCAST = 0x35

    # 4xx - Network Operations
    SYS_FETCH = 0x40
    SYS_CONNECT = 0x41
    SYS_DISCONNECT = 0x42
    SYS_STREAM = 0x43

    # 5xx - AI Operations (SIRE-specific)
    SYS_THINK = 0x50        # Call LLM
    SYS_REMEMBER = 0x51     # Store to memory (RAG)
    SYS_RECALL = 0x52       # Query memory
    SYS_LEARN = 0x53        # Update knowledge
    SYS_EMBED = 0x54        # Create embedding
    SYS_COMPRESS = 0x55     # Semantic compression
    SYS_EXPAND = 0x56       # Expand intent

    # 6xx - Transaction Operations
    SYS_TX_BEGIN = 0x60
    SYS_TX_COMMIT = 0x61
    SYS_TX_ROLLBACK = 0x62
    SYS_CHECKPOINT = 0x63
    SYS_RECOVER = 0x64


class SyscallCategory(IntEnum):
    """Syscall categories for permission grouping"""
    FILE = 0
    MEMORY = 1
    PROCESS = 2
    IPC = 3
    NETWORK = 4
    AI = 5
    TRANSACTION = 6


def get_syscall_category(syscall: SyscallNumber) -> SyscallCategory:
    """Get category for a syscall number"""
    num = syscall.value
    if num < 0x10:
        return SyscallCategory.FILE
    elif num < 0x20:
        return SyscallCategory.MEMORY
    elif num < 0x30:
        return SyscallCategory.PROCESS
    elif num < 0x40:
        return SyscallCategory.IPC
    elif num < 0x50:
        return SyscallCategory.NETWORK
    elif num < 0x60:
        return SyscallCategory.AI
    else:
        return SyscallCategory.TRANSACTION


# =============================================================================
# Syscall Request/Response
# =============================================================================


@dataclass
class SyscallRequest:
    """Request to execute a syscall"""
    syscall: SyscallNumber
    args: Dict[str, Any] = field(default_factory=dict)

    # Caller identification
    agent_id: str = ""
    pid: int = 0

    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Transaction context
    tx_id: Optional[str] = None


@dataclass
class SyscallResponse:
    """Response from syscall execution"""
    request_id: str
    syscall: SyscallNumber

    # Result
    success: bool = True
    result: Any = None
    error: Optional[str] = None
    error_code: int = 0

    # Metrics
    start_time: float = 0
    end_time: float = 0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


# =============================================================================
# Error Codes (like errno)
# =============================================================================


class ErrorCode(IntEnum):
    """Syscall error codes"""
    SUCCESS = 0

    # Permission errors (1-10)
    EPERM = 1          # Operation not permitted
    EACCES = 2         # Permission denied
    EQUOTA = 3         # Quota exceeded
    ERATE = 4          # Rate limit exceeded

    # Resource errors (11-20)
    ENOENT = 11        # No such entry
    EEXIST = 12        # Entry already exists
    ENOMEM = 13        # Out of memory
    ENOSPC = 14        # No space left
    EBUSY = 15         # Resource busy

    # Operation errors (21-30)
    EINVAL = 21        # Invalid argument
    ETIMEOUT = 22      # Operation timed out
    ECANCELED = 23     # Operation canceled
    EFAULT = 24        # Bad address
    ENOTIMPL = 25      # Not implemented

    # AI-specific errors (51-60)
    EMODEL = 51        # Model error
    ECTX = 52          # Context overflow
    EHALLUC = 53       # Hallucination detected
    ECONFIDENCE = 54   # Low confidence result

    # Transaction errors (61-70)
    ETXFAIL = 61       # Transaction failed
    ETXCONFLICT = 62   # Transaction conflict
    ETXROLLBACK = 63   # Rollback required


# =============================================================================
# Security Monitor (eBPF-inspired)
# =============================================================================


@dataclass
class Permission:
    """Permission definition"""
    syscall: Optional[SyscallNumber] = None  # None = all
    category: Optional[SyscallCategory] = None  # None = all categories
    allow: bool = True
    paths: List[str] = field(default_factory=list)  # Allowed paths (for file ops)
    rate_limit: Optional[int] = None  # Max calls per minute
    resource_limit: Optional[int] = None  # Max resource usage


@dataclass
class SecurityPolicy:
    """Security policy for an agent"""
    agent_id: str
    permissions: List[Permission] = field(default_factory=list)

    # Default behavior
    default_allow: bool = False

    # Resource limits
    max_memory_mb: int = 1024
    max_spawned_agents: int = 10
    max_open_channels: int = 100
    max_tokens_per_minute: int = 100000

    # Audit settings
    audit_all: bool = False
    audit_categories: Set[SyscallCategory] = field(default_factory=set)


@dataclass
class AuditEntry:
    """Audit log entry"""
    timestamp: float
    agent_id: str
    syscall: SyscallNumber
    args_hash: str  # Hash of args for privacy
    success: bool
    error_code: int
    duration_ms: float
    tx_id: Optional[str] = None


class SecurityMonitor:
    """
    eBPF-inspired security monitor for syscalls.

    Every syscall passes through here for:
    - Permission checks
    - Rate limiting
    - Resource enforcement
    - Audit logging
    """

    def __init__(self):
        # Policies by agent
        self._policies: Dict[str, SecurityPolicy] = {}

        # Rate limiting state
        self._rate_counters: Dict[str, Dict[SyscallNumber, List[float]]] = {}

        # Resource tracking
        self._resources: Dict[str, Dict[str, int]] = {}

        # Audit log
        self._audit_log: List[AuditEntry] = []
        self._max_audit_entries = 10000

        # Hooks (eBPF-style programmable checks)
        self._pre_hooks: Dict[SyscallNumber, List[Callable]] = {}
        self._post_hooks: Dict[SyscallNumber, List[Callable]] = {}

        # Default policy
        self._default_policy = SecurityPolicy(
            agent_id="default",
            default_allow=True,
            permissions=[
                Permission(category=SyscallCategory.AI, allow=True),
                Permission(category=SyscallCategory.MEMORY, allow=True),
                Permission(syscall=SyscallNumber.SYS_THINK, rate_limit=60),
            ]
        )

    def register_policy(self, policy: SecurityPolicy):
        """Register security policy for an agent"""
        self._policies[policy.agent_id] = policy
        self._rate_counters[policy.agent_id] = {}
        self._resources[policy.agent_id] = {
            "memory_mb": 0,
            "spawned_agents": 0,
            "open_channels": 0,
            "tokens_used": 0,
        }

    def register_hook(
        self,
        syscall: SyscallNumber,
        hook: Callable[[SyscallRequest], Optional[str]],
        pre: bool = True
    ):
        """
        Register eBPF-style hook.

        Hook receives request, returns None to allow or error message to deny.
        """
        hooks = self._pre_hooks if pre else self._post_hooks
        if syscall not in hooks:
            hooks[syscall] = []
        hooks[syscall].append(hook)

    async def check(self, request: SyscallRequest) -> Optional[str]:
        """
        Check if syscall is allowed.

        Returns None if allowed, error message if denied.
        """
        policy = self._policies.get(request.agent_id, self._default_policy)

        # 1. Run pre-hooks
        if request.syscall in self._pre_hooks:
            for hook in self._pre_hooks[request.syscall]:
                try:
                    error = hook(request)
                    if error:
                        return error
                except Exception as e:
                    logger.warning(f"Hook error: {e}")

        # 2. Check permissions
        category = get_syscall_category(request.syscall)
        allowed = policy.default_allow
        rate_limit = None

        for perm in policy.permissions:
            # Check if permission applies
            if perm.syscall is not None and perm.syscall != request.syscall:
                continue
            if perm.category is not None and perm.category != category:
                continue

            allowed = perm.allow
            rate_limit = perm.rate_limit

            # Check path restrictions for file ops
            if perm.paths and category == SyscallCategory.FILE:
                path = request.args.get("path", "")
                if not any(path.startswith(p) for p in perm.paths):
                    return f"Path not allowed: {path}"

        if not allowed:
            return f"Permission denied for {request.syscall.name}"

        # 3. Check rate limit
        if rate_limit:
            error = self._check_rate_limit(request, rate_limit)
            if error:
                return error

        # 4. Check resource limits
        error = self._check_resources(request, policy)
        if error:
            return error

        return None

    def _check_rate_limit(self, request: SyscallRequest, limit: int) -> Optional[str]:
        """Check rate limit for syscall"""
        counters = self._rate_counters.get(request.agent_id, {})
        syscall_times = counters.get(request.syscall, [])

        # Clean old entries (older than 1 minute)
        now = time.time()
        syscall_times = [t for t in syscall_times if now - t < 60]

        if len(syscall_times) >= limit:
            return f"Rate limit exceeded for {request.syscall.name}: {limit}/min"

        syscall_times.append(now)
        counters[request.syscall] = syscall_times
        self._rate_counters[request.agent_id] = counters

        return None

    def _check_resources(self, request: SyscallRequest, policy: SecurityPolicy) -> Optional[str]:
        """Check resource limits"""
        resources = self._resources.get(request.agent_id, {})

        # Memory check for alloc
        if request.syscall == SyscallNumber.SYS_ALLOC:
            size_mb = request.args.get("size_mb", 0)
            if resources.get("memory_mb", 0) + size_mb > policy.max_memory_mb:
                return f"Memory limit exceeded: {policy.max_memory_mb}MB"

        # Agent count check for spawn
        if request.syscall == SyscallNumber.SYS_SPAWN:
            if resources.get("spawned_agents", 0) >= policy.max_spawned_agents:
                return f"Agent limit exceeded: {policy.max_spawned_agents}"

        # Channel check for IPC
        if request.syscall == SyscallNumber.SYS_IPC_CREATE_CHANNEL:
            if resources.get("open_channels", 0) >= policy.max_open_channels:
                return f"Channel limit exceeded: {policy.max_open_channels}"

        return None

    def record_audit(self, request: SyscallRequest, response: SyscallResponse):
        """Record syscall in audit log"""
        policy = self._policies.get(request.agent_id, self._default_policy)
        category = get_syscall_category(request.syscall)

        # Check if we should audit
        if not policy.audit_all and category not in policy.audit_categories:
            return

        entry = AuditEntry(
            timestamp=request.timestamp,
            agent_id=request.agent_id,
            syscall=request.syscall,
            args_hash=hashlib.sha256(json.dumps(request.args, default=str).encode()).hexdigest()[:16],
            success=response.success,
            error_code=response.error_code,
            duration_ms=response.duration_ms,
            tx_id=request.tx_id,
        )

        self._audit_log.append(entry)

        # Trim if too large
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]

    def update_resource(self, agent_id: str, resource: str, delta: int):
        """Update resource usage"""
        if agent_id not in self._resources:
            self._resources[agent_id] = {}

        current = self._resources[agent_id].get(resource, 0)
        self._resources[agent_id][resource] = max(0, current + delta)

    def get_audit_log(
        self,
        agent_id: Optional[str] = None,
        syscall: Optional[SyscallNumber] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Get audit log entries"""
        entries = self._audit_log

        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        if syscall:
            entries = [e for e in entries if e.syscall == syscall]

        return entries[-limit:]


# =============================================================================
# Syscall Handlers
# =============================================================================


class SyscallHandler:
    """Base class for syscall handlers"""

    @property
    def syscall(self) -> SyscallNumber:
        raise NotImplementedError

    async def execute(self, request: SyscallRequest) -> Any:
        raise NotImplementedError


class ThinkHandler(SyscallHandler):
    """Handler for SYS_THINK - call LLM"""

    @property
    def syscall(self) -> SyscallNumber:
        return SyscallNumber.SYS_THINK

    async def execute(self, request: SyscallRequest) -> Any:
        prompt = request.args.get("prompt", "")
        model = request.args.get("model", "default")
        max_tokens = request.args.get("max_tokens", 4096)

        # This would call actual LLM
        # For now, return placeholder
        return {
            "response": f"[LLM response to: {prompt[:50]}...]",
            "model": model,
            "tokens_used": len(prompt.split()),
        }


class RememberHandler(SyscallHandler):
    """Handler for SYS_REMEMBER - store to memory"""

    @property
    def syscall(self) -> SyscallNumber:
        return SyscallNumber.SYS_REMEMBER

    async def execute(self, request: SyscallRequest) -> Any:
        content = request.args.get("content", "")
        metadata = request.args.get("metadata", {})

        # Store to semantic memory
        memory_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        return {
            "memory_id": memory_id,
            "stored": True,
        }


class RecallHandler(SyscallHandler):
    """Handler for SYS_RECALL - query memory"""

    @property
    def syscall(self) -> SyscallNumber:
        return SyscallNumber.SYS_RECALL

    async def execute(self, request: SyscallRequest) -> Any:
        query = request.args.get("query", "")
        top_k = request.args.get("top_k", 5)

        # Query semantic memory
        return {
            "results": [],
            "query": query,
        }


class SpawnHandler(SyscallHandler):
    """Handler for SYS_SPAWN - create new agent"""

    @property
    def syscall(self) -> SyscallNumber:
        return SyscallNumber.SYS_SPAWN

    async def execute(self, request: SyscallRequest) -> Any:
        agent_type = request.args.get("agent_type", "worker")
        config = request.args.get("config", {})

        new_pid = hash(f"{time.time()}{agent_type}") & 0xFFFFFFFF

        return {
            "pid": new_pid,
            "agent_type": agent_type,
        }


# =============================================================================
# Syscall Table
# =============================================================================


class SyscallTable:
    """
    The syscall table - central dispatch for all agent operations.

    Like Linux syscall table but for AI agents.
    Every operation goes through here for security and tracking.
    """

    def __init__(self):
        # Handler registry
        self._handlers: Dict[SyscallNumber, SyscallHandler] = {}

        # Security monitor
        self.security = SecurityMonitor()

        # Statistics
        self._stats: Dict[SyscallNumber, Dict[str, Any]] = {}

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register built-in handlers"""
        handlers = [
            ThinkHandler(),
            RememberHandler(),
            RecallHandler(),
            SpawnHandler(),
        ]

        for handler in handlers:
            self.register_handler(handler)

    def register_handler(self, handler: SyscallHandler):
        """Register a syscall handler"""
        self._handlers[handler.syscall] = handler
        self._stats[handler.syscall] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_time_ms": 0,
        }

    async def syscall(self, request: SyscallRequest) -> SyscallResponse:
        """
        Execute a syscall.

        This is the main entry point for all agent operations.
        """
        response = SyscallResponse(
            request_id=request.request_id,
            syscall=request.syscall,
            start_time=time.time(),
        )

        try:
            # 1. Security check
            error = await self.security.check(request)
            if error:
                response.success = False
                response.error = error
                response.error_code = ErrorCode.EPERM
                response.end_time = time.time()
                self.security.record_audit(request, response)
                return response

            # 2. Find handler
            handler = self._handlers.get(request.syscall)
            if not handler:
                response.success = False
                response.error = f"No handler for {request.syscall.name}"
                response.error_code = ErrorCode.ENOTIMPL
                response.end_time = time.time()
                return response

            # 3. Execute
            result = await handler.execute(request)
            response.result = result
            response.success = True

        except Exception as e:
            response.success = False
            response.error = str(e)
            response.error_code = ErrorCode.EFAULT

        response.end_time = time.time()

        # 4. Update stats
        stats = self._stats.get(request.syscall, {})
        stats["calls"] = stats.get("calls", 0) + 1
        if response.success:
            stats["successes"] = stats.get("successes", 0) + 1
        else:
            stats["failures"] = stats.get("failures", 0) + 1
        stats["total_time_ms"] = stats.get("total_time_ms", 0) + response.duration_ms
        self._stats[request.syscall] = stats

        # 5. Audit
        self.security.record_audit(request, response)

        return response

    # =========================================================================
    # Convenience methods (syntactic sugar over raw syscalls)
    # =========================================================================

    async def read(self, agent_id: str, path: str) -> SyscallResponse:
        """Read file"""
        return await self.syscall(SyscallRequest(
            syscall=SyscallNumber.SYS_READ,
            agent_id=agent_id,
            args={"path": path}
        ))

    async def write(self, agent_id: str, path: str, content: str) -> SyscallResponse:
        """Write file"""
        return await self.syscall(SyscallRequest(
            syscall=SyscallNumber.SYS_WRITE,
            agent_id=agent_id,
            args={"path": path, "content": content}
        ))

    async def think(
        self,
        agent_id: str,
        prompt: str,
        model: str = "default",
        max_tokens: int = 4096
    ) -> SyscallResponse:
        """Call LLM"""
        return await self.syscall(SyscallRequest(
            syscall=SyscallNumber.SYS_THINK,
            agent_id=agent_id,
            args={"prompt": prompt, "model": model, "max_tokens": max_tokens}
        ))

    async def remember(
        self,
        agent_id: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> SyscallResponse:
        """Store to memory"""
        return await self.syscall(SyscallRequest(
            syscall=SyscallNumber.SYS_REMEMBER,
            agent_id=agent_id,
            args={"content": content, "metadata": metadata or {}}
        ))

    async def recall(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> SyscallResponse:
        """Query memory"""
        return await self.syscall(SyscallRequest(
            syscall=SyscallNumber.SYS_RECALL,
            agent_id=agent_id,
            args={"query": query, "top_k": top_k}
        ))

    async def spawn(
        self,
        agent_id: str,
        agent_type: str,
        config: Optional[Dict] = None
    ) -> SyscallResponse:
        """Spawn new agent"""
        return await self.syscall(SyscallRequest(
            syscall=SyscallNumber.SYS_SPAWN,
            agent_id=agent_id,
            args={"agent_type": agent_type, "config": config or {}}
        ))

    async def ipc_send(
        self,
        agent_id: str,
        target: str,
        message: Any
    ) -> SyscallResponse:
        """Send IPC message"""
        return await self.syscall(SyscallRequest(
            syscall=SyscallNumber.SYS_IPC_SEND,
            agent_id=agent_id,
            args={"target": target, "message": message}
        ))

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get syscall statistics"""
        result = {}
        for syscall, stats in self._stats.items():
            result[syscall.name] = {
                **stats,
                "avg_time_ms": stats["total_time_ms"] / max(1, stats["calls"]),
                "success_rate": stats["successes"] / max(1, stats["calls"]) * 100,
            }
        return result


# =============================================================================
# Global Instance
# =============================================================================


_syscall_table: Optional[SyscallTable] = None


def get_syscall_table() -> SyscallTable:
    """Get global syscall table instance"""
    global _syscall_table
    if _syscall_table is None:
        _syscall_table = SyscallTable()
    return _syscall_table


# =============================================================================
# Syscall decorator (for easy handler definition)
# =============================================================================


def syscall_handler(syscall_num: SyscallNumber):
    """Decorator to register a syscall handler"""
    def decorator(func: Callable):
        class DecoratedHandler(SyscallHandler):
            @property
            def syscall(self) -> SyscallNumber:
                return syscall_num

            async def execute(self, request: SyscallRequest) -> Any:
                return await func(request)

        handler = DecoratedHandler()
        get_syscall_table().register_handler(handler)
        return func

    return decorator
