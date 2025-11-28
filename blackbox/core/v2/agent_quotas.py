# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX AgentQuotas - Cgroups v2-inspired Resource Limits for AI Agents

Provides hierarchical resource management for agents:
- CPU time limits per agent/group
- Memory/context budget allocation
- I/O operations quotas (via AgentRing)
- External API rate limits
- Hierarchical groups with inheritance

Inspired by Linux Cgroups v2:
- Unified hierarchy → Agent/Group/Org hierarchy
- Resource controllers → CPU, Memory, IO, API controllers
- Limits and accounting → Quotas and usage tracking

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Organization (root cgroup)                                 │
    │  └─ Project Group                                           │
    │     ├─ Agent Instance 1 (leaf)                              │
    │     ├─ Agent Instance 2 (leaf)                              │
    │     └─ Sub-Group                                            │
    │        └─ Agent Instance 3                                  │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
import threading

logger = logging.getLogger("bbx.agent_quotas")


# =============================================================================
# Enums
# =============================================================================

class ResourceType(Enum):
    """Types of resources that can be limited"""
    CPU_TIME = "cpu_time"           # CPU time in seconds
    MEMORY = "memory"               # Memory in bytes
    CONTEXT_SIZE = "context_size"   # Context tiering budget
    IO_OPS = "io_ops"               # I/O operations count
    IO_BANDWIDTH = "io_bandwidth"   # I/O bandwidth in bytes/sec
    API_CALLS = "api_calls"         # External API calls
    TOOL_CALLS = "tool_calls"       # Tool invocations
    TOKENS = "tokens"               # LLM tokens
    NETWORK = "network"             # Network bandwidth


class QuotaAction(Enum):
    """Action when quota is exceeded"""
    ALLOW = auto()      # Log and allow
    THROTTLE = auto()   # Slow down
    QUEUE = auto()      # Queue for later
    REJECT = auto()     # Reject immediately
    NOTIFY = auto()     # Notify and allow


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceLimit:
    """Defines a resource limit"""
    resource_type: ResourceType
    limit: float  # The limit value
    period_sec: float = 60.0  # Period for rate limits
    burst: float = 0.0  # Allowed burst above limit
    action: QuotaAction = QuotaAction.THROTTLE
    priority: int = 0  # Higher = more important (gets resources first)


@dataclass
class ResourceUsage:
    """Tracks resource usage"""
    resource_type: ResourceType
    current: float = 0.0
    peak: float = 0.0
    total: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    violations: int = 0

    def record(self, amount: float):
        self.current += amount
        self.total += amount
        if self.current > self.peak:
            self.peak = self.current

    def reset_period(self):
        self.current = 0.0
        self.last_reset = datetime.now()


@dataclass
class QuotaConfig:
    """Configuration for agent quotas"""
    # CPU limits
    cpu_time_sec: float = 300.0  # 5 minutes per period
    cpu_period_sec: float = 60.0

    # Memory limits
    memory_bytes: int = 512 * 1024 * 1024  # 512MB
    context_bytes: int = 100 * 1024 * 1024  # 100MB context

    # I/O limits
    io_ops_per_sec: int = 100
    io_bandwidth_bytes: int = 10 * 1024 * 1024  # 10MB/s

    # API limits
    api_calls_per_min: int = 60
    tool_calls_per_min: int = 100
    tokens_per_min: int = 100000

    # Network
    network_bytes_per_sec: int = 1024 * 1024  # 1MB/s

    # Behavior
    default_action: QuotaAction = QuotaAction.THROTTLE
    enable_burst: bool = True
    burst_multiplier: float = 1.5


# =============================================================================
# Quota Group (like cgroup)
# =============================================================================

class QuotaGroup:
    """
    A quota group - like a cgroup in Linux.

    Provides hierarchical resource management with inheritance.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[QuotaGroup] = None,
        config: Optional[QuotaConfig] = None,
    ):
        self.name = name
        self.parent = parent
        self.children: Dict[str, QuotaGroup] = {}
        self.agents: Set[str] = set()

        # Limits (can be inherited from parent)
        self._limits: Dict[ResourceType, ResourceLimit] = {}

        # Usage tracking
        self._usage: Dict[ResourceType, ResourceUsage] = {
            rt: ResourceUsage(resource_type=rt) for rt in ResourceType
        }

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Lock for thread safety
        self._lock = threading.RLock()

        # Apply config if provided
        if config:
            self._apply_config(config)

    def _apply_config(self, config: QuotaConfig):
        """Apply configuration as limits"""
        self.set_limit(ResourceLimit(
            ResourceType.CPU_TIME,
            limit=config.cpu_time_sec,
            period_sec=config.cpu_period_sec,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.MEMORY,
            limit=config.memory_bytes,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.CONTEXT_SIZE,
            limit=config.context_bytes,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.IO_OPS,
            limit=config.io_ops_per_sec,
            period_sec=1.0,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.IO_BANDWIDTH,
            limit=config.io_bandwidth_bytes,
            period_sec=1.0,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.API_CALLS,
            limit=config.api_calls_per_min,
            period_sec=60.0,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.TOOL_CALLS,
            limit=config.tool_calls_per_min,
            period_sec=60.0,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.TOKENS,
            limit=config.tokens_per_min,
            period_sec=60.0,
        ))
        self.set_limit(ResourceLimit(
            ResourceType.NETWORK,
            limit=config.network_bytes_per_sec,
            period_sec=1.0,
        ))

    def set_limit(self, limit: ResourceLimit):
        """Set a resource limit"""
        with self._lock:
            self._limits[limit.resource_type] = limit

    def get_limit(self, resource_type: ResourceType) -> Optional[ResourceLimit]:
        """Get limit, inheriting from parent if not set"""
        with self._lock:
            if resource_type in self._limits:
                return self._limits[resource_type]
            if self.parent:
                return self.parent.get_limit(resource_type)
            return None

    def get_effective_limit(self, resource_type: ResourceType) -> float:
        """Get effective limit value (considering hierarchy)"""
        limit = self.get_limit(resource_type)
        if limit:
            return limit.limit
        return float('inf')

    def add_child(self, name: str, config: Optional[QuotaConfig] = None) -> QuotaGroup:
        """Add a child group"""
        with self._lock:
            child = QuotaGroup(name, parent=self, config=config)
            self.children[name] = child
            return child

    def add_agent(self, agent_id: str):
        """Add an agent to this group"""
        with self._lock:
            self.agents.add(agent_id)

    def remove_agent(self, agent_id: str):
        """Remove an agent from this group"""
        with self._lock:
            self.agents.discard(agent_id)

    def get_usage(self, resource_type: ResourceType) -> ResourceUsage:
        """Get usage for a resource type"""
        return self._usage[resource_type]

    def get_total_usage(self, resource_type: ResourceType) -> float:
        """Get total usage including children"""
        with self._lock:
            total = self._usage[resource_type].current
            for child in self.children.values():
                total += child.get_total_usage(resource_type)
            return total

    def record_usage(self, resource_type: ResourceType, amount: float) -> bool:
        """
        Record resource usage.

        Returns True if within limits, False if quota exceeded.
        """
        with self._lock:
            limit = self.get_limit(resource_type)
            usage = self._usage[resource_type]

            # Check if period needs reset
            if limit and limit.period_sec > 0:
                elapsed = (datetime.now() - usage.last_reset).total_seconds()
                if elapsed >= limit.period_sec:
                    usage.reset_period()

            # Record usage
            usage.record(amount)

            # Check limit
            if limit:
                effective_limit = limit.limit
                if limit.burst > 0:
                    effective_limit += limit.burst

                if usage.current > effective_limit:
                    usage.violations += 1
                    return False

            # Propagate to parent
            if self.parent:
                return self.parent.record_usage(resource_type, amount)

            return True

    def check_quota(self, resource_type: ResourceType, amount: float) -> Tuple[bool, QuotaAction]:
        """
        Check if an operation would exceed quota.

        Returns (allowed, action) tuple.
        """
        with self._lock:
            limit = self.get_limit(resource_type)
            if not limit:
                return True, QuotaAction.ALLOW

            usage = self._usage[resource_type]
            projected = usage.current + amount

            effective_limit = limit.limit
            if limit.burst > 0:
                effective_limit += limit.burst

            if projected > effective_limit:
                return False, limit.action

            # Check parent
            if self.parent:
                return self.parent.check_quota(resource_type, amount)

            return True, QuotaAction.ALLOW

    def get_stats(self) -> Dict[str, Any]:
        """Get quota statistics"""
        with self._lock:
            return {
                "name": self.name,
                "agents": list(self.agents),
                "children": list(self.children.keys()),
                "usage": {
                    rt.value: {
                        "current": self._usage[rt].current,
                        "peak": self._usage[rt].peak,
                        "total": self._usage[rt].total,
                        "violations": self._usage[rt].violations,
                        "limit": self.get_effective_limit(rt),
                    }
                    for rt in ResourceType
                },
            }


# =============================================================================
# Quota Manager
# =============================================================================

class QuotaManager:
    """
    Central manager for agent quotas.

    Manages the hierarchy of quota groups and provides
    quota checking/enforcement APIs.
    """

    def __init__(self, root_config: Optional[QuotaConfig] = None):
        self.root = QuotaGroup("root", config=root_config or QuotaConfig())
        self._agent_groups: Dict[str, QuotaGroup] = {}
        self._throttle_state: Dict[str, Dict[ResourceType, float]] = defaultdict(dict)
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

    def create_group(
        self,
        path: str,
        config: Optional[QuotaConfig] = None,
    ) -> QuotaGroup:
        """
        Create a quota group at the specified path.

        Path format: "org/project/team"
        """
        parts = path.split("/")
        current = self.root

        for part in parts:
            if part not in current.children:
                current.add_child(part, config if part == parts[-1] else None)
            current = current.children[part]

        return current

    def get_group(self, path: str) -> Optional[QuotaGroup]:
        """Get a quota group by path"""
        parts = path.split("/")
        current = self.root

        for part in parts:
            if part not in current.children:
                return None
            current = current.children[part]

        return current

    def assign_agent(self, agent_id: str, group_path: str):
        """Assign an agent to a quota group"""
        group = self.get_group(group_path)
        if not group:
            group = self.create_group(group_path)

        # Remove from old group if any
        if agent_id in self._agent_groups:
            self._agent_groups[agent_id].remove_agent(agent_id)

        group.add_agent(agent_id)
        self._agent_groups[agent_id] = group

    def get_agent_group(self, agent_id: str) -> QuotaGroup:
        """Get the quota group for an agent"""
        return self._agent_groups.get(agent_id, self.root)

    async def check_and_record(
        self,
        agent_id: str,
        resource_type: ResourceType,
        amount: float,
    ) -> Tuple[bool, Optional[float]]:
        """
        Check quota and record usage.

        Returns (allowed, throttle_delay) tuple.
        throttle_delay is None if no throttling needed.
        """
        group = self.get_agent_group(agent_id)

        # Check quota
        allowed, action = group.check_quota(resource_type, amount)

        if allowed:
            group.record_usage(resource_type, amount)
            return True, None

        # Handle action
        if action == QuotaAction.ALLOW:
            group.record_usage(resource_type, amount)
            return True, None

        if action == QuotaAction.THROTTLE:
            # Calculate throttle delay
            delay = self._calculate_throttle_delay(agent_id, resource_type, group)
            return True, delay

        if action == QuotaAction.QUEUE:
            # For now, treat as throttle with longer delay
            delay = self._calculate_throttle_delay(agent_id, resource_type, group) * 2
            return True, delay

        if action == QuotaAction.NOTIFY:
            await self._emit_event("quota_exceeded", {
                "agent_id": agent_id,
                "resource_type": resource_type,
                "amount": amount,
            })
            group.record_usage(resource_type, amount)
            return True, None

        # REJECT
        return False, None

    def _calculate_throttle_delay(
        self,
        agent_id: str,
        resource_type: ResourceType,
        group: QuotaGroup,
    ) -> float:
        """Calculate throttle delay based on usage"""
        limit = group.get_limit(resource_type)
        if not limit:
            return 0.0

        usage = group.get_usage(resource_type)
        overage = usage.current - limit.limit

        if overage <= 0:
            return 0.0

        # Calculate delay to wait until quota resets
        if limit.period_sec > 0:
            elapsed = (datetime.now() - usage.last_reset).total_seconds()
            remaining = limit.period_sec - elapsed
            if remaining > 0:
                return min(remaining, 10.0)  # Cap at 10 seconds

        # Default delay
        return 1.0

    async def _emit_event(self, event: str, data: Any):
        """Emit an event"""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Quota callback error: {e}")

    def register_callback(self, event: str, callback: Callable):
        """Register an event callback"""
        self._callbacks[event].append(callback)

    def get_agent_usage(self, agent_id: str) -> Dict[str, Any]:
        """Get usage statistics for an agent"""
        group = self.get_agent_group(agent_id)
        return group.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get overall quota statistics"""
        def collect_stats(group: QuotaGroup, depth: int = 0) -> Dict:
            result = {
                "name": group.name,
                "agents_count": len(group.agents),
                "children_count": len(group.children),
                "total_violations": sum(
                    group._usage[rt].violations for rt in ResourceType
                ),
            }
            if depth < 3:  # Limit depth for performance
                result["children"] = {
                    name: collect_stats(child, depth + 1)
                    for name, child in group.children.items()
                }
            return result

        return {
            "root": collect_stats(self.root),
            "total_agents": len(self._agent_groups),
        }

    def reset_usage(self, group_path: Optional[str] = None):
        """Reset usage counters"""
        group = self.get_group(group_path) if group_path else self.root

        def reset_recursive(g: QuotaGroup):
            for rt in ResourceType:
                g._usage[rt].reset_period()
            for child in g.children.values():
                reset_recursive(child)

        reset_recursive(group)


# =============================================================================
# Quota Decorator
# =============================================================================

def quota_limited(
    resource_type: ResourceType,
    amount_param: str = "amount",
    default_amount: float = 1.0,
):
    """
    Decorator to apply quota limits to a function.

    Usage:
        @quota_limited(ResourceType.API_CALLS)
        async def call_api(self, ...):
            ...
    """
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Get agent ID from self or kwargs
            agent_id = getattr(self, 'agent_id', None) or kwargs.get('agent_id', 'default')

            # Get amount from kwargs or use default
            amount = kwargs.get(amount_param, default_amount)

            # Get quota manager
            manager = get_quota_manager()

            # Check and record
            allowed, delay = await manager.check_and_record(
                agent_id, resource_type, amount
            )

            if not allowed:
                raise QuotaExceededError(
                    f"Quota exceeded for {resource_type.value}"
                )

            if delay:
                logger.debug(f"Throttling {agent_id} for {delay:.2f}s")
                await asyncio.sleep(delay)

            return await func(self, *args, **kwargs)

        return wrapper
    return decorator


class QuotaExceededError(Exception):
    """Raised when quota is exceeded"""
    pass


# =============================================================================
# Global Instance
# =============================================================================

_quota_manager: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    """Get global QuotaManager instance"""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager


# Missing import
from typing import Tuple
