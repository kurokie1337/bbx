# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Agent Filter Stack - Windows Filter Drivers Inspired

Implements extensible message processing pipelines similar to
Windows I/O Filter Driver model (minifilter architecture).

Key concepts:
- FilterManager: Central manager for filter registration
- Filter: Base class for all filters (like minifilter drivers)
- FilterContext: Request context passing through the filter stack
- Altitude: Priority ordering for filters (higher = closer to user)
- PreOperation/PostOperation: Before/after processing callbacks

Filter Stack Order (by altitude):
  900+ : Audit filters (logging, tracing)
  800-899: Security filters (ACL, authentication)
  700-799: Policy filters (quota, rate limiting)
  600-699: Transform filters (encryption, compression)
  500-599: Cache filters
  400-499: Routing filters
  300-399: Protocol filters
  200-299: Validation filters
  100-199: Core processing
  0-99   : Device/Backend filters
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import asyncio
import hashlib
import logging
import time
import traceback
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class FilterResult(Enum):
    """Result of filter operation."""
    SUCCESS_WITH_CALLBACK = auto()  # Continue, call post-operation
    SUCCESS_NO_CALLBACK = auto()    # Continue, skip post-operation
    COMPLETE = auto()               # Complete request immediately
    DISALLOW = auto()               # Reject request
    PENDING = auto()                # Request is pending (async)
    REPARSE = auto()                # Reparse with modified context


class FilterStatus(Enum):
    """Filter registration status."""
    UNLOADED = auto()
    LOADING = auto()
    LOADED = auto()
    UNLOADING = auto()
    ERROR = auto()


class OperationClass(Enum):
    """Classes of operations for filtering."""
    # Agent operations
    AGENT_CREATE = auto()
    AGENT_INVOKE = auto()
    AGENT_TERMINATE = auto()
    AGENT_QUERY = auto()
    AGENT_MODIFY = auto()

    # Context operations
    CONTEXT_CREATE = auto()
    CONTEXT_READ = auto()
    CONTEXT_WRITE = auto()
    CONTEXT_DELETE = auto()

    # Tool operations
    TOOL_CALL = auto()
    TOOL_RESULT = auto()

    # Workflow operations
    WORKFLOW_START = auto()
    WORKFLOW_STEP = auto()
    WORKFLOW_COMPLETE = auto()
    WORKFLOW_ERROR = auto()

    # Message operations
    MESSAGE_SEND = auto()
    MESSAGE_RECEIVE = auto()

    # State operations
    STATE_READ = auto()
    STATE_WRITE = auto()
    STATE_SNAPSHOT = auto()

    # Network operations
    NETWORK_CONNECT = auto()
    NETWORK_SEND = auto()
    NETWORK_RECEIVE = auto()

    # Generic
    CUSTOM = auto()


class FilterAltitude:
    """Standard altitude ranges for filters."""
    AUDIT_HIGH = 950
    AUDIT_LOW = 900
    SECURITY_HIGH = 880
    SECURITY_LOW = 800
    POLICY_HIGH = 780
    POLICY_LOW = 700
    TRANSFORM_HIGH = 680
    TRANSFORM_LOW = 600
    CACHE_HIGH = 580
    CACHE_LOW = 500
    ROUTING_HIGH = 480
    ROUTING_LOW = 400
    PROTOCOL_HIGH = 380
    PROTOCOL_LOW = 300
    VALIDATION_HIGH = 280
    VALIDATION_LOW = 200
    CORE_HIGH = 180
    CORE_LOW = 100
    DEVICE_HIGH = 80
    DEVICE_LOW = 0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilterContext:
    """
    Context passed through the filter stack.
    Similar to FLT_CALLBACK_DATA in Windows.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: OperationClass = OperationClass.CUSTOM
    operation_name: str = ""

    # Request data
    caller_id: str = ""
    target_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout_ms: Optional[int] = None

    # Security context
    security_context: Dict[str, Any] = field(default_factory=dict)

    # Routing
    source_path: str = ""
    target_path: str = ""

    # Filter state
    filter_state: Dict[str, Any] = field(default_factory=dict)

    # Result
    result: Optional[Any] = None
    error: Optional[str] = None
    status: FilterResult = FilterResult.SUCCESS_WITH_CALLBACK

    # Flags
    flags: Set[str] = field(default_factory=set)

    # Reparse
    reparse_count: int = 0
    max_reparse: int = 10

    def set_result(self, result: Any) -> None:
        """Set successful result."""
        self.result = result
        self.status = FilterResult.SUCCESS_WITH_CALLBACK

    def set_error(self, error: str) -> None:
        """Set error result."""
        self.error = error
        self.status = FilterResult.DISALLOW

    def complete(self, result: Any = None) -> None:
        """Complete request immediately."""
        self.result = result
        self.status = FilterResult.COMPLETE

    def request_reparse(self) -> bool:
        """Request reparse with modified context."""
        if self.reparse_count >= self.max_reparse:
            return False
        self.reparse_count += 1
        self.status = FilterResult.REPARSE
        return True


@dataclass
class FilterInstance:
    """Instance of a filter attached to a volume/path."""
    id: str
    filter_id: str
    path_pattern: str
    altitude: float
    enabled: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FilterStats:
    """Statistics for a filter."""
    filter_id: str
    pre_operations: int = 0
    post_operations: int = 0
    operations_blocked: int = 0
    operations_completed: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    last_operation_at: Optional[datetime] = None

    @property
    def avg_latency_ms(self) -> float:
        total = self.pre_operations + self.post_operations
        if total == 0:
            return 0.0
        return self.total_latency_ms / total


@dataclass
class FilterRegistration:
    """Filter registration information."""
    id: str
    name: str
    altitude: float
    operations: Set[OperationClass]
    filter: "Filter"
    status: FilterStatus = FilterStatus.UNLOADED
    instances: List[FilterInstance] = field(default_factory=list)
    stats: FilterStats = field(default_factory=lambda: FilterStats(""))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.stats.filter_id == "":
            self.stats.filter_id = self.id


# =============================================================================
# Filter Base Class
# =============================================================================

class Filter(ABC):
    """
    Base class for all filters.

    Implements minifilter pattern with pre/post operation callbacks.
    """

    def __init__(self, name: str, altitude: float):
        self.name = name
        self.altitude = altitude
        self._operations: Set[OperationClass] = set()
        self._enabled = True

    @property
    def operations(self) -> Set[OperationClass]:
        """Operations this filter handles."""
        return self._operations

    @operations.setter
    def operations(self, ops: Set[OperationClass]) -> None:
        self._operations = ops

    def handles_operation(self, op: OperationClass) -> bool:
        """Check if filter handles this operation."""
        if not self._operations:
            return True  # Handle all if not specified
        return op in self._operations

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        """
        Called before operation is processed.

        Return values:
        - SUCCESS_WITH_CALLBACK: Continue, call post_operation after
        - SUCCESS_NO_CALLBACK: Continue, don't call post_operation
        - COMPLETE: Stop processing, use ctx.result
        - DISALLOW: Block the operation
        - REPARSE: Restart filter stack with modified ctx
        """
        return FilterResult.SUCCESS_WITH_CALLBACK

    async def post_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        """
        Called after operation is processed.

        Only called if pre_operation returned SUCCESS_WITH_CALLBACK.
        """
        return FilterResult.SUCCESS_WITH_CALLBACK

    async def instance_setup(self, instance: FilterInstance) -> bool:
        """Called when filter instance is attached."""
        return True

    async def instance_teardown(self, instance: FilterInstance) -> None:
        """Called when filter instance is detached."""
        pass

    def on_load(self) -> None:
        """Called when filter is loaded."""
        pass

    def on_unload(self) -> None:
        """Called when filter is unloaded."""
        pass


# =============================================================================
# Built-in Filters
# =============================================================================

class AuditFilter(Filter):
    """
    Audit filter - logs all operations.

    Highest altitude (900+) to see all traffic.
    """

    def __init__(
        self,
        name: str = "AuditFilter",
        altitude: float = FilterAltitude.AUDIT_HIGH,
        log_func: Optional[Callable[[str], None]] = None
    ):
        super().__init__(name, altitude)
        self._log_func = log_func or logger.info
        self._audit_log: List[Dict[str, Any]] = []
        self._max_log_size = 10000

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "pre",
            "operation": ctx.operation.name,
            "operation_name": ctx.operation_name,
            "caller": ctx.caller_id,
            "target": ctx.target_id,
            "source_path": ctx.source_path,
            "target_path": ctx.target_path,
            "context_id": ctx.id,
        }

        self._audit_log.append(entry)
        if len(self._audit_log) > self._max_log_size:
            self._audit_log = self._audit_log[-self._max_log_size // 2:]

        self._log_func(f"[AUDIT PRE] {ctx.operation.name}: {ctx.caller_id} -> {ctx.target_id}")

        # Store start time in filter state
        ctx.filter_state[f"{self.name}_start"] = time.time()

        return FilterResult.SUCCESS_WITH_CALLBACK

    async def post_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        start_time = ctx.filter_state.get(f"{self.name}_start", time.time())
        duration_ms = (time.time() - start_time) * 1000

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "post",
            "operation": ctx.operation.name,
            "context_id": ctx.id,
            "status": ctx.status.name,
            "duration_ms": duration_ms,
            "has_error": ctx.error is not None,
        }

        self._audit_log.append(entry)

        self._log_func(
            f"[AUDIT POST] {ctx.operation.name}: status={ctx.status.name}, "
            f"duration={duration_ms:.2f}ms, error={ctx.error}"
        )

        return FilterResult.SUCCESS_WITH_CALLBACK

    def get_audit_log(
        self,
        operation: Optional[OperationClass] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        entries = self._audit_log
        if operation:
            entries = [e for e in entries if e.get("operation") == operation.name]
        return entries[-limit:]


class SecurityFilter(Filter):
    """
    Security filter - checks ACLs and permissions.

    High altitude (800-880) for early security checks.
    """

    def __init__(
        self,
        name: str = "SecurityFilter",
        altitude: float = FilterAltitude.SECURITY_HIGH,
        acl_checker: Optional[Callable[[str, str, OperationClass], Awaitable[bool]]] = None
    ):
        super().__init__(name, altitude)
        self._acl_checker = acl_checker
        self._denied_operations: List[Dict[str, Any]] = []

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        # Skip if no ACL checker configured
        if not self._acl_checker:
            return FilterResult.SUCCESS_WITH_CALLBACK

        # Check access
        allowed = await self._acl_checker(
            ctx.caller_id,
            ctx.target_id,
            ctx.operation
        )

        if not allowed:
            self._denied_operations.append({
                "timestamp": datetime.utcnow().isoformat(),
                "caller": ctx.caller_id,
                "target": ctx.target_id,
                "operation": ctx.operation.name,
            })

            ctx.set_error(f"Access denied: {ctx.caller_id} cannot perform {ctx.operation.name} on {ctx.target_id}")
            logger.warning(f"[SECURITY] Access denied: {ctx.caller_id} -> {ctx.target_id} ({ctx.operation.name})")
            return FilterResult.DISALLOW

        return FilterResult.SUCCESS_WITH_CALLBACK


class QuotaFilter(Filter):
    """
    Quota filter - enforces resource limits.

    Medium-high altitude (700-780) after security.
    """

    def __init__(
        self,
        name: str = "QuotaFilter",
        altitude: float = FilterAltitude.POLICY_HIGH,
        quota_checker: Optional[Callable[[str, OperationClass], Awaitable[Tuple[bool, str]]]] = None
    ):
        super().__init__(name, altitude)
        self._quota_checker = quota_checker
        self._quota_exceeded: List[Dict[str, Any]] = []

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        if not self._quota_checker:
            return FilterResult.SUCCESS_WITH_CALLBACK

        allowed, reason = await self._quota_checker(ctx.caller_id, ctx.operation)

        if not allowed:
            self._quota_exceeded.append({
                "timestamp": datetime.utcnow().isoformat(),
                "caller": ctx.caller_id,
                "operation": ctx.operation.name,
                "reason": reason,
            })

            ctx.set_error(f"Quota exceeded: {reason}")
            logger.warning(f"[QUOTA] Exceeded: {ctx.caller_id} - {reason}")
            return FilterResult.DISALLOW

        return FilterResult.SUCCESS_WITH_CALLBACK


class RateLimitFilter(Filter):
    """
    Rate limit filter - throttles request frequency.

    Token bucket algorithm implementation.
    """

    def __init__(
        self,
        name: str = "RateLimitFilter",
        altitude: float = FilterAltitude.POLICY_LOW,
        rate_per_second: float = 10.0,
        burst_size: int = 20
    ):
        super().__init__(name, altitude)
        self._rate = rate_per_second
        self._burst = burst_size
        self._buckets: Dict[str, Tuple[float, float]] = {}  # caller -> (tokens, last_update)

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        caller = ctx.caller_id
        now = time.time()

        # Get or create bucket
        if caller in self._buckets:
            tokens, last_update = self._buckets[caller]
            # Add tokens based on elapsed time
            elapsed = now - last_update
            tokens = min(self._burst, tokens + elapsed * self._rate)
        else:
            tokens = float(self._burst)

        # Try to consume token
        if tokens >= 1.0:
            self._buckets[caller] = (tokens - 1.0, now)
            return FilterResult.SUCCESS_WITH_CALLBACK
        else:
            self._buckets[caller] = (tokens, now)
            retry_after = (1.0 - tokens) / self._rate
            ctx.set_error(f"Rate limit exceeded. Retry after {retry_after:.2f}s")
            logger.warning(f"[RATE LIMIT] {caller} exceeded rate limit")
            return FilterResult.DISALLOW


class TransformFilter(Filter):
    """
    Transform filter - modifies request/response data.

    Can encrypt, compress, or transform data.
    """

    def __init__(
        self,
        name: str = "TransformFilter",
        altitude: float = FilterAltitude.TRANSFORM_HIGH,
        pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        post_transform: Optional[Callable[[Any], Any]] = None
    ):
        super().__init__(name, altitude)
        self._pre_transform = pre_transform
        self._post_transform = post_transform

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        if self._pre_transform:
            try:
                ctx.data = self._pre_transform(ctx.data)
            except Exception as e:
                logger.error(f"[TRANSFORM] Pre-transform error: {e}")
        return FilterResult.SUCCESS_WITH_CALLBACK

    async def post_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        if self._post_transform and ctx.result is not None:
            try:
                ctx.result = self._post_transform(ctx.result)
            except Exception as e:
                logger.error(f"[TRANSFORM] Post-transform error: {e}")
        return FilterResult.SUCCESS_WITH_CALLBACK


class CacheFilter(Filter):
    """
    Cache filter - caches operation results.

    Simple LRU cache implementation.
    """

    def __init__(
        self,
        name: str = "CacheFilter",
        altitude: float = FilterAltitude.CACHE_HIGH,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
        cacheable_operations: Optional[Set[OperationClass]] = None
    ):
        super().__init__(name, altitude)
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cacheable = cacheable_operations or {
            OperationClass.CONTEXT_READ,
            OperationClass.STATE_READ,
            OperationClass.AGENT_QUERY,
        }
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.operations = self._cacheable

    def _cache_key(self, ctx: FilterContext) -> str:
        """Generate cache key from context."""
        key_data = f"{ctx.operation.name}:{ctx.caller_id}:{ctx.target_id}:{ctx.target_path}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        if ctx.operation not in self._cacheable:
            return FilterResult.SUCCESS_NO_CALLBACK

        key = self._cache_key(ctx)
        now = time.time()

        if key in self._cache:
            result, timestamp = self._cache[key]
            if now - timestamp < self._ttl:
                ctx.result = result
                ctx.filter_state["cache_hit"] = True
                logger.debug(f"[CACHE] Hit for {key[:8]}...")
                return FilterResult.COMPLETE
            else:
                del self._cache[key]

        ctx.filter_state["cache_key"] = key
        return FilterResult.SUCCESS_WITH_CALLBACK

    async def post_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        if ctx.status != FilterResult.SUCCESS_WITH_CALLBACK:
            return FilterResult.SUCCESS_WITH_CALLBACK

        key = ctx.filter_state.get("cache_key")
        if key and ctx.result is not None and ctx.error is None:
            # Evict oldest if full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (ctx.result, time.time())
            logger.debug(f"[CACHE] Stored {key[:8]}...")

        return FilterResult.SUCCESS_WITH_CALLBACK

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries."""
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        count = 0
        keys_to_delete = [k for k in self._cache if pattern in k]
        for key in keys_to_delete:
            del self._cache[key]
            count += 1
        return count


class ValidationFilter(Filter):
    """
    Validation filter - validates request data.

    Lower altitude (200-280) before core processing.
    """

    def __init__(
        self,
        name: str = "ValidationFilter",
        altitude: float = FilterAltitude.VALIDATION_HIGH,
        validators: Optional[Dict[OperationClass, Callable[[Dict[str, Any]], Tuple[bool, str]]]] = None
    ):
        super().__init__(name, altitude)
        self._validators = validators or {}

    def add_validator(
        self,
        operation: OperationClass,
        validator: Callable[[Dict[str, Any]], Tuple[bool, str]]
    ) -> None:
        """Add validator for operation."""
        self._validators[operation] = validator

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        validator = self._validators.get(ctx.operation)
        if not validator:
            return FilterResult.SUCCESS_WITH_CALLBACK

        try:
            valid, error = validator(ctx.data)
            if not valid:
                ctx.set_error(f"Validation failed: {error}")
                return FilterResult.DISALLOW
        except Exception as e:
            ctx.set_error(f"Validation error: {e}")
            return FilterResult.DISALLOW

        return FilterResult.SUCCESS_WITH_CALLBACK


class MetricsFilter(Filter):
    """
    Metrics filter - collects operation metrics.

    Similar to audit but focused on performance metrics.
    """

    def __init__(
        self,
        name: str = "MetricsFilter",
        altitude: float = FilterAltitude.AUDIT_LOW
    ):
        super().__init__(name, altitude)
        self._metrics: Dict[str, Dict[str, Any]] = {}

    async def pre_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        ctx.filter_state[f"{self.name}_start"] = time.time()
        return FilterResult.SUCCESS_WITH_CALLBACK

    async def post_operation(
        self,
        ctx: FilterContext,
        instance: FilterInstance
    ) -> FilterResult:
        start = ctx.filter_state.get(f"{self.name}_start", time.time())
        duration_ms = (time.time() - start) * 1000

        op_name = ctx.operation.name
        if op_name not in self._metrics:
            self._metrics[op_name] = {
                "count": 0,
                "success": 0,
                "errors": 0,
                "total_ms": 0.0,
                "min_ms": float("inf"),
                "max_ms": 0.0,
            }

        m = self._metrics[op_name]
        m["count"] += 1
        m["total_ms"] += duration_ms
        m["min_ms"] = min(m["min_ms"], duration_ms)
        m["max_ms"] = max(m["max_ms"], duration_ms)

        if ctx.error:
            m["errors"] += 1
        else:
            m["success"] += 1

        return FilterResult.SUCCESS_WITH_CALLBACK

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics."""
        result = {}
        for op, m in self._metrics.items():
            avg = m["total_ms"] / m["count"] if m["count"] > 0 else 0
            result[op] = {
                **m,
                "avg_ms": avg,
            }
        return result


# =============================================================================
# Filter Manager
# =============================================================================

class FilterManager:
    """
    Central manager for filter registration and invocation.

    Similar to Windows Filter Manager (fltmgr.sys).
    """

    def __init__(self):
        self._filters: Dict[str, FilterRegistration] = {}
        self._altitude_order: List[str] = []  # Filter IDs sorted by altitude (descending)
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        """Start the filter manager."""
        self._started = True
        logger.info("FilterManager started")

    async def stop(self) -> None:
        """Stop the filter manager."""
        for reg in self._filters.values():
            if reg.status == FilterStatus.LOADED:
                await self.unload_filter(reg.id)
        self._started = False
        logger.info("FilterManager stopped")

    async def register_filter(
        self,
        filter_obj: Filter,
        filter_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a filter.

        Returns filter ID.
        """
        async with self._lock:
            fid = filter_id or str(uuid.uuid4())

            reg = FilterRegistration(
                id=fid,
                name=filter_obj.name,
                altitude=filter_obj.altitude,
                operations=filter_obj.operations,
                filter=filter_obj,
                status=FilterStatus.UNLOADED,
                metadata=metadata or {},
            )

            self._filters[fid] = reg
            self._rebuild_altitude_order()

            logger.info(f"Filter registered: {filter_obj.name} (altitude={filter_obj.altitude})")
            return fid

    async def unregister_filter(self, filter_id: str) -> bool:
        """Unregister a filter."""
        async with self._lock:
            if filter_id not in self._filters:
                return False

            reg = self._filters[filter_id]
            if reg.status == FilterStatus.LOADED:
                await self.unload_filter(filter_id)

            del self._filters[filter_id]
            self._rebuild_altitude_order()

            logger.info(f"Filter unregistered: {reg.name}")
            return True

    async def load_filter(self, filter_id: str) -> bool:
        """Load a filter (make it active)."""
        if filter_id not in self._filters:
            return False

        reg = self._filters[filter_id]
        if reg.status == FilterStatus.LOADED:
            return True

        reg.status = FilterStatus.LOADING

        try:
            reg.filter.on_load()
            reg.status = FilterStatus.LOADED
            logger.info(f"Filter loaded: {reg.name}")
            return True
        except Exception as e:
            reg.status = FilterStatus.ERROR
            logger.error(f"Failed to load filter {reg.name}: {e}")
            return False

    async def unload_filter(self, filter_id: str) -> bool:
        """Unload a filter."""
        if filter_id not in self._filters:
            return False

        reg = self._filters[filter_id]
        if reg.status != FilterStatus.LOADED:
            return True

        reg.status = FilterStatus.UNLOADING

        try:
            # Teardown all instances
            for instance in reg.instances:
                await reg.filter.instance_teardown(instance)
            reg.instances.clear()

            reg.filter.on_unload()
            reg.status = FilterStatus.UNLOADED
            logger.info(f"Filter unloaded: {reg.name}")
            return True
        except Exception as e:
            reg.status = FilterStatus.ERROR
            logger.error(f"Failed to unload filter {reg.name}: {e}")
            return False

    async def attach_filter(
        self,
        filter_id: str,
        path_pattern: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Attach a filter instance to a path pattern.

        Returns instance ID.
        """
        if filter_id not in self._filters:
            return None

        reg = self._filters[filter_id]
        if reg.status != FilterStatus.LOADED:
            return None

        instance = FilterInstance(
            id=str(uuid.uuid4()),
            filter_id=filter_id,
            path_pattern=path_pattern,
            altitude=reg.altitude,
            context=context or {},
        )

        if not await reg.filter.instance_setup(instance):
            return None

        reg.instances.append(instance)
        logger.debug(f"Filter instance attached: {reg.name} -> {path_pattern}")
        return instance.id

    async def detach_filter(self, filter_id: str, instance_id: str) -> bool:
        """Detach a filter instance."""
        if filter_id not in self._filters:
            return False

        reg = self._filters[filter_id]
        for i, instance in enumerate(reg.instances):
            if instance.id == instance_id:
                await reg.filter.instance_teardown(instance)
                reg.instances.pop(i)
                return True

        return False

    def _rebuild_altitude_order(self) -> None:
        """Rebuild altitude-sorted filter order."""
        self._altitude_order = sorted(
            self._filters.keys(),
            key=lambda fid: self._filters[fid].altitude,
            reverse=True  # Higher altitude first
        )

    def _get_matching_filters(
        self,
        ctx: FilterContext
    ) -> List[Tuple[FilterRegistration, FilterInstance]]:
        """Get filters matching the context, ordered by altitude."""
        matches = []

        for fid in self._altitude_order:
            reg = self._filters[fid]

            # Skip if not loaded
            if reg.status != FilterStatus.LOADED:
                continue

            # Skip if filter doesn't handle this operation
            if not reg.filter.handles_operation(ctx.operation):
                continue

            # Find matching instances
            for instance in reg.instances:
                if not instance.enabled:
                    continue

                # Match path pattern
                if self._matches_pattern(ctx.target_path, instance.path_pattern):
                    matches.append((reg, instance))
                    break  # Only one instance per filter

        return matches

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches pattern."""
        if pattern == "*" or pattern == "/*":
            return True

        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])

        if pattern.startswith("*"):
            return path.endswith(pattern[1:])

        return path == pattern

    async def process(
        self,
        ctx: FilterContext,
        processor: Optional[Callable[[FilterContext], Awaitable[Any]]] = None
    ) -> FilterContext:
        """
        Process context through the filter stack.

        Args:
            ctx: Filter context
            processor: Optional core processor function

        Returns:
            Updated context with result or error
        """
        if not self._started:
            if processor:
                ctx.result = await processor(ctx)
            return ctx

        # Handle reparse loop
        while True:
            result = await self._process_once(ctx, processor)
            if result.status != FilterResult.REPARSE:
                break
            if result.reparse_count > result.max_reparse:
                result.set_error("Maximum reparse count exceeded")
                break

        return result

    async def _process_once(
        self,
        ctx: FilterContext,
        processor: Optional[Callable[[FilterContext], Awaitable[Any]]]
    ) -> FilterContext:
        """Single pass through filter stack."""
        filters = self._get_matching_filters(ctx)
        post_callbacks: List[Tuple[FilterRegistration, FilterInstance]] = []

        # Pre-operation phase (high altitude to low)
        for reg, instance in filters:
            start = time.time()
            try:
                result = await reg.filter.pre_operation(ctx, instance)

                # Update stats
                reg.stats.pre_operations += 1
                reg.stats.total_latency_ms += (time.time() - start) * 1000
                reg.stats.last_operation_at = datetime.utcnow()

                if result == FilterResult.SUCCESS_WITH_CALLBACK:
                    post_callbacks.append((reg, instance))
                elif result == FilterResult.SUCCESS_NO_CALLBACK:
                    pass
                elif result == FilterResult.COMPLETE:
                    reg.stats.operations_completed += 1
                    return ctx
                elif result == FilterResult.DISALLOW:
                    reg.stats.operations_blocked += 1
                    return ctx
                elif result == FilterResult.REPARSE:
                    return ctx

            except Exception as e:
                reg.stats.errors += 1
                logger.error(f"Filter {reg.name} pre_operation error: {e}\n{traceback.format_exc()}")
                ctx.set_error(f"Filter error: {e}")
                return ctx

        # Core processing
        if processor:
            try:
                ctx.result = await processor(ctx)
            except Exception as e:
                ctx.set_error(f"Processor error: {e}")
                logger.error(f"Core processor error: {e}\n{traceback.format_exc()}")

        # Post-operation phase (low altitude to high - reverse order)
        for reg, instance in reversed(post_callbacks):
            start = time.time()
            try:
                await reg.filter.post_operation(ctx, instance)

                reg.stats.post_operations += 1
                reg.stats.total_latency_ms += (time.time() - start) * 1000

            except Exception as e:
                reg.stats.errors += 1
                logger.error(f"Filter {reg.name} post_operation error: {e}")

        return ctx

    def get_filter_stats(self, filter_id: str) -> Optional[FilterStats]:
        """Get statistics for a filter."""
        if filter_id not in self._filters:
            return None
        return self._filters[filter_id].stats

    def get_all_stats(self) -> Dict[str, FilterStats]:
        """Get statistics for all filters."""
        return {fid: reg.stats for fid, reg in self._filters.items()}

    def list_filters(self) -> List[Dict[str, Any]]:
        """List all registered filters."""
        return [
            {
                "id": reg.id,
                "name": reg.name,
                "altitude": reg.altitude,
                "status": reg.status.name,
                "operations": [op.name for op in reg.operations],
                "instances": len(reg.instances),
            }
            for reg in self._filters.values()
        ]


# =============================================================================
# Filter Stack Builder
# =============================================================================

class FilterStackBuilder:
    """Builder for creating filter stacks with common patterns."""

    def __init__(self, manager: FilterManager):
        self._manager = manager
        self._filters: List[Tuple[Filter, Optional[str]]] = []

    def add_audit(
        self,
        log_func: Optional[Callable[[str], None]] = None
    ) -> "FilterStackBuilder":
        """Add audit filter."""
        self._filters.append((AuditFilter(log_func=log_func), None))
        return self

    def add_metrics(self) -> "FilterStackBuilder":
        """Add metrics filter."""
        self._filters.append((MetricsFilter(), None))
        return self

    def add_security(
        self,
        acl_checker: Callable[[str, str, OperationClass], Awaitable[bool]]
    ) -> "FilterStackBuilder":
        """Add security filter."""
        self._filters.append((SecurityFilter(acl_checker=acl_checker), None))
        return self

    def add_quota(
        self,
        quota_checker: Callable[[str, OperationClass], Awaitable[Tuple[bool, str]]]
    ) -> "FilterStackBuilder":
        """Add quota filter."""
        self._filters.append((QuotaFilter(quota_checker=quota_checker), None))
        return self

    def add_rate_limit(
        self,
        rate_per_second: float = 10.0,
        burst_size: int = 20
    ) -> "FilterStackBuilder":
        """Add rate limit filter."""
        self._filters.append((
            RateLimitFilter(rate_per_second=rate_per_second, burst_size=burst_size),
            None
        ))
        return self

    def add_cache(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0
    ) -> "FilterStackBuilder":
        """Add cache filter."""
        self._filters.append((
            CacheFilter(max_size=max_size, ttl_seconds=ttl_seconds),
            None
        ))
        return self

    def add_validation(
        self,
        validators: Dict[OperationClass, Callable[[Dict[str, Any]], Tuple[bool, str]]]
    ) -> "FilterStackBuilder":
        """Add validation filter."""
        f = ValidationFilter()
        for op, validator in validators.items():
            f.add_validator(op, validator)
        self._filters.append((f, None))
        return self

    def add_transform(
        self,
        pre_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        post_transform: Optional[Callable[[Any], Any]] = None
    ) -> "FilterStackBuilder":
        """Add transform filter."""
        self._filters.append((
            TransformFilter(pre_transform=pre_transform, post_transform=post_transform),
            None
        ))
        return self

    def add_custom(
        self,
        filter_obj: Filter,
        filter_id: Optional[str] = None
    ) -> "FilterStackBuilder":
        """Add custom filter."""
        self._filters.append((filter_obj, filter_id))
        return self

    async def build(self) -> List[str]:
        """Build and register all filters. Returns filter IDs."""
        ids = []
        for filter_obj, filter_id in self._filters:
            fid = await self._manager.register_filter(filter_obj, filter_id)
            await self._manager.load_filter(fid)
            await self._manager.attach_filter(fid, "*")
            ids.append(fid)
        return ids


# =============================================================================
# Singleton Access
# =============================================================================

_filter_manager: Optional[FilterManager] = None


def get_filter_manager() -> FilterManager:
    """Get or create the global filter manager."""
    global _filter_manager
    if _filter_manager is None:
        _filter_manager = FilterManager()
    return _filter_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "FilterResult",
    "FilterStatus",
    "OperationClass",
    "FilterAltitude",

    # Data classes
    "FilterContext",
    "FilterInstance",
    "FilterStats",
    "FilterRegistration",

    # Base class
    "Filter",

    # Built-in filters
    "AuditFilter",
    "SecurityFilter",
    "QuotaFilter",
    "RateLimitFilter",
    "TransformFilter",
    "CacheFilter",
    "ValidationFilter",
    "MetricsFilter",

    # Manager
    "FilterManager",
    "FilterStackBuilder",

    # Singleton
    "get_filter_manager",
]
