# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 AgentRing - io_uring-inspired batch operation system for AI agents.

This module provides efficient batch submission and completion handling,
minimizing overhead for high-throughput agent operations.

Key concepts from Linux io_uring:
- Submission Queue (SQ): Agent submits operations here
- Completion Queue (CQ): Results appear here when operations complete
- Zero-copy where possible
- Completion-based (not polling)

Example usage:
    ring = AgentRing()
    await ring.start(adapters)

    # Submit batch
    ops = [Operation(adapter="http", method="get", args={"url": url}) for url in urls]
    op_ids = await ring.submit_batch(ops)

    # Wait for all
    completions = await ring.wait_batch(op_ids)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("bbx.ring")


# =============================================================================
# Enums
# =============================================================================


class OperationType(Enum):
    """Types of operations that can be submitted to AgentRing"""
    ADAPTER_CALL = auto()
    WORKFLOW_EXEC = auto()
    STATE_OP = auto()
    CONTEXT_OP = auto()
    HOOK_TRIGGER = auto()


class OperationPriority(Enum):
    """Operation priority for scheduling"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    REALTIME = 3


class OperationStatus(Enum):
    """Status of an operation in the ring"""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Operation:
    """Single operation to submit to AgentRing"""
    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_num: int = 0

    # Operation details
    op_type: OperationType = OperationType.ADAPTER_CALL
    adapter: str = ""
    method: str = ""
    args: Dict[str, Any] = field(default_factory=dict)

    # Execution control
    priority: OperationPriority = OperationPriority.NORMAL
    timeout_ms: int = 30000
    retry_count: int = 0
    retry_delay_ms: int = 1000

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Metadata
    submitted_at: Optional[datetime] = None
    user_data: Any = None

    def __lt__(self, other: "Operation") -> bool:
        """Comparison for PriorityQueue - higher priority and lower sequence_num first"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.sequence_num < other.sequence_num


@dataclass
class Completion:
    """Completion result from AgentRing"""
    operation_id: str
    sequence_num: int = 0

    status: OperationStatus = OperationStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    duration_ms: float = 0
    retry_attempts: int = 0

    @property
    def latency_ms(self) -> float:
        """Total latency from submission to completion"""
        if self.completed_at and self.submitted_at:
            return (self.completed_at - self.submitted_at).total_seconds() * 1000
        return 0


@dataclass
class RingConfig:
    """Configuration for AgentRing"""
    submission_queue_size: int = 4096
    completion_queue_size: int = 4096
    min_workers: int = 4
    max_workers: int = 32
    worker_idle_timeout: float = 60.0
    max_batch_size: int = 256
    batch_timeout_ms: float = 10.0
    enable_prioritization: bool = True
    enable_dependency_tracking: bool = True
    default_timeout_ms: int = 30000


@dataclass
class RingStats:
    """Statistics for AgentRing"""
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0
    total_timeout: int = 0
    pending_count: int = 0
    processing_count: int = 0
    active_workers: int = 0


# =============================================================================
# AgentRing Implementation
# =============================================================================


class AgentRing:
    """
    io_uring-inspired batch operation system for AI agents.

    Features:
    - Batch submission: Submit multiple operations in one call
    - Batch completion: Retrieve multiple results in one call
    - Priority scheduling
    - Dependency tracking between operations
    """

    def __init__(self, config: Optional[RingConfig] = None):
        self.config = config or RingConfig()

        # Queues
        self._submission_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.submission_queue_size
        )
        self._completion_queue: asyncio.Queue[Completion] = asyncio.Queue(
            maxsize=self.config.completion_queue_size
        )

        # Tracking
        self._pending: Dict[str, Operation] = {}
        self._processing: Set[str] = set()
        self._completed: Dict[str, Completion] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}

        # Workers
        self._workers: List[asyncio.Task] = []
        self._shutdown = False

        # Stats
        self.stats = RingStats()

        # Adapters
        self._adapters: Dict[str, Any] = {}

        # Waiters
        self._completion_waiters: Dict[str, asyncio.Event] = {}

    async def start(self, adapters: Dict[str, Any]):
        """Start the AgentRing with given adapters"""
        self._adapters = adapters
        self._shutdown = False

        for _ in range(self.config.min_workers):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)

        self.stats.active_workers = len(self._workers)
        logger.info(f"AgentRing started with {len(self._workers)} workers")

    async def stop(self):
        """Stop the AgentRing gracefully"""
        self._shutdown = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        self.stats.active_workers = 0
        logger.info("AgentRing stopped")

    # =========================================================================
    # Submission API
    # =========================================================================

    async def submit(self, operation: Operation) -> str:
        """Submit a single operation"""
        operation.submitted_at = datetime.now()
        self._pending[operation.id] = operation

        if operation.depends_on:
            self._dependency_graph[operation.id] = set(operation.depends_on)
        else:
            priority = -operation.priority.value
            await self._submission_queue.put((priority, operation.sequence_num, operation))

        self.stats.total_submitted += 1
        self.stats.pending_count = len(self._pending)
        return operation.id

    async def submit_batch(
        self,
        operations: List[Operation],
        ordered: bool = False
    ) -> List[str]:
        """Submit multiple operations in one call"""
        op_ids = []
        for i, op in enumerate(operations):
            op.sequence_num = i
            if ordered and i > 0:
                op.depends_on.append(operations[i - 1].id)
            op_id = await self.submit(op)
            op_ids.append(op_id)
        return op_ids

    # =========================================================================
    # Completion API
    # =========================================================================

    async def wait_completion(
        self,
        operation_id: str,
        timeout: Optional[float] = None
    ) -> Completion:
        """Wait for a specific operation to complete"""
        if operation_id in self._completed:
            return self._completed[operation_id]

        event = asyncio.Event()
        self._completion_waiters[operation_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._completed[operation_id]
        except asyncio.TimeoutError:
            return Completion(
                operation_id=operation_id,
                status=OperationStatus.TIMEOUT,
                error="Operation timed out"
            )
        finally:
            self._completion_waiters.pop(operation_id, None)

    async def wait_batch(
        self,
        operation_ids: List[str],
        timeout: Optional[float] = None
    ) -> List[Completion]:
        """Wait for multiple operations to complete"""
        tasks = [self.wait_completion(op_id, timeout) for op_id in operation_ids]
        completions = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for op_id, comp in zip(operation_ids, completions):
            if isinstance(comp, Exception):
                results.append(Completion(
                    operation_id=op_id,
                    status=OperationStatus.FAILED,
                    error=str(comp)
                ))
            else:
                results.append(comp)
        return results

    async def drain_completions(
        self,
        max_count: int = 100,
        timeout_ms: float = 0
    ) -> List[Completion]:
        """Non-blocking drain of completion queue"""
        results = []
        deadline = datetime.now().timestamp() + (timeout_ms / 1000) if timeout_ms > 0 else 0

        while len(results) < max_count:
            try:
                if timeout_ms > 0:
                    remaining = deadline - datetime.now().timestamp()
                    if remaining <= 0:
                        break
                    completion = await asyncio.wait_for(
                        self._completion_queue.get(),
                        timeout=remaining
                    )
                else:
                    completion = self._completion_queue.get_nowait()
                results.append(completion)
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                break
        return results

    # =========================================================================
    # Cancellation
    # =========================================================================

    async def cancel(self, operation_id: str) -> bool:
        """Cancel a pending operation"""
        if operation_id in self._pending and operation_id not in self._processing:
            op = self._pending.pop(operation_id)
            completion = Completion(
                operation_id=operation_id,
                sequence_num=op.sequence_num,
                status=OperationStatus.CANCELLED,
                error="Cancelled by agent",
                completed_at=datetime.now()
            )
            await self._complete_operation(operation_id, completion)
            self.stats.total_cancelled += 1
            return True
        return False

    # =========================================================================
    # Internal Workers
    # =========================================================================

    async def _worker_loop(self):
        """Worker coroutine that processes operations"""
        while not self._shutdown:
            try:
                _, _, operation = await asyncio.wait_for(
                    self._submission_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._processing.add(operation.id)
            self.stats.processing_count = len(self._processing)

            completion = await self._execute_operation(operation)
            await self._complete_operation(operation.id, completion)

    async def _execute_operation(self, operation: Operation) -> Completion:
        """Execute a single operation"""
        started_at = datetime.now()
        completion = Completion(
            operation_id=operation.id,
            sequence_num=operation.sequence_num,
            submitted_at=operation.submitted_at or started_at,
            started_at=started_at,
        )

        try:
            adapter = self._adapters.get(operation.adapter)
            if not adapter:
                raise ValueError(f"Unknown adapter: {operation.adapter}")

            timeout_sec = operation.timeout_ms / 1000
            result = await asyncio.wait_for(
                adapter.execute(operation.method, operation.args),
                timeout=timeout_sec
            )

            completion.status = OperationStatus.COMPLETED
            completion.result = result

        except asyncio.TimeoutError:
            completion.status = OperationStatus.TIMEOUT
            completion.error = f"Operation timed out after {operation.timeout_ms}ms"
            self.stats.total_timeout += 1

        except Exception as e:
            completion.status = OperationStatus.FAILED
            completion.error = str(e)
            self.stats.total_failed += 1

        completion.completed_at = datetime.now()
        completion.duration_ms = (completion.completed_at - started_at).total_seconds() * 1000
        return completion

    async def _complete_operation(self, operation_id: str, completion: Completion):
        """Mark operation as complete"""
        self._completed[operation_id] = completion
        self._pending.pop(operation_id, None)
        self._processing.discard(operation_id)

        await self._completion_queue.put(completion)

        if operation_id in self._completion_waiters:
            self._completion_waiters[operation_id].set()

        await self._check_dependents(operation_id)

        self.stats.total_completed += 1
        self.stats.pending_count = len(self._pending)
        self.stats.processing_count = len(self._processing)

    async def _check_dependents(self, completed_op_id: str):
        """Check if any operations were waiting on this completion"""
        for op_id, waiting_for in list(self._dependency_graph.items()):
            if completed_op_id in waiting_for:
                waiting_for.discard(completed_op_id)

                if not waiting_for:
                    del self._dependency_graph[op_id]
                    op = self._pending.get(op_id)
                    if op:
                        priority = -op.priority.value
                        await self._submission_queue.put((priority, op.sequence_num, op))

    def get_stats(self) -> "ExtendedRingStats":
        """Get extended ring statistics"""
        return ExtendedRingStats(
            operations_submitted=self.stats.total_submitted,
            operations_completed=self.stats.total_completed,
            operations_failed=self.stats.total_failed,
            operations_cancelled=self.stats.total_cancelled,
            operations_timeout=self.stats.total_timeout,
            pending_count=self.stats.pending_count,
            processing_count=self.stats.processing_count,
            active_workers=self.stats.active_workers,
            worker_pool_size=self.config.max_workers,
            submission_queue_size=self._submission_queue.qsize(),
            completion_queue_size=self._completion_queue.qsize(),
            # Default latency values (would need actual tracking)
            throughput_ops_sec=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            worker_utilization=self.stats.active_workers / max(self.config.max_workers, 1) * 100,
        )


@dataclass
class ExtendedRingStats:
    """Extended statistics for AgentRing (for CLI display)"""
    operations_submitted: int = 0
    operations_completed: int = 0
    operations_failed: int = 0
    operations_cancelled: int = 0
    operations_timeout: int = 0
    pending_count: int = 0
    processing_count: int = 0
    active_workers: int = 0
    worker_pool_size: int = 0
    submission_queue_size: int = 0
    completion_queue_size: int = 0
    throughput_ops_sec: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    worker_utilization: float = 0.0


# =============================================================================
# Convenience Functions
# =============================================================================


_global_ring: Optional[AgentRing] = None


def get_ring() -> AgentRing:
    """Get the global AgentRing instance"""
    global _global_ring
    if _global_ring is None:
        _global_ring = AgentRing()
    return _global_ring


async def submit_batch(operations: List[Dict[str, Any]]) -> List[str]:
    """Convenience function to submit a batch of operations"""
    ring = get_ring()
    ops = [
        Operation(
            adapter=op.get("adapter", ""),
            method=op.get("method", ""),
            args=op.get("args", {}),
            priority=OperationPriority[op.get("priority", "NORMAL").upper()],
            timeout_ms=op.get("timeout_ms", 30000),
        )
        for op in operations
    ]
    return await ring.submit_batch(ops)
