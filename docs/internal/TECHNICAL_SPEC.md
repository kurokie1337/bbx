# BBX 2.0 Technical Specification

> **Detailed implementation specifications for BBX 2.0 core systems**

```
Document: BBX 2.0 Technical Specification
Version: 1.0
Status: DRAFT
Date: November 2025
```

---

# 1. AgentRing System

## 1.1 Architectural Overview

AgentRing is inspired by Linux's io_uring, providing batch operation submission and completion handling for AI agents.

### Problem Statement

Current BBX 1.0 operation model:
```
Agent → Operation → JSON Parse → Adapter Lookup → Execute → Result → Agent
       ↑_____________________________↓ (repeat for each operation)
```

Each operation incurs:
- JSON serialization/deserialization overhead
- Adapter registry lookup
- Context switch costs
- Individual error handling

### Solution: Ring Buffer Model

```
Agent → [Op1, Op2, Op3, ... OpN] → BATCH SUBMIT → AgentRing
                                                      ↓
                                              [Worker Pool]
                                                      ↓
Agent ← [R1, R2, R3, ... RN] ←── BATCH COMPLETE ←────┘
```

## 1.2 Data Structures

### Operation Types

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import uuid
from datetime import datetime


class OperationType(Enum):
    """Types of operations that can be submitted to AgentRing"""
    ADAPTER_CALL = auto()      # Call an adapter method
    WORKFLOW_EXEC = auto()     # Execute a sub-workflow
    STATE_OP = auto()          # State operation (get/set/etc)
    CONTEXT_OP = auto()        # Context memory operation
    HOOK_TRIGGER = auto()      # Trigger a hook manually


class OperationPriority(Enum):
    """Operation priority for scheduling"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    REALTIME = 3  # For time-critical operations


class OperationStatus(Enum):
    """Status of an operation in the ring"""
    PENDING = auto()           # In submission queue
    PROCESSING = auto()        # Being processed by worker
    COMPLETED = auto()         # Successfully completed
    FAILED = auto()            # Failed with error
    CANCELLED = auto()         # Cancelled by agent
    TIMEOUT = auto()           # Timed out


@dataclass
class Operation:
    """Single operation to submit to AgentRing"""
    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_num: int = 0  # Order in submission batch

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

    # Dependencies (for ordered execution within batch)
    depends_on: List[str] = field(default_factory=list)

    # Metadata
    submitted_at: Optional[datetime] = None
    user_data: Any = None  # Arbitrary user data passed through


@dataclass
class Completion:
    """Completion result from AgentRing"""
    # Link to original operation
    operation_id: str
    sequence_num: int

    # Result
    status: OperationStatus = OperationStatus.PENDING
    result: Any = None
    error: Optional[str] = None

    # Timing
    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics
    duration_ms: float = 0
    retry_attempts: int = 0

    @property
    def latency_ms(self) -> float:
        """Total latency from submission to completion"""
        if self.completed_at and self.submitted_at:
            return (self.completed_at - self.submitted_at).total_seconds() * 1000
        return 0

    @property
    def processing_ms(self) -> float:
        """Actual processing time"""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0
```

### Ring Buffers

```python
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set


@dataclass
class RingConfig:
    """Configuration for AgentRing"""
    # Ring sizes
    submission_queue_size: int = 4096
    completion_queue_size: int = 4096

    # Worker pool
    min_workers: int = 4
    max_workers: int = 32
    worker_idle_timeout: float = 60.0

    # Batching
    max_batch_size: int = 256
    batch_timeout_ms: float = 10.0  # Max wait for batch to fill

    # Features
    enable_prioritization: bool = True
    enable_dependency_tracking: bool = True

    # Timeouts
    default_timeout_ms: int = 30000
    max_timeout_ms: int = 600000


@dataclass
class RingStats:
    """Statistics for AgentRing performance monitoring"""
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0
    total_timeout: int = 0

    # Latency tracking (in ms)
    avg_latency_ms: float = 0
    p50_latency_ms: float = 0
    p95_latency_ms: float = 0
    p99_latency_ms: float = 0

    # Throughput
    ops_per_second: float = 0

    # Current state
    pending_count: int = 0
    processing_count: int = 0
    active_workers: int = 0
```

## 1.3 Core Implementation

```python
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime
import heapq

logger = logging.getLogger("bbx.ring")


class AgentRing:
    """
    io_uring-inspired batch operation system for AI agents.

    Key features:
    - Batch submission: Submit multiple operations in one call
    - Batch completion: Retrieve multiple results in one call
    - Priority scheduling: Operations can have different priorities
    - Dependency tracking: Operations can depend on other operations
    - Zero-copy where possible: Minimize data copying between queues
    """

    def __init__(self, config: Optional[RingConfig] = None):
        self.config = config or RingConfig()

        # Submission queue (priority queue for scheduling)
        self._submission_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.submission_queue_size
        )

        # Completion queue
        self._completion_queue: asyncio.Queue[Completion] = asyncio.Queue(
            maxsize=self.config.completion_queue_size
        )

        # Tracking
        self._pending: Dict[str, Operation] = {}
        self._processing: Set[str] = set()
        self._completed: Dict[str, Completion] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}  # op_id -> waiting_for

        # Workers
        self._workers: List[asyncio.Task] = []
        self._shutdown = False

        # Statistics
        self.stats = RingStats()

        # Adapters registry reference
        self._adapters: Dict[str, Any] = {}

        # Completion waiters (for wait_completion)
        self._completion_waiters: Dict[str, asyncio.Event] = {}

    async def start(self, adapters: Dict[str, Any]):
        """Start the AgentRing with given adapters"""
        self._adapters = adapters
        self._shutdown = False

        # Start initial worker pool
        for _ in range(self.config.min_workers):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)

        logger.info(f"AgentRing started with {len(self._workers)} workers")

    async def stop(self):
        """Stop the AgentRing gracefully"""
        self._shutdown = True

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("AgentRing stopped")

    # =========================================================================
    # SUBMISSION API
    # =========================================================================

    async def submit(self, operation: Operation) -> str:
        """
        Submit a single operation.

        Returns:
            Operation ID
        """
        operation.submitted_at = datetime.now()
        self._pending[operation.id] = operation

        # Track dependencies
        if operation.depends_on:
            self._dependency_graph[operation.id] = set(operation.depends_on)
        else:
            # No dependencies, queue immediately
            priority = -operation.priority.value  # Negative for max-heap behavior
            await self._submission_queue.put((priority, operation.sequence_num, operation))

        self.stats.total_submitted += 1
        self.stats.pending_count = len(self._pending)

        return operation.id

    async def submit_batch(
        self,
        operations: List[Operation],
        ordered: bool = False
    ) -> List[str]:
        """
        Submit multiple operations in one call.

        Args:
            operations: List of operations to submit
            ordered: If True, operations depend on previous operation

        Returns:
            List of operation IDs
        """
        op_ids = []

        for i, op in enumerate(operations):
            op.sequence_num = i

            # If ordered, each op depends on previous
            if ordered and i > 0:
                op.depends_on.append(operations[i - 1].id)

            op_id = await self.submit(op)
            op_ids.append(op_id)

        return op_ids

    # =========================================================================
    # COMPLETION API
    # =========================================================================

    async def wait_completion(
        self,
        operation_id: str,
        timeout: Optional[float] = None
    ) -> Completion:
        """
        Wait for a specific operation to complete.

        Args:
            operation_id: ID of operation to wait for
            timeout: Timeout in seconds

        Returns:
            Completion result
        """
        # Check if already completed
        if operation_id in self._completed:
            return self._completed[operation_id]

        # Create waiter event
        event = asyncio.Event()
        self._completion_waiters[operation_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._completed[operation_id]
        except asyncio.TimeoutError:
            # Create timeout completion
            completion = Completion(
                operation_id=operation_id,
                sequence_num=self._pending.get(operation_id, Operation()).sequence_num,
                status=OperationStatus.TIMEOUT,
                error="Operation timed out waiting for completion",
            )
            self._completed[operation_id] = completion
            return completion
        finally:
            self._completion_waiters.pop(operation_id, None)

    async def wait_batch(
        self,
        operation_ids: List[str],
        timeout: Optional[float] = None
    ) -> List[Completion]:
        """
        Wait for multiple operations to complete.

        Args:
            operation_ids: List of operation IDs
            timeout: Timeout in seconds (for entire batch)

        Returns:
            List of completions (in same order as input IDs)
        """
        async def wait_single(op_id: str) -> Completion:
            return await self.wait_completion(op_id, timeout=timeout)

        completions = await asyncio.gather(
            *[wait_single(op_id) for op_id in operation_ids],
            return_exceptions=True
        )

        results = []
        for op_id, completion in zip(operation_ids, completions):
            if isinstance(completion, Exception):
                results.append(Completion(
                    operation_id=op_id,
                    sequence_num=0,
                    status=OperationStatus.FAILED,
                    error=str(completion)
                ))
            else:
                results.append(completion)

        return results

    async def drain_completions(
        self,
        max_count: int = 100,
        timeout_ms: float = 0
    ) -> List[Completion]:
        """
        Non-blocking drain of completion queue.

        Args:
            max_count: Maximum completions to retrieve
            timeout_ms: How long to wait for completions (0 = no wait)

        Returns:
            List of available completions
        """
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
    # CANCELLATION
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

    async def cancel_batch(self, operation_ids: List[str]) -> int:
        """Cancel multiple operations, return count of successfully cancelled"""
        cancelled = 0
        for op_id in operation_ids:
            if await self.cancel(op_id):
                cancelled += 1
        return cancelled

    # =========================================================================
    # INTERNAL: Worker Loop
    # =========================================================================

    async def _worker_loop(self):
        """Worker coroutine that processes operations from submission queue"""
        while not self._shutdown:
            try:
                # Get next operation (with timeout to check shutdown)
                try:
                    _, _, operation = await asyncio.wait_for(
                        self._submission_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Mark as processing
                self._processing.add(operation.id)
                self.stats.processing_count = len(self._processing)

                # Execute operation
                completion = await self._execute_operation(operation)

                # Complete
                await self._complete_operation(operation.id, completion)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")

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
            # Get adapter
            adapter = self._adapters.get(operation.adapter)
            if not adapter:
                raise ValueError(f"Unknown adapter: {operation.adapter}")

            # Execute with timeout
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
        """Mark operation as complete and handle dependents"""
        # Store completion
        self._completed[operation_id] = completion
        self._pending.pop(operation_id, None)
        self._processing.discard(operation_id)

        # Add to completion queue
        await self._completion_queue.put(completion)

        # Notify waiters
        if operation_id in self._completion_waiters:
            self._completion_waiters[operation_id].set()

        # Check dependent operations
        await self._check_dependents(operation_id)

        # Update stats
        self.stats.total_completed += 1
        self.stats.pending_count = len(self._pending)
        self.stats.processing_count = len(self._processing)

    async def _check_dependents(self, completed_op_id: str):
        """Check if any operations were waiting on this completion"""
        for op_id, waiting_for in list(self._dependency_graph.items()):
            if completed_op_id in waiting_for:
                waiting_for.discard(completed_op_id)

                # If no more dependencies, queue the operation
                if not waiting_for:
                    del self._dependency_graph[op_id]
                    op = self._pending.get(op_id)
                    if op:
                        priority = -op.priority.value
                        await self._submission_queue.put((priority, op.sequence_num, op))
```

## 1.4 Usage Examples

### Basic Batch Submission

```python
# Create AgentRing
ring = AgentRing(RingConfig(
    submission_queue_size=1024,
    max_workers=16
))

# Start with adapters
await ring.start(adapters={
    "http": HttpAdapter(),
    "database": DatabaseAdapter(),
    "logger": LoggerAdapter(),
})

# Submit batch of HTTP requests
operations = [
    Operation(
        adapter="http",
        method="get",
        args={"url": f"https://api.example.com/users/{i}"}
    )
    for i in range(100)
]

op_ids = await ring.submit_batch(operations)

# Wait for all completions
completions = await ring.wait_batch(op_ids, timeout=30.0)

# Process results
for completion in completions:
    if completion.status == OperationStatus.COMPLETED:
        print(f"Got user: {completion.result}")
    else:
        print(f"Error: {completion.error}")
```

### Priority-Based Operations

```python
# High priority operation
urgent_op = Operation(
    adapter="http",
    method="post",
    args={"url": "https://api.example.com/critical"},
    priority=OperationPriority.HIGH
)

# Normal priority batch
normal_ops = [
    Operation(
        adapter="http",
        method="get",
        args={"url": f"https://api.example.com/data/{i}"},
        priority=OperationPriority.NORMAL
    )
    for i in range(50)
]

# Submit all - urgent will be processed first
await ring.submit(urgent_op)
await ring.submit_batch(normal_ops)
```

### Dependent Operations

```python
# Operation chain: fetch -> transform -> store
fetch_op = Operation(
    id="fetch",
    adapter="http",
    method="get",
    args={"url": "https://api.example.com/data"}
)

transform_op = Operation(
    id="transform",
    adapter="transform",
    method="map",
    args={"function": "process_data"},
    depends_on=["fetch"]  # Wait for fetch to complete
)

store_op = Operation(
    id="store",
    adapter="database",
    method="insert",
    args={"table": "results"},
    depends_on=["transform"]  # Wait for transform
)

# Submit all at once - dependencies handled automatically
await ring.submit_batch([fetch_op, transform_op, store_op])
```

---

# 2. BBX Hooks System

## 2.1 Architecture

BBX Hooks provide eBPF-like dynamic programming capabilities for workflows.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Hook System Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        Hook Manager                              │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│   │  │ Hook        │  │ Hook        │  │ Hook        │             │   │
│   │  │ Registry    │  │ Verifier    │  │ Executor    │             │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘             │   │
│   └──────────────────────────┬──────────────────────────────────────┘   │
│                              │                                           │
│                              ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      Attach Points                               │   │
│   │                                                                  │   │
│   │  workflow.start    step.pre    step.post    workflow.end        │   │
│   │       │               │            │              │              │   │
│   │       ▼               ▼            ▼              ▼              │   │
│   │  [hooks...]       [hooks...]   [hooks...]    [hooks...]         │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Data Structures

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import re


class HookType(Enum):
    """Types of hooks (like eBPF program types)"""
    PROBE = auto()        # Observe execution, emit metrics/logs
    FILTER = auto()       # Block/allow operations
    TRANSFORM = auto()    # Modify data in-flight
    SECURITY = auto()     # Enforce security policies
    SCHEDULER = auto()    # Custom scheduling decisions


class AttachPoint(Enum):
    """Points where hooks can attach (like eBPF attach points)"""
    # Workflow lifecycle
    WORKFLOW_START = "workflow.start"
    WORKFLOW_END = "workflow.end"
    WORKFLOW_ERROR = "workflow.error"

    # Step lifecycle
    STEP_PRE_EXECUTE = "step.pre_execute"
    STEP_POST_EXECUTE = "step.post_execute"
    STEP_ERROR = "step.error"
    STEP_RETRY = "step.retry"

    # Adapter calls
    ADAPTER_PRE_CALL = "adapter.pre_call"
    ADAPTER_POST_CALL = "adapter.post_call"

    # Resource access
    FILE_ACCESS = "file.access"
    NETWORK_CONNECT = "network.connect"
    STATE_ACCESS = "state.access"

    # Context operations
    CONTEXT_GET = "context.get"
    CONTEXT_SET = "context.set"


class HookAction(Enum):
    """Actions a hook can return"""
    CONTINUE = auto()     # Continue normal execution
    SKIP = auto()         # Skip this operation
    BLOCK = auto()        # Block and return error
    RETRY = auto()        # Retry the operation
    TRANSFORM = auto()    # Use transformed data


@dataclass
class HookContext:
    """
    Context passed to hook functions.
    Like eBPF's ctx pointer - provides access to execution state.
    """
    # Current workflow
    workflow_id: str = ""
    workflow_name: str = ""

    # Current step (if applicable)
    step_id: Optional[str] = None
    step_type: Optional[str] = None
    step_inputs: Dict[str, Any] = field(default_factory=dict)
    step_outputs: Optional[Any] = None
    step_error: Optional[str] = None
    step_duration_ms: float = 0

    # Adapter info (if applicable)
    adapter_name: Optional[str] = None
    adapter_method: Optional[str] = None

    # Resource info (if applicable)
    resource_type: Optional[str] = None
    resource_path: Optional[str] = None

    # Global context
    variables: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: float = 0
    trace_id: str = ""

    # For transform hooks - modified data
    transformed_data: Optional[Any] = None


@dataclass
class HookResult:
    """Result returned by a hook"""
    action: HookAction = HookAction.CONTINUE
    data: Optional[Any] = None  # For TRANSFORM action
    error: Optional[str] = None  # For BLOCK action
    metrics: Dict[str, Any] = field(default_factory=dict)  # Emitted metrics
    logs: List[Dict[str, Any]] = field(default_factory=list)  # Emitted logs


@dataclass
class HookFilter:
    """Filter to match when hook should trigger"""
    # Step filters
    step_ids: Optional[List[str]] = None  # Match specific step IDs
    step_types: Optional[List[str]] = None  # Match step types (glob patterns)

    # Workflow filters
    workflow_ids: Optional[List[str]] = None
    workflow_tags: Optional[List[str]] = None

    # Adapter filters
    adapters: Optional[List[str]] = None
    methods: Optional[List[str]] = None

    # Resource filters
    paths: Optional[List[str]] = None  # Glob patterns

    def matches(self, ctx: HookContext) -> bool:
        """Check if context matches filter"""
        # Check step filters
        if self.step_ids and ctx.step_id not in self.step_ids:
            return False

        if self.step_types:
            if not any(self._glob_match(ctx.step_type or "", p) for p in self.step_types):
                return False

        # Check workflow filters
        if self.workflow_ids and ctx.workflow_id not in self.workflow_ids:
            return False

        # Check adapter filters
        if self.adapters:
            if not any(self._glob_match(ctx.adapter_name or "", p) for p in self.adapters):
                return False

        if self.methods:
            if not any(self._glob_match(ctx.adapter_method or "", p) for p in self.methods):
                return False

        # Check path filters
        if self.paths:
            if not any(self._glob_match(ctx.resource_path or "", p) for p in self.paths):
                return False

        return True

    def _glob_match(self, text: str, pattern: str) -> bool:
        """Simple glob matching (* and **)"""
        regex = pattern.replace(".", r"\.").replace("**", ".*").replace("*", "[^.]*")
        return bool(re.match(f"^{regex}$", text))


@dataclass
class HookDefinition:
    """Complete hook definition"""
    id: str
    name: str
    type: HookType
    attach_points: List[AttachPoint]
    filter: Optional[HookFilter] = None

    # Execution
    handler: Optional[Callable[[HookContext], HookResult]] = None
    code: Optional[str] = None  # Inline code (verified before execution)

    # Configuration
    priority: int = 0  # Higher = runs first
    enabled: bool = True
    timeout_ms: int = 1000  # Max execution time

    # Metadata
    version: str = "1.0"
    author: str = ""
    description: str = ""
```

## 2.3 Hook Verifier

```python
import ast
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class VerificationResult:
    """Result of hook verification"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class HookVerifier:
    """
    Verifies hook programs before execution.

    Like eBPF verifier, ensures:
    - No infinite loops
    - Bounded execution time
    - Memory safety
    - Valid access patterns
    - No dangerous operations
    """

    # Allowed built-in functions
    ALLOWED_BUILTINS: Set[str] = {
        "len", "str", "int", "float", "bool", "list", "dict", "set",
        "min", "max", "sum", "abs", "round", "sorted", "reversed",
        "any", "all", "zip", "enumerate", "range", "map", "filter",
        "isinstance", "hasattr", "getattr",
    }

    # Forbidden AST node types
    FORBIDDEN_NODES: Set[type] = {
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.Exec,  # Python 2 remnant
    }

    # Max AST depth
    MAX_DEPTH = 10

    # Max loop iterations (detected statically where possible)
    MAX_LOOP_ITERATIONS = 1000

    def verify(self, hook: HookDefinition) -> VerificationResult:
        """Verify a hook definition"""
        errors: List[str] = []
        warnings: List[str] = []

        # Verify basic structure
        if not hook.id:
            errors.append("Hook ID is required")

        if not hook.attach_points:
            errors.append("At least one attach point is required")

        # Verify handler
        if hook.code:
            code_result = self._verify_code(hook.code)
            errors.extend(code_result.errors)
            warnings.extend(code_result.warnings)
        elif not hook.handler:
            errors.append("Either handler function or code is required")

        # Verify type-specific requirements
        if hook.type == HookType.SECURITY:
            type_result = self._verify_security_hook(hook)
            errors.extend(type_result.errors)
            warnings.extend(type_result.warnings)

        return VerificationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _verify_code(self, code: str) -> VerificationResult:
        """Verify inline hook code"""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return VerificationResult(passed=False, errors=errors)

        # Check for forbidden nodes
        for node in ast.walk(tree):
            if type(node) in self.FORBIDDEN_NODES:
                errors.append(f"Forbidden operation: {type(node).__name__}")

            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "__import__"):
                        errors.append(f"Forbidden function call: {node.func.id}")

                    if node.func.id not in self.ALLOWED_BUILTINS and not node.func.id.startswith("ctx."):
                        warnings.append(f"Unknown function: {node.func.id}")

            # Check loop bounds
            if isinstance(node, (ast.For, ast.While)):
                if not self._has_bounded_iterations(node):
                    warnings.append("Potentially unbounded loop detected")

        # Check AST depth
        depth = self._get_max_depth(tree)
        if depth > self.MAX_DEPTH:
            errors.append(f"Code too complex (depth {depth} > {self.MAX_DEPTH})")

        return VerificationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _has_bounded_iterations(self, node: ast.AST) -> bool:
        """Check if a loop has bounded iterations"""
        if isinstance(node, ast.For):
            # Check if iterating over range with literal
            if isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                    # Check if range argument is a literal within bounds
                    if node.iter.args:
                        arg = node.iter.args[-1]  # Last arg is the upper bound
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                            return arg.value <= self.MAX_LOOP_ITERATIONS
            return False

        elif isinstance(node, ast.While):
            # While loops are harder to verify statically
            # For now, require explicit break or limit
            return False

        return True

    def _get_max_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Get maximum depth of AST"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_max_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth

    def _verify_security_hook(self, hook: HookDefinition) -> VerificationResult:
        """Additional verification for security hooks"""
        errors: List[str] = []
        warnings: List[str] = []

        # Security hooks should have filters
        if not hook.filter:
            warnings.append("Security hook without filter will apply to all operations")

        return VerificationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
```

## 2.4 Hook Manager

```python
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbx.hooks")


class HookManager:
    """
    Manages hook registration and execution.

    Provides eBPF-like capabilities for BBX workflows.
    """

    def __init__(self):
        # Registered hooks by attach point
        self._hooks: Dict[AttachPoint, List[HookDefinition]] = {
            point: [] for point in AttachPoint
        }

        # Hook registry by ID
        self._registry: Dict[str, HookDefinition] = {}

        # Verifier
        self._verifier = HookVerifier()

        # Metrics
        self._execution_count: Dict[str, int] = {}
        self._execution_time: Dict[str, float] = {}

    def register(self, hook: HookDefinition) -> bool:
        """
        Register a hook.

        Returns True if registration successful, False otherwise.
        """
        # Verify hook
        result = self._verifier.verify(hook)
        if not result.passed:
            logger.error(f"Hook verification failed for {hook.id}: {result.errors}")
            return False

        if result.warnings:
            logger.warning(f"Hook warnings for {hook.id}: {result.warnings}")

        # Compile inline code if present
        if hook.code and not hook.handler:
            hook.handler = self._compile_handler(hook.code)

        # Register at each attach point
        for attach_point in hook.attach_points:
            self._hooks[attach_point].append(hook)
            # Sort by priority (higher first)
            self._hooks[attach_point].sort(key=lambda h: -h.priority)

        self._registry[hook.id] = hook
        logger.info(f"Registered hook: {hook.id} at {[ap.value for ap in hook.attach_points]}")
        return True

    def unregister(self, hook_id: str) -> bool:
        """Unregister a hook by ID"""
        if hook_id not in self._registry:
            return False

        hook = self._registry.pop(hook_id)
        for attach_point in hook.attach_points:
            self._hooks[attach_point] = [
                h for h in self._hooks[attach_point] if h.id != hook_id
            ]

        logger.info(f"Unregistered hook: {hook_id}")
        return True

    async def trigger(
        self,
        attach_point: AttachPoint,
        context: HookContext
    ) -> HookResult:
        """
        Trigger all hooks at an attach point.

        Hooks are executed in priority order.
        First hook to return non-CONTINUE action stops the chain.
        """
        hooks = self._hooks.get(attach_point, [])
        combined_result = HookResult(action=HookAction.CONTINUE)

        for hook in hooks:
            if not hook.enabled:
                continue

            # Check filter
            if hook.filter and not hook.filter.matches(context):
                continue

            try:
                # Execute hook with timeout
                start_time = time.time()
                result = await asyncio.wait_for(
                    self._execute_hook(hook, context),
                    timeout=hook.timeout_ms / 1000
                )
                duration = time.time() - start_time

                # Track metrics
                self._execution_count[hook.id] = self._execution_count.get(hook.id, 0) + 1
                self._execution_time[hook.id] = self._execution_time.get(hook.id, 0) + duration

                # Collect metrics and logs from hook
                combined_result.metrics.update(result.metrics)
                combined_result.logs.extend(result.logs)

                # Check action
                if result.action != HookAction.CONTINUE:
                    combined_result.action = result.action
                    combined_result.data = result.data
                    combined_result.error = result.error
                    break

            except asyncio.TimeoutError:
                logger.warning(f"Hook {hook.id} timed out after {hook.timeout_ms}ms")

            except Exception as e:
                logger.error(f"Hook {hook.id} error: {e}")

        return combined_result

    async def _execute_hook(
        self,
        hook: HookDefinition,
        context: HookContext
    ) -> HookResult:
        """Execute a single hook"""
        if hook.handler:
            # If handler is async
            if asyncio.iscoroutinefunction(hook.handler):
                return await hook.handler(context)
            else:
                return hook.handler(context)

        return HookResult(action=HookAction.CONTINUE)

    def _compile_handler(self, code: str) -> Callable[[HookContext], HookResult]:
        """Compile inline code into a handler function"""
        # Create a safe execution environment
        safe_globals = {
            # Built-ins
            "len": len, "str": str, "int": int, "float": float,
            "bool": bool, "list": list, "dict": dict, "set": set,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "round": round, "sorted": sorted, "any": any, "all": all,
            "zip": zip, "enumerate": enumerate, "range": range,
            "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,

            # Hook-specific
            "HookResult": HookResult,
            "HookAction": HookAction,
        }

        # Wrap code in function
        wrapped_code = f"""
def _hook_handler(ctx):
    # User code
{chr(10).join('    ' + line for line in code.split(chr(10)))}

    # Default return
    return HookResult(action=HookAction.CONTINUE)
"""

        # Compile and extract handler
        exec(wrapped_code, safe_globals)
        return safe_globals["_hook_handler"]

    def get_stats(self) -> Dict[str, Any]:
        """Get hook execution statistics"""
        stats = {}
        for hook_id, hook in self._registry.items():
            count = self._execution_count.get(hook_id, 0)
            total_time = self._execution_time.get(hook_id, 0)
            stats[hook_id] = {
                "execution_count": count,
                "total_time_ms": total_time * 1000,
                "avg_time_ms": (total_time / count * 1000) if count > 0 else 0,
                "attach_points": [ap.value for ap in hook.attach_points],
                "enabled": hook.enabled,
            }
        return stats
```

## 2.5 Usage Examples

### Metrics Probe Hook

```python
# Define metrics hook
metrics_hook = HookDefinition(
    id="metrics_collector",
    name="Step Metrics Collector",
    type=HookType.PROBE,
    attach_points=[
        AttachPoint.STEP_PRE_EXECUTE,
        AttachPoint.STEP_POST_EXECUTE,
    ],
    code="""
# Emit step duration metric
if ctx.step_duration_ms > 0:
    emit_metric("bbx_step_duration_ms", ctx.step_duration_ms, {
        "step_id": ctx.step_id,
        "step_type": ctx.step_type,
        "workflow_id": ctx.workflow_id,
    })

# Emit error metric if failed
if ctx.step_error:
    emit_metric("bbx_step_errors_total", 1, {
        "step_id": ctx.step_id,
        "error_type": type(ctx.step_error).__name__,
    })

return HookResult(action=HookAction.CONTINUE)
"""
)

# Register hook
manager.register(metrics_hook)
```

### Security Filter Hook

```python
# Define security hook
security_hook = HookDefinition(
    id="file_access_control",
    name="File Access Control",
    type=HookType.SECURITY,
    attach_points=[AttachPoint.FILE_ACCESS],
    filter=HookFilter(
        adapters=["file.*"],
    ),
    code="""
# Block access to sensitive paths
blocked_paths = ["/etc/passwd", "/etc/shadow", "~/.ssh/*", "*.key", "*.pem"]

for pattern in blocked_paths:
    if glob_match(ctx.resource_path, pattern):
        return HookResult(
            action=HookAction.BLOCK,
            error=f"Access to {ctx.resource_path} is blocked by security policy"
        )

return HookResult(action=HookAction.CONTINUE)
"""
)

# Register hook
manager.register(security_hook)
```

### Data Transform Hook

```python
# Define transform hook
pii_masking_hook = HookDefinition(
    id="pii_masking",
    name="PII Data Masking",
    type=HookType.TRANSFORM,
    attach_points=[AttachPoint.ADAPTER_POST_CALL],
    filter=HookFilter(
        adapters=["http", "database"],
        methods=["get", "query"],
    ),
    code="""
import re

def mask_pii(data):
    if isinstance(data, str):
        # Mask email addresses
        data = re.sub(r'[\\w.-]+@[\\w.-]+', '[EMAIL]', data)
        # Mask phone numbers
        data = re.sub(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b', '[PHONE]', data)
        # Mask SSN
        data = re.sub(r'\\b\\d{3}-\\d{2}-\\d{4}\\b', '[SSN]', data)
    elif isinstance(data, dict):
        return {k: mask_pii(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_pii(item) for item in data]
    return data

ctx.transformed_data = mask_pii(ctx.step_outputs)
return HookResult(action=HookAction.TRANSFORM, data=ctx.transformed_data)
"""
)

# Register hook
manager.register(pii_masking_hook)
```

---

# 3. ContextTiering System

## 3.1 Implementation

```python
import asyncio
import hashlib
import json
import lz4.frame  # For compression
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger("bbx.context_tiering")


class GenerationTier(Enum):
    """Memory tiers for context (like MGLRU generations)"""
    HOT = 0       # In-memory, uncompressed
    WARM = 1      # In-memory, compressed
    COOL = 2      # On disk, compressed
    COLD = 3      # Vector DB / long-term archive


@dataclass
class ContextItem:
    """Single item in context memory"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    pinned: bool = False  # If True, won't be demoted
    generation: GenerationTier = GenerationTier.HOT

    # For tracking refaults (like MGLRU)
    last_generation_change: datetime = field(default_factory=datetime.now)
    promotion_count: int = 0


@dataclass
class TieringConfig:
    """Configuration for context tiering"""
    # Generation limits
    hot_max_size: int = 100 * 1024  # 100KB
    warm_max_size: int = 1 * 1024 * 1024  # 1MB
    cool_max_size: int = 100 * 1024 * 1024  # 100MB

    # Aging
    hot_max_age: timedelta = timedelta(minutes=5)
    warm_max_age: timedelta = timedelta(hours=1)
    cool_max_age: timedelta = timedelta(days=1)

    # MGLRU-style parameters
    aging_interval: float = 30.0  # Seconds between aging cycles
    refault_distance: int = 5  # Generations before promotion on access
    min_ttl: float = 10.0  # Minimum seconds before demotion


class Generation:
    """Single generation in the tiering hierarchy"""

    def __init__(
        self,
        tier: GenerationTier,
        max_size: int,
        storage_path: Optional[Path] = None
    ):
        self.tier = tier
        self.max_size = max_size
        self.storage_path = storage_path
        self._items: Dict[str, ContextItem] = {}
        self._current_size = 0
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return self._current_size

    @property
    def count(self) -> int:
        return len(self._items)

    async def get(self, key: str) -> Optional[ContextItem]:
        """Get item from this generation"""
        async with self._lock:
            item = self._items.get(key)
            if item:
                item.last_accessed = datetime.now()
                item.access_count += 1
            return item

    async def add(self, item: ContextItem) -> bool:
        """Add item to this generation"""
        async with self._lock:
            # Serialize and measure size
            serialized = self._serialize(item.value)
            item.size_bytes = len(serialized)

            # Check if fits
            if self._current_size + item.size_bytes > self.max_size:
                return False

            item.generation = self.tier
            item.last_generation_change = datetime.now()
            self._items[item.key] = item
            self._current_size += item.size_bytes

            # If cool/cold tier, persist to disk
            if self.tier in (GenerationTier.COOL, GenerationTier.COLD) and self.storage_path:
                await self._persist(item)

            return True

    async def remove(self, key: str) -> Optional[ContextItem]:
        """Remove item from this generation"""
        async with self._lock:
            item = self._items.pop(key, None)
            if item:
                self._current_size -= item.size_bytes

                # Remove from disk if persisted
                if self.tier in (GenerationTier.COOL, GenerationTier.COLD) and self.storage_path:
                    await self._unpersist(item.key)

            return item

    async def get_aged_items(self, max_age: timedelta) -> List[ContextItem]:
        """Get items older than max_age since last access"""
        cutoff = datetime.now() - max_age
        aged = []
        async with self._lock:
            for item in self._items.values():
                if not item.pinned and item.last_accessed < cutoff:
                    aged.append(item)
        return aged

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        json_bytes = json.dumps(value, default=str).encode()
        if self.tier in (GenerationTier.WARM, GenerationTier.COOL, GenerationTier.COLD):
            return lz4.frame.compress(json_bytes)
        return json_bytes

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if self.tier in (GenerationTier.WARM, GenerationTier.COOL, GenerationTier.COLD):
            data = lz4.frame.decompress(data)
        return json.loads(data.decode())

    async def _persist(self, item: ContextItem):
        """Persist item to disk"""
        if not self.storage_path:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        file_path = self.storage_path / f"{hashlib.md5(item.key.encode()).hexdigest()}.ctx"
        serialized = self._serialize(item.value)
        file_path.write_bytes(serialized)

    async def _unpersist(self, key: str):
        """Remove persisted item from disk"""
        if not self.storage_path:
            return
        file_path = self.storage_path / f"{hashlib.md5(key.encode()).hexdigest()}.ctx"
        if file_path.exists():
            file_path.unlink()


class RefaultTracker:
    """
    Tracks "refaults" - when a demoted item is accessed again.
    Used to make smart promotion decisions (like MGLRU).
    """

    def __init__(self, max_history: int = 10000):
        self._access_history: Dict[str, List[datetime]] = {}
        self._generation_history: Dict[str, List[int]] = {}
        self._max_history = max_history

    def record_access(self, key: str, generation: int):
        """Record an access from a specific generation"""
        now = datetime.now()

        if key not in self._access_history:
            self._access_history[key] = []
            self._generation_history[key] = []

        self._access_history[key].append(now)
        self._generation_history[key].append(generation)

        # Trim history
        if len(self._access_history[key]) > self._max_history:
            self._access_history[key] = self._access_history[key][-self._max_history:]
            self._generation_history[key] = self._generation_history[key][-self._max_history:]

    def get_distance(self, key: str) -> int:
        """
        Get "distance" since last access from hot tier.
        Lower distance = more likely to be accessed soon.
        """
        if key not in self._generation_history:
            return float('inf')

        history = self._generation_history[key]
        # Count generations since last access from Gen 0
        for i, gen in enumerate(reversed(history)):
            if gen == 0:
                return i
        return len(history)


class ContextTiering:
    """
    Multi-Generation LRU for AI agent context.

    Like Linux MGLRU, uses generations instead of simple LRU.
    Context items age through generations, with smart promotion/demotion.
    """

    def __init__(self, config: Optional[TieringConfig] = None, base_path: Optional[Path] = None):
        self.config = config or TieringConfig()
        self.base_path = base_path or Path.home() / ".bbx" / "context"

        # Initialize generations
        self.generations = [
            Generation(GenerationTier.HOT, self.config.hot_max_size),
            Generation(GenerationTier.WARM, self.config.warm_max_size),
            Generation(
                GenerationTier.COOL,
                self.config.cool_max_size,
                self.base_path / "cool"
            ),
            Generation(
                GenerationTier.COLD,
                float('inf'),  # Unlimited
                self.base_path / "cold"
            ),
        ]

        self.refault_tracker = RefaultTracker()
        self._aging_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self):
        """Start the aging background task"""
        self._shutdown = False
        self._aging_task = asyncio.create_task(self._aging_loop())
        logger.info("ContextTiering started")

    async def stop(self):
        """Stop the aging background task"""
        self._shutdown = True
        if self._aging_task:
            self._aging_task.cancel()
            try:
                await self._aging_task
            except asyncio.CancelledError:
                pass
        logger.info("ContextTiering stopped")

    async def get(self, key: str) -> Optional[Any]:
        """Get context item, promoting if accessed from cold generation"""
        for gen_idx, gen in enumerate(self.generations):
            item = await gen.get(key)
            if item:
                # Track access for promotion decision
                self.refault_tracker.record_access(key, gen_idx)

                # Promote if accessed from cold generation
                if gen_idx > 0 and self._should_promote(key, gen_idx):
                    await self._promote(item, from_gen=gen_idx, to_gen=0)

                return item.value
        return None

    async def set(self, key: str, value: Any, pinned: bool = False):
        """Add/update item in hot generation"""
        # Remove from all generations first
        for gen in self.generations:
            await gen.remove(key)

        # Add to hot generation
        item = ContextItem(key=key, value=value, pinned=pinned)
        success = await self.generations[0].add(item)

        if not success:
            # Hot generation full, trigger aging
            await self._age_generation(0)
            success = await self.generations[0].add(item)

            if not success:
                logger.warning(f"Failed to add item {key} to context (too large?)")

    async def delete(self, key: str):
        """Remove item from all generations"""
        for gen in self.generations:
            await gen.remove(key)

    async def pin(self, key: str):
        """Pin an item to prevent demotion"""
        for gen in self.generations:
            item = await gen.get(key)
            if item:
                item.pinned = True
                return
        raise KeyError(f"Item {key} not found")

    async def unpin(self, key: str):
        """Unpin an item to allow demotion"""
        for gen in self.generations:
            item = await gen.get(key)
            if item:
                item.pinned = False
                return
        raise KeyError(f"Item {key} not found")

    def _should_promote(self, key: str, current_gen: int) -> bool:
        """Decide if item should be promoted based on access patterns"""
        refault_distance = self.refault_tracker.get_distance(key)
        return refault_distance < self.config.refault_distance

    async def _promote(self, item: ContextItem, from_gen: int, to_gen: int):
        """Promote item to a hotter generation"""
        # Remove from current generation
        await self.generations[from_gen].remove(item.key)

        # Update item
        item.promotion_count += 1
        item.last_generation_change = datetime.now()

        # Add to target generation
        success = await self.generations[to_gen].add(item)
        if not success:
            # Target full, try next colder generation
            for gen_idx in range(to_gen + 1, from_gen):
                success = await self.generations[gen_idx].add(item)
                if success:
                    break

            if not success:
                # Put back in original
                await self.generations[from_gen].add(item)

    async def _demote(self, item: ContextItem, from_gen: int):
        """Demote item to next colder generation"""
        if from_gen >= len(self.generations) - 1:
            return  # Already in coldest

        # Remove from current
        await self.generations[from_gen].remove(item.key)

        # Update item
        item.last_generation_change = datetime.now()

        # Add to next colder generation
        to_gen = from_gen + 1
        success = await self.generations[to_gen].add(item)

        if not success:
            # Recursively demote from target
            await self._age_generation(to_gen)
            await self.generations[to_gen].add(item)

    async def _age_generation(self, gen_idx: int):
        """Age items in a generation (demote old items)"""
        gen = self.generations[gen_idx]

        max_ages = [
            self.config.hot_max_age,
            self.config.warm_max_age,
            self.config.cool_max_age,
            timedelta.max,  # Cold never ages out
        ]

        aged_items = await gen.get_aged_items(max_ages[gen_idx])

        for item in aged_items:
            # Check refault distance before demoting
            distance = self.refault_tracker.get_distance(item.key)
            if distance < self.config.refault_distance:
                # Likely to be accessed soon, keep in current generation
                continue

            # Check min TTL
            time_since_change = datetime.now() - item.last_generation_change
            if time_since_change.total_seconds() < self.config.min_ttl:
                continue

            await self._demote(item, gen_idx)

    async def _aging_loop(self):
        """Background loop that runs aging on all generations"""
        while not self._shutdown:
            try:
                for gen_idx in range(len(self.generations) - 1):  # Don't age coldest
                    await self._age_generation(gen_idx)

                await asyncio.sleep(self.config.aging_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aging loop error: {e}")
                await asyncio.sleep(self.config.aging_interval)

    def get_stats(self) -> Dict[str, Any]:
        """Get tiering statistics"""
        return {
            "generations": [
                {
                    "tier": gen.tier.name,
                    "size_bytes": gen.size,
                    "max_size_bytes": gen.max_size,
                    "utilization": gen.size / gen.max_size if gen.max_size > 0 else 0,
                    "item_count": gen.count,
                }
                for gen in self.generations
            ],
            "total_items": sum(gen.count for gen in self.generations),
            "total_size": sum(gen.size for gen in self.generations),
        }
```

---

# Summary

This technical specification details three core BBX 2.0 systems:

1. **AgentRing**: io_uring-inspired batch operations for efficient agent execution
2. **BBX Hooks**: eBPF-inspired dynamic programming for observability and security
3. **ContextTiering**: MGLRU-inspired multi-generation context memory management

Each system brings Linux kernel innovations to the AI agent domain, providing:
- **Performance**: Batch operations, zero-copy where possible
- **Flexibility**: Dynamic hooks without code changes
- **Intelligence**: Smart memory management with access pattern tracking
- **Safety**: Verification, sandboxing, and resource limits

---

*Document Version: 1.0*
*Date: November 2025*
