# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 AgentRing Enhanced - Production-ready io_uring-inspired system.

Improvements over base AgentRing:
- WAL (Write-Ahead Log) for persistence - operations survive crashes
- Shared memory (mmap) for cross-process communication
- Idempotency keys for exactly-once semantics
- Distributed mode with Redis backend
- Backpressure handling with circuit breaker
- Metrics and observability hooks

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    AgentRing Enhanced                        │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
    │  │   WAL   │  │ IdempKey│  │ Circuit │  │  SharedMemory   │ │
    │  │ Manager │  │ Manager │  │ Breaker │  │    (mmap)       │ │
    │  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
    │       │            │            │                │          │
    │  ┌────▼────────────▼────────────▼────────────────▼────────┐ │
    │  │              Submission Queue (Priority)               │ │
    │  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ... ┌─────┐         │ │
    │  │   │REAL │ │HIGH │ │NORM │ │LOW  │     │BACK │         │ │
    │  │   │TIME │ │     │ │     │ │     │     │LOG  │         │ │
    │  │   └─────┘ └─────┘ └─────┘ └─────┘     └─────┘         │ │
    │  └────────────────────────────────────────────────────────┘ │
    │                              │                              │
    │  ┌───────────────────────────▼───────────────────────────┐  │
    │  │                   Worker Pool                          │  │
    │  │   ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐     │  │
    │  │   │Worker 1│ │Worker 2│ │Worker 3│ ... │Worker N│     │  │
    │  │   └────────┘ └────────┘ └────────┘     └────────┘     │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                              │                              │
    │  ┌───────────────────────────▼───────────────────────────┐  │
    │  │              Completion Queue + Metrics                │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mmap
import os
import pickle
import struct
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.ring.enhanced")


# =============================================================================
# Enums
# =============================================================================


class OperationType(Enum):
    """Types of operations that can be submitted"""
    ADAPTER_CALL = auto()
    WORKFLOW_EXEC = auto()
    STATE_OP = auto()
    CONTEXT_OP = auto()
    HOOK_TRIGGER = auto()
    RPC_CALL = auto()
    BATCH_OP = auto()


class OperationPriority(Enum):
    """Operation priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    REALTIME = 3


class OperationStatus(Enum):
    """Status of an operation"""
    PENDING = auto()
    QUEUED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()
    RETRYING = auto()


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if recovered


# =============================================================================
# WAL (Write-Ahead Log)
# =============================================================================


@dataclass
class WALEntry:
    """Single entry in the Write-Ahead Log"""
    sequence_id: int
    timestamp: float
    operation_id: str
    idempotency_key: Optional[str]
    operation_data: bytes
    entry_type: str  # 'submit', 'complete', 'cancel'
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        data = f"{self.sequence_id}:{self.timestamp}:{self.operation_id}:{self.entry_type}"
        return hashlib.sha256(data.encode() + self.operation_data).hexdigest()[:16]

    def verify(self) -> bool:
        return self.checksum == self._compute_checksum()

    def to_bytes(self) -> bytes:
        """Serialize WAL entry to bytes"""
        header = struct.pack(
            ">Q d 36s 64s B 16s",
            self.sequence_id,
            self.timestamp,
            self.operation_id.encode()[:36].ljust(36),
            (self.idempotency_key or "").encode()[:64].ljust(64),
            len(self.entry_type),
            self.checksum.encode()[:16].ljust(16)
        )
        entry_type_bytes = self.entry_type.encode()
        data_len = struct.pack(">I", len(self.operation_data))
        return header + entry_type_bytes + data_len + self.operation_data

    @classmethod
    def from_bytes(cls, data: bytes) -> "WALEntry":
        """Deserialize WAL entry from bytes"""
        header_size = struct.calcsize(">Q d 36s 64s B 16s")
        header = data[:header_size]

        seq_id, ts, op_id_bytes, idem_bytes, entry_type_len, checksum_bytes = struct.unpack(
            ">Q d 36s 64s B 16s", header
        )

        offset = header_size
        entry_type = data[offset:offset + entry_type_len].decode()
        offset += entry_type_len

        data_len = struct.unpack(">I", data[offset:offset + 4])[0]
        offset += 4

        operation_data = data[offset:offset + data_len]

        return cls(
            sequence_id=seq_id,
            timestamp=ts,
            operation_id=op_id_bytes.decode().strip(),
            idempotency_key=idem_bytes.decode().strip() or None,
            operation_data=operation_data,
            entry_type=entry_type,
            checksum=checksum_bytes.decode().strip()
        )


class WALManager:
    """
    Write-Ahead Log Manager for durability.

    All operations are logged before execution, enabling:
    - Crash recovery
    - Point-in-time replay
    - Exactly-once delivery (with idempotency keys)
    """

    def __init__(
        self,
        wal_dir: Optional[Path] = None,
        max_wal_size: int = 100 * 1024 * 1024,  # 100MB
        sync_on_write: bool = True,
        checkpoint_interval: int = 10000,  # entries
    ):
        self.wal_dir = wal_dir or Path(tempfile.gettempdir()) / "bbx_wal"
        self.wal_dir.mkdir(parents=True, exist_ok=True)

        self.max_wal_size = max_wal_size
        self.sync_on_write = sync_on_write
        self.checkpoint_interval = checkpoint_interval

        self._sequence_id = 0
        self._current_wal_file: Optional[Path] = None
        self._wal_handle: Optional[Any] = None
        self._entries_since_checkpoint = 0
        self._lock = asyncio.Lock()

        # Recovery state
        self._recovered_ops: Dict[str, WALEntry] = {}

    async def initialize(self) -> int:
        """Initialize WAL and recover any uncommitted operations"""
        recovered_count = await self._recover()
        await self._open_new_wal()
        return recovered_count

    async def _recover(self) -> int:
        """Recover operations from existing WAL files"""
        recovered = 0
        wal_files = sorted(self.wal_dir.glob("wal_*.log"))

        for wal_file in wal_files:
            try:
                entries = await self._read_wal_file(wal_file)
                for entry in entries:
                    if entry.verify():
                        if entry.entry_type == 'submit':
                            self._recovered_ops[entry.operation_id] = entry
                        elif entry.entry_type in ('complete', 'cancel'):
                            self._recovered_ops.pop(entry.operation_id, None)
                        self._sequence_id = max(self._sequence_id, entry.sequence_id)
                        recovered += 1
            except Exception as e:
                logger.error(f"Error recovering WAL file {wal_file}: {e}")

        logger.info(f"WAL recovery complete: {recovered} entries, {len(self._recovered_ops)} pending")
        return len(self._recovered_ops)

    async def _read_wal_file(self, path: Path) -> List[WALEntry]:
        """Read all entries from a WAL file"""
        entries = []
        try:
            with open(path, 'rb') as f:
                while True:
                    # Read entry length prefix
                    len_bytes = f.read(4)
                    if not len_bytes:
                        break
                    entry_len = struct.unpack(">I", len_bytes)[0]
                    entry_data = f.read(entry_len)
                    if len(entry_data) < entry_len:
                        break  # Incomplete entry (crash during write)
                    entries.append(WALEntry.from_bytes(entry_data))
        except Exception as e:
            logger.error(f"Error reading WAL file: {e}")
        return entries

    async def _open_new_wal(self):
        """Open a new WAL file"""
        if self._wal_handle:
            self._wal_handle.close()

        timestamp = int(time.time() * 1000)
        self._current_wal_file = self.wal_dir / f"wal_{timestamp}.log"
        self._wal_handle = open(self._current_wal_file, 'ab')
        self._entries_since_checkpoint = 0

    async def append(
        self,
        operation_id: str,
        operation_data: bytes,
        entry_type: str,
        idempotency_key: Optional[str] = None
    ) -> int:
        """Append an entry to the WAL"""
        async with self._lock:
            self._sequence_id += 1
            entry = WALEntry(
                sequence_id=self._sequence_id,
                timestamp=time.time(),
                operation_id=operation_id,
                idempotency_key=idempotency_key,
                operation_data=operation_data,
                entry_type=entry_type
            )

            entry_bytes = entry.to_bytes()
            # Write length prefix + entry
            self._wal_handle.write(struct.pack(">I", len(entry_bytes)))
            self._wal_handle.write(entry_bytes)

            if self.sync_on_write:
                self._wal_handle.flush()
                os.fsync(self._wal_handle.fileno())

            self._entries_since_checkpoint += 1

            # Check if we need to rotate or checkpoint
            if self._entries_since_checkpoint >= self.checkpoint_interval:
                await self._checkpoint()

            return self._sequence_id

    async def _checkpoint(self):
        """Create a checkpoint and rotate WAL"""
        # Write checkpoint marker
        checkpoint_data = json.dumps({
            "type": "checkpoint",
            "sequence_id": self._sequence_id,
            "timestamp": time.time(),
            "pending_ops": list(self._recovered_ops.keys())
        }).encode()

        checkpoint_file = self.wal_dir / f"checkpoint_{self._sequence_id}.json"
        with open(checkpoint_file, 'wb') as f:
            f.write(checkpoint_data)

        # Rotate to new WAL
        await self._open_new_wal()

        # Clean up old WAL files (keep last 3)
        wal_files = sorted(self.wal_dir.glob("wal_*.log"))[:-3]
        for old_file in wal_files:
            try:
                old_file.unlink()
            except Exception:
                pass

    def get_recovered_operations(self) -> Dict[str, bytes]:
        """Get operations that need to be replayed after recovery"""
        return {
            op_id: entry.operation_data
            for op_id, entry in self._recovered_ops.items()
        }

    async def mark_complete(self, operation_id: str):
        """Mark an operation as complete in WAL"""
        await self.append(operation_id, b"", "complete")
        self._recovered_ops.pop(operation_id, None)

    async def close(self):
        """Close WAL manager"""
        if self._wal_handle:
            self._wal_handle.close()
            self._wal_handle = None


# =============================================================================
# Idempotency Manager
# =============================================================================


@dataclass
class IdempotencyRecord:
    """Record of a completed operation for idempotency"""
    key: str
    operation_id: str
    result: Any
    completed_at: float
    expires_at: float


class IdempotencyManager:
    """
    Manages idempotency keys for exactly-once semantics.

    Features:
    - Deduplication of requests with same idempotency key
    - Configurable TTL for stored results
    - Memory-efficient with LRU eviction
    - Optional Redis backend for distributed scenarios
    """

    def __init__(
        self,
        ttl_seconds: int = 86400,  # 24 hours
        max_entries: int = 100000,
        redis_client: Optional[Any] = None
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.redis_client = redis_client

        self._local_cache: Dict[str, IdempotencyRecord] = {}
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()

    async def check_and_store(
        self,
        idempotency_key: str,
        operation_id: str
    ) -> Optional[IdempotencyRecord]:
        """
        Check if operation already exists, store if new.

        Returns existing record if duplicate, None if new operation.
        """
        async with self._lock:
            # Check local cache
            if idempotency_key in self._local_cache:
                record = self._local_cache[idempotency_key]
                if record.expires_at > time.time():
                    # Update access order
                    if idempotency_key in self._access_order:
                        self._access_order.remove(idempotency_key)
                    self._access_order.append(idempotency_key)
                    return record
                else:
                    # Expired, remove
                    del self._local_cache[idempotency_key]

            # Check Redis if available
            if self.redis_client:
                try:
                    redis_data = await self.redis_client.get(f"idem:{idempotency_key}")
                    if redis_data:
                        record_data = json.loads(redis_data)
                        record = IdempotencyRecord(**record_data)
                        if record.expires_at > time.time():
                            self._local_cache[idempotency_key] = record
                            return record
                except Exception as e:
                    logger.warning(f"Redis check failed: {e}")

            # New operation - create placeholder record
            now = time.time()
            placeholder = IdempotencyRecord(
                key=idempotency_key,
                operation_id=operation_id,
                result=None,  # Will be filled when complete
                completed_at=0,
                expires_at=now + self.ttl_seconds
            )

            # Evict if needed
            while len(self._local_cache) >= self.max_entries and self._access_order:
                oldest = self._access_order.pop(0)
                self._local_cache.pop(oldest, None)

            self._local_cache[idempotency_key] = placeholder
            self._access_order.append(idempotency_key)

            return None

    async def store_result(
        self,
        idempotency_key: str,
        result: Any
    ):
        """Store the result for an idempotency key"""
        async with self._lock:
            if idempotency_key in self._local_cache:
                record = self._local_cache[idempotency_key]
                record.result = result
                record.completed_at = time.time()

                # Store in Redis if available
                if self.redis_client:
                    try:
                        await self.redis_client.setex(
                            f"idem:{idempotency_key}",
                            self.ttl_seconds,
                            json.dumps({
                                "key": record.key,
                                "operation_id": record.operation_id,
                                "result": record.result,
                                "completed_at": record.completed_at,
                                "expires_at": record.expires_at
                            })
                        )
                    except Exception as e:
                        logger.warning(f"Redis store failed: {e}")

    async def remove(self, idempotency_key: str):
        """Remove an idempotency record"""
        async with self._lock:
            self._local_cache.pop(idempotency_key, None)
            if idempotency_key in self._access_order:
                self._access_order.remove(idempotency_key)

            if self.redis_client:
                try:
                    await self.redis_client.delete(f"idem:{idempotency_key}")
                except Exception:
                    pass


# =============================================================================
# Circuit Breaker
# =============================================================================


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests rejected immediately
    - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def can_execute(self) -> bool:
        """Check if request can proceed"""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self):
        """Record a successful operation"""
        async with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info(f"Circuit {self.name} CLOSED after recovery")

    async def record_failure(self):
        """Record a failed operation"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(f"Circuit {self.name} OPEN after half-open failure")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name} OPEN after {self._failure_count} failures")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time,
        }


# =============================================================================
# Shared Memory Ring Buffer
# =============================================================================


class SharedMemoryRingBuffer:
    """
    Lock-free ring buffer using shared memory (mmap).

    Enables zero-copy data sharing between processes.
    Used for high-performance operation submission/completion.

    Memory Layout:
    [Header: 64 bytes]
        - magic: 4 bytes (BBX1)
        - version: 4 bytes
        - capacity: 8 bytes
        - head: 8 bytes (atomic)
        - tail: 8 bytes (atomic)
        - entry_size: 8 bytes
        - flags: 8 bytes
        - reserved: 16 bytes
    [Entries: capacity * entry_size bytes]
    """

    MAGIC = b'BBX1'
    HEADER_SIZE = 64

    def __init__(
        self,
        name: str,
        capacity: int = 4096,
        entry_size: int = 1024,
        create: bool = True
    ):
        self.name = name
        self.capacity = capacity
        self.entry_size = entry_size

        self._total_size = self.HEADER_SIZE + (capacity * entry_size)
        self._mmap: Optional[mmap.mmap] = None
        self._file_path = Path(tempfile.gettempdir()) / f"bbx_shm_{name}"

        if create:
            self._create()
        else:
            self._attach()

    def _create(self):
        """Create new shared memory region"""
        # Create/truncate file
        with open(self._file_path, 'wb') as f:
            f.write(b'\x00' * self._total_size)

        # Map the file
        fd = os.open(str(self._file_path), os.O_RDWR)
        try:
            self._mmap = mmap.mmap(fd, self._total_size)
        finally:
            os.close(fd)

        # Write header
        self._write_header()

    def _attach(self):
        """Attach to existing shared memory"""
        fd = os.open(str(self._file_path), os.O_RDWR)
        try:
            self._mmap = mmap.mmap(fd, 0)  # Map entire file
        finally:
            os.close(fd)

        # Verify header
        if not self._verify_header():
            raise ValueError("Invalid shared memory header")

    def _write_header(self):
        """Write header to shared memory"""
        header = struct.pack(
            ">4s I Q Q Q Q Q 16s",
            self.MAGIC,
            1,  # version
            self.capacity,
            0,  # head
            0,  # tail
            self.entry_size,
            0,  # flags
            b'\x00' * 16  # reserved
        )
        self._mmap[0:self.HEADER_SIZE] = header

    def _verify_header(self) -> bool:
        """Verify shared memory header"""
        magic = self._mmap[0:4]
        return magic == self.MAGIC

    def _get_head(self) -> int:
        """Get current head position (atomic read)"""
        return struct.unpack(">Q", self._mmap[16:24])[0]

    def _set_head(self, value: int):
        """Set head position (atomic write)"""
        self._mmap[16:24] = struct.pack(">Q", value)

    def _get_tail(self) -> int:
        """Get current tail position (atomic read)"""
        return struct.unpack(">Q", self._mmap[24:32])[0]

    def _set_tail(self, value: int):
        """Set tail position (atomic write)"""
        self._mmap[24:32] = struct.pack(">Q", value)

    def _entry_offset(self, index: int) -> int:
        """Calculate offset for entry at index"""
        return self.HEADER_SIZE + (index % self.capacity) * self.entry_size

    def push(self, data: bytes) -> bool:
        """
        Push data to the ring buffer.
        Returns False if buffer is full.
        """
        if len(data) > self.entry_size - 4:  # 4 bytes for length prefix
            raise ValueError(f"Data too large: {len(data)} > {self.entry_size - 4}")

        head = self._get_head()
        tail = self._get_tail()

        # Check if full
        next_head = (head + 1) % self.capacity
        if next_head == tail:
            return False  # Buffer full

        # Write entry
        offset = self._entry_offset(head)
        entry = struct.pack(">I", len(data)) + data
        entry = entry.ljust(self.entry_size, b'\x00')
        self._mmap[offset:offset + self.entry_size] = entry

        # Update head
        self._set_head(next_head)
        return True

    def pop(self) -> Optional[bytes]:
        """
        Pop data from the ring buffer.
        Returns None if buffer is empty.
        """
        head = self._get_head()
        tail = self._get_tail()

        # Check if empty
        if head == tail:
            return None

        # Read entry
        offset = self._entry_offset(tail)
        length = struct.unpack(">I", self._mmap[offset:offset + 4])[0]
        data = bytes(self._mmap[offset + 4:offset + 4 + length])

        # Update tail
        self._set_tail((tail + 1) % self.capacity)
        return data

    def size(self) -> int:
        """Get current number of entries in buffer"""
        head = self._get_head()
        tail = self._get_tail()
        if head >= tail:
            return head - tail
        return self.capacity - tail + head

    def close(self):
        """Close shared memory"""
        if self._mmap:
            self._mmap.close()
            self._mmap = None

    def unlink(self):
        """Remove shared memory file"""
        self.close()
        try:
            self._file_path.unlink()
        except Exception:
            pass


# =============================================================================
# Enhanced Operation and Completion
# =============================================================================


@dataclass
class EnhancedOperation:
    """Enhanced operation with additional features"""
    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: Optional[str] = None
    sequence_num: int = 0

    # Operation details
    op_type: OperationType = OperationType.ADAPTER_CALL
    adapter: str = ""
    method: str = ""
    args: Dict[str, Any] = field(default_factory=dict)

    # Execution control
    priority: OperationPriority = OperationPriority.NORMAL
    timeout_ms: int = 30000
    max_retries: int = 3
    retry_delay_ms: int = 1000
    retry_backoff: float = 2.0  # Exponential backoff multiplier

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Circuit breaker
    circuit_breaker: Optional[str] = None

    # Metadata
    submitted_at: Optional[datetime] = None
    user_data: Any = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_bytes(self) -> bytes:
        """Serialize operation to bytes"""
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EnhancedOperation":
        """Deserialize operation from bytes"""
        return pickle.loads(data)


@dataclass
class EnhancedCompletion:
    """Enhanced completion with metrics"""
    operation_id: str
    idempotency_key: Optional[str] = None
    sequence_num: int = 0

    status: OperationStatus = OperationStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metrics
    queue_time_ms: float = 0
    execution_time_ms: float = 0
    total_time_ms: float = 0
    retry_count: int = 0

    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EnhancedCompletion":
        return pickle.loads(data)


# =============================================================================
# Enhanced Ring Configuration
# =============================================================================


@dataclass
class EnhancedRingConfig:
    """Configuration for EnhancedAgentRing"""
    # Queue sizes
    submission_queue_size: int = 4096
    completion_queue_size: int = 4096

    # Workers
    min_workers: int = 4
    max_workers: int = 64
    worker_idle_timeout: float = 60.0
    worker_scale_up_threshold: float = 0.8  # Scale up when queue > 80% full
    worker_scale_down_threshold: float = 0.2  # Scale down when queue < 20% full

    # Batching
    max_batch_size: int = 256
    batch_timeout_ms: float = 10.0

    # Features
    enable_wal: bool = True
    enable_idempotency: bool = True
    enable_circuit_breaker: bool = True
    enable_shared_memory: bool = False
    enable_metrics: bool = True

    # WAL
    wal_dir: Optional[Path] = None
    wal_sync_on_write: bool = True

    # Idempotency
    idempotency_ttl_seconds: int = 86400

    # Timeouts
    default_timeout_ms: int = 30000
    shutdown_timeout_seconds: float = 30.0


@dataclass
class EnhancedRingStats:
    """Statistics for EnhancedAgentRing"""
    # Counters
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_cancelled: int = 0
    total_timeout: int = 0
    total_retried: int = 0
    total_deduplicated: int = 0
    total_circuit_rejected: int = 0

    # Current state
    pending_count: int = 0
    processing_count: int = 0
    active_workers: int = 0

    # Performance
    avg_queue_time_ms: float = 0
    avg_execution_time_ms: float = 0
    p99_latency_ms: float = 0

    # WAL
    wal_entries: int = 0
    wal_recovered: int = 0

    # Circuit breakers
    circuit_breakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# =============================================================================
# Enhanced AgentRing
# =============================================================================


class EnhancedAgentRing:
    """
    Production-ready io_uring-inspired batch operation system.

    Features:
    - WAL for durability
    - Idempotency for exactly-once
    - Circuit breakers for fault tolerance
    - Shared memory for cross-process
    - Auto-scaling workers
    - Comprehensive metrics
    """

    def __init__(self, config: Optional[EnhancedRingConfig] = None):
        self.config = config or EnhancedRingConfig()

        # Core components
        self._wal: Optional[WALManager] = None
        self._idempotency: Optional[IdempotencyManager] = None
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._shared_memory: Optional[SharedMemoryRingBuffer] = None

        # Queues
        self._submission_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.config.submission_queue_size
        )
        self._completion_queue: asyncio.Queue[EnhancedCompletion] = asyncio.Queue(
            maxsize=self.config.completion_queue_size
        )

        # Tracking
        self._pending: Dict[str, EnhancedOperation] = {}
        self._processing: Set[str] = set()
        self._completed: Dict[str, EnhancedCompletion] = {}
        self._dependency_graph: Dict[str, Set[str]] = {}

        # Workers
        self._workers: List[asyncio.Task] = []
        self._shutdown = False
        self._scaling_task: Optional[asyncio.Task] = None

        # Adapters
        self._adapters: Dict[str, Any] = {}

        # Waiters
        self._completion_waiters: Dict[str, asyncio.Event] = {}

        # Stats
        self.stats = EnhancedRingStats()
        self._latencies: List[float] = []

        # Metrics callbacks
        self._on_submit: List[Callable] = []
        self._on_complete: List[Callable] = []

    async def start(self, adapters: Dict[str, Any]):
        """Start the enhanced ring"""
        self._adapters = adapters
        self._shutdown = False

        # Initialize WAL
        if self.config.enable_wal:
            self._wal = WALManager(
                wal_dir=self.config.wal_dir,
                sync_on_write=self.config.wal_sync_on_write
            )
            recovered = await self._wal.initialize()
            self.stats.wal_recovered = recovered

            # Replay recovered operations
            for op_id, op_data in self._wal.get_recovered_operations().items():
                try:
                    op = EnhancedOperation.from_bytes(op_data)
                    await self._enqueue_operation(op)
                except Exception as e:
                    logger.error(f"Failed to replay operation {op_id}: {e}")

        # Initialize idempotency manager
        if self.config.enable_idempotency:
            self._idempotency = IdempotencyManager(
                ttl_seconds=self.config.idempotency_ttl_seconds
            )

        # Initialize shared memory
        if self.config.enable_shared_memory:
            self._shared_memory = SharedMemoryRingBuffer(
                name="agent_ring",
                capacity=self.config.submission_queue_size
            )

        # Start workers
        for _ in range(self.config.min_workers):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)

        # Start auto-scaling
        self._scaling_task = asyncio.create_task(self._auto_scale_loop())

        self.stats.active_workers = len(self._workers)
        logger.info(f"EnhancedAgentRing started with {len(self._workers)} workers")

    async def stop(self):
        """Stop gracefully"""
        self._shutdown = True

        # Stop scaling
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass

        # Wait for pending operations
        deadline = time.time() + self.config.shutdown_timeout_seconds
        while self._pending and time.time() < deadline:
            await asyncio.sleep(0.1)

        # Stop workers
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        # Close WAL
        if self._wal:
            await self._wal.close()

        # Close shared memory
        if self._shared_memory:
            self._shared_memory.close()

        self.stats.active_workers = 0
        logger.info("EnhancedAgentRing stopped")

    # =========================================================================
    # Submission API
    # =========================================================================

    async def submit(
        self,
        operation: EnhancedOperation,
        idempotency_key: Optional[str] = None
    ) -> str:
        """Submit a single operation"""
        operation.submitted_at = datetime.now()

        # Use provided key or operation's key
        idem_key = idempotency_key or operation.idempotency_key

        # Check idempotency
        if self._idempotency and idem_key:
            existing = await self._idempotency.check_and_store(idem_key, operation.id)
            if existing and existing.result is not None:
                # Return cached result
                self.stats.total_deduplicated += 1
                logger.debug(f"Deduplicated operation with key {idem_key}")

                # Create completion from cached result
                completion = EnhancedCompletion(
                    operation_id=operation.id,
                    idempotency_key=idem_key,
                    status=OperationStatus.COMPLETED,
                    result=existing.result,
                    submitted_at=operation.submitted_at,
                    completed_at=datetime.fromtimestamp(existing.completed_at)
                )
                self._completed[operation.id] = completion
                return operation.id

        # Write to WAL
        if self._wal:
            await self._wal.append(
                operation.id,
                operation.to_bytes(),
                "submit",
                idem_key
            )
            self.stats.wal_entries += 1

        # Add to pending
        self._pending[operation.id] = operation

        # Enqueue
        await self._enqueue_operation(operation)

        self.stats.total_submitted += 1
        self.stats.pending_count = len(self._pending)

        # Metrics callback
        for callback in self._on_submit:
            try:
                callback(operation)
            except Exception:
                pass

        return operation.id

    async def _enqueue_operation(self, operation: EnhancedOperation):
        """Add operation to submission queue"""
        if operation.depends_on:
            self._dependency_graph[operation.id] = set(operation.depends_on)
        else:
            priority = -operation.priority.value
            await self._submission_queue.put((priority, operation.sequence_num, operation))

    async def submit_batch(
        self,
        operations: List[EnhancedOperation],
        ordered: bool = False
    ) -> List[str]:
        """Submit multiple operations"""
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
    ) -> EnhancedCompletion:
        """Wait for specific operation"""
        if operation_id in self._completed:
            return self._completed[operation_id]

        event = asyncio.Event()
        self._completion_waiters[operation_id] = event

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return self._completed[operation_id]
        except asyncio.TimeoutError:
            return EnhancedCompletion(
                operation_id=operation_id,
                status=OperationStatus.TIMEOUT,
                error="Wait timed out"
            )
        finally:
            self._completion_waiters.pop(operation_id, None)

    async def wait_batch(
        self,
        operation_ids: List[str],
        timeout: Optional[float] = None
    ) -> List[EnhancedCompletion]:
        """Wait for multiple operations"""
        tasks = [self.wait_completion(op_id, timeout) for op_id in operation_ids]
        completions = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for op_id, comp in zip(operation_ids, completions):
            if isinstance(comp, Exception):
                results.append(EnhancedCompletion(
                    operation_id=op_id,
                    status=OperationStatus.FAILED,
                    error=str(comp)
                ))
            else:
                results.append(comp)
        return results

    # =========================================================================
    # Circuit Breaker API
    # =========================================================================

    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]

    # =========================================================================
    # Workers
    # =========================================================================

    async def _worker_loop(self):
        """Worker coroutine"""
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

            completion = await self._execute_with_retry(operation)
            await self._complete_operation(operation.id, completion)

    async def _execute_with_retry(
        self,
        operation: EnhancedOperation
    ) -> EnhancedCompletion:
        """Execute operation with retries"""
        completion = EnhancedCompletion(
            operation_id=operation.id,
            idempotency_key=operation.idempotency_key,
            sequence_num=operation.sequence_num,
            submitted_at=operation.submitted_at or datetime.now(),
            trace_id=operation.trace_id,
            span_id=operation.span_id
        )

        retry_count = 0
        last_error = None
        retry_delay = operation.retry_delay_ms / 1000

        while retry_count <= operation.max_retries:
            # Check circuit breaker
            if self.config.enable_circuit_breaker and operation.circuit_breaker:
                cb = self.get_circuit_breaker(operation.circuit_breaker)
                if not await cb.can_execute():
                    completion.status = OperationStatus.FAILED
                    completion.error = f"Circuit breaker {operation.circuit_breaker} is OPEN"
                    self.stats.total_circuit_rejected += 1
                    return completion

            started_at = datetime.now()
            completion.started_at = completion.started_at or started_at

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

                # Record circuit breaker success
                if self.config.enable_circuit_breaker and operation.circuit_breaker:
                    cb = self.get_circuit_breaker(operation.circuit_breaker)
                    await cb.record_success()

                break

            except asyncio.TimeoutError:
                last_error = f"Timeout after {operation.timeout_ms}ms"
                self.stats.total_timeout += 1

            except Exception as e:
                last_error = str(e)
                completion.error_type = type(e).__name__

            # Record circuit breaker failure
            if self.config.enable_circuit_breaker and operation.circuit_breaker:
                cb = self.get_circuit_breaker(operation.circuit_breaker)
                await cb.record_failure()

            retry_count += 1
            if retry_count <= operation.max_retries:
                completion.status = OperationStatus.RETRYING
                self.stats.total_retried += 1
                await asyncio.sleep(retry_delay)
                retry_delay *= operation.retry_backoff

        if completion.status != OperationStatus.COMPLETED:
            completion.status = OperationStatus.FAILED
            completion.error = last_error
            self.stats.total_failed += 1

        completion.completed_at = datetime.now()
        completion.retry_count = retry_count

        # Calculate metrics
        if completion.started_at:
            completion.queue_time_ms = (
                completion.started_at - completion.submitted_at
            ).total_seconds() * 1000
        completion.execution_time_ms = (
            completion.completed_at - (completion.started_at or completion.submitted_at)
        ).total_seconds() * 1000
        completion.total_time_ms = (
            completion.completed_at - completion.submitted_at
        ).total_seconds() * 1000

        return completion

    async def _complete_operation(
        self,
        operation_id: str,
        completion: EnhancedCompletion
    ):
        """Mark operation as complete"""
        self._completed[operation_id] = completion
        self._pending.pop(operation_id, None)
        self._processing.discard(operation_id)

        # Update WAL
        if self._wal:
            await self._wal.mark_complete(operation_id)

        # Store idempotency result
        if self._idempotency and completion.idempotency_key:
            await self._idempotency.store_result(
                completion.idempotency_key,
                completion.result
            )

        # Notify waiters
        await self._completion_queue.put(completion)

        if operation_id in self._completion_waiters:
            self._completion_waiters[operation_id].set()

        # Check dependents
        await self._check_dependents(operation_id)

        # Update stats
        self.stats.total_completed += 1
        self.stats.pending_count = len(self._pending)
        self.stats.processing_count = len(self._processing)

        # Track latency for percentiles
        self._latencies.append(completion.total_time_ms)
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]

        # Update circuit breaker stats
        for name, cb in self._circuit_breakers.items():
            self.stats.circuit_breakers[name] = cb.get_stats()

        # Metrics callback
        for callback in self._on_complete:
            try:
                callback(completion)
            except Exception:
                pass

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

    async def _auto_scale_loop(self):
        """Auto-scale workers based on queue utilization"""
        while not self._shutdown:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                queue_size = self._submission_queue.qsize()
                utilization = queue_size / self.config.submission_queue_size

                if utilization > self.config.worker_scale_up_threshold:
                    # Scale up
                    if len(self._workers) < self.config.max_workers:
                        worker = asyncio.create_task(self._worker_loop())
                        self._workers.append(worker)
                        self.stats.active_workers = len(self._workers)
                        logger.info(f"Scaled up to {len(self._workers)} workers")

                elif utilization < self.config.worker_scale_down_threshold:
                    # Scale down
                    if len(self._workers) > self.config.min_workers:
                        # Cancel one worker
                        worker = self._workers.pop()
                        worker.cancel()
                        self.stats.active_workers = len(self._workers)
                        logger.info(f"Scaled down to {len(self._workers)} workers")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scale error: {e}")

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_stats(self) -> EnhancedRingStats:
        """Get current statistics"""
        # Calculate percentiles
        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            p99_idx = int(len(sorted_latencies) * 0.99)
            self.stats.p99_latency_ms = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
            self.stats.avg_queue_time_ms = sum(self._latencies) / len(self._latencies)

        return self.stats

    def on_submit(self, callback: Callable[[EnhancedOperation], None]):
        """Register callback for operation submission"""
        self._on_submit.append(callback)

    def on_complete(self, callback: Callable[[EnhancedCompletion], None]):
        """Register callback for operation completion"""
        self._on_complete.append(callback)


# =============================================================================
# Factory and Global Instance
# =============================================================================


_global_enhanced_ring: Optional[EnhancedAgentRing] = None


def get_enhanced_ring() -> EnhancedAgentRing:
    """Get global enhanced ring instance"""
    global _global_enhanced_ring
    if _global_enhanced_ring is None:
        _global_enhanced_ring = EnhancedAgentRing()
    return _global_enhanced_ring


async def create_enhanced_ring(
    config: Optional[EnhancedRingConfig] = None,
    adapters: Optional[Dict[str, Any]] = None
) -> EnhancedAgentRing:
    """Create and start an enhanced ring"""
    ring = EnhancedAgentRing(config)
    await ring.start(adapters or {})
    return ring
