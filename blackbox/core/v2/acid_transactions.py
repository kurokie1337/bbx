# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX SIRE ACID Transactions - Reliable AI Operations.

ACID for AI Agents:
- Atomic: All operations succeed or all rollback
- Consistent: System always in valid state
- Isolated: Concurrent agents don't interfere
- Durable: Committed changes survive crashes

This is what makes AI RELIABLE (like WinRAR's Recovery Record).

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                  Transaction Manager                         │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────────┐  ┌─────────────────┐                   │
    │  │  Transaction    │  │  Lock Manager   │                   │
    │  │  Coordinator    │  │  (2PL Protocol) │                   │
    │  └────────┬────────┘  └────────┬────────┘                   │
    │           │                    │                             │
    │  ┌────────▼────────────────────▼────────────────────────┐   │
    │  │                    WAL (Write-Ahead Log)              │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │   │
    │  │  │ BEGIN   │ │ WRITE   │ │ WRITE   │ │ COMMIT  │     │   │
    │  │  │ tx_001  │ │ file A  │ │ file B  │ │ tx_001  │     │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │   │
    │  └──────────────────────────────────────────────────────┘   │
    │                              │                               │
    │  ┌───────────────────────────▼───────────────────────────┐   │
    │  │                 Snapshot Manager                       │   │
    │  │  - Before-images for rollback                         │   │
    │  │  - After-images for redo                              │   │
    │  │  - Checkpoint management                              │   │
    │  └───────────────────────────────────────────────────────┘   │
    │                              │                               │
    │  ┌───────────────────────────▼───────────────────────────┐   │
    │  │                 Recovery Manager                       │   │
    │  │  - Crash recovery (ARIES algorithm inspired)          │   │
    │  │  - Point-in-time recovery                             │   │
    │  │  - Transaction rollback                               │   │
    │  └───────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘

Transaction Flow:
    1. BEGIN - Allocate transaction ID, acquire resources
    2. OPERATIONS - Each op writes to WAL first, then executes
    3. COMMIT - Flush WAL, release locks
    4. ROLLBACK - Undo using before-images
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.sire.acid")


# =============================================================================
# Enums
# =============================================================================


class TransactionState(Enum):
    """Transaction lifecycle states"""
    PENDING = auto()      # Not yet started
    ACTIVE = auto()       # In progress
    PREPARING = auto()    # Preparing to commit (2PC)
    COMMITTED = auto()    # Successfully committed
    ABORTED = auto()      # Rolled back
    RECOVERING = auto()   # Being recovered after crash


class LockMode(Enum):
    """Lock modes for 2PL protocol"""
    SHARED = "S"          # Read lock (multiple holders)
    EXCLUSIVE = "X"       # Write lock (single holder)
    INTENT_SHARED = "IS"  # Intent to get shared lock on children
    INTENT_EXCLUSIVE = "IX"  # Intent to get exclusive lock on children


class IsolationLevel(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = 0  # Dirty reads allowed
    READ_COMMITTED = 1    # No dirty reads
    REPEATABLE_READ = 2   # No phantom reads
    SERIALIZABLE = 3      # Full isolation


class WALRecordType(Enum):
    """Types of WAL records"""
    BEGIN = "BEGIN"
    WRITE = "WRITE"
    DELETE = "DELETE"
    CHECKPOINT = "CHECKPOINT"
    COMMIT = "COMMIT"
    ABORT = "ABORT"
    COMPENSATE = "COMPENSATE"  # Undo operation


# =============================================================================
# WAL (Write-Ahead Log)
# =============================================================================


@dataclass
class WALRecord:
    """Single record in WAL"""
    lsn: int                    # Log Sequence Number
    tx_id: str                  # Transaction ID
    record_type: WALRecordType
    timestamp: float = field(default_factory=time.time)

    # For WRITE/DELETE
    resource_id: Optional[str] = None
    before_image: Optional[bytes] = None  # State before change
    after_image: Optional[bytes] = None   # State after change

    # For COMPENSATE
    undo_lsn: Optional[int] = None  # LSN being undone

    # Checksum for integrity
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        data = f"{self.lsn}:{self.tx_id}:{self.record_type.value}:{self.resource_id}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify(self) -> bool:
        return self.checksum == self._compute_checksum()


class WriteAheadLog:
    """
    WAL implementation for durability.

    Guarantees:
    - All changes written to log BEFORE applied
    - Log survives crashes
    - Can recover to any point
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path(".bbx/wal")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._lsn_counter = 0
        self._records: List[WALRecord] = []
        self._lock = asyncio.Lock()

        # Index for fast lookup
        self._tx_records: Dict[str, List[int]] = {}  # tx_id -> [lsn]
        self._resource_records: Dict[str, List[int]] = {}  # resource_id -> [lsn]

        # Persistence
        self._log_file = self.log_dir / "wal.log"
        self._checkpoint_file = self.log_dir / "checkpoint.json"

    async def initialize(self) -> int:
        """Initialize WAL and recover if needed"""
        if self._log_file.exists():
            await self._recover_from_disk()
        return len(self._records)

    async def append(
        self,
        tx_id: str,
        record_type: WALRecordType,
        resource_id: Optional[str] = None,
        before_image: Optional[bytes] = None,
        after_image: Optional[bytes] = None,
        undo_lsn: Optional[int] = None
    ) -> int:
        """Append record to WAL, returns LSN"""
        async with self._lock:
            self._lsn_counter += 1
            lsn = self._lsn_counter

            record = WALRecord(
                lsn=lsn,
                tx_id=tx_id,
                record_type=record_type,
                resource_id=resource_id,
                before_image=before_image,
                after_image=after_image,
                undo_lsn=undo_lsn,
            )

            self._records.append(record)

            # Update indices
            if tx_id not in self._tx_records:
                self._tx_records[tx_id] = []
            self._tx_records[tx_id].append(lsn)

            if resource_id:
                if resource_id not in self._resource_records:
                    self._resource_records[resource_id] = []
                self._resource_records[resource_id].append(lsn)

            # Persist to disk
            await self._persist_record(record)

            return lsn

    async def _persist_record(self, record: WALRecord):
        """Persist record to disk"""
        data = {
            "lsn": record.lsn,
            "tx_id": record.tx_id,
            "record_type": record.record_type.value,
            "timestamp": record.timestamp,
            "resource_id": record.resource_id,
            "before_image": record.before_image.hex() if record.before_image else None,
            "after_image": record.after_image.hex() if record.after_image else None,
            "undo_lsn": record.undo_lsn,
            "checksum": record.checksum,
        }

        with open(self._log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()

    async def _recover_from_disk(self):
        """Recover WAL from disk"""
        with open(self._log_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    record = WALRecord(
                        lsn=data["lsn"],
                        tx_id=data["tx_id"],
                        record_type=WALRecordType(data["record_type"]),
                        timestamp=data["timestamp"],
                        resource_id=data["resource_id"],
                        before_image=bytes.fromhex(data["before_image"]) if data["before_image"] else None,
                        after_image=bytes.fromhex(data["after_image"]) if data["after_image"] else None,
                        undo_lsn=data["undo_lsn"],
                        checksum=data["checksum"],
                    )

                    if record.verify():
                        self._records.append(record)
                        self._lsn_counter = max(self._lsn_counter, record.lsn)

                        # Update indices
                        if record.tx_id not in self._tx_records:
                            self._tx_records[record.tx_id] = []
                        self._tx_records[record.tx_id].append(record.lsn)

                        if record.resource_id:
                            if record.resource_id not in self._resource_records:
                                self._resource_records[record.resource_id] = []
                            self._resource_records[record.resource_id].append(record.lsn)

        logger.info(f"Recovered {len(self._records)} WAL records")

    def get_records_for_tx(self, tx_id: str) -> List[WALRecord]:
        """Get all records for a transaction"""
        lsns = self._tx_records.get(tx_id, [])
        return [r for r in self._records if r.lsn in lsns]

    def get_uncommitted_transactions(self) -> List[str]:
        """Get transactions that started but didn't commit/abort"""
        begun = set()
        ended = set()

        for record in self._records:
            if record.record_type == WALRecordType.BEGIN:
                begun.add(record.tx_id)
            elif record.record_type in (WALRecordType.COMMIT, WALRecordType.ABORT):
                ended.add(record.tx_id)

        return list(begun - ended)

    async def checkpoint(self):
        """Create checkpoint for faster recovery"""
        checkpoint_data = {
            "lsn": self._lsn_counter,
            "timestamp": time.time(),
            "active_transactions": self.get_uncommitted_transactions(),
        }

        with open(self._checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Append checkpoint record
        await self.append("SYSTEM", WALRecordType.CHECKPOINT)


# =============================================================================
# Lock Manager
# =============================================================================


@dataclass
class LockRequest:
    """Request for a lock"""
    tx_id: str
    resource_id: str
    mode: LockMode
    granted: bool = False
    granted_at: Optional[float] = None


class LockManager:
    """
    2PL (Two-Phase Locking) implementation.

    Guarantees serializability through:
    - Growing phase: Acquire locks
    - Shrinking phase: Release locks (only after commit/abort)

    Deadlock detection via timeout (could add waits-for graph).
    """

    def __init__(self, lock_timeout: float = 30.0):
        self.lock_timeout = lock_timeout

        # Current lock holders
        # resource_id -> {tx_id -> LockMode}
        self._locks: Dict[str, Dict[str, LockMode]] = {}

        # Waiting queue
        # resource_id -> [LockRequest]
        self._wait_queue: Dict[str, List[LockRequest]] = {}

        # Locks held by each transaction
        self._tx_locks: Dict[str, Set[str]] = {}

        self._lock = asyncio.Lock()

    async def acquire(
        self,
        tx_id: str,
        resource_id: str,
        mode: LockMode,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire lock on resource.

        Returns True if lock acquired, False on timeout.
        """
        timeout = timeout or self.lock_timeout

        async with self._lock:
            # Check if we can grant immediately
            if self._can_grant(resource_id, tx_id, mode):
                self._grant_lock(resource_id, tx_id, mode)
                return True

            # Add to wait queue
            request = LockRequest(tx_id=tx_id, resource_id=resource_id, mode=mode)
            if resource_id not in self._wait_queue:
                self._wait_queue[resource_id] = []
            self._wait_queue[resource_id].append(request)

        # Wait for lock
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(0.1)

            async with self._lock:
                if self._can_grant(resource_id, tx_id, mode):
                    # Remove from wait queue
                    self._wait_queue[resource_id] = [
                        r for r in self._wait_queue.get(resource_id, [])
                        if r.tx_id != tx_id
                    ]
                    self._grant_lock(resource_id, tx_id, mode)
                    return True

        # Timeout - remove from wait queue
        async with self._lock:
            self._wait_queue[resource_id] = [
                r for r in self._wait_queue.get(resource_id, [])
                if r.tx_id != tx_id
            ]

        return False

    def _can_grant(self, resource_id: str, tx_id: str, mode: LockMode) -> bool:
        """Check if lock can be granted"""
        holders = self._locks.get(resource_id, {})

        # Already hold the lock?
        if tx_id in holders:
            current_mode = holders[tx_id]
            # Upgrade S -> X allowed if we're only holder
            if current_mode == LockMode.SHARED and mode == LockMode.EXCLUSIVE:
                return len(holders) == 1
            return True

        if not holders:
            return True

        if mode == LockMode.SHARED:
            # S compatible with S
            return all(m == LockMode.SHARED for m in holders.values())

        # X requires no other holders
        return False

    def _grant_lock(self, resource_id: str, tx_id: str, mode: LockMode):
        """Grant lock to transaction"""
        if resource_id not in self._locks:
            self._locks[resource_id] = {}
        self._locks[resource_id][tx_id] = mode

        if tx_id not in self._tx_locks:
            self._tx_locks[tx_id] = set()
        self._tx_locks[tx_id].add(resource_id)

    async def release(self, tx_id: str, resource_id: str):
        """Release a specific lock"""
        async with self._lock:
            if resource_id in self._locks:
                self._locks[resource_id].pop(tx_id, None)
                if not self._locks[resource_id]:
                    del self._locks[resource_id]

            if tx_id in self._tx_locks:
                self._tx_locks[tx_id].discard(resource_id)

    async def release_all(self, tx_id: str):
        """Release all locks held by transaction"""
        async with self._lock:
            resources = list(self._tx_locks.get(tx_id, set()))

            for resource_id in resources:
                if resource_id in self._locks:
                    self._locks[resource_id].pop(tx_id, None)
                    if not self._locks[resource_id]:
                        del self._locks[resource_id]

            self._tx_locks.pop(tx_id, None)

    def get_locks_held(self, tx_id: str) -> Set[str]:
        """Get resources locked by transaction"""
        return self._tx_locks.get(tx_id, set()).copy()


# =============================================================================
# Transaction
# =============================================================================


@dataclass
class Transaction:
    """A single transaction"""
    tx_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TransactionState = TransactionState.PENDING
    isolation_level: IsolationLevel = IsolationLevel.REPEATABLE_READ

    # Timing
    started_at: Optional[float] = None
    committed_at: Optional[float] = None
    aborted_at: Optional[float] = None

    # Changes
    changes: Dict[str, Tuple[bytes, bytes]] = field(default_factory=dict)  # resource -> (before, after)

    # WAL position
    first_lsn: Optional[int] = None
    last_lsn: Optional[int] = None

    # Savepoints
    savepoints: Dict[str, int] = field(default_factory=dict)  # name -> lsn

    # Parent transaction (for nested transactions)
    parent_tx_id: Optional[str] = None


# =============================================================================
# Transaction Manager
# =============================================================================


class TransactionManager:
    """
    Central transaction coordinator.

    Provides ACID guarantees for all agent operations.
    """

    def __init__(
        self,
        wal_dir: Optional[Path] = None,
        isolation_level: IsolationLevel = IsolationLevel.REPEATABLE_READ
    ):
        self.default_isolation = isolation_level

        # Components
        self.wal = WriteAheadLog(wal_dir)
        self.lock_manager = LockManager()

        # Active transactions
        self._transactions: Dict[str, Transaction] = {}

        # Resource store (actual data)
        self._resources: Dict[str, bytes] = {}

        # Statistics
        self._stats = {
            "started": 0,
            "committed": 0,
            "aborted": 0,
            "active": 0,
        }

        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize and recover from WAL"""
        recovered = await self.wal.initialize()

        # Recover uncommitted transactions
        uncommitted = self.wal.get_uncommitted_transactions()
        for tx_id in uncommitted:
            await self._rollback_transaction(tx_id)

        logger.info(f"TransactionManager initialized, recovered {recovered} records")

    async def begin(
        self,
        isolation_level: Optional[IsolationLevel] = None,
        parent_tx_id: Optional[str] = None
    ) -> str:
        """Begin a new transaction"""
        async with self._lock:
            tx = Transaction(
                isolation_level=isolation_level or self.default_isolation,
                state=TransactionState.ACTIVE,
                started_at=time.time(),
                parent_tx_id=parent_tx_id,
            )

            # Log BEGIN
            lsn = await self.wal.append(tx.tx_id, WALRecordType.BEGIN)
            tx.first_lsn = lsn

            self._transactions[tx.tx_id] = tx
            self._stats["started"] += 1
            self._stats["active"] += 1

            logger.debug(f"Transaction {tx.tx_id} started")
            return tx.tx_id

    async def read(self, tx_id: str, resource_id: str) -> Optional[bytes]:
        """Read resource within transaction"""
        tx = self._transactions.get(tx_id)
        if not tx or tx.state != TransactionState.ACTIVE:
            raise ValueError(f"Invalid transaction: {tx_id}")

        # Acquire shared lock
        if not await self.lock_manager.acquire(tx_id, resource_id, LockMode.SHARED):
            raise TimeoutError(f"Lock timeout on {resource_id}")

        # Check if we have uncommitted change
        if resource_id in tx.changes:
            return tx.changes[resource_id][1]  # After image

        # Read from store
        return self._resources.get(resource_id)

    async def write(self, tx_id: str, resource_id: str, data: bytes):
        """Write resource within transaction"""
        tx = self._transactions.get(tx_id)
        if not tx or tx.state != TransactionState.ACTIVE:
            raise ValueError(f"Invalid transaction: {tx_id}")

        # Acquire exclusive lock
        if not await self.lock_manager.acquire(tx_id, resource_id, LockMode.EXCLUSIVE):
            raise TimeoutError(f"Lock timeout on {resource_id}")

        # Get before image
        if resource_id in tx.changes:
            before_image = tx.changes[resource_id][0]  # Original before
        else:
            before_image = self._resources.get(resource_id, b"")

        # Log WRITE
        lsn = await self.wal.append(
            tx_id,
            WALRecordType.WRITE,
            resource_id=resource_id,
            before_image=before_image,
            after_image=data,
        )
        tx.last_lsn = lsn

        # Store change (not yet committed)
        tx.changes[resource_id] = (before_image, data)

    async def delete(self, tx_id: str, resource_id: str):
        """Delete resource within transaction"""
        tx = self._transactions.get(tx_id)
        if not tx or tx.state != TransactionState.ACTIVE:
            raise ValueError(f"Invalid transaction: {tx_id}")

        # Acquire exclusive lock
        if not await self.lock_manager.acquire(tx_id, resource_id, LockMode.EXCLUSIVE):
            raise TimeoutError(f"Lock timeout on {resource_id}")

        # Get before image
        before_image = self._resources.get(resource_id, b"")

        # Log DELETE
        lsn = await self.wal.append(
            tx_id,
            WALRecordType.DELETE,
            resource_id=resource_id,
            before_image=before_image,
        )
        tx.last_lsn = lsn

        # Mark for deletion
        tx.changes[resource_id] = (before_image, None)

    async def savepoint(self, tx_id: str, name: str):
        """Create savepoint within transaction"""
        tx = self._transactions.get(tx_id)
        if not tx or tx.state != TransactionState.ACTIVE:
            raise ValueError(f"Invalid transaction: {tx_id}")

        tx.savepoints[name] = tx.last_lsn or tx.first_lsn or 0

    async def rollback_to_savepoint(self, tx_id: str, name: str):
        """Rollback to savepoint"""
        tx = self._transactions.get(tx_id)
        if not tx or tx.state != TransactionState.ACTIVE:
            raise ValueError(f"Invalid transaction: {tx_id}")

        if name not in tx.savepoints:
            raise ValueError(f"Unknown savepoint: {name}")

        savepoint_lsn = tx.savepoints[name]

        # Undo changes after savepoint
        records = self.wal.get_records_for_tx(tx_id)
        for record in reversed(records):
            if record.lsn <= savepoint_lsn:
                break
            if record.record_type == WALRecordType.WRITE:
                # Restore before image
                if record.resource_id:
                    tx.changes[record.resource_id] = (
                        record.before_image,
                        record.before_image
                    )
                    # Log compensate
                    await self.wal.append(
                        tx_id,
                        WALRecordType.COMPENSATE,
                        resource_id=record.resource_id,
                        before_image=record.after_image,
                        after_image=record.before_image,
                        undo_lsn=record.lsn,
                    )

    async def commit(self, tx_id: str):
        """Commit transaction"""
        async with self._lock:
            tx = self._transactions.get(tx_id)
            if not tx or tx.state != TransactionState.ACTIVE:
                raise ValueError(f"Invalid transaction: {tx_id}")

            tx.state = TransactionState.PREPARING

            # Apply changes to actual store
            for resource_id, (before, after) in tx.changes.items():
                if after is None:
                    self._resources.pop(resource_id, None)
                else:
                    self._resources[resource_id] = after

            # Log COMMIT
            await self.wal.append(tx_id, WALRecordType.COMMIT)

            # Release all locks
            await self.lock_manager.release_all(tx_id)

            tx.state = TransactionState.COMMITTED
            tx.committed_at = time.time()

            self._stats["committed"] += 1
            self._stats["active"] -= 1

            logger.debug(f"Transaction {tx_id} committed")

    async def rollback(self, tx_id: str):
        """Rollback transaction"""
        await self._rollback_transaction(tx_id)

    async def _rollback_transaction(self, tx_id: str):
        """Internal rollback implementation"""
        async with self._lock:
            tx = self._transactions.get(tx_id)

            if tx:
                tx.state = TransactionState.ABORTED
                tx.aborted_at = time.time()

            # Log ABORT
            await self.wal.append(tx_id, WALRecordType.ABORT)

            # Release all locks
            await self.lock_manager.release_all(tx_id)

            if tx:
                self._stats["aborted"] += 1
                self._stats["active"] -= 1

            logger.debug(f"Transaction {tx_id} rolled back")

    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: Optional[IsolationLevel] = None
    ):
        """Context manager for transactions"""
        tx_id = await self.begin(isolation_level)
        try:
            yield TransactionContext(self, tx_id)
            await self.commit(tx_id)
        except Exception as e:
            await self.rollback(tx_id)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get transaction statistics"""
        return {
            **self._stats,
            "resources": len(self._resources),
            "active_locks": sum(len(h) for h in self.lock_manager._locks.values()),
        }


@dataclass
class TransactionContext:
    """Context for transaction operations"""
    manager: TransactionManager
    tx_id: str

    async def read(self, resource_id: str) -> Optional[bytes]:
        return await self.manager.read(self.tx_id, resource_id)

    async def write(self, resource_id: str, data: bytes):
        await self.manager.write(self.tx_id, resource_id, data)

    async def delete(self, resource_id: str):
        await self.manager.delete(self.tx_id, resource_id)

    async def savepoint(self, name: str):
        await self.manager.savepoint(self.tx_id, name)

    async def rollback_to(self, name: str):
        await self.manager.rollback_to_savepoint(self.tx_id, name)


# =============================================================================
# Agent-Level Transaction API
# =============================================================================


class AgentTransactionManager:
    """
    High-level transaction API for agents.

    Makes ACID transactions easy to use:
    - Simple begin/commit/rollback
    - Automatic resource serialization
    - Supports Python objects, not just bytes
    """

    def __init__(self, base_manager: Optional[TransactionManager] = None):
        self.manager = base_manager or TransactionManager()

    async def initialize(self):
        await self.manager.initialize()

    @asynccontextmanager
    async def atomic(self, agent_id: str = "default"):
        """
        Execute operations atomically.

        Example:
            async with tx_manager.atomic("agent_1") as tx:
                await tx.set("config", {"key": "value"})
                await tx.set("state", {"running": True})
                # Either both succeed or both fail
        """
        async with self.manager.transaction() as ctx:
            yield AgentTransactionContext(ctx, agent_id)

    async def checkpoint(self):
        """Create checkpoint for faster recovery"""
        await self.manager.wal.checkpoint()


@dataclass
class AgentTransactionContext:
    """High-level transaction context for agents"""
    ctx: TransactionContext
    agent_id: str

    def _key(self, name: str) -> str:
        return f"{self.agent_id}:{name}"

    async def get(self, name: str) -> Optional[Any]:
        """Get value by name"""
        data = await self.ctx.read(self._key(name))
        if data:
            return pickle.loads(data)
        return None

    async def set(self, name: str, value: Any):
        """Set value"""
        data = pickle.dumps(value)
        await self.ctx.write(self._key(name), data)

    async def delete(self, name: str):
        """Delete value"""
        await self.ctx.delete(self._key(name))

    async def savepoint(self, name: str):
        """Create savepoint"""
        await self.ctx.savepoint(name)

    async def rollback_to(self, name: str):
        """Rollback to savepoint"""
        await self.ctx.rollback_to(name)


# =============================================================================
# Global Instance
# =============================================================================


_tx_manager: Optional[AgentTransactionManager] = None


async def get_transaction_manager() -> AgentTransactionManager:
    """Get global transaction manager"""
    global _tx_manager
    if _tx_manager is None:
        _tx_manager = AgentTransactionManager()
        await _tx_manager.initialize()
    return _tx_manager


# =============================================================================
# Convenience Functions
# =============================================================================


async def atomic(agent_id: str = "default"):
    """Get atomic context for transactions"""
    manager = await get_transaction_manager()
    return manager.atomic(agent_id)
