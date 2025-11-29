# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX StateSnapshots - XFS Reflink-inspired Copy-on-Write State Management

Provides efficient state versioning and branching for AI agents:
- Copy-on-Write state snapshots (like XFS reflink)
- Transactional state modifications
- Multi-branch state trees (like git)
- Time-travel debugging
- Efficient storage with deduplication

Inspired by XFS Reflink and Btrfs:
- CoW semantics → Only changed data is copied
- Reflinks → Multiple snapshots share unchanged data
- Snapshots → Point-in-time captures of state
- Subvolumes → Agent state branches

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  StateSnapshots Engine                                      │
    │  ├─ SnapshotStore: CoW storage with deduplication          │
    │  ├─ BranchManager: Multi-branch state trees                │
    │  ├─ TransactionManager: ACID state transactions            │
    │  └─ TimeTravel: Debug/replay capabilities                  │
    └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import pickle
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union
import threading
import uuid

logger = logging.getLogger("bbx.state_snapshots")

T = TypeVar('T')


# =============================================================================
# Enums
# =============================================================================

class SnapshotType(Enum):
    """Types of snapshots"""
    MANUAL = "manual"           # User-created snapshot
    AUTO = "auto"               # Auto-created (e.g., before risky op)
    CHECKPOINT = "checkpoint"   # Periodic checkpoint
    BRANCH = "branch"           # Branch point
    SHUTDOWN = "shutdown"       # Created on system shutdown


class TransactionState(Enum):
    """States of a transaction"""
    PENDING = auto()
    ACTIVE = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()
    FAILED = auto()


class MergeStrategy(Enum):
    """Strategies for merging branches"""
    OURS = "ours"           # Keep our version
    THEIRS = "theirs"       # Keep their version
    UNION = "union"         # Union of both
    MANUAL = "manual"       # Manual resolution required


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DataBlock:
    """
    A block of data in the CoW store.

    Uses content-addressable storage - hash is the key.
    """
    hash: str
    data: bytes
    ref_count: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    size: int = 0

    def __post_init__(self):
        if not self.size:
            self.size = len(self.data)


@dataclass
class Snapshot:
    """A point-in-time snapshot of agent state"""
    id: str
    agent_id: str
    parent_id: Optional[str]
    branch: str
    snapshot_type: SnapshotType
    created_at: datetime
    description: str = ""

    # State reference (content-addressed)
    state_hash: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For efficient diff
    changed_keys: Set[str] = field(default_factory=set)


@dataclass
class Branch:
    """A branch in the state tree"""
    name: str
    agent_id: str
    head_snapshot_id: str
    created_at: datetime = field(default_factory=datetime.now)
    parent_branch: Optional[str] = None
    fork_point: Optional[str] = None  # Snapshot ID where branched
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transaction:
    """A state transaction"""
    id: str
    agent_id: str
    branch: str
    state: TransactionState = TransactionState.PENDING
    started_at: Optional[datetime] = None
    committed_at: Optional[datetime] = None
    pre_snapshot_id: Optional[str] = None
    post_snapshot_id: Optional[str] = None
    changes: Dict[str, Any] = field(default_factory=dict)
    rollback_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SnapshotConfig:
    """Configuration for StateSnapshots"""
    enable_compression: bool = True
    compression_level: int = 6
    max_snapshots_per_agent: int = 100
    auto_snapshot_interval_sec: float = 300.0
    gc_interval_sec: float = 3600.0
    storage_path: Optional[Path] = None
    enable_persistence: bool = True


# =============================================================================
# Copy-on-Write Store
# =============================================================================

class CoWStore:
    """
    Copy-on-Write storage with content-addressable blocks.

    Like XFS reflink, multiple snapshots share unchanged data blocks.
    """

    def __init__(self, config: SnapshotConfig):
        self.config = config
        self._blocks: Dict[str, DataBlock] = {}
        self._lock = threading.RLock()

    def _hash_data(self, data: bytes) -> str:
        """Compute hash of data"""
        return hashlib.sha256(data).hexdigest()

    def _serialize(self, obj: Any) -> bytes:
        """Serialize object to bytes"""
        data = pickle.dumps(obj)
        if self.config.enable_compression:
            data = zlib.compress(data, self.config.compression_level)
        return data

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to object"""
        if self.config.enable_compression:
            data = zlib.decompress(data)
        return pickle.loads(data)

    def store(self, obj: Any) -> str:
        """
        Store object and return content hash.

        If identical data exists, just increment ref count.
        """
        data = self._serialize(obj)
        hash_key = self._hash_data(data)

        with self._lock:
            if hash_key in self._blocks:
                self._blocks[hash_key].ref_count += 1
            else:
                self._blocks[hash_key] = DataBlock(
                    hash=hash_key,
                    data=data,
                    ref_count=1,
                )

        return hash_key

    def retrieve(self, hash_key: str) -> Optional[Any]:
        """Retrieve object by hash"""
        with self._lock:
            block = self._blocks.get(hash_key)
            if block:
                return self._deserialize(block.data)
        return None

    def release(self, hash_key: str):
        """Release a reference to a block"""
        with self._lock:
            if hash_key in self._blocks:
                self._blocks[hash_key].ref_count -= 1
                if self._blocks[hash_key].ref_count <= 0:
                    del self._blocks[hash_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self._lock:
            total_size = sum(b.size for b in self._blocks.values())
            total_refs = sum(b.ref_count for b in self._blocks.values())
            return {
                "blocks_count": len(self._blocks),
                "total_size_bytes": total_size,
                "total_refs": total_refs,
                "dedup_ratio": total_refs / max(len(self._blocks), 1),
            }


# =============================================================================
# State Manager
# =============================================================================

class AgentState:
    """
    Manages state for a single agent with CoW semantics.

    Provides dict-like interface with automatic change tracking.
    """

    def __init__(self, agent_id: str, cow_store: CoWStore):
        self.agent_id = agent_id
        self._cow_store = cow_store
        self._data: Dict[str, Any] = {}
        self._dirty_keys: Set[str] = set()
        self._lock = threading.RLock()

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)

    def __setitem__(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value
            self._dirty_keys.add(key)

    def __delitem__(self, key: str):
        with self._lock:
            if key in self._data:
                del self._data[key]
                self._dirty_keys.add(key)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self[key] = value

    def delete(self, key: str):
        del self[key]

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._data.keys())

    def values(self) -> List[Any]:
        with self._lock:
            return list(self._data.values())

    def items(self) -> List[tuple]:
        with self._lock:
            return list(self._data.items())

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def from_dict(self, data: Dict[str, Any]):
        with self._lock:
            self._data = copy.deepcopy(data)
            self._dirty_keys.clear()

    def get_dirty_keys(self) -> Set[str]:
        """Get keys that have been modified"""
        with self._lock:
            return self._dirty_keys.copy()

    def clear_dirty(self):
        """Clear dirty tracking"""
        with self._lock:
            self._dirty_keys.clear()

    def snapshot(self) -> str:
        """Create a snapshot and return hash"""
        with self._lock:
            return self._cow_store.store(self._data)

    def restore(self, hash_key: str) -> bool:
        """Restore state from snapshot hash"""
        data = self._cow_store.retrieve(hash_key)
        if data is not None:
            with self._lock:
                self._data = data
                self._dirty_keys.clear()
            return True
        return False


# =============================================================================
# Branch Manager
# =============================================================================

class BranchManager:
    """
    Manages state branches for agents.

    Like git branches, allows parallel state evolution.
    """

    def __init__(self):
        self._branches: Dict[str, Dict[str, Branch]] = {}  # agent_id -> {branch_name -> Branch}
        self._lock = threading.RLock()

    def create_branch(
        self,
        agent_id: str,
        branch_name: str,
        head_snapshot_id: str,
        parent_branch: Optional[str] = None,
        fork_point: Optional[str] = None,
    ) -> Branch:
        """Create a new branch"""
        with self._lock:
            if agent_id not in self._branches:
                self._branches[agent_id] = {}

            branch = Branch(
                name=branch_name,
                agent_id=agent_id,
                head_snapshot_id=head_snapshot_id,
                parent_branch=parent_branch,
                fork_point=fork_point,
            )
            self._branches[agent_id][branch_name] = branch
            return branch

    def get_branch(self, agent_id: str, branch_name: str) -> Optional[Branch]:
        """Get a branch"""
        with self._lock:
            return self._branches.get(agent_id, {}).get(branch_name)

    def list_branches(self, agent_id: str) -> List[Branch]:
        """List all branches for an agent"""
        with self._lock:
            return list(self._branches.get(agent_id, {}).values())

    def update_head(self, agent_id: str, branch_name: str, snapshot_id: str):
        """Update branch head"""
        with self._lock:
            if branch_name in self._branches.get(agent_id, {}):
                self._branches[agent_id][branch_name].head_snapshot_id = snapshot_id

    def delete_branch(self, agent_id: str, branch_name: str) -> bool:
        """Delete a branch"""
        with self._lock:
            if branch_name in self._branches.get(agent_id, {}):
                del self._branches[agent_id][branch_name]
                return True
            return False


# =============================================================================
# Transaction Manager
# =============================================================================

class TransactionManager:
    """
    Manages state transactions with ACID properties.

    Provides begin/commit/rollback semantics.
    """

    def __init__(self, snapshot_engine: StateSnapshotEngine):
        self._engine = snapshot_engine
        self._active_txns: Dict[str, Transaction] = {}
        self._lock = threading.RLock()

    def begin(self, agent_id: str, branch: str = "main") -> Transaction:
        """Begin a new transaction"""
        txn_id = str(uuid.uuid4())

        with self._lock:
            # Create pre-snapshot
            pre_snapshot_id = self._engine.create_snapshot(
                agent_id,
                branch,
                SnapshotType.AUTO,
                "Transaction begin",
            )

            txn = Transaction(
                id=txn_id,
                agent_id=agent_id,
                branch=branch,
                state=TransactionState.ACTIVE,
                started_at=datetime.now(),
                pre_snapshot_id=pre_snapshot_id,
            )
            self._active_txns[txn_id] = txn
            return txn

    def commit(self, txn_id: str) -> bool:
        """Commit a transaction"""
        with self._lock:
            txn = self._active_txns.get(txn_id)
            if not txn or txn.state != TransactionState.ACTIVE:
                return False

            # Create post-snapshot
            post_snapshot_id = self._engine.create_snapshot(
                txn.agent_id,
                txn.branch,
                SnapshotType.AUTO,
                "Transaction commit",
            )

            txn.post_snapshot_id = post_snapshot_id
            txn.state = TransactionState.COMMITTED
            txn.committed_at = datetime.now()

            del self._active_txns[txn_id]
            return True

    def rollback(self, txn_id: str) -> bool:
        """Rollback a transaction"""
        with self._lock:
            txn = self._active_txns.get(txn_id)
            if not txn or txn.state != TransactionState.ACTIVE:
                return False

            # Restore pre-snapshot
            if txn.pre_snapshot_id:
                self._engine.restore_snapshot(txn.agent_id, txn.pre_snapshot_id)

            txn.state = TransactionState.ROLLED_BACK
            del self._active_txns[txn_id]
            return True

    def get_active_transactions(self, agent_id: Optional[str] = None) -> List[Transaction]:
        """Get active transactions"""
        with self._lock:
            txns = list(self._active_txns.values())
            if agent_id:
                txns = [t for t in txns if t.agent_id == agent_id]
            return txns


# =============================================================================
# State Snapshot Engine
# =============================================================================

class StateSnapshotEngine:
    """
    Main engine for state snapshots.

    Coordinates CoW storage, branching, and transactions.
    """

    def __init__(self, config: Optional[SnapshotConfig] = None):
        self.config = config or SnapshotConfig()
        self._cow_store = CoWStore(self.config)
        self._branch_manager = BranchManager()
        self._snapshots: Dict[str, Snapshot] = {}
        self._agent_states: Dict[str, AgentState] = {}
        self._current_branch: Dict[str, str] = {}  # agent_id -> branch_name
        self._lock = threading.RLock()
        self._transaction_manager: Optional[TransactionManager] = None

    @property
    def transactions(self) -> TransactionManager:
        """Get transaction manager"""
        if self._transaction_manager is None:
            self._transaction_manager = TransactionManager(self)
        return self._transaction_manager

    def get_state(self, agent_id: str) -> AgentState:
        """Get or create state for an agent"""
        with self._lock:
            if agent_id not in self._agent_states:
                self._agent_states[agent_id] = AgentState(agent_id, self._cow_store)
                # Create main branch
                self._current_branch[agent_id] = "main"
            return self._agent_states[agent_id]

    def set_state(self, agent_id: str, key: str, value: Any):
        """Set a state value for an agent"""
        state = self.get_state(agent_id)
        state.set(key, value)

    def create_snapshot(
        self,
        agent_id: str,
        branch: Optional[str] = None,
        snapshot_type: SnapshotType = SnapshotType.MANUAL,
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> str:
        """Create a new snapshot"""
        with self._lock:
            state = self.get_state(agent_id)
            branch = branch or self._current_branch.get(agent_id, "main")

            # Get parent snapshot
            branch_obj = self._branch_manager.get_branch(agent_id, branch)
            parent_id = branch_obj.head_snapshot_id if branch_obj else None

            # Store state
            state_hash = state.snapshot()

            # Create snapshot record
            snapshot_id = str(uuid.uuid4())
            snapshot = Snapshot(
                id=snapshot_id,
                agent_id=agent_id,
                parent_id=parent_id,
                branch=branch,
                snapshot_type=snapshot_type,
                created_at=datetime.now(),
                description=description,
                state_hash=state_hash,
                changed_keys=state.get_dirty_keys(),
                metadata=metadata or {},
            )

            self._snapshots[snapshot_id] = snapshot
            state.clear_dirty()

            # Update branch head
            if branch_obj:
                self._branch_manager.update_head(agent_id, branch, snapshot_id)
            else:
                self._branch_manager.create_branch(
                    agent_id, branch, snapshot_id
                )

            logger.debug(f"Created snapshot {snapshot_id[:8]} for {agent_id}/{branch}")
            return snapshot_id

    def restore_snapshot(self, agent_id: str, snapshot_id: str) -> bool:
        """Restore state from a snapshot"""
        with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot or snapshot.agent_id != agent_id:
                return False

            state = self.get_state(agent_id)
            success = state.restore(snapshot.state_hash)

            if success:
                self._current_branch[agent_id] = snapshot.branch
                logger.debug(f"Restored {agent_id} to snapshot {snapshot_id[:8]}")

            return success

    def fork_branch(
        self,
        agent_id: str,
        new_branch: str,
        from_snapshot: Optional[str] = None,
    ) -> Branch:
        """Fork a new branch from current or specified snapshot"""
        with self._lock:
            current_branch = self._current_branch.get(agent_id, "main")

            if from_snapshot:
                snapshot_id = from_snapshot
            else:
                # Use current head
                branch_obj = self._branch_manager.get_branch(agent_id, current_branch)
                snapshot_id = branch_obj.head_snapshot_id if branch_obj else None

                if not snapshot_id:
                    # Create initial snapshot
                    snapshot_id = self.create_snapshot(
                        agent_id, current_branch,
                        SnapshotType.BRANCH,
                        f"Branch point for {new_branch}"
                    )

            return self._branch_manager.create_branch(
                agent_id,
                new_branch,
                snapshot_id,
                parent_branch=current_branch,
                fork_point=snapshot_id,
            )

    def switch_branch(self, agent_id: str, branch_name: str) -> bool:
        """Switch to a different branch"""
        with self._lock:
            branch = self._branch_manager.get_branch(agent_id, branch_name)
            if not branch:
                return False

            # Restore to branch head
            success = self.restore_snapshot(agent_id, branch.head_snapshot_id)
            if success:
                self._current_branch[agent_id] = branch_name
            return success

    def get_snapshot(self, snapshot_id: str) -> Optional[Snapshot]:
        """Get snapshot by ID"""
        return self._snapshots.get(snapshot_id)

    def list_snapshots(
        self,
        agent_id: str,
        branch: Optional[str] = None,
        limit: int = 50,
    ) -> List[Snapshot]:
        """List snapshots for an agent"""
        with self._lock:
            snapshots = [
                s for s in self._snapshots.values()
                if s.agent_id == agent_id
                and (branch is None or s.branch == branch)
            ]
            snapshots.sort(key=lambda s: s.created_at, reverse=True)
            return snapshots[:limit]

    def get_history(
        self,
        agent_id: str,
        snapshot_id: str,
        limit: int = 20,
    ) -> List[Snapshot]:
        """Get snapshot history (parent chain)"""
        history = []
        current_id = snapshot_id

        while current_id and len(history) < limit:
            snapshot = self._snapshots.get(current_id)
            if not snapshot:
                break
            history.append(snapshot)
            current_id = snapshot.parent_id

        return history

    def diff_snapshots(
        self,
        snapshot_id_1: str,
        snapshot_id_2: str,
    ) -> Dict[str, Any]:
        """Compute diff between two snapshots"""
        s1 = self._snapshots.get(snapshot_id_1)
        s2 = self._snapshots.get(snapshot_id_2)

        if not s1 or not s2:
            return {"error": "Snapshot not found"}

        data1 = self._cow_store.retrieve(s1.state_hash) or {}
        data2 = self._cow_store.retrieve(s2.state_hash) or {}

        added = {k: data2[k] for k in data2 if k not in data1}
        removed = {k: data1[k] for k in data1 if k not in data2}
        changed = {
            k: {"old": data1[k], "new": data2[k]}
            for k in data1
            if k in data2 and data1[k] != data2[k]
        }

        return {
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def merge_branches(
        self,
        agent_id: str,
        source_branch: str,
        target_branch: str,
        strategy: MergeStrategy = MergeStrategy.THEIRS,
    ) -> Optional[str]:
        """Merge source branch into target branch"""
        with self._lock:
            source = self._branch_manager.get_branch(agent_id, source_branch)
            target = self._branch_manager.get_branch(agent_id, target_branch)

            if not source or not target:
                return None

            # Get states
            source_data = self._cow_store.retrieve(
                self._snapshots[source.head_snapshot_id].state_hash
            ) or {}
            target_data = self._cow_store.retrieve(
                self._snapshots[target.head_snapshot_id].state_hash
            ) or {}

            # Merge based on strategy
            if strategy == MergeStrategy.OURS:
                merged = target_data.copy()
            elif strategy == MergeStrategy.THEIRS:
                merged = target_data.copy()
                merged.update(source_data)
            elif strategy == MergeStrategy.UNION:
                merged = target_data.copy()
                for k, v in source_data.items():
                    if k not in merged:
                        merged[k] = v
            else:
                # MANUAL - return None, user must resolve
                return None

            # Update target state
            state = self.get_state(agent_id)
            state.from_dict(merged)

            # Create merge snapshot
            snapshot_id = self.create_snapshot(
                agent_id,
                target_branch,
                SnapshotType.MANUAL,
                f"Merge {source_branch} into {target_branch}",
                {"merge_source": source_branch, "strategy": strategy.value},
            )

            return snapshot_id

    def gc(self, agent_id: Optional[str] = None):
        """Garbage collect unreferenced snapshots"""
        with self._lock:
            # Find referenced snapshots
            referenced = set()

            for branch_dict in self._branch_manager._branches.values():
                for branch in branch_dict.values():
                    if agent_id and branch.agent_id != agent_id:
                        continue
                    # Walk history from head
                    current = branch.head_snapshot_id
                    while current:
                        referenced.add(current)
                        snap = self._snapshots.get(current)
                        current = snap.parent_id if snap else None

            # Remove unreferenced
            to_remove = [
                sid for sid in self._snapshots
                if sid not in referenced
                and (agent_id is None or self._snapshots[sid].agent_id == agent_id)
            ]

            for sid in to_remove:
                snap = self._snapshots[sid]
                self._cow_store.release(snap.state_hash)
                del self._snapshots[sid]

            logger.info(f"GC removed {len(to_remove)} snapshots")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self._lock:
            return {
                "total_snapshots": len(self._snapshots),
                "agents_count": len(self._agent_states),
                "cow_store": self._cow_store.get_stats(),
                "branches": sum(
                    len(branches)
                    for branches in self._branch_manager._branches.values()
                ),
            }


# =============================================================================
# Context Manager for Transactions
# =============================================================================

class StateTransaction:
    """Context manager for state transactions"""

    def __init__(self, engine: StateSnapshotEngine, agent_id: str, branch: str = "main"):
        self._engine = engine
        self._agent_id = agent_id
        self._branch = branch
        self._txn: Optional[Transaction] = None

    async def __aenter__(self) -> AgentState:
        self._txn = self._engine.transactions.begin(self._agent_id, self._branch)
        return self._engine.get_state(self._agent_id)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._engine.transactions.rollback(self._txn.id)
            return False
        self._engine.transactions.commit(self._txn.id)
        return True


# =============================================================================
# Global Instance
# =============================================================================

_snapshot_engine: Optional[StateSnapshotEngine] = None


def get_snapshot_engine() -> StateSnapshotEngine:
    """Get global StateSnapshotEngine instance"""
    global _snapshot_engine
    if _snapshot_engine is None:
        _snapshot_engine = StateSnapshotEngine()
    return _snapshot_engine
