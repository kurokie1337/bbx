# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Agent Working Set Manager - Windows Memory Management Inspired

Implements working set and memory compression concepts from Windows:

1. Working Set (WS): Active context items an agent uses frequently
2. Standby List: Soft-faulted items that can be quickly restored
3. Modified List: Items pending compression/persistence
4. Compressed Store: ZRAM-like compressed context storage
5. Page File: Persistent cold storage (disk/database)

Key concepts:
- VirtualAlloc Reserve/Commit analogy for context allocation
- Soft/Hard page faults for context access
- Memory pressure responses
- Trim operations (like MmWorkingSetManager)

Working Set States:
  ACTIVE     -> In working set, immediately accessible
  STANDBY    -> Soft-faulted, in memory but not in WS
  MODIFIED   -> Changed, pending write-back
  COMPRESSED -> In compressed store
  PAGED_OUT  -> In persistent storage (cold)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)
import asyncio
import gzip
import hashlib
import json
import logging
import pickle
import threading
import time
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class PageState(Enum):
    """State of a context page."""
    ACTIVE = auto()       # In working set
    STANDBY = auto()      # Soft-faulted, easily restorable
    MODIFIED = auto()     # Changed, needs write-back
    COMPRESSED = auto()   # In compressed store
    PAGED_OUT = auto()    # On disk/persistent storage
    RESERVED = auto()     # Reserved but not committed
    FREE = auto()         # Deallocated


class MemoryPressure(Enum):
    """System memory pressure levels."""
    LOW = auto()          # Plenty of memory
    MEDIUM = auto()       # Some pressure
    HIGH = auto()         # High pressure, start trimming
    CRITICAL = auto()     # Critical, aggressive trimming


class FaultType(Enum):
    """Type of page fault."""
    SOFT = auto()   # Page in standby/modified list
    HARD = auto()   # Page in compressed store or paged out


class TrimPriority(Enum):
    """Priority for trimming."""
    BACKGROUND = auto()   # Normal background trimming
    FOREGROUND = auto()   # Trim to make room for new allocation
    URGENT = auto()       # Trim due to memory pressure


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PageMetadata:
    """Metadata for a context page."""
    id: str
    owner_id: str
    key: str
    size_bytes: int
    state: PageState = PageState.FREE
    priority: int = 5  # 0-10, higher = more important
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: Optional[datetime] = None
    compressed_size: Optional[int] = None
    checksum: Optional[str] = None
    generation: int = 0  # For MGLRU integration


@dataclass
class WorkingSetLimits:
    """Working set size limits."""
    minimum: int = 100          # Minimum items
    maximum: int = 10000        # Maximum items
    soft_limit: int = 5000      # Soft limit (trigger trimming)
    hard_limit: int = 8000      # Hard limit (block new allocations)


@dataclass
class CompressionStats:
    """Statistics for compression operations."""
    total_compressed: int = 0
    total_decompressed: int = 0
    bytes_before_compression: int = 0
    bytes_after_compression: int = 0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0

    @property
    def compression_ratio(self) -> float:
        if self.bytes_before_compression == 0:
            return 0.0
        return self.bytes_after_compression / self.bytes_before_compression


@dataclass
class WorkingSetStats:
    """Statistics for working set manager."""
    active_pages: int = 0
    standby_pages: int = 0
    modified_pages: int = 0
    compressed_pages: int = 0
    paged_out_pages: int = 0

    soft_faults: int = 0
    hard_faults: int = 0

    trims_performed: int = 0
    pages_trimmed: int = 0

    memory_used_bytes: int = 0
    compressed_bytes: int = 0

    current_pressure: MemoryPressure = MemoryPressure.LOW


@dataclass
class AllocationRequest:
    """Request to allocate context memory."""
    owner_id: str
    key: str
    size_hint: int = 0
    priority: int = 5
    reserve_only: bool = False  # True = VirtualAlloc MEM_RESERVE


@dataclass
class AllocationResult:
    """Result of allocation request."""
    success: bool
    page_id: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# Compressed Store
# =============================================================================

class CompressedStore:
    """
    ZRAM-like compressed context storage.

    Stores compressed pages in memory, trading CPU for memory.
    """

    def __init__(
        self,
        max_size_bytes: int = 100_000_000,  # 100MB
        compression_level: int = 6
    ):
        self._max_size = max_size_bytes
        self._compression_level = compression_level
        self._store: Dict[str, bytes] = {}
        self._metadata: Dict[str, PageMetadata] = {}
        self._current_size = 0
        self._lock = threading.Lock()
        self._stats = CompressionStats()

    def compress(self, page_id: str, data: Any, metadata: PageMetadata) -> bool:
        """Compress and store data."""
        start = time.time()

        try:
            # Serialize
            serialized = pickle.dumps(data)
            original_size = len(serialized)

            # Compress
            compressed = gzip.compress(serialized, compresslevel=self._compression_level)
            compressed_size = len(compressed)

            with self._lock:
                # Check space
                if self._current_size + compressed_size > self._max_size:
                    return False

                # Store
                self._store[page_id] = compressed
                metadata.compressed_size = compressed_size
                metadata.state = PageState.COMPRESSED
                self._metadata[page_id] = metadata
                self._current_size += compressed_size

            # Update stats
            self._stats.total_compressed += 1
            self._stats.bytes_before_compression += original_size
            self._stats.bytes_after_compression += compressed_size
            self._stats.compression_time_ms += (time.time() - start) * 1000

            logger.debug(
                f"Compressed page {page_id[:8]}: {original_size} -> {compressed_size} "
                f"({compressed_size / original_size * 100:.1f}%)"
            )
            return True

        except Exception as e:
            logger.error(f"Compression failed for {page_id}: {e}")
            return False

    def decompress(self, page_id: str) -> Optional[Tuple[Any, PageMetadata]]:
        """Decompress and retrieve data."""
        start = time.time()

        with self._lock:
            if page_id not in self._store:
                return None

            compressed = self._store[page_id]
            metadata = self._metadata[page_id]

        try:
            # Decompress
            serialized = gzip.decompress(compressed)

            # Deserialize
            data = pickle.loads(serialized)

            # Update stats
            self._stats.total_decompressed += 1
            self._stats.decompression_time_ms += (time.time() - start) * 1000

            return data, metadata

        except Exception as e:
            logger.error(f"Decompression failed for {page_id}: {e}")
            return None

    def remove(self, page_id: str) -> bool:
        """Remove compressed page."""
        with self._lock:
            if page_id not in self._store:
                return False

            compressed = self._store.pop(page_id)
            del self._metadata[page_id]
            self._current_size -= len(compressed)
            return True

    def get_stats(self) -> CompressionStats:
        """Get compression statistics."""
        return self._stats

    @property
    def size(self) -> int:
        """Current size of compressed store."""
        return self._current_size

    @property
    def count(self) -> int:
        """Number of compressed pages."""
        return len(self._store)


# =============================================================================
# Page File (Persistent Storage)
# =============================================================================

class PageFile(ABC):
    """Abstract base class for persistent page storage."""

    @abstractmethod
    async def write_page(self, page_id: str, data: Any, metadata: PageMetadata) -> bool:
        """Write page to persistent storage."""
        pass

    @abstractmethod
    async def read_page(self, page_id: str) -> Optional[Tuple[Any, PageMetadata]]:
        """Read page from persistent storage."""
        pass

    @abstractmethod
    async def delete_page(self, page_id: str) -> bool:
        """Delete page from persistent storage."""
        pass

    @abstractmethod
    async def list_pages(self, owner_id: Optional[str] = None) -> List[str]:
        """List page IDs in storage."""
        pass


class InMemoryPageFile(PageFile):
    """In-memory page file (for testing)."""

    def __init__(self):
        self._pages: Dict[str, Tuple[Any, PageMetadata]] = {}

    async def write_page(self, page_id: str, data: Any, metadata: PageMetadata) -> bool:
        metadata.state = PageState.PAGED_OUT
        self._pages[page_id] = (data, metadata)
        return True

    async def read_page(self, page_id: str) -> Optional[Tuple[Any, PageMetadata]]:
        return self._pages.get(page_id)

    async def delete_page(self, page_id: str) -> bool:
        if page_id in self._pages:
            del self._pages[page_id]
            return True
        return False

    async def list_pages(self, owner_id: Optional[str] = None) -> List[str]:
        if owner_id is None:
            return list(self._pages.keys())
        return [
            pid for pid, (_, meta) in self._pages.items()
            if meta.owner_id == owner_id
        ]


class FileSystemPageFile(PageFile):
    """File system-based page file."""

    def __init__(self, base_path: str):
        import os
        self._base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _page_path(self, page_id: str) -> str:
        import os
        return os.path.join(self._base_path, f"{page_id}.page")

    def _meta_path(self, page_id: str) -> str:
        import os
        return os.path.join(self._base_path, f"{page_id}.meta")

    async def write_page(self, page_id: str, data: Any, metadata: PageMetadata) -> bool:
        try:
            # Write data
            serialized = pickle.dumps(data)
            compressed = gzip.compress(serialized)

            with open(self._page_path(page_id), "wb") as f:
                f.write(compressed)

            # Write metadata
            metadata.state = PageState.PAGED_OUT
            meta_dict = {
                "id": metadata.id,
                "owner_id": metadata.owner_id,
                "key": metadata.key,
                "size_bytes": metadata.size_bytes,
                "state": metadata.state.name,
                "priority": metadata.priority,
                "access_count": metadata.access_count,
                "last_access": metadata.last_access.isoformat(),
                "created_at": metadata.created_at.isoformat(),
                "generation": metadata.generation,
            }
            with open(self._meta_path(page_id), "w") as f:
                json.dump(meta_dict, f)

            return True
        except Exception as e:
            logger.error(f"Failed to write page {page_id}: {e}")
            return False

    async def read_page(self, page_id: str) -> Optional[Tuple[Any, PageMetadata]]:
        import os
        try:
            if not os.path.exists(self._page_path(page_id)):
                return None

            # Read data
            with open(self._page_path(page_id), "rb") as f:
                compressed = f.read()
            serialized = gzip.decompress(compressed)
            data = pickle.loads(serialized)

            # Read metadata
            with open(self._meta_path(page_id), "r") as f:
                meta_dict = json.load(f)

            metadata = PageMetadata(
                id=meta_dict["id"],
                owner_id=meta_dict["owner_id"],
                key=meta_dict["key"],
                size_bytes=meta_dict["size_bytes"],
                state=PageState[meta_dict["state"]],
                priority=meta_dict.get("priority", 5),
                access_count=meta_dict.get("access_count", 0),
                last_access=datetime.fromisoformat(meta_dict["last_access"]),
                created_at=datetime.fromisoformat(meta_dict["created_at"]),
                generation=meta_dict.get("generation", 0),
            )

            return data, metadata
        except Exception as e:
            logger.error(f"Failed to read page {page_id}: {e}")
            return None

    async def delete_page(self, page_id: str) -> bool:
        import os
        try:
            if os.path.exists(self._page_path(page_id)):
                os.remove(self._page_path(page_id))
            if os.path.exists(self._meta_path(page_id)):
                os.remove(self._meta_path(page_id))
            return True
        except Exception as e:
            logger.error(f"Failed to delete page {page_id}: {e}")
            return False

    async def list_pages(self, owner_id: Optional[str] = None) -> List[str]:
        import os
        pages = []
        for fname in os.listdir(self._base_path):
            if fname.endswith(".page"):
                page_id = fname[:-5]
                if owner_id is None:
                    pages.append(page_id)
                else:
                    # Check owner
                    result = await self.read_page(page_id)
                    if result and result[1].owner_id == owner_id:
                        pages.append(page_id)
        return pages


# =============================================================================
# Working Set Manager
# =============================================================================

class WorkingSetManager:
    """
    Manages working sets for agents.

    Similar to Windows MmWorkingSetManager.
    """

    def __init__(
        self,
        limits: Optional[WorkingSetLimits] = None,
        compressed_store: Optional[CompressedStore] = None,
        page_file: Optional[PageFile] = None,
        pressure_check_interval: float = 5.0
    ):
        self._limits = limits or WorkingSetLimits()
        self._compressed_store = compressed_store or CompressedStore()
        self._page_file = page_file or InMemoryPageFile()
        self._pressure_interval = pressure_check_interval

        # Working set (active pages)
        self._working_set: Dict[str, Tuple[Any, PageMetadata]] = {}

        # Standby list (soft-faulted)
        self._standby_list: Dict[str, Tuple[Any, PageMetadata]] = {}

        # Modified list (pending write-back)
        self._modified_list: Dict[str, Tuple[Any, PageMetadata]] = {}

        # Reserved pages (VirtualAlloc MEM_RESERVE)
        self._reserved: Dict[str, PageMetadata] = {}

        # Owner index for fast lookup
        self._owner_index: Dict[str, Set[str]] = {}

        # Statistics
        self._stats = WorkingSetStats()

        # Locks
        self._lock = asyncio.Lock()

        # Background tasks
        self._running = False
        self._pressure_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the working set manager."""
        self._running = True
        self._pressure_task = asyncio.create_task(self._pressure_monitor())
        logger.info("WorkingSetManager started")

    async def stop(self) -> None:
        """Stop the working set manager."""
        self._running = False
        if self._pressure_task:
            self._pressure_task.cancel()
            try:
                await self._pressure_task
            except asyncio.CancelledError:
                pass
        logger.info("WorkingSetManager stopped")

    # =========================================================================
    # Allocation (VirtualAlloc-like)
    # =========================================================================

    async def reserve(self, request: AllocationRequest) -> AllocationResult:
        """
        Reserve context memory (like VirtualAlloc MEM_RESERVE).

        Reserves address space without committing memory.
        """
        async with self._lock:
            page_id = str(uuid.uuid4())

            metadata = PageMetadata(
                id=page_id,
                owner_id=request.owner_id,
                key=request.key,
                size_bytes=request.size_hint,
                state=PageState.RESERVED,
                priority=request.priority,
            )

            self._reserved[page_id] = metadata
            self._add_to_owner_index(request.owner_id, page_id)

            return AllocationResult(success=True, page_id=page_id)

    async def commit(
        self,
        page_id: str,
        data: Any,
        size_bytes: Optional[int] = None
    ) -> AllocationResult:
        """
        Commit reserved memory (like VirtualAlloc MEM_COMMIT).

        Actually allocates the memory and stores data.
        """
        async with self._lock:
            # Check if reserved
            if page_id not in self._reserved:
                return AllocationResult(
                    success=False,
                    error="Page not reserved or already committed"
                )

            # Check limits
            current_count = len(self._working_set)
            if current_count >= self._limits.hard_limit:
                return AllocationResult(
                    success=False,
                    error="Hard limit reached"
                )

            metadata = self._reserved.pop(page_id)
            metadata.state = PageState.ACTIVE

            if size_bytes:
                metadata.size_bytes = size_bytes

            self._working_set[page_id] = (data, metadata)
            self._stats.active_pages += 1
            self._stats.memory_used_bytes += metadata.size_bytes

            return AllocationResult(success=True, page_id=page_id)

    async def allocate(
        self,
        request: AllocationRequest
    ) -> AllocationResult:
        """
        Allocate context memory (reserve + commit in one step).
        """
        async with self._lock:
            # Check limits
            current_count = len(self._working_set)
            if current_count >= self._limits.hard_limit:
                # Try to make room
                await self._trim_internal(TrimPriority.FOREGROUND, 10)

                if len(self._working_set) >= self._limits.hard_limit:
                    return AllocationResult(
                        success=False,
                        error="Working set hard limit reached"
                    )

            page_id = str(uuid.uuid4())

            metadata = PageMetadata(
                id=page_id,
                owner_id=request.owner_id,
                key=request.key,
                size_bytes=request.size_hint,
                state=PageState.ACTIVE,
                priority=request.priority,
            )

            self._working_set[page_id] = (None, metadata)
            self._add_to_owner_index(request.owner_id, page_id)
            self._stats.active_pages += 1

            return AllocationResult(success=True, page_id=page_id)

    async def free(self, page_id: str) -> bool:
        """Free allocated memory."""
        async with self._lock:
            return await self._free_internal(page_id)

    async def _free_internal(self, page_id: str) -> bool:
        """Internal free without lock."""
        # Check all lists
        if page_id in self._working_set:
            data, metadata = self._working_set.pop(page_id)
            self._stats.active_pages -= 1
            self._stats.memory_used_bytes -= metadata.size_bytes
            self._remove_from_owner_index(metadata.owner_id, page_id)
            return True

        if page_id in self._standby_list:
            data, metadata = self._standby_list.pop(page_id)
            self._stats.standby_pages -= 1
            self._remove_from_owner_index(metadata.owner_id, page_id)
            return True

        if page_id in self._modified_list:
            data, metadata = self._modified_list.pop(page_id)
            self._stats.modified_pages -= 1
            self._remove_from_owner_index(metadata.owner_id, page_id)
            return True

        if page_id in self._reserved:
            metadata = self._reserved.pop(page_id)
            self._remove_from_owner_index(metadata.owner_id, page_id)
            return True

        # Try compressed store
        if self._compressed_store.remove(page_id):
            self._stats.compressed_pages -= 1
            return True

        # Try page file
        return await self._page_file.delete_page(page_id)

    # =========================================================================
    # Read/Write Operations
    # =========================================================================

    async def read(self, page_id: str) -> Optional[Any]:
        """
        Read page data.

        May cause page fault if not in working set.
        """
        async with self._lock:
            # Check working set
            if page_id in self._working_set:
                data, metadata = self._working_set[page_id]
                metadata.access_count += 1
                metadata.last_access = datetime.utcnow()
                return data

            # Soft fault - check standby list
            if page_id in self._standby_list:
                self._stats.soft_faults += 1
                data, metadata = self._standby_list.pop(page_id)
                self._stats.standby_pages -= 1

                metadata.state = PageState.ACTIVE
                metadata.access_count += 1
                metadata.last_access = datetime.utcnow()

                self._working_set[page_id] = (data, metadata)
                self._stats.active_pages += 1
                self._stats.memory_used_bytes += metadata.size_bytes

                return data

            # Soft fault - check modified list
            if page_id in self._modified_list:
                self._stats.soft_faults += 1
                data, metadata = self._modified_list.pop(page_id)
                self._stats.modified_pages -= 1

                metadata.state = PageState.ACTIVE
                metadata.access_count += 1
                metadata.last_access = datetime.utcnow()

                self._working_set[page_id] = (data, metadata)
                self._stats.active_pages += 1
                self._stats.memory_used_bytes += metadata.size_bytes

                return data

            # Hard fault - check compressed store
            result = self._compressed_store.decompress(page_id)
            if result:
                self._stats.hard_faults += 1
                data, metadata = result
                self._compressed_store.remove(page_id)
                self._stats.compressed_pages -= 1

                metadata.state = PageState.ACTIVE
                metadata.access_count += 1
                metadata.last_access = datetime.utcnow()

                self._working_set[page_id] = (data, metadata)
                self._stats.active_pages += 1
                self._stats.memory_used_bytes += metadata.size_bytes

                return data

            # Hard fault - check page file
            result = await self._page_file.read_page(page_id)
            if result:
                self._stats.hard_faults += 1
                data, metadata = result
                await self._page_file.delete_page(page_id)
                self._stats.paged_out_pages -= 1

                metadata.state = PageState.ACTIVE
                metadata.access_count += 1
                metadata.last_access = datetime.utcnow()

                self._working_set[page_id] = (data, metadata)
                self._stats.active_pages += 1
                self._stats.memory_used_bytes += metadata.size_bytes

                return data

            return None

    async def write(self, page_id: str, data: Any, size_bytes: int = 0) -> bool:
        """Write data to page."""
        async with self._lock:
            # Check working set
            if page_id in self._working_set:
                _, metadata = self._working_set[page_id]
                metadata.modified_at = datetime.utcnow()
                metadata.last_access = datetime.utcnow()
                metadata.access_count += 1
                if size_bytes:
                    self._stats.memory_used_bytes -= metadata.size_bytes
                    metadata.size_bytes = size_bytes
                    self._stats.memory_used_bytes += size_bytes
                self._working_set[page_id] = (data, metadata)
                return True

            # Fault in from other lists
            await self.read(page_id)
            if page_id in self._working_set:
                return await self.write(page_id, data, size_bytes)

            return False

    # =========================================================================
    # Trimming (MmWorkingSetManager)
    # =========================================================================

    async def trim(
        self,
        priority: TrimPriority = TrimPriority.BACKGROUND,
        count: int = 100
    ) -> int:
        """
        Trim working set.

        Moves least recently used pages to standby list.
        """
        async with self._lock:
            return await self._trim_internal(priority, count)

    async def _trim_internal(
        self,
        priority: TrimPriority,
        count: int
    ) -> int:
        """Internal trim without lock."""
        if not self._working_set:
            return 0

        # Sort by priority (low first) then by last access (oldest first)
        candidates = sorted(
            self._working_set.items(),
            key=lambda x: (x[1][1].priority, x[1][1].last_access)
        )

        trimmed = 0
        for page_id, (data, metadata) in candidates[:count]:
            # Don't trim if under minimum
            if len(self._working_set) <= self._limits.minimum:
                break

            # Move to standby or modified
            del self._working_set[page_id]
            self._stats.active_pages -= 1
            self._stats.memory_used_bytes -= metadata.size_bytes

            if metadata.modified_at:
                metadata.state = PageState.MODIFIED
                self._modified_list[page_id] = (data, metadata)
                self._stats.modified_pages += 1
            else:
                metadata.state = PageState.STANDBY
                self._standby_list[page_id] = (data, metadata)
                self._stats.standby_pages += 1

            trimmed += 1

        self._stats.trims_performed += 1
        self._stats.pages_trimmed += trimmed

        logger.debug(f"Trimmed {trimmed} pages (priority={priority.name})")
        return trimmed

    async def compress_standby(self, count: int = 50) -> int:
        """
        Compress pages from standby list.

        Moves to compressed store.
        """
        async with self._lock:
            if not self._standby_list:
                return 0

            # Sort by last access (oldest first)
            candidates = sorted(
                self._standby_list.items(),
                key=lambda x: x[1][1].last_access
            )

            compressed = 0
            for page_id, (data, metadata) in candidates[:count]:
                if self._compressed_store.compress(page_id, data, metadata):
                    del self._standby_list[page_id]
                    self._stats.standby_pages -= 1
                    self._stats.compressed_pages += 1
                    self._stats.compressed_bytes += metadata.compressed_size or 0
                    compressed += 1

            logger.debug(f"Compressed {compressed} pages")
            return compressed

    async def page_out(self, count: int = 50) -> int:
        """
        Page out compressed pages to persistent storage.
        """
        # This would need access to compressed store internals
        # For now, page out from modified list
        async with self._lock:
            if not self._modified_list:
                return 0

            candidates = sorted(
                self._modified_list.items(),
                key=lambda x: x[1][1].last_access
            )

            paged_out = 0
            for page_id, (data, metadata) in candidates[:count]:
                if await self._page_file.write_page(page_id, data, metadata):
                    del self._modified_list[page_id]
                    self._stats.modified_pages -= 1
                    self._stats.paged_out_pages += 1
                    paged_out += 1

            logger.debug(f"Paged out {paged_out} pages")
            return paged_out

    # =========================================================================
    # Pressure Monitoring
    # =========================================================================

    async def _pressure_monitor(self) -> None:
        """Background task to monitor memory pressure."""
        while self._running:
            try:
                await asyncio.sleep(self._pressure_interval)

                pressure = self._calculate_pressure()
                self._stats.current_pressure = pressure

                if pressure == MemoryPressure.HIGH:
                    await self.trim(TrimPriority.BACKGROUND, 50)
                    await self.compress_standby(25)
                elif pressure == MemoryPressure.CRITICAL:
                    await self.trim(TrimPriority.URGENT, 100)
                    await self.compress_standby(50)
                    await self.page_out(25)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pressure monitor error: {e}")

    def _calculate_pressure(self) -> MemoryPressure:
        """Calculate current memory pressure."""
        ws_count = len(self._working_set)

        if ws_count >= self._limits.hard_limit * 0.95:
            return MemoryPressure.CRITICAL
        elif ws_count >= self._limits.soft_limit:
            return MemoryPressure.HIGH
        elif ws_count >= self._limits.soft_limit * 0.7:
            return MemoryPressure.MEDIUM
        else:
            return MemoryPressure.LOW

    # =========================================================================
    # Owner Operations
    # =========================================================================

    def _add_to_owner_index(self, owner_id: str, page_id: str) -> None:
        """Add page to owner index."""
        if owner_id not in self._owner_index:
            self._owner_index[owner_id] = set()
        self._owner_index[owner_id].add(page_id)

    def _remove_from_owner_index(self, owner_id: str, page_id: str) -> None:
        """Remove page from owner index."""
        if owner_id in self._owner_index:
            self._owner_index[owner_id].discard(page_id)
            if not self._owner_index[owner_id]:
                del self._owner_index[owner_id]

    async def get_owner_pages(self, owner_id: str) -> List[str]:
        """Get all page IDs for an owner."""
        async with self._lock:
            if owner_id not in self._owner_index:
                return []
            return list(self._owner_index[owner_id])

    async def free_owner_pages(self, owner_id: str) -> int:
        """Free all pages for an owner."""
        async with self._lock:
            if owner_id not in self._owner_index:
                return 0

            page_ids = list(self._owner_index[owner_id])
            freed = 0
            for page_id in page_ids:
                if await self._free_internal(page_id):
                    freed += 1

            return freed

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> WorkingSetStats:
        """Get working set statistics."""
        return self._stats

    def get_compression_stats(self) -> CompressionStats:
        """Get compression statistics."""
        return self._compressed_store.get_stats()

    # =========================================================================
    # Queries
    # =========================================================================

    async def get_page_metadata(self, page_id: str) -> Optional[PageMetadata]:
        """Get metadata for a page."""
        async with self._lock:
            if page_id in self._working_set:
                return self._working_set[page_id][1]
            if page_id in self._standby_list:
                return self._standby_list[page_id][1]
            if page_id in self._modified_list:
                return self._modified_list[page_id][1]
            if page_id in self._reserved:
                return self._reserved[page_id]
            return None

    async def list_pages(
        self,
        owner_id: Optional[str] = None,
        state: Optional[PageState] = None
    ) -> List[PageMetadata]:
        """List pages with optional filtering."""
        async with self._lock:
            pages = []

            # Collect from all lists
            for page_id, (_, metadata) in self._working_set.items():
                if owner_id and metadata.owner_id != owner_id:
                    continue
                if state and metadata.state != state:
                    continue
                pages.append(metadata)

            for page_id, (_, metadata) in self._standby_list.items():
                if owner_id and metadata.owner_id != owner_id:
                    continue
                if state and metadata.state != state:
                    continue
                pages.append(metadata)

            for page_id, (_, metadata) in self._modified_list.items():
                if owner_id and metadata.owner_id != owner_id:
                    continue
                if state and metadata.state != state:
                    continue
                pages.append(metadata)

            for page_id, metadata in self._reserved.items():
                if owner_id and metadata.owner_id != owner_id:
                    continue
                if state and metadata.state != state:
                    continue
                pages.append(metadata)

            return pages


# =============================================================================
# Agent Working Set
# =============================================================================

class AgentWorkingSet:
    """
    Per-agent working set wrapper.

    Provides a simplified interface for agents to manage their context.
    """

    def __init__(
        self,
        agent_id: str,
        manager: WorkingSetManager,
        limits: Optional[WorkingSetLimits] = None
    ):
        self._agent_id = agent_id
        self._manager = manager
        self._limits = limits or WorkingSetLimits(
            minimum=10,
            maximum=1000,
            soft_limit=500,
            hard_limit=800
        )
        self._key_to_page: Dict[str, str] = {}

    async def get(self, key: str) -> Optional[Any]:
        """Get context by key."""
        page_id = self._key_to_page.get(key)
        if page_id:
            return await self._manager.read(page_id)
        return None

    async def set(self, key: str, value: Any, priority: int = 5) -> bool:
        """Set context by key."""
        if key in self._key_to_page:
            page_id = self._key_to_page[key]
            return await self._manager.write(page_id, value)
        else:
            # Allocate new page
            result = await self._manager.allocate(AllocationRequest(
                owner_id=self._agent_id,
                key=key,
                priority=priority,
            ))
            if result.success and result.page_id:
                self._key_to_page[key] = result.page_id
                return await self._manager.write(result.page_id, value)
            return False

    async def delete(self, key: str) -> bool:
        """Delete context by key."""
        page_id = self._key_to_page.pop(key, None)
        if page_id:
            return await self._manager.free(page_id)
        return False

    async def keys(self) -> List[str]:
        """Get all keys."""
        return list(self._key_to_page.keys())

    async def clear(self) -> int:
        """Clear all context."""
        return await self._manager.free_owner_pages(self._agent_id)


# =============================================================================
# Singleton Access
# =============================================================================

_working_set_manager: Optional[WorkingSetManager] = None


def get_working_set_manager() -> WorkingSetManager:
    """Get or create the global working set manager."""
    global _working_set_manager
    if _working_set_manager is None:
        _working_set_manager = WorkingSetManager()
    return _working_set_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "PageState",
    "MemoryPressure",
    "FaultType",
    "TrimPriority",

    # Data classes
    "PageMetadata",
    "WorkingSetLimits",
    "CompressionStats",
    "WorkingSetStats",
    "AllocationRequest",
    "AllocationResult",

    # Components
    "CompressedStore",
    "PageFile",
    "InMemoryPageFile",
    "FileSystemPageFile",

    # Manager
    "WorkingSetManager",
    "AgentWorkingSet",

    # Singleton
    "get_working_set_manager",
]
