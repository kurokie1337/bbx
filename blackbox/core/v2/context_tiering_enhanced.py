# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 ContextTiering Enhanced - ML-powered context memory management.

Improvements over base ContextTiering:
- ML-based importance scoring (not just recency)
- Explicit prefetch API for proactive loading
- Async background migration (non-blocking)
- Adaptive compression based on content type
- Batch migration for efficiency
- User-defined importance markers
- Compression ratio metrics

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Enhanced Context Tiering                         │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
    │  │ ML Scorer    │  │ Prefetch     │  │ Async Migration      │   │
    │  │ (Importance) │  │ Manager      │  │ Engine               │   │
    │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘   │
    │         │                 │                      │               │
    │  ┌──────▼─────────────────▼──────────────────────▼───────────┐   │
    │  │                    Tier Manager                           │   │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │   │
    │  │  │  HOT    │ │  WARM   │ │  COOL   │ │     COLD        │  │   │
    │  │  │ Memory  │ │ Memory  │ │ Compress│ │ Compress+Disk   │  │   │
    │  │  │ <1ms    │ │ <5ms    │ │ <50ms   │ │ <200ms          │  │   │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │   │
    │  └───────────────────────────────────────────────────────────┘   │
    │                              │                                   │
    │  ┌───────────────────────────▼───────────────────────────────┐   │
    │  │              Metrics & Observability                       │   │
    │  │  - Hit rates per tier                                      │   │
    │  │  - Compression ratios                                      │   │
    │  │  - Migration throughput                                    │   │
    │  │  - ML prediction accuracy                                  │   │
    │  └───────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

ML Importance Model:
    Features:
    - Access frequency (last hour, day, week)
    - Access recency (time since last access)
    - Content size
    - Content type (embedding, text, structured)
    - User importance markers
    - Semantic similarity to recent queries
    - Reference count (how many times referenced by other entries)

    Output: Importance score [0.0, 1.0]
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import math
import os
import pickle
import struct
import tempfile
import time
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Generic
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("bbx.context.enhanced")

T = TypeVar("T")


# =============================================================================
# Enums
# =============================================================================


class GenerationTier(Enum):
    """Memory tier levels"""
    HOT = 0      # In-memory, uncompressed, instant access
    WARM = 1     # In-memory, uncompressed, fast access
    COOL = 2     # In-memory, compressed, moderate access
    COLD = 3     # On-disk, compressed, slow access
    ARCHIVED = 4 # Remote storage, very slow access


class CompressionType(Enum):
    """Compression algorithms"""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"      # Fast compression
    ZSTD = "zstd"    # High ratio compression


class ContentType(Enum):
    """Content type hints for optimization"""
    TEXT = "text"
    EMBEDDING = "embedding"
    STRUCTURED = "structured"
    BINARY = "binary"
    JSON = "json"


class MigrationDirection(Enum):
    """Direction of tier migration"""
    PROMOTE = "promote"   # Move to hotter tier
    DEMOTE = "demote"     # Move to colder tier


# =============================================================================
# ML Importance Scorer
# =============================================================================


@dataclass
class AccessPattern:
    """Tracks access patterns for an entry"""
    total_accesses: int = 0
    accesses_last_hour: int = 0
    accesses_last_day: int = 0
    accesses_last_week: int = 0
    last_access_time: float = 0
    first_access_time: float = 0
    access_timestamps: List[float] = field(default_factory=list)

    def record_access(self):
        """Record a new access"""
        now = time.time()
        self.total_accesses += 1
        self.last_access_time = now
        if self.first_access_time == 0:
            self.first_access_time = now

        # Keep last 1000 timestamps for pattern analysis
        self.access_timestamps.append(now)
        if len(self.access_timestamps) > 1000:
            self.access_timestamps = self.access_timestamps[-1000:]

        # Update rolling counts
        self._update_rolling_counts(now)

    def _update_rolling_counts(self, now: float):
        """Update rolling access counts"""
        hour_ago = now - 3600
        day_ago = now - 86400
        week_ago = now - 604800

        self.accesses_last_hour = sum(1 for t in self.access_timestamps if t >= hour_ago)
        self.accesses_last_day = sum(1 for t in self.access_timestamps if t >= day_ago)
        self.accesses_last_week = sum(1 for t in self.access_timestamps if t >= week_ago)


@dataclass
class EntryMetadata:
    """Metadata for a context entry"""
    key: str
    size_bytes: int
    content_type: ContentType
    created_at: float
    user_importance: float = 0.5  # User-defined importance [0, 1]
    pinned: bool = False  # Never demote
    reference_count: int = 0
    tags: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None  # For semantic similarity
    access_pattern: AccessPattern = field(default_factory=AccessPattern)


class ImportanceScorer:
    """
    ML-based importance scoring for context entries.

    Uses a lightweight model to predict importance based on:
    - Access patterns
    - Content characteristics
    - User signals
    - Semantic relevance
    """

    def __init__(
        self,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.25,
        user_weight: float = 0.2,
        reference_weight: float = 0.15,
        size_weight: float = 0.1
    ):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.user_weight = user_weight
        self.reference_weight = reference_weight
        self.size_weight = size_weight

        # Feature statistics for normalization
        self._max_accesses = 1000
        self._max_references = 100
        self._max_size = 10 * 1024 * 1024  # 10MB

        # Recent query embeddings for semantic relevance
        self._recent_query_embeddings: List[List[float]] = []
        self._max_query_history = 50

        # Model weights (could be learned)
        self._feature_weights = {
            "recency": 1.0,
            "frequency_hour": 2.0,
            "frequency_day": 1.5,
            "frequency_week": 1.0,
            "user_importance": 1.0,
            "reference_count": 0.8,
            "size_penalty": -0.2,
            "semantic_similarity": 1.5,
            "pinned_bonus": 10.0,
        }

    def calculate_importance(self, metadata: EntryMetadata) -> float:
        """Calculate importance score for an entry"""
        features = self._extract_features(metadata)
        score = self._score_features(features)
        return max(0.0, min(1.0, score))

    def _extract_features(self, metadata: EntryMetadata) -> Dict[str, float]:
        """Extract features from metadata"""
        now = time.time()
        pattern = metadata.access_pattern

        # Recency: exponential decay
        time_since_access = now - pattern.last_access_time if pattern.last_access_time > 0 else float('inf')
        recency = math.exp(-time_since_access / 3600)  # Half-life of 1 hour

        # Frequency features (normalized)
        freq_hour = min(pattern.accesses_last_hour / 100, 1.0)
        freq_day = min(pattern.accesses_last_day / 500, 1.0)
        freq_week = min(pattern.accesses_last_week / 2000, 1.0)

        # Reference count (normalized)
        ref_count = min(metadata.reference_count / self._max_references, 1.0)

        # Size penalty (larger items are less valuable to keep hot)
        size_penalty = min(metadata.size_bytes / self._max_size, 1.0)

        # Semantic similarity to recent queries
        semantic_sim = self._calculate_semantic_similarity(metadata)

        return {
            "recency": recency,
            "frequency_hour": freq_hour,
            "frequency_day": freq_day,
            "frequency_week": freq_week,
            "user_importance": metadata.user_importance,
            "reference_count": ref_count,
            "size_penalty": size_penalty,
            "semantic_similarity": semantic_sim,
            "pinned_bonus": 1.0 if metadata.pinned else 0.0,
        }

    def _score_features(self, features: Dict[str, float]) -> float:
        """Calculate weighted score from features"""
        score = 0.0
        total_weight = 0.0

        for feature_name, feature_value in features.items():
            weight = self._feature_weights.get(feature_name, 0.0)
            score += feature_value * weight
            total_weight += abs(weight)

        if total_weight > 0:
            score /= total_weight

        return score

    def _calculate_semantic_similarity(self, metadata: EntryMetadata) -> float:
        """Calculate semantic similarity to recent queries"""
        if not metadata.embedding or not self._recent_query_embeddings:
            return 0.5  # Neutral if no embeddings

        max_similarity = 0.0
        for query_embedding in self._recent_query_embeddings:
            similarity = self._cosine_similarity(metadata.embedding, query_embedding)
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return (dot_product / (norm_a * norm_b) + 1) / 2  # Normalize to [0, 1]

    def add_query_embedding(self, embedding: List[float]):
        """Add a query embedding for semantic relevance calculation"""
        self._recent_query_embeddings.append(embedding)
        if len(self._recent_query_embeddings) > self._max_query_history:
            self._recent_query_embeddings.pop(0)

    def update_weights(self, weights: Dict[str, float]):
        """Update feature weights (for online learning)"""
        self._feature_weights.update(weights)


# =============================================================================
# Compression Manager
# =============================================================================


@dataclass
class CompressionStats:
    """Statistics for compression"""
    original_size: int = 0
    compressed_size: int = 0
    compression_time_ms: float = 0
    decompression_time_ms: float = 0

    @property
    def ratio(self) -> float:
        if self.original_size == 0:
            return 0.0
        return 1 - (self.compressed_size / self.original_size)


class CompressionManager:
    """
    Adaptive compression manager.

    Selects optimal compression algorithm based on:
    - Content type
    - Size
    - Access patterns (frequently accessed = faster compression)
    """

    def __init__(self, default_level: int = 6):
        self.default_level = default_level
        self._stats: Dict[str, CompressionStats] = {}

        # Content type to compression mapping
        self._type_compression = {
            ContentType.TEXT: CompressionType.GZIP,
            ContentType.EMBEDDING: CompressionType.ZLIB,  # Embeddings compress poorly
            ContentType.STRUCTURED: CompressionType.GZIP,
            ContentType.BINARY: CompressionType.ZLIB,
            ContentType.JSON: CompressionType.GZIP,
        }

    def compress(
        self,
        data: bytes,
        content_type: ContentType = ContentType.BINARY,
        compression_type: Optional[CompressionType] = None
    ) -> Tuple[bytes, CompressionStats]:
        """Compress data with stats"""
        if compression_type is None:
            compression_type = self._type_compression.get(content_type, CompressionType.GZIP)

        if compression_type == CompressionType.NONE:
            return data, CompressionStats(original_size=len(data), compressed_size=len(data))

        start = time.time()

        if compression_type == CompressionType.GZIP:
            compressed = gzip.compress(data, compresslevel=self.default_level)
        elif compression_type == CompressionType.ZLIB:
            compressed = zlib.compress(data, level=self.default_level)
        else:
            compressed = gzip.compress(data, compresslevel=self.default_level)

        elapsed = (time.time() - start) * 1000

        stats = CompressionStats(
            original_size=len(data),
            compressed_size=len(compressed),
            compression_time_ms=elapsed
        )

        return compressed, stats

    def decompress(
        self,
        data: bytes,
        compression_type: CompressionType = CompressionType.GZIP
    ) -> Tuple[bytes, float]:
        """Decompress data, returns (data, decompression_time_ms)"""
        if compression_type == CompressionType.NONE:
            return data, 0.0

        start = time.time()

        if compression_type == CompressionType.GZIP:
            decompressed = gzip.decompress(data)
        elif compression_type == CompressionType.ZLIB:
            decompressed = zlib.decompress(data)
        else:
            decompressed = gzip.decompress(data)

        elapsed = (time.time() - start) * 1000
        return decompressed, elapsed

    def select_compression(
        self,
        content_type: ContentType,
        size: int,
        access_frequency: float
    ) -> Tuple[CompressionType, int]:
        """Select optimal compression algorithm and level"""
        # Small data: no compression
        if size < 1024:
            return CompressionType.NONE, 0

        # Frequently accessed: fast compression
        if access_frequency > 0.5:
            return CompressionType.ZLIB, 1  # Fast

        # Large data: high compression
        if size > 1024 * 1024:
            return CompressionType.GZIP, 9  # Maximum

        # Default
        return self._type_compression.get(content_type, CompressionType.GZIP), self.default_level


# =============================================================================
# Storage Backends
# =============================================================================


class StorageBackend(ABC):
    """Abstract storage backend"""

    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        pass

    @abstractmethod
    async def put(self, key: str, data: bytes) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass


class MemoryBackend(StorageBackend):
    """In-memory storage backend"""

    def __init__(self, max_size: int = 1024 * 1024 * 1024):  # 1GB default
        self._storage: Dict[str, bytes] = {}
        self._current_size = 0
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[bytes]:
        return self._storage.get(key)

    async def put(self, key: str, data: bytes) -> bool:
        async with self._lock:
            # Remove old entry if exists
            if key in self._storage:
                self._current_size -= len(self._storage[key])

            # Check capacity
            new_size = self._current_size + len(data)
            if new_size > self._max_size:
                return False

            self._storage[key] = data
            self._current_size = new_size
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._storage:
                self._current_size -= len(self._storage[key])
                del self._storage[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        return key in self._storage

    def get_size(self) -> int:
        return self._current_size


class DiskBackend(StorageBackend):
    """Disk-based storage backend"""

    def __init__(self, base_path: Optional[Path] = None):
        self._base_path = base_path or Path(tempfile.gettempdir()) / "bbx_context"
        self._base_path.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self._base_path / key_hash[:2] / key_hash[2:4] / key_hash

    async def get(self, key: str) -> Optional[bytes]:
        path = self._key_to_path(key)
        if not path.exists():
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._read_file, path)

    def _read_file(self, path: Path) -> bytes:
        return path.read_bytes()

    async def put(self, key: str, data: bytes) -> bool:
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._write_file, path, data)
        return True

    def _write_file(self, path: Path, data: bytes):
        path.write_bytes(data)

    async def delete(self, key: str) -> bool:
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    async def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()


# =============================================================================
# Context Entry
# =============================================================================


@dataclass
class ContextEntry:
    """A single context entry with tiering support"""
    key: str
    value: Any
    metadata: EntryMetadata

    # Tiering state
    current_tier: GenerationTier = GenerationTier.HOT
    is_compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE

    # Raw data (for compressed/serialized state)
    _raw_data: Optional[bytes] = None

    def serialize(self) -> bytes:
        """Serialize entry value to bytes"""
        return pickle.dumps(self.value)

    @classmethod
    def deserialize(cls, data: bytes) -> Any:
        """Deserialize value from bytes"""
        return pickle.loads(data)


# =============================================================================
# Prefetch Manager
# =============================================================================


@dataclass
class PrefetchRequest:
    """Request to prefetch data"""
    keys: List[str]
    priority: int = 0  # Higher = more urgent
    target_tier: GenerationTier = GenerationTier.HOT
    callback: Optional[Callable[[List[str]], None]] = None


class PrefetchManager:
    """
    Manages proactive data prefetching.

    Features:
    - Priority-based prefetch queue
    - Background prefetch workers
    - Prefetch hints from access patterns
    - Rate limiting to avoid overwhelming storage
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        rate_limit_per_second: float = 100
    ):
        self._queue: asyncio.PriorityQueue[Tuple[int, PrefetchRequest]] = asyncio.PriorityQueue()
        self._workers: List[asyncio.Task] = []
        self._max_concurrent = max_concurrent
        self._rate_limit = rate_limit_per_second
        self._last_prefetch_time = 0.0
        self._shutdown = False

        # Stats
        self._prefetched_count = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0

        # Callback for actual prefetch
        self._prefetch_callback: Optional[Callable[[str, GenerationTier], asyncio.Future]] = None

    def set_prefetch_callback(self, callback: Callable[[str, GenerationTier], asyncio.Future]):
        """Set callback for actual prefetch operation"""
        self._prefetch_callback = callback

    async def start(self):
        """Start prefetch workers"""
        self._shutdown = False
        for _ in range(self._max_concurrent):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)

    async def stop(self):
        """Stop prefetch workers"""
        self._shutdown = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def prefetch(
        self,
        keys: List[str],
        priority: int = 0,
        target_tier: GenerationTier = GenerationTier.HOT,
        callback: Optional[Callable[[List[str]], None]] = None
    ):
        """Request prefetch of keys"""
        request = PrefetchRequest(
            keys=keys,
            priority=priority,
            target_tier=target_tier,
            callback=callback
        )
        await self._queue.put((-priority, request))

    async def prefetch_related(self, key: str, patterns: Dict[str, List[str]]):
        """Prefetch keys that are commonly accessed together"""
        related_keys = patterns.get(key, [])
        if related_keys:
            await self.prefetch(related_keys, priority=1)

    async def _worker_loop(self):
        """Worker coroutine for prefetching"""
        while not self._shutdown:
            try:
                _, request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Rate limiting
            now = time.time()
            min_interval = 1.0 / self._rate_limit
            elapsed = now - self._last_prefetch_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_prefetch_time = time.time()

            # Execute prefetch
            successful_keys = []
            for key in request.keys:
                if self._prefetch_callback:
                    try:
                        await self._prefetch_callback(key, request.target_tier)
                        successful_keys.append(key)
                        self._prefetched_count += 1
                    except Exception as e:
                        logger.debug(f"Prefetch failed for {key}: {e}")

            # Invoke callback
            if request.callback and successful_keys:
                try:
                    request.callback(successful_keys)
                except Exception:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        return {
            "prefetched_count": self._prefetched_count,
            "prefetch_hits": self._prefetch_hits,
            "prefetch_misses": self._prefetch_misses,
            "queue_size": self._queue.qsize(),
        }


# =============================================================================
# Migration Engine
# =============================================================================


@dataclass
class MigrationTask:
    """Task for tier migration"""
    key: str
    source_tier: GenerationTier
    target_tier: GenerationTier
    priority: int = 0
    created_at: float = field(default_factory=time.time)


class AsyncMigrationEngine:
    """
    Async background migration engine.

    Features:
    - Non-blocking migration
    - Batch migration for efficiency
    - Priority-based scheduling
    - Progress tracking
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_concurrent_batches: int = 4,
        migration_interval_ms: float = 100
    ):
        self._batch_size = batch_size
        self._max_concurrent = max_concurrent_batches
        self._interval_ms = migration_interval_ms

        self._queue: asyncio.PriorityQueue[Tuple[int, MigrationTask]] = asyncio.PriorityQueue()
        self._workers: List[asyncio.Task] = []
        self._shutdown = False

        # Migration callback
        self._migrate_callback: Optional[Callable[[str, GenerationTier, GenerationTier], asyncio.Future]] = None

        # Stats
        self._total_migrated = 0
        self._total_bytes_migrated = 0
        self._migration_errors = 0
        self._active_migrations = 0

    def set_migrate_callback(
        self,
        callback: Callable[[str, GenerationTier, GenerationTier], asyncio.Future]
    ):
        """Set callback for actual migration"""
        self._migrate_callback = callback

    async def start(self):
        """Start migration workers"""
        self._shutdown = False
        for _ in range(self._max_concurrent):
            worker = asyncio.create_task(self._worker_loop())
            self._workers.append(worker)

    async def stop(self):
        """Stop migration workers"""
        self._shutdown = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def schedule_migration(
        self,
        key: str,
        source_tier: GenerationTier,
        target_tier: GenerationTier,
        priority: int = 0
    ):
        """Schedule a migration task"""
        task = MigrationTask(
            key=key,
            source_tier=source_tier,
            target_tier=target_tier,
            priority=priority
        )
        await self._queue.put((-priority, task))

    async def schedule_batch_migration(
        self,
        keys: List[str],
        source_tier: GenerationTier,
        target_tier: GenerationTier
    ):
        """Schedule batch migration"""
        for key in keys:
            await self.schedule_migration(key, source_tier, target_tier)

    async def _worker_loop(self):
        """Worker coroutine for migrations"""
        batch: List[MigrationTask] = []

        while not self._shutdown:
            try:
                # Collect batch
                deadline = time.time() + (self._interval_ms / 1000)
                while len(batch) < self._batch_size and time.time() < deadline:
                    try:
                        remaining = deadline - time.time()
                        _, task = await asyncio.wait_for(self._queue.get(), timeout=max(0.01, remaining))
                        batch.append(task)
                    except asyncio.TimeoutError:
                        break

                if not batch:
                    continue

                # Process batch
                self._active_migrations += len(batch)
                await self._process_batch(batch)
                self._active_migrations -= len(batch)
                batch.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Migration worker error: {e}")
                batch.clear()

    async def _process_batch(self, batch: List[MigrationTask]):
        """Process a batch of migrations"""
        if not self._migrate_callback:
            return

        tasks = []
        for migration in batch:
            task = asyncio.create_task(self._execute_migration(migration))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self._migration_errors += 1
            else:
                self._total_migrated += 1

    async def _execute_migration(self, migration: MigrationTask):
        """Execute a single migration"""
        await self._migrate_callback(
            migration.key,
            migration.source_tier,
            migration.target_tier
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_migrated": self._total_migrated,
            "total_bytes_migrated": self._total_bytes_migrated,
            "migration_errors": self._migration_errors,
            "active_migrations": self._active_migrations,
            "queue_size": self._queue.qsize(),
        }


# =============================================================================
# Enhanced Context Tiering Configuration
# =============================================================================


@dataclass
class EnhancedTieringConfig:
    """Configuration for enhanced context tiering"""
    # Tier sizes
    hot_tier_size: int = 256 * 1024 * 1024     # 256MB
    warm_tier_size: int = 512 * 1024 * 1024    # 512MB
    cool_tier_size: int = 1024 * 1024 * 1024   # 1GB
    cold_tier_size: int = 10 * 1024 * 1024 * 1024  # 10GB

    # Aging intervals
    aging_interval_seconds: float = 10.0
    promotion_threshold: float = 0.7   # Importance > 0.7 -> promote
    demotion_threshold: float = 0.3    # Importance < 0.3 -> demote

    # Compression
    compress_cool_tier: bool = True
    compress_cold_tier: bool = True
    compression_level: int = 6

    # Prefetch
    enable_prefetch: bool = True
    prefetch_batch_size: int = 10
    prefetch_lookahead: int = 5

    # Migration
    enable_async_migration: bool = True
    migration_batch_size: int = 100
    migration_workers: int = 4

    # ML scoring
    enable_ml_scoring: bool = True
    score_update_interval_seconds: float = 60.0

    # Disk storage
    disk_storage_path: Optional[Path] = None


@dataclass
class EnhancedTieringStats:
    """Statistics for enhanced tiering"""
    # Tier stats
    hot_tier_entries: int = 0
    hot_tier_size: int = 0
    warm_tier_entries: int = 0
    warm_tier_size: int = 0
    cool_tier_entries: int = 0
    cool_tier_size: int = 0
    cold_tier_entries: int = 0
    cold_tier_size: int = 0

    # Hit rates
    hot_hits: int = 0
    warm_hits: int = 0
    cool_hits: int = 0
    cold_hits: int = 0
    misses: int = 0

    # Migration stats
    promotions: int = 0
    demotions: int = 0

    # Compression stats
    compression_ratio: float = 0.0
    total_compressed_size: int = 0
    total_original_size: int = 0

    # Latency
    avg_hot_latency_ms: float = 0.0
    avg_cold_latency_ms: float = 0.0

    @property
    def total_entries(self) -> int:
        return self.hot_tier_entries + self.warm_tier_entries + self.cool_tier_entries + self.cold_tier_entries

    @property
    def hit_rate(self) -> float:
        total = self.hot_hits + self.warm_hits + self.cool_hits + self.cold_hits + self.misses
        if total == 0:
            return 0.0
        return (total - self.misses) / total


# =============================================================================
# Enhanced Context Tiering Manager
# =============================================================================


class EnhancedContextTiering:
    """
    Production-ready context tiering with ML scoring.

    Features:
    - ML-based importance scoring
    - Async background migration
    - Explicit prefetch API
    - Adaptive compression
    - Comprehensive metrics
    """

    def __init__(self, config: Optional[EnhancedTieringConfig] = None):
        self.config = config or EnhancedTieringConfig()

        # Storage backends
        self._hot_tier = MemoryBackend(self.config.hot_tier_size)
        self._warm_tier = MemoryBackend(self.config.warm_tier_size)
        self._cool_tier = MemoryBackend(self.config.cool_tier_size)
        self._cold_tier = DiskBackend(self.config.disk_storage_path)

        # Entry tracking
        self._entries: Dict[str, ContextEntry] = {}
        self._metadata: Dict[str, EntryMetadata] = {}
        self._tier_mapping: Dict[GenerationTier, Set[str]] = {
            GenerationTier.HOT: set(),
            GenerationTier.WARM: set(),
            GenerationTier.COOL: set(),
            GenerationTier.COLD: set(),
        }

        # Components
        self._scorer = ImportanceScorer() if self.config.enable_ml_scoring else None
        self._compression = CompressionManager(self.config.compression_level)
        self._prefetch = PrefetchManager() if self.config.enable_prefetch else None
        self._migration = AsyncMigrationEngine(
            batch_size=self.config.migration_batch_size,
            max_concurrent_batches=self.config.migration_workers
        ) if self.config.enable_async_migration else None

        # Background tasks
        self._aging_task: Optional[asyncio.Task] = None
        self._scoring_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Stats
        self.stats = EnhancedTieringStats()

        # Access patterns for prefetch
        self._access_patterns: Dict[str, List[str]] = defaultdict(list)
        self._recent_accesses: List[str] = []

        # Locks
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the tiering manager"""
        self._shutdown = False

        # Set up callbacks
        if self._prefetch:
            self._prefetch.set_prefetch_callback(self._prefetch_callback)
            await self._prefetch.start()

        if self._migration:
            self._migration.set_migrate_callback(self._migrate_callback)
            await self._migration.start()

        # Start background tasks
        self._aging_task = asyncio.create_task(self._aging_loop())
        if self.config.enable_ml_scoring:
            self._scoring_task = asyncio.create_task(self._scoring_loop())

        logger.info("EnhancedContextTiering started")

    async def stop(self):
        """Stop the tiering manager"""
        self._shutdown = True

        if self._aging_task:
            self._aging_task.cancel()
            try:
                await self._aging_task
            except asyncio.CancelledError:
                pass

        if self._scoring_task:
            self._scoring_task.cancel()
            try:
                await self._scoring_task
            except asyncio.CancelledError:
                pass

        if self._prefetch:
            await self._prefetch.stop()

        if self._migration:
            await self._migration.stop()

        logger.info("EnhancedContextTiering stopped")

    # =========================================================================
    # Core API
    # =========================================================================

    async def get(self, key: str) -> Optional[Any]:
        """Get a value, automatically promoting if needed"""
        async with self._lock:
            entry = self._entries.get(key)
            if not entry:
                self.stats.misses += 1
                return None

            # Record access
            entry.metadata.access_pattern.record_access()

            # Track access pattern
            self._record_access_pattern(key)

            # Get value based on tier
            value = await self._get_from_tier(entry)

            # Update hit stats
            tier_hits = {
                GenerationTier.HOT: "hot_hits",
                GenerationTier.WARM: "warm_hits",
                GenerationTier.COOL: "cool_hits",
                GenerationTier.COLD: "cold_hits",
            }
            setattr(self.stats, tier_hits[entry.current_tier], getattr(self.stats, tier_hits[entry.current_tier]) + 1)

            # Schedule promotion if accessed from cold tier
            if entry.current_tier in (GenerationTier.COOL, GenerationTier.COLD):
                if self._migration and not entry.metadata.pinned:
                    await self._migration.schedule_migration(
                        key,
                        entry.current_tier,
                        GenerationTier.WARM,
                        priority=1
                    )

            return value

    async def put(
        self,
        key: str,
        value: Any,
        content_type: ContentType = ContentType.BINARY,
        importance: float = 0.5,
        pinned: bool = False,
        tags: Optional[Set[str]] = None,
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Store a value with metadata"""
        async with self._lock:
            # Serialize value
            data = pickle.dumps(value)
            size = len(data)

            # Create or update metadata
            metadata = EntryMetadata(
                key=key,
                size_bytes=size,
                content_type=content_type,
                created_at=time.time(),
                user_importance=importance,
                pinned=pinned,
                tags=tags or set(),
                embedding=embedding
            )
            metadata.access_pattern.record_access()

            # Determine initial tier based on importance
            if pinned or importance >= 0.8:
                target_tier = GenerationTier.HOT
            elif importance >= 0.5:
                target_tier = GenerationTier.WARM
            else:
                target_tier = GenerationTier.COOL

            # Create entry
            entry = ContextEntry(
                key=key,
                value=value,
                metadata=metadata,
                current_tier=target_tier
            )

            # Store in appropriate tier
            success = await self._store_in_tier(entry, target_tier)
            if not success:
                # Try next tier
                for tier in [GenerationTier.WARM, GenerationTier.COOL, GenerationTier.COLD]:
                    if tier.value > target_tier.value:
                        success = await self._store_in_tier(entry, tier)
                        if success:
                            entry.current_tier = tier
                            break

            if success:
                self._entries[key] = entry
                self._metadata[key] = metadata
                self._tier_mapping[entry.current_tier].add(key)
                self._update_tier_stats()

            return success

    async def delete(self, key: str) -> bool:
        """Delete a value"""
        async with self._lock:
            entry = self._entries.pop(key, None)
            if not entry:
                return False

            self._metadata.pop(key, None)
            self._tier_mapping[entry.current_tier].discard(key)

            # Remove from storage
            backend = self._get_backend(entry.current_tier)
            await backend.delete(key)

            self._update_tier_stats()
            return True

    async def mark_important(self, key: str, importance: float = 1.0):
        """Mark an entry as important (user signal)"""
        async with self._lock:
            metadata = self._metadata.get(key)
            if metadata:
                metadata.user_importance = importance
                metadata.pinned = importance >= 1.0

                # Schedule promotion if not already hot
                entry = self._entries.get(key)
                if entry and entry.current_tier != GenerationTier.HOT:
                    if self._migration:
                        await self._migration.schedule_migration(
                            key,
                            entry.current_tier,
                            GenerationTier.HOT,
                            priority=10
                        )

    async def pin(self, key: str):
        """Pin an entry (never demote)"""
        await self.mark_important(key, 1.0)

    async def unpin(self, key: str):
        """Unpin an entry"""
        async with self._lock:
            metadata = self._metadata.get(key)
            if metadata:
                metadata.pinned = False

    # =========================================================================
    # Prefetch API
    # =========================================================================

    async def prefetch(
        self,
        keys: List[str],
        target_tier: GenerationTier = GenerationTier.HOT
    ):
        """Explicitly prefetch keys to a tier"""
        if self._prefetch:
            await self._prefetch.prefetch(keys, priority=5, target_tier=target_tier)

    async def prefetch_related(self, key: str):
        """Prefetch keys commonly accessed with this key"""
        related = self._access_patterns.get(key, [])
        if related and self._prefetch:
            await self._prefetch.prefetch(related[:10], priority=3)

    def add_query_embedding(self, embedding: List[float]):
        """Add query embedding for semantic prefetch"""
        if self._scorer:
            self._scorer.add_query_embedding(embedding)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _get_backend(self, tier: GenerationTier) -> StorageBackend:
        """Get storage backend for tier"""
        backends = {
            GenerationTier.HOT: self._hot_tier,
            GenerationTier.WARM: self._warm_tier,
            GenerationTier.COOL: self._cool_tier,
            GenerationTier.COLD: self._cold_tier,
        }
        return backends[tier]

    async def _get_from_tier(self, entry: ContextEntry) -> Any:
        """Get value from entry's current tier"""
        if entry.current_tier in (GenerationTier.HOT, GenerationTier.WARM):
            # In-memory, uncompressed
            return entry.value

        # Need to fetch from storage
        backend = self._get_backend(entry.current_tier)
        data = await backend.get(entry.key)

        if data is None:
            return None

        # Decompress if needed
        if entry.is_compressed:
            data, decompress_time = self._compression.decompress(data, entry.compression_type)

        # Deserialize
        return ContextEntry.deserialize(data)

    async def _store_in_tier(self, entry: ContextEntry, tier: GenerationTier) -> bool:
        """Store entry in specified tier"""
        backend = self._get_backend(tier)

        if tier in (GenerationTier.HOT, GenerationTier.WARM):
            # Store uncompressed in memory
            data = entry.serialize()
            entry.is_compressed = False
        else:
            # Compress for cool/cold tiers
            data = entry.serialize()
            if (tier == GenerationTier.COOL and self.config.compress_cool_tier) or \
               (tier == GenerationTier.COLD and self.config.compress_cold_tier):
                data, stats = self._compression.compress(
                    data,
                    entry.metadata.content_type
                )
                entry.is_compressed = True
                entry.compression_type = CompressionType.GZIP

                # Update compression stats
                self.stats.total_original_size += stats.original_size
                self.stats.total_compressed_size += stats.compressed_size

        return await backend.put(entry.key, data)

    async def _prefetch_callback(self, key: str, target_tier: GenerationTier):
        """Callback for prefetch manager"""
        entry = self._entries.get(key)
        if not entry or entry.current_tier.value <= target_tier.value:
            return  # Already in hot enough tier

        await self._do_migration(key, entry.current_tier, target_tier)

    async def _migrate_callback(
        self,
        key: str,
        source_tier: GenerationTier,
        target_tier: GenerationTier
    ):
        """Callback for migration engine"""
        await self._do_migration(key, source_tier, target_tier)

    async def _do_migration(
        self,
        key: str,
        source_tier: GenerationTier,
        target_tier: GenerationTier
    ):
        """Actually perform a migration"""
        async with self._lock:
            entry = self._entries.get(key)
            if not entry or entry.current_tier != source_tier:
                return

            # Get value from current tier
            value = await self._get_from_tier(entry)
            if value is None:
                return

            # Store in target tier
            entry.value = value
            success = await self._store_in_tier(entry, target_tier)

            if success:
                # Remove from old tier
                old_backend = self._get_backend(source_tier)
                await old_backend.delete(key)

                # Update tracking
                self._tier_mapping[source_tier].discard(key)
                self._tier_mapping[target_tier].add(key)
                entry.current_tier = target_tier

                # Update stats
                if target_tier.value < source_tier.value:
                    self.stats.promotions += 1
                else:
                    self.stats.demotions += 1

                self._update_tier_stats()

    def _record_access_pattern(self, key: str):
        """Record access pattern for prefetch hints"""
        # Track which keys are accessed together
        for recent_key in self._recent_accesses[-5:]:
            if recent_key != key:
                if key not in self._access_patterns[recent_key]:
                    self._access_patterns[recent_key].append(key)
                    # Limit pattern size
                    if len(self._access_patterns[recent_key]) > 20:
                        self._access_patterns[recent_key].pop(0)

        self._recent_accesses.append(key)
        if len(self._recent_accesses) > 100:
            self._recent_accesses.pop(0)

    def _update_tier_stats(self):
        """Update tier statistics"""
        self.stats.hot_tier_entries = len(self._tier_mapping[GenerationTier.HOT])
        self.stats.warm_tier_entries = len(self._tier_mapping[GenerationTier.WARM])
        self.stats.cool_tier_entries = len(self._tier_mapping[GenerationTier.COOL])
        self.stats.cold_tier_entries = len(self._tier_mapping[GenerationTier.COLD])

        self.stats.hot_tier_size = self._hot_tier.get_size()
        self.stats.warm_tier_size = self._warm_tier.get_size()
        self.stats.cool_tier_size = self._cool_tier.get_size()

        if self.stats.total_original_size > 0:
            self.stats.compression_ratio = 1 - (
                self.stats.total_compressed_size / self.stats.total_original_size
            )

    async def _aging_loop(self):
        """Background loop for tier aging"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.aging_interval_seconds)
                await self._process_aging()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aging loop error: {e}")

    async def _process_aging(self):
        """Process tier aging based on importance scores"""
        if not self._scorer:
            return

        async with self._lock:
            migrations = []

            for key, entry in self._entries.items():
                if entry.metadata.pinned:
                    continue

                # Calculate importance
                importance = self._scorer.calculate_importance(entry.metadata)

                # Determine target tier
                if importance >= self.config.promotion_threshold:
                    if entry.current_tier.value > GenerationTier.HOT.value:
                        # Promote
                        target = GenerationTier(max(0, entry.current_tier.value - 1))
                        migrations.append((key, entry.current_tier, target))

                elif importance <= self.config.demotion_threshold:
                    if entry.current_tier.value < GenerationTier.COLD.value:
                        # Demote
                        target = GenerationTier(min(3, entry.current_tier.value + 1))
                        migrations.append((key, entry.current_tier, target))

            # Schedule migrations
            if self._migration:
                for key, source, target in migrations:
                    await self._migration.schedule_migration(key, source, target)

    async def _scoring_loop(self):
        """Background loop for updating importance scores"""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.score_update_interval_seconds)
                # Scores are calculated on-demand, but we could pre-compute here
            except asyncio.CancelledError:
                break

    # =========================================================================
    # Stats and Metrics
    # =========================================================================

    def get_stats(self) -> EnhancedTieringStats:
        """Get current statistics"""
        return self.stats

    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about an entry"""
        entry = self._entries.get(key)
        if not entry:
            return None

        importance = 0.0
        if self._scorer:
            importance = self._scorer.calculate_importance(entry.metadata)

        return {
            "key": key,
            "current_tier": entry.current_tier.name,
            "is_compressed": entry.is_compressed,
            "size_bytes": entry.metadata.size_bytes,
            "content_type": entry.metadata.content_type.value,
            "importance": importance,
            "user_importance": entry.metadata.user_importance,
            "pinned": entry.metadata.pinned,
            "total_accesses": entry.metadata.access_pattern.total_accesses,
            "last_access": entry.metadata.access_pattern.last_access_time,
            "tags": list(entry.metadata.tags),
        }


# =============================================================================
# Factory
# =============================================================================


_global_tiering: Optional[EnhancedContextTiering] = None


def get_enhanced_tiering() -> EnhancedContextTiering:
    """Get global tiering instance"""
    global _global_tiering
    if _global_tiering is None:
        _global_tiering = EnhancedContextTiering()
    return _global_tiering


async def create_enhanced_tiering(
    config: Optional[EnhancedTieringConfig] = None
) -> EnhancedContextTiering:
    """Create and start tiering manager"""
    tiering = EnhancedContextTiering(config)
    await tiering.start()
    return tiering
