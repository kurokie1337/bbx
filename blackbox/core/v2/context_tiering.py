# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 Context Tiering - MGLRU-inspired multi-generation context memory.

This module provides intelligent context memory management for AI agents,
automatically moving data between fast (hot) and slow (cold) storage tiers.

Key concepts from Linux MGLRU:
- Generations: Context ages through generations (0=hot to N=cold)
- Refault tracking: Track when cold data is accessed (promotes on access)
- Smart demotion: Only demote data unlikely to be needed soon
- Feedback loop: Monitor and adjust based on access patterns

Example usage:
    tiering = ContextTiering()
    await tiering.start()

    # Add context (goes to hot tier)
    await tiering.set("user_prefs", {"theme": "dark"})

    # Get context (promotes from cold if needed)
    value = await tiering.get("user_prefs")

    # Pin important context
    await tiering.pin("critical_data")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbx.context_tiering")


# =============================================================================
# Enums
# =============================================================================


class GenerationTier(Enum):
    """Memory tiers for context (like MGLRU generations)"""
    HOT = 0       # In-memory, uncompressed, instant access
    WARM = 1      # In-memory, compressed
    COOL = 2      # On disk, compressed
    COLD = 3      # Archive / vector store


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ContextItem:
    """Single item in context memory"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    size_bytes: int = 0
    pinned: bool = False
    generation: GenerationTier = GenerationTier.HOT
    last_generation_change: datetime = field(default_factory=datetime.now)
    promotion_count: int = 0


@dataclass
class TieringConfig:
    """Configuration for context tiering"""
    # Generation size limits
    hot_max_size: int = 100 * 1024  # 100KB
    warm_max_size: int = 1 * 1024 * 1024  # 1MB
    cool_max_size: int = 100 * 1024 * 1024  # 100MB

    # Aging parameters
    hot_max_age: timedelta = timedelta(minutes=5)
    warm_max_age: timedelta = timedelta(hours=1)
    cool_max_age: timedelta = timedelta(days=1)

    # MGLRU parameters
    aging_interval: float = 30.0  # Seconds between aging cycles
    refault_distance: int = 5  # Generations before promotion
    min_ttl: float = 10.0  # Minimum seconds before demotion


@dataclass
class TieringStats:
    """Statistics for context tiering"""
    total_items: int = 0
    total_size_bytes: int = 0

    hot_items: int = 0
    hot_size_bytes: int = 0

    warm_items: int = 0
    warm_size_bytes: int = 0

    cool_items: int = 0
    cool_size_bytes: int = 0

    cold_items: int = 0
    cold_size_bytes: int = 0

    promotions: int = 0
    demotions: int = 0

    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


# =============================================================================
# Generation
# =============================================================================


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

    @property
    def utilization(self) -> float:
        return self._current_size / self.max_size if self.max_size > 0 else 0.0

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
            serialized = self._serialize(item.value)
            item.size_bytes = len(serialized)

            if self._current_size + item.size_bytes > self.max_size:
                return False

            item.generation = self.tier
            item.last_generation_change = datetime.now()
            self._items[item.key] = item
            self._current_size += item.size_bytes

            # Persist to disk for cool/cold tiers
            if self.tier in (GenerationTier.COOL, GenerationTier.COLD):
                await self._persist(item)

            return True

    async def remove(self, key: str) -> Optional[ContextItem]:
        """Remove item from this generation"""
        async with self._lock:
            item = self._items.pop(key, None)
            if item:
                self._current_size -= item.size_bytes

                if self.tier in (GenerationTier.COOL, GenerationTier.COLD):
                    await self._unpersist(key)

            return item

    async def get_aged_items(self, max_age: timedelta) -> List[ContextItem]:
        """Get items older than max_age"""
        cutoff = datetime.now() - max_age
        aged = []
        async with self._lock:
            for item in self._items.values():
                if not item.pinned and item.last_accessed < cutoff:
                    aged.append(item)
        return aged

    async def get_all_keys(self) -> List[str]:
        """Get all keys in this generation"""
        async with self._lock:
            return list(self._items.keys())

    def _serialize(self, value: Any) -> bytes:
        """Serialize value"""
        json_bytes = json.dumps(value, default=str).encode()
        if self.tier in (GenerationTier.WARM, GenerationTier.COOL, GenerationTier.COLD):
            return zlib.compress(json_bytes)
        return json_bytes

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value"""
        if self.tier in (GenerationTier.WARM, GenerationTier.COOL, GenerationTier.COLD):
            data = zlib.decompress(data)
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
        """Remove persisted item"""
        if not self.storage_path:
            return
        file_path = self.storage_path / f"{hashlib.md5(key.encode()).hexdigest()}.ctx"
        if file_path.exists():
            file_path.unlink()


# =============================================================================
# Refault Tracker
# =============================================================================


class RefaultTracker:
    """
    Tracks refaults - when demoted items are accessed again.

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
        Get distance since last access from hot tier.
        Lower = more likely to be accessed soon.
        """
        if key not in self._generation_history:
            return float('inf')

        history = self._generation_history[key]
        for i, gen in enumerate(reversed(history)):
            if gen == 0:
                return i
        return len(history)

    def get_access_frequency(self, key: str, window: timedelta = timedelta(hours=1)) -> float:
        """Get access frequency in the given time window"""
        if key not in self._access_history:
            return 0.0

        cutoff = datetime.now() - window
        recent_accesses = sum(1 for t in self._access_history[key] if t > cutoff)
        return recent_accesses / window.total_seconds() * 3600  # Per hour


# =============================================================================
# Context Tiering
# =============================================================================


class ContextTiering:
    """
    Multi-Generation LRU for AI agent context.

    Like Linux MGLRU, uses generations instead of simple LRU.
    """

    def __init__(
        self,
        config: Optional[TieringConfig] = None,
        base_path: Optional[Path] = None
    ):
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
                int(1e12),  # Effectively unlimited
                self.base_path / "cold"
            ),
        ]

        self.refault_tracker = RefaultTracker()
        self.stats = TieringStats()
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
        """Get context item, promoting from cold if needed"""
        for gen_idx, gen in enumerate(self.generations):
            item = await gen.get(key)
            if item:
                self.stats.cache_hits += 1
                self.refault_tracker.record_access(key, gen_idx)

                # Promote if accessed from cold generation
                if gen_idx > 0 and self._should_promote(key, gen_idx):
                    await self._promote(item, from_gen=gen_idx, to_gen=0)

                return item.value

        self.stats.cache_misses += 1
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
            # Hot full, trigger aging
            await self._age_generation(0)
            success = await self.generations[0].add(item)

            if not success:
                logger.warning(f"Failed to add {key} to context")

        self._update_stats()

    async def delete(self, key: str):
        """Remove item from all generations"""
        for gen in self.generations:
            await gen.remove(key)
        self._update_stats()

    async def pin(self, key: str):
        """Pin item to prevent demotion"""
        for gen in self.generations:
            item = await gen.get(key)
            if item:
                item.pinned = True
                return
        raise KeyError(f"Item {key} not found")

    async def unpin(self, key: str):
        """Unpin item to allow demotion"""
        for gen in self.generations:
            item = await gen.get(key)
            if item:
                item.pinned = False
                return
        raise KeyError(f"Item {key} not found")

    async def keys(self) -> List[str]:
        """Get all keys across all generations"""
        all_keys = []
        for gen in self.generations:
            all_keys.extend(await gen.get_all_keys())
        return all_keys

    def _should_promote(self, key: str, current_gen: int) -> bool:
        """Decide if item should be promoted"""
        distance = self.refault_tracker.get_distance(key)
        return distance < self.config.refault_distance

    async def _promote(self, item: ContextItem, from_gen: int, to_gen: int):
        """Promote item to hotter generation"""
        await self.generations[from_gen].remove(item.key)
        item.promotion_count += 1
        item.last_generation_change = datetime.now()

        success = await self.generations[to_gen].add(item)
        if not success:
            # Target full, try next colder
            for gen_idx in range(to_gen + 1, from_gen):
                success = await self.generations[gen_idx].add(item)
                if success:
                    break

            if not success:
                await self.generations[from_gen].add(item)
        else:
            self.stats.promotions += 1

        self._update_stats()

    async def _demote(self, item: ContextItem, from_gen: int):
        """Demote item to colder generation"""
        if from_gen >= len(self.generations) - 1:
            return

        await self.generations[from_gen].remove(item.key)
        item.last_generation_change = datetime.now()

        to_gen = from_gen + 1
        success = await self.generations[to_gen].add(item)

        if not success:
            await self._age_generation(to_gen)
            await self.generations[to_gen].add(item)

        self.stats.demotions += 1
        self._update_stats()

    async def _age_generation(self, gen_idx: int):
        """Age items in generation (demote old items)"""
        gen = self.generations[gen_idx]

        max_ages = [
            self.config.hot_max_age,
            self.config.warm_max_age,
            self.config.cool_max_age,
            timedelta.max,
        ]

        aged_items = await gen.get_aged_items(max_ages[gen_idx])

        for item in aged_items:
            distance = self.refault_tracker.get_distance(item.key)
            if distance < self.config.refault_distance:
                continue

            time_since_change = datetime.now() - item.last_generation_change
            if time_since_change.total_seconds() < self.config.min_ttl:
                continue

            await self._demote(item, gen_idx)

    async def _aging_loop(self):
        """Background aging loop"""
        while not self._shutdown:
            try:
                for gen_idx in range(len(self.generations) - 1):
                    await self._age_generation(gen_idx)

                await asyncio.sleep(self.config.aging_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aging loop error: {e}")
                await asyncio.sleep(self.config.aging_interval)

    def _update_stats(self):
        """Update statistics"""
        self.stats.total_items = sum(gen.count for gen in self.generations)
        self.stats.total_size_bytes = sum(gen.size for gen in self.generations)

        self.stats.hot_items = self.generations[0].count
        self.stats.hot_size_bytes = self.generations[0].size

        self.stats.warm_items = self.generations[1].count
        self.stats.warm_size_bytes = self.generations[1].size

        self.stats.cool_items = self.generations[2].count
        self.stats.cool_size_bytes = self.generations[2].size

        self.stats.cold_items = self.generations[3].count
        self.stats.cold_size_bytes = self.generations[3].size

    def get_stats(self) -> Dict[str, Any]:
        """Get tiering statistics"""
        self._update_stats()
        return {
            "generations": [
                {
                    "tier": gen.tier.name,
                    "items": gen.count,
                    "size_bytes": gen.size,
                    "max_size_bytes": gen.max_size,
                    "utilization": gen.utilization,
                }
                for gen in self.generations
            ],
            "total_items": self.stats.total_items,
            "total_size_bytes": self.stats.total_size_bytes,
            "promotions": self.stats.promotions,
            "demotions": self.stats.demotions,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "hit_rate": self.stats.hit_rate,
        }


# =============================================================================
# Global Instance
# =============================================================================


_global_tiering: Optional[ContextTiering] = None


def get_context_tiering() -> ContextTiering:
    """Get global ContextTiering instance"""
    global _global_tiering
    if _global_tiering is None:
        _global_tiering = ContextTiering()
    return _global_tiering
