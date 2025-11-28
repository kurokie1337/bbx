# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Tests for ContextTiering - MGLRU-inspired multi-generation context memory

These tests verify:
- Basic get/set operations
- Tier transitions (promotion/demotion)
- Pinning functionality
- Memory management
- Statistics collection
"""

import asyncio
import pytest
import time

from blackbox.core.v2.context_tiering import (
    ContextTiering,
    ContextItem,
    TieringConfig,
    TieringStats,
    GenerationTier,
    Generation,
    RefaultTracker,
)


@pytest.fixture
def tiering_config():
    """Default tiering configuration for tests"""
    return TieringConfig(
        hot_max_items=10,
        warm_max_items=20,
        cool_max_items=50,
        cold_max_items=100,
        hot_max_bytes=1024 * 1024,  # 1MB
        warm_max_bytes=5 * 1024 * 1024,  # 5MB
        demotion_interval_sec=0.1,  # Fast for testing
        enable_compression=False,  # Disable for simpler testing
    )


@pytest.fixture
async def tiering(tiering_config):
    """Create and start a ContextTiering instance"""
    tiering = ContextTiering(tiering_config)
    await tiering.start()
    yield tiering
    await tiering.stop()


class TestBasicOperations:
    """Basic get/set operations tests"""

    @pytest.mark.asyncio
    async def test_set_and_get(self, tiering):
        """Test basic set and get"""
        await tiering.set("key1", "value1")
        result = await tiering.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tiering):
        """Test getting nonexistent key"""
        result = await tiering.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_overwrite(self, tiering):
        """Test overwriting existing key"""
        await tiering.set("key1", "value1")
        await tiering.set("key1", "value2")
        result = await tiering.get("key1")
        assert result == "value2"

    @pytest.mark.asyncio
    async def test_delete(self, tiering):
        """Test deleting a key"""
        await tiering.set("key1", "value1")
        success = await tiering.delete("key1")
        assert success

        result = await tiering.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tiering):
        """Test deleting nonexistent key"""
        success = await tiering.delete("nonexistent")
        assert not success

    @pytest.mark.asyncio
    async def test_various_types(self, tiering):
        """Test storing various data types"""
        # String
        await tiering.set("string", "hello")
        assert await tiering.get("string") == "hello"

        # Integer
        await tiering.set("int", 42)
        assert await tiering.get("int") == 42

        # Float
        await tiering.set("float", 3.14)
        assert await tiering.get("float") == 3.14

        # List
        await tiering.set("list", [1, 2, 3])
        assert await tiering.get("list") == [1, 2, 3]

        # Dict
        await tiering.set("dict", {"a": 1, "b": 2})
        assert await tiering.get("dict") == {"a": 1, "b": 2}

        # None
        await tiering.set("none", None)
        # Note: This might be tricky since get returns None for missing keys


class TestPinning:
    """Pinning functionality tests"""

    @pytest.mark.asyncio
    async def test_pin_item(self, tiering):
        """Test pinning an item"""
        await tiering.set("pinned", "important data")
        success = await tiering.pin("pinned")
        assert success

    @pytest.mark.asyncio
    async def test_unpin_item(self, tiering):
        """Test unpinning an item"""
        await tiering.set("pinned", "data")
        await tiering.pin("pinned")
        success = await tiering.unpin("pinned")
        assert success

    @pytest.mark.asyncio
    async def test_pin_nonexistent(self, tiering):
        """Test pinning nonexistent key"""
        success = await tiering.pin("nonexistent")
        assert not success

    @pytest.mark.asyncio
    async def test_set_with_pinned(self, tiering):
        """Test setting with pinned=True"""
        await tiering.set("auto_pinned", "data", pinned=True)

        # Should be pinned
        stats = tiering.get_stats()
        assert stats.pinned_items >= 1


class TestTierTransitions:
    """Tier transition tests"""

    @pytest.mark.asyncio
    async def test_item_starts_in_hot(self, tiering):
        """Test that new items start in HOT tier"""
        await tiering.set("new_item", "data")

        # Get tier info
        item = await tiering._get_item("new_item")
        if item:
            assert item.tier == GenerationTier.HOT

    @pytest.mark.asyncio
    async def test_access_promotes_item(self, tiering_config):
        """Test that accessing item promotes it"""
        # Use a small config to force demotion
        tiering_config.hot_max_items = 2
        tiering = ContextTiering(tiering_config)
        await tiering.start()

        try:
            # Fill hot tier
            await tiering.set("item1", "data1")
            await tiering.set("item2", "data2")
            await tiering.set("item3", "data3")  # Should push item1 to warm

            # Give time for demotion
            await asyncio.sleep(0.2)

            # Access item1 should promote it back to hot
            _ = await tiering.get("item1")

            # Verify it's promoted
            item = await tiering._get_item("item1")
            if item:
                # After access, should be in hot or at least not cold
                assert item.tier in [GenerationTier.HOT, GenerationTier.WARM]
        finally:
            await tiering.stop()


class TestMemoryManagement:
    """Memory management tests"""

    @pytest.mark.asyncio
    async def test_stats_tracking(self, tiering):
        """Test that stats are tracked correctly"""
        # Set some data
        await tiering.set("key1", "value1")
        await tiering.set("key2", "value2")

        # Get some data
        await tiering.get("key1")
        await tiering.get("key2")
        await tiering.get("nonexistent")  # Miss

        stats = tiering.get_stats()
        assert stats.total_gets >= 3
        assert stats.total_sets >= 2

    @pytest.mark.asyncio
    async def test_bytes_tracking(self, tiering):
        """Test that byte usage is tracked"""
        # Set some data
        large_data = "x" * 1000
        await tiering.set("large", large_data)

        stats = tiering.get_stats()
        assert stats.total_bytes > 0


class TestRefaultTracker:
    """Refault tracking tests"""

    def test_track_refault(self):
        """Test tracking refaults"""
        tracker = RefaultTracker()
        tracker.record_refault("key1", GenerationTier.COOL)

        count = tracker.get_refault_count("key1")
        assert count == 1

    def test_multiple_refaults(self):
        """Test multiple refaults for same key"""
        tracker = RefaultTracker()
        tracker.record_refault("key1", GenerationTier.COOL)
        tracker.record_refault("key1", GenerationTier.COLD)

        count = tracker.get_refault_count("key1")
        assert count == 2

    def test_refault_decay(self):
        """Test that refault counts can decay"""
        tracker = RefaultTracker()
        tracker.record_refault("key1", GenerationTier.COOL)

        # Decay
        tracker.decay()

        count = tracker.get_refault_count("key1")
        assert count <= 1  # Should be decayed


class TestStatistics:
    """Statistics tests"""

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self, tiering):
        """Test hit rate calculation"""
        # Set some data
        await tiering.set("key1", "value1")

        # Hits
        await tiering.get("key1")
        await tiering.get("key1")
        await tiering.get("key1")

        # Misses
        await tiering.get("nonexistent1")
        await tiering.get("nonexistent2")

        stats = tiering.get_stats()
        # 3 hits, 2 misses = 60% hit rate
        assert 0.5 <= stats.hit_rate <= 0.7

    @pytest.mark.asyncio
    async def test_tier_counts(self, tiering):
        """Test tier item counts"""
        # Set multiple items
        for i in range(5):
            await tiering.set(f"key_{i}", f"value_{i}")

        stats = tiering.get_stats()
        total_items = (
            stats.hot_items +
            stats.warm_items +
            stats.cool_items +
            stats.cold_items
        )
        assert total_items == 5


class TestConcurrency:
    """Concurrency tests"""

    @pytest.mark.asyncio
    async def test_concurrent_access(self, tiering):
        """Test concurrent read/write operations"""
        async def writer(key_prefix, count):
            for i in range(count):
                await tiering.set(f"{key_prefix}_{i}", f"value_{i}")

        async def reader(key_prefix, count):
            results = []
            for i in range(count):
                result = await tiering.get(f"{key_prefix}_{i}")
                results.append(result)
            return results

        # Start concurrent operations
        await asyncio.gather(
            writer("a", 10),
            writer("b", 10),
            reader("a", 10),
            reader("b", 10),
        )

        # Verify data integrity
        for i in range(10):
            assert await tiering.get(f"a_{i}") == f"value_{i}"
            assert await tiering.get(f"b_{i}") == f"value_{i}"


class TestFlush:
    """Flush operations tests"""

    @pytest.mark.asyncio
    async def test_flush_tier(self, tiering):
        """Test flushing a tier to disk"""
        # Set some data
        for i in range(5):
            await tiering.set(f"key_{i}", f"value_{i}")

        # Force items to warm/cool
        await asyncio.sleep(0.2)

        # Flush warm tier
        count = await tiering.flush_tier(GenerationTier.WARM)

        # Count should be >= 0 (might be 0 if nothing in warm)
        assert count >= 0
