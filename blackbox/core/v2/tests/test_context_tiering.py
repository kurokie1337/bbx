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
        hot_max_size=100 * 1024,  # 100KB
        warm_max_size=1 * 1024 * 1024,  # 1MB
        cool_max_size=10 * 1024 * 1024,  # 10MB
        aging_interval=0.1,  # Fast for testing
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
        await tiering.delete("key1")  # delete returns None

        result = await tiering.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tiering):
        """Test deleting nonexistent key (no error)"""
        await tiering.delete("nonexistent")  # delete returns None, no error

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
        await tiering.pin("pinned")  # No return value

    @pytest.mark.asyncio
    async def test_unpin_item(self, tiering):
        """Test unpinning an item"""
        await tiering.set("pinned", "data")
        await tiering.pin("pinned")
        await tiering.unpin("pinned")  # No return value

    @pytest.mark.asyncio
    async def test_pin_nonexistent(self, tiering):
        """Test pinning nonexistent key raises KeyError"""
        with pytest.raises(KeyError):
            await tiering.pin("nonexistent")

    @pytest.mark.asyncio
    async def test_set_with_pinned(self, tiering):
        """Test setting with pinned=True"""
        await tiering.set("auto_pinned", "data", pinned=True)

        # Check stats (dict access)
        stats = tiering.get_stats()
        assert stats.get("total_items", 0) >= 1


class TestTierTransitions:
    """Tier transition tests"""

    @pytest.mark.asyncio
    async def test_item_starts_in_hot(self, tiering):
        """Test that new items start in HOT tier"""
        await tiering.set("new_item", "data")

        # Verify item is in hot tier by checking stats
        stats = tiering.get_stats()
        assert stats.get("generations", [])[0].get("items", 0) >= 1

    @pytest.mark.asyncio
    async def test_access_promotes_item(self, tiering_config):
        """Test that accessing item promotes it"""
        # Use a small config to force demotion
        tiering_config.hot_max_size = 100  # Very small to force demotion
        tiering = ContextTiering(tiering_config)
        await tiering.start()

        try:
            # Fill hot tier
            await tiering.set("item1", "data1")
            await tiering.set("item2", "data2")
            await tiering.set("item3", "data3")

            # Give time for demotion
            await asyncio.sleep(0.2)

            # Access item1 should promote it back to hot
            result = await tiering.get("item1")
            assert result == "data1"

            # Verify total items are still there
            stats = tiering.get_stats()
            assert stats.get("total_items", 0) >= 1
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
        # Stats track cache_hits and cache_misses
        total_accesses = stats.get("cache_hits", 0) + stats.get("cache_misses", 0)
        assert total_accesses >= 3

    @pytest.mark.asyncio
    async def test_bytes_tracking(self, tiering):
        """Test that byte usage is tracked"""
        # Set some data
        large_data = "x" * 1000
        await tiering.set("large", large_data)

        stats = tiering.get_stats()
        assert stats.get("total_size_bytes", 0) > 0


class TestRefaultTracker:
    """Refault tracking tests"""

    def test_track_refault(self):
        """Test tracking refaults (via record_access)"""
        tracker = RefaultTracker()
        # record_access with generation index 2 (COOL tier)
        tracker.record_access("key1", 2)

        distance = tracker.get_distance("key1")
        assert distance >= 0  # Distance should be trackable

    def test_multiple_refaults(self):
        """Test multiple refaults for same key"""
        tracker = RefaultTracker()
        tracker.record_access("key1", 2)  # COOL
        tracker.record_access("key1", 3)  # COLD

        distance = tracker.get_distance("key1")
        assert distance >= 0

    def test_access_frequency(self):
        """Test access frequency calculation"""
        from datetime import timedelta
        tracker = RefaultTracker()
        tracker.record_access("key1", 0)  # HOT access
        tracker.record_access("key1", 0)  # Another HOT access

        frequency = tracker.get_access_frequency("key1", timedelta(hours=1))
        assert frequency >= 0


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
        hit_rate = stats.get("hit_rate", 0)
        assert 0.5 <= hit_rate <= 0.7

    @pytest.mark.asyncio
    async def test_tier_counts(self, tiering):
        """Test tier item counts"""
        # Set multiple items
        for i in range(5):
            await tiering.set(f"key_{i}", f"value_{i}")

        stats = tiering.get_stats()
        total_items = stats.get("total_items", 0)
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
    async def test_context_persistence(self, tiering):
        """Test that context data persists across access"""
        # Set some data
        for i in range(5):
            await tiering.set(f"key_{i}", f"value_{i}")

        # Wait for aging to potentially move items
        await asyncio.sleep(0.2)

        # Verify all items still accessible
        for i in range(5):
            result = await tiering.get(f"key_{i}")
            assert result == f"value_{i}"
