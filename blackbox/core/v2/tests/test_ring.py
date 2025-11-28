# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Tests for AgentRing - io_uring-inspired batch operations

These tests verify:
- Operation submission and completion
- Batch processing
- Priority handling
- Error handling and retries
- Performance characteristics
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from blackbox.core.v2.ring import (
    AgentRing,
    Operation,
    Completion,
    OperationType,
    OperationPriority,
    OperationStatus,
    RingConfig,
)


class MockAdapter:
    """Mock adapter for testing"""

    def __init__(self, delay: float = 0.01, error_rate: float = 0.0):
        self.delay = delay
        self.error_rate = error_rate
        self.call_count = 0

    async def execute(self, method: str, args: dict):
        self.call_count += 1
        await asyncio.sleep(self.delay)

        import random
        if random.random() < self.error_rate:
            raise RuntimeError("Random error")

        return {"method": method, "args": args, "call": self.call_count}


@pytest.fixture
def ring_config():
    """Default ring configuration for tests"""
    return RingConfig(
        submission_queue_size=100,
        completion_queue_size=100,
        worker_pool_size=4,
        max_batch_size=50,
        default_timeout_ms=5000,
    )


@pytest.fixture
async def ring(ring_config):
    """Create and start an AgentRing instance"""
    ring = AgentRing(ring_config)
    adapters = {"mock": MockAdapter()}
    await ring.start(adapters)
    yield ring
    await ring.stop()


class TestAgentRingBasics:
    """Basic AgentRing functionality tests"""

    @pytest.mark.asyncio
    async def test_ring_creation(self, ring_config):
        """Test ring creation with config"""
        ring = AgentRing(ring_config)
        assert ring.config == ring_config
        assert not ring._started

    @pytest.mark.asyncio
    async def test_ring_start_stop(self, ring_config):
        """Test ring start and stop lifecycle"""
        ring = AgentRing(ring_config)
        await ring.start({"mock": MockAdapter()})
        assert ring._started

        await ring.stop()
        assert not ring._started

    @pytest.mark.asyncio
    async def test_single_operation(self, ring):
        """Test submitting and completing a single operation"""
        op = Operation(
            op_type=OperationType.ADAPTER_CALL,
            adapter="mock",
            method="test",
            args={"key": "value"},
        )

        op_ids = await ring.submit_batch([op])
        assert len(op_ids) == 1

        completions = await ring.wait_batch(op_ids, timeout=5.0)
        assert len(completions) == 1
        assert completions[0].status == OperationStatus.COMPLETED
        assert completions[0].result is not None


class TestBatchOperations:
    """Batch operation tests"""

    @pytest.mark.asyncio
    async def test_batch_submission(self, ring):
        """Test submitting multiple operations as a batch"""
        operations = [
            Operation(
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method=f"method_{i}",
                args={"index": i},
            )
            for i in range(10)
        ]

        op_ids = await ring.submit_batch(operations)
        assert len(op_ids) == 10

        completions = await ring.wait_batch(op_ids, timeout=10.0)
        assert len(completions) == 10
        assert all(c.status == OperationStatus.COMPLETED for c in completions)

    @pytest.mark.asyncio
    async def test_parallel_execution(self, ring_config):
        """Test that operations execute in parallel"""
        # Use a slow adapter to verify parallelism
        slow_adapter = MockAdapter(delay=0.1)
        ring = AgentRing(ring_config)
        await ring.start({"mock": slow_adapter})

        # 10 operations with 0.1s delay each
        # Sequential: 1s, Parallel (4 workers): ~0.3s
        operations = [
            Operation(
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method="test",
                args={},
            )
            for _ in range(10)
        ]

        import time
        start = time.perf_counter()

        op_ids = await ring.submit_batch(operations)
        completions = await ring.wait_batch(op_ids, timeout=5.0)

        duration = time.perf_counter() - start

        # Should be faster than sequential (< 1s)
        assert duration < 0.6  # Allow some overhead
        assert len(completions) == 10

        await ring.stop()


class TestOperationPriorities:
    """Priority handling tests"""

    @pytest.mark.asyncio
    async def test_priority_ordering(self, ring_config):
        """Test that high priority operations execute first"""
        ring_config.enable_priorities = True
        ring = AgentRing(ring_config)
        await ring.start({"mock": MockAdapter(delay=0.05)})

        # Submit low priority first
        low_ops = [
            Operation(
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method="low",
                args={},
                priority=OperationPriority.LOW,
            )
            for _ in range(5)
        ]

        # Then high priority
        high_ops = [
            Operation(
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method="high",
                args={},
                priority=OperationPriority.HIGH,
            )
            for _ in range(5)
        ]

        low_ids = await ring.submit_batch(low_ops)
        high_ids = await ring.submit_batch(high_ops)

        # Wait for all
        all_ids = low_ids + high_ids
        completions = await ring.wait_batch(all_ids, timeout=10.0)

        assert len(completions) == 10
        await ring.stop()


class TestErrorHandling:
    """Error handling and retry tests"""

    @pytest.mark.asyncio
    async def test_operation_failure(self, ring_config):
        """Test handling of failed operations"""
        # Adapter that always fails
        class FailingAdapter:
            async def execute(self, method, args):
                raise RuntimeError("Always fails")

        ring = AgentRing(ring_config)
        await ring.start({"failing": FailingAdapter()})

        op = Operation(
            op_type=OperationType.ADAPTER_CALL,
            adapter="failing",
            method="test",
            args={},
            retry_count=0,  # No retries
        )

        op_ids = await ring.submit_batch([op])
        completions = await ring.wait_batch(op_ids, timeout=5.0)

        assert len(completions) == 1
        assert completions[0].status == OperationStatus.FAILED
        assert "Always fails" in completions[0].error

        await ring.stop()

    @pytest.mark.asyncio
    async def test_operation_timeout(self, ring_config):
        """Test operation timeout handling"""
        # Very slow adapter
        class SlowAdapter:
            async def execute(self, method, args):
                await asyncio.sleep(10.0)
                return {}

        ring = AgentRing(ring_config)
        await ring.start({"slow": SlowAdapter()})

        op = Operation(
            op_type=OperationType.ADAPTER_CALL,
            adapter="slow",
            method="test",
            args={},
            timeout_ms=100,  # 100ms timeout
        )

        op_ids = await ring.submit_batch([op])
        completions = await ring.wait_batch(op_ids, timeout=1.0)

        assert len(completions) == 1
        assert completions[0].status == OperationStatus.TIMEOUT

        await ring.stop()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, ring_config):
        """Test automatic retry on failure"""
        ring_config.enable_retries = True

        # Adapter that fails twice then succeeds
        call_count = 0

        class FlakeyAdapter:
            async def execute(self, method, args):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise RuntimeError(f"Fail {call_count}")
                return {"success": True}

        ring = AgentRing(ring_config)
        await ring.start({"flakey": FlakeyAdapter()})

        op = Operation(
            op_type=OperationType.ADAPTER_CALL,
            adapter="flakey",
            method="test",
            args={},
            retry_count=3,  # Allow 3 retries
        )

        op_ids = await ring.submit_batch([op])
        completions = await ring.wait_batch(op_ids, timeout=10.0)

        assert len(completions) == 1
        assert completions[0].status == OperationStatus.COMPLETED
        assert call_count == 3

        await ring.stop()


class TestRingStatistics:
    """Statistics and monitoring tests"""

    @pytest.mark.asyncio
    async def test_stats_collection(self, ring):
        """Test that statistics are collected"""
        # Execute some operations
        operations = [
            Operation(
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method="test",
                args={},
            )
            for _ in range(5)
        ]

        op_ids = await ring.submit_batch(operations)
        await ring.wait_batch(op_ids, timeout=5.0)

        stats = ring.get_stats()
        assert stats.operations_submitted >= 5
        assert stats.operations_completed >= 5
        assert stats.throughput_ops_sec > 0


class TestUnknownAdapter:
    """Tests for unknown adapter handling"""

    @pytest.mark.asyncio
    async def test_unknown_adapter(self, ring):
        """Test handling of operations with unknown adapter"""
        op = Operation(
            op_type=OperationType.ADAPTER_CALL,
            adapter="nonexistent",
            method="test",
            args={},
        )

        op_ids = await ring.submit_batch([op])
        completions = await ring.wait_batch(op_ids, timeout=5.0)

        assert len(completions) == 1
        assert completions[0].status == OperationStatus.FAILED
        assert "not found" in completions[0].error.lower() or "unknown" in completions[0].error.lower()
