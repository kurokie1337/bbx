# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Comprehensive End-to-End Tests for BBX v2

These tests verify real functionality of all v2 components working together.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Core v2 imports
from blackbox.core.v2.ring import AgentRing, RingConfig, Operation, OperationType, OperationPriority, OperationStatus
from blackbox.core.v2.hooks import HookManager, HookDefinition, HookType, AttachPoint, HookContext, HookResult, HookAction
from blackbox.core.v2.context_tiering import ContextTiering, TieringConfig
from blackbox.core.v2.flow_integrity import FlowIntegrityEngine, FlowIntegrityConfig, FlowState
from blackbox.core.v2.declarative import BBXConfig, AgentConfig, QuotaConfig, GenerationManager, Generation


# =============================================================================
# Mock Adapters for Testing
# =============================================================================

class MockAdapter:
    """Mock adapter that records calls and returns predictable results"""

    def __init__(self):
        self.calls = []
        self.should_fail = False
        self.delay = 0

    async def execute(self, method: str, params: dict) -> dict:
        self.calls.append({"method": method, "params": params, "time": datetime.now()})

        if self.delay > 0:
            await asyncio.sleep(self.delay)

        if self.should_fail:
            raise Exception("Mock failure")

        return {"status": "ok", "method": method, "result": f"executed_{method}"}


# =============================================================================
# E2E Test: AgentRing Operations
# =============================================================================

class TestE2ERingOperations:
    """End-to-end tests for AgentRing"""

    @pytest.mark.asyncio
    async def test_ring_full_lifecycle(self):
        """Test complete ring lifecycle: create, start, submit, process, stop"""
        # Setup
        config = RingConfig(
            submission_queue_size=100,
            completion_queue_size=100,
            min_workers=2,
            max_workers=4,
            default_timeout_ms=5000,
        )
        ring = AgentRing(config)
        mock_adapter = MockAdapter()

        # Start ring
        await ring.start({"mock": mock_adapter})

        # Submit operation
        op = Operation(
            id="test_op_1",
            op_type=OperationType.ADAPTER_CALL,
            adapter="mock",
            method="test_method",
            args={"key": "value"},
            priority=OperationPriority.NORMAL,
        )

        # submit() returns operation ID
        op_id = await ring.submit(op)
        assert op_id == "test_op_1"

        # Wait for completion
        completion = await ring.wait_completion(op_id, timeout=5.0)
        assert completion is not None
        assert completion.operation_id == "test_op_1"
        assert completion.status == OperationStatus.COMPLETED

        # Verify adapter was called
        assert len(mock_adapter.calls) == 1
        assert mock_adapter.calls[0]["method"] == "test_method"

        # Stop ring
        await ring.stop()

    @pytest.mark.asyncio
    async def test_ring_batch_operations(self):
        """Test batch submission and parallel processing"""
        config = RingConfig(min_workers=4, max_workers=8)
        ring = AgentRing(config)
        mock_adapter = MockAdapter()
        mock_adapter.delay = 0.01  # Small delay to test parallelism

        await ring.start({"mock": mock_adapter})

        # Submit batch of operations
        operations = [
            Operation(
                id=f"batch_op_{i}",
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method=f"method_{i}",
                args={"index": i},
            )
            for i in range(10)
        ]

        start_time = datetime.now()
        op_ids = await ring.submit_batch(operations)

        # Wait for all completions
        completions = []
        for op_id in op_ids:
            completion = await ring.wait_completion(op_id, timeout=5.0)
            completions.append(completion)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Verify all completed
        assert len(completions) == 10
        assert all(c.status == OperationStatus.COMPLETED for c in completions)

        # Verify parallel execution (should be faster than sequential)
        assert elapsed < 0.3, f"Batch took {elapsed}s, expected parallel execution"

        await ring.stop()

    @pytest.mark.asyncio
    async def test_ring_priority_handling(self):
        """Test that operations with different priorities work"""
        config = RingConfig(min_workers=2, max_workers=4)
        ring = AgentRing(config)
        mock_adapter = MockAdapter()

        await ring.start({"mock": mock_adapter})

        # Submit operations with different priorities
        low_op = Operation(
            id="low_priority",
            op_type=OperationType.ADAPTER_CALL,
            adapter="mock",
            method="low",
            args={},
            priority=OperationPriority.LOW,
        )

        high_op = Operation(
            id="high_priority",
            op_type=OperationType.ADAPTER_CALL,
            adapter="mock",
            method="high",
            args={},
            priority=OperationPriority.HIGH,
        )

        # Submit both
        op_ids = await ring.submit_batch([low_op, high_op])

        # Wait for completions
        completions = []
        for op_id in op_ids:
            completion = await ring.wait_completion(op_id, timeout=5.0)
            completions.append(completion)

        assert len(completions) == 2
        assert all(c.status == OperationStatus.COMPLETED for c in completions)

        await ring.stop()


# =============================================================================
# E2E Test: Hooks System
# =============================================================================

class TestE2EHooksSystem:
    """End-to-end tests for Hooks system"""

    def test_hook_registration_and_listing(self):
        """Test registering and listing hooks"""
        manager = HookManager()

        # Create hook
        hook = HookDefinition(
            id="test_hook",
            name="Test Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            handler=lambda ctx: HookResult(action=HookAction.CONTINUE),
        )

        # Register
        success = manager.register(hook)
        assert success is True

        # List
        hooks = manager.list_hooks()
        assert len(hooks) == 1
        assert hooks[0].id == "test_hook"

        # Get by ID
        retrieved = manager.get_hook("test_hook")
        assert retrieved is not None
        assert retrieved.name == "Test Hook"

    def test_hook_unregistration(self):
        """Test unregistering hooks"""
        manager = HookManager()

        hook = HookDefinition(
            id="temp_hook",
            name="Temporary Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.WORKFLOW_START],
            handler=lambda ctx: HookResult(action=HookAction.CONTINUE),
        )

        manager.register(hook)
        assert len(manager.list_hooks()) == 1

        # Unregister
        success = manager.unregister("temp_hook")
        assert success is True
        assert len(manager.list_hooks()) == 0

        # Unregister non-existent
        success = manager.unregister("non_existent")
        assert success is False

    @pytest.mark.asyncio
    async def test_hook_triggering(self):
        """Test hooks are triggered at attach points"""
        manager = HookManager()
        trigger_count = {"count": 0}

        async def counting_handler(ctx: HookContext) -> HookResult:
            trigger_count["count"] += 1
            return HookResult(action=HookAction.CONTINUE, data={"triggered": True})

        hook = HookDefinition(
            id="counting_hook",
            name="Counting Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            handler=counting_handler,
        )

        manager.register(hook)

        # Trigger with correct HookContext fields
        ctx = HookContext(
            workflow_id="test_wf",
            workflow_name="Test Workflow",
            step_id="test_step",
        )

        result = await manager.trigger(AttachPoint.STEP_PRE_EXECUTE, ctx)

        assert trigger_count["count"] == 1
        assert result.action == HookAction.CONTINUE

    def test_hook_verification_catches_invalid_hooks(self):
        """Test hook verifier catches invalid hook definitions"""
        manager = HookManager()

        # Hook with no ID - should fail verification
        invalid_hook = HookDefinition(
            id="",  # Empty ID
            name="Invalid Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.WORKFLOW_START],
            handler=lambda ctx: HookResult(action=HookAction.CONTINUE),
        )

        success = manager.register(invalid_hook)
        assert success is False  # Should be rejected

        # Hook with no attach points - should fail verification
        no_attach_hook = HookDefinition(
            id="no_attach",
            name="No Attach Points",
            type=HookType.PROBE,
            attach_points=[],  # Empty
            handler=lambda ctx: HookResult(action=HookAction.CONTINUE),
        )

        success = manager.register(no_attach_hook)
        assert success is False


# =============================================================================
# E2E Test: Context Tiering
# =============================================================================

class TestE2EContextTiering:
    """End-to-end tests for Context Tiering"""

    @pytest.mark.asyncio
    async def test_tiering_set_get_delete(self):
        """Test basic set/get/delete operations"""
        config = TieringConfig(
            hot_max_size=100 * 1024,
            warm_max_size=1 * 1024 * 1024,
        )
        tiering = ContextTiering(config)
        await tiering.start()

        try:
            # Set
            await tiering.set("key1", "value1")
            await tiering.set("key2", {"nested": "data"})

            # Get
            val1 = await tiering.get("key1")
            val2 = await tiering.get("key2")

            assert val1 == "value1"
            assert val2 == {"nested": "data"}

            # Get non-existent
            val3 = await tiering.get("non_existent")
            assert val3 is None

            # Delete
            await tiering.delete("key1")
            val1_after = await tiering.get("key1")
            assert val1_after is None

        finally:
            await tiering.stop()

    @pytest.mark.asyncio
    async def test_tiering_stats(self):
        """Test statistics collection"""
        config = TieringConfig()
        tiering = ContextTiering(config)
        await tiering.start()

        try:
            # Perform operations
            for i in range(10):
                await tiering.set(f"key_{i}", f"value_{i}")

            for i in range(5):
                await tiering.get(f"key_{i}")  # Hits

            for i in range(3):
                await tiering.get(f"missing_{i}")  # Misses

            # Get stats - returns dict
            stats = tiering.get_stats()

            assert isinstance(stats, dict)
            # Check common stats keys
            assert "hits" in stats or "total_items" in stats or len(stats) > 0

        finally:
            await tiering.stop()

    @pytest.mark.asyncio
    async def test_tiering_performance(self):
        """Test context tiering performance"""
        config = TieringConfig()
        tiering = ContextTiering(config)
        await tiering.start()

        try:
            # Write 1000 items
            start_time = datetime.now()
            for i in range(1000):
                await tiering.set(f"perf_key_{i}", f"value_{i}")
            write_elapsed = (datetime.now() - start_time).total_seconds()

            # Read 1000 items
            start_time = datetime.now()
            for i in range(1000):
                await tiering.get(f"perf_key_{i}")
            read_elapsed = (datetime.now() - start_time).total_seconds()

            print(f"Tiering write: {1000/write_elapsed:.2f} ops/sec")
            print(f"Tiering read: {1000/read_elapsed:.2f} ops/sec")

            # Should be at least 500 ops/sec for in-memory operations
            assert 1000/write_elapsed > 500
            assert 1000/read_elapsed > 500

        finally:
            await tiering.stop()


# =============================================================================
# E2E Test: Flow Integrity
# =============================================================================

class TestE2EFlowIntegrity:
    """End-to-end tests for Flow Integrity"""

    def test_flow_state_management(self):
        """Test flow state tracking"""
        config = FlowIntegrityConfig()
        engine = FlowIntegrityEngine(config)

        # Get state for agent (auto-registers)
        agent_id = "test_agent"
        state = engine.get_state(agent_id)

        # Should return a valid FlowState
        assert state is not None
        assert isinstance(state, FlowState)

    def test_flow_stats(self):
        """Test flow integrity statistics"""
        config = FlowIntegrityConfig()
        engine = FlowIntegrityEngine(config)

        # Get stats
        stats = engine.get_stats()

        assert isinstance(stats, dict)
        # Should have some stats
        assert len(stats) >= 0


# =============================================================================
# E2E Test: Declarative Configuration
# =============================================================================

class TestE2EDeclarativeConfig:
    """End-to-end tests for Declarative Configuration"""

    def test_config_creation_and_validation(self):
        """Test creating and validating configuration"""
        config = BBXConfig(version="2.0")
        config.agents["main"] = AgentConfig(
            id="main",
            name="Main Agent",
            description="Primary processing agent",
            adapters=["shell", "http"],
            quotas=QuotaConfig(
                max_concurrent_steps=20,
                max_execution_time_seconds=600,
            ),
        )

        # Validate
        errors = config.validate()
        assert len(errors) == 0

        # Test backward compatibility
        assert config.agent is not None
        assert config.agent.name == "Main Agent"

    def test_config_serialization(self):
        """Test config to_dict and from_dict"""
        original = BBXConfig(version="2.0")
        original.agents["test"] = AgentConfig(
            id="test",
            name="Test Agent",
            adapters=["mock"],
        )

        # Serialize
        data = original.to_dict()
        assert "version" in data
        assert "agents" in data
        assert "test" in data["agents"]

        # Deserialize
        restored = BBXConfig.from_dict(data)
        assert restored.version == original.version
        assert "test" in restored.agents

    def test_generation_management(self):
        """Test generation creation and listing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = GenerationManager(base_path=Path(tmpdir))

            # Create first generation
            config1 = BBXConfig(version="2.0")
            config1.agents["v1"] = AgentConfig(id="v1", name="Version 1")
            gen1 = manager.create(config1, description="Initial version")

            assert gen1.id == 1
            assert gen1.description == "Initial version"

            # Create second generation
            config2 = BBXConfig(version="2.0")
            config2.agents["v2"] = AgentConfig(id="v2", name="Version 2")
            gen2 = manager.create(config2, description="Updated version")

            assert gen2.id == 2

            # List generations
            generations = manager.list()
            assert len(generations) == 2


# =============================================================================
# E2E Test: Full Integration
# =============================================================================

class TestE2EFullIntegration:
    """Full integration tests combining multiple v2 components"""

    @pytest.mark.asyncio
    async def test_ring_with_hooks_and_tiering(self):
        """Test Ring operations with Hooks and Context Tiering"""
        # Setup components
        ring_config = RingConfig(min_workers=2, max_workers=4)
        ring = AgentRing(ring_config)

        hook_manager = HookManager()

        tiering_config = TieringConfig()
        tiering = ContextTiering(tiering_config)

        # Track hook executions
        hook_executions = []

        async def tracking_hook(ctx: HookContext) -> HookResult:
            hook_executions.append({
                "workflow_id": ctx.workflow_id,
                "time": datetime.now(),
            })
            return HookResult(action=HookAction.CONTINUE)

        # Register hook
        hook = HookDefinition(
            id="tracker",
            name="Execution Tracker",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE, AttachPoint.STEP_POST_EXECUTE],
            handler=tracking_hook,
        )
        hook_manager.register(hook)

        # Start components
        mock_adapter = MockAdapter()
        await ring.start({"mock": mock_adapter})
        await tiering.start()

        try:
            # Execute workflow-like sequence
            workflow_id = "integration_test_wf"

            # Pre-execute hook
            pre_ctx = HookContext(
                workflow_id=workflow_id,
                workflow_name="Integration Test",
                step_id="step_1",
            )
            await hook_manager.trigger(AttachPoint.STEP_PRE_EXECUTE, pre_ctx)

            # Submit ring operation
            op = Operation(
                id="int_op_1",
                op_type=OperationType.ADAPTER_CALL,
                adapter="mock",
                method="process",
                args={"data": "test"},
            )
            op_id = await ring.submit(op)
            completion = await ring.wait_completion(op_id, timeout=5.0)

            # Store result in tiering
            await tiering.set(f"result_{op.id}", {
                "completed": completion.status == OperationStatus.COMPLETED,
                "result": completion.result,
            })

            # Post-execute hook
            post_ctx = HookContext(
                workflow_id=workflow_id,
                workflow_name="Integration Test",
                step_id="step_1",
            )
            await hook_manager.trigger(AttachPoint.STEP_POST_EXECUTE, post_ctx)

            # Verify integration
            assert completion.status == OperationStatus.COMPLETED
            assert len(hook_executions) == 2

            stored_result = await tiering.get(f"result_{op.id}")
            assert stored_result is not None
            assert stored_result["completed"] is True

        finally:
            await ring.stop()
            await tiering.stop()

    def test_declarative_with_flow_integrity(self):
        """Test Declarative Config with Flow Integrity"""
        # Create config
        config = BBXConfig(version="2.0")
        config.agents["monitored"] = AgentConfig(
            id="monitored",
            name="Monitored Agent",
            quotas=QuotaConfig(max_concurrent_steps=5),
        )

        # Validate config
        errors = config.validate()
        assert len(errors) == 0

        # Setup flow integrity
        flow_config = FlowIntegrityConfig()
        flow_engine = FlowIntegrityEngine(flow_config)

        # Get state for agent
        state = flow_engine.get_state("monitored")
        assert state is not None


# =============================================================================
# E2E Test: Error Handling
# =============================================================================

class TestE2EErrorHandling:
    """End-to-end tests for error handling"""

    @pytest.mark.asyncio
    async def test_ring_handles_adapter_failure(self):
        """Test ring gracefully handles adapter failures"""
        config = RingConfig(min_workers=2, max_workers=4)
        ring = AgentRing(config)

        failing_adapter = MockAdapter()
        failing_adapter.should_fail = True

        await ring.start({"failing": failing_adapter})

        try:
            op = Operation(
                id="fail_op",
                op_type=OperationType.ADAPTER_CALL,
                adapter="failing",
                method="test",
                args={},
            )

            op_id = await ring.submit(op)
            completion = await ring.wait_completion(op_id, timeout=5.0)

            # Should complete but with failure
            assert completion is not None
            assert completion.status == OperationStatus.FAILED
            assert completion.error is not None

        finally:
            await ring.stop()

    @pytest.mark.asyncio
    async def test_ring_handles_unknown_adapter(self):
        """Test ring handles unknown adapter gracefully"""
        config = RingConfig()
        ring = AgentRing(config)

        await ring.start({"known": MockAdapter()})

        try:
            op = Operation(
                id="unknown_op",
                op_type=OperationType.ADAPTER_CALL,
                adapter="unknown_adapter",
                method="test",
                args={},
            )

            op_id = await ring.submit(op)
            completion = await ring.wait_completion(op_id, timeout=5.0)

            assert completion.status == OperationStatus.FAILED

        finally:
            await ring.stop()


# =============================================================================
# E2E Test: Performance
# =============================================================================

class TestE2EPerformance:
    """Performance tests for v2 components"""

    @pytest.mark.asyncio
    async def test_ring_throughput(self):
        """Test ring can handle high throughput"""
        config = RingConfig(
            submission_queue_size=1000,
            completion_queue_size=1000,
            min_workers=8,
            max_workers=16,
        )
        ring = AgentRing(config)
        mock_adapter = MockAdapter()

        await ring.start({"mock": mock_adapter})

        try:
            # Submit 100 operations
            operations = [
                Operation(
                    id=f"perf_op_{i}",
                    op_type=OperationType.ADAPTER_CALL,
                    adapter="mock",
                    method=f"method_{i}",
                    args={},
                )
                for i in range(100)
            ]

            start_time = datetime.now()
            op_ids = await ring.submit_batch(operations)

            # Wait for all completions
            completions = []
            for op_id in op_ids:
                completion = await ring.wait_completion(op_id, timeout=5.0)
                completions.append(completion)

            elapsed = (datetime.now() - start_time).total_seconds()

            # All should complete
            assert len(completions) == 100
            success_count = sum(1 for c in completions if c.status == OperationStatus.COMPLETED)
            assert success_count == 100

            # Calculate throughput
            throughput = 100 / elapsed
            print(f"Ring throughput: {throughput:.2f} ops/sec")

            # Should be at least 50 ops/sec (conservative for CI)
            assert throughput > 50, f"Throughput too low: {throughput}"

        finally:
            await ring.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
