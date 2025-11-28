# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Integration Tests for BBX 2.0

These tests verify that all BBX 2.0 components work together:
- Runtime with AgentRing, Hooks, and ContextTiering
- End-to-end workflow execution
- Component interaction
"""

import asyncio
import pytest
import tempfile
from pathlib import Path

from blackbox.core.v2.ring import AgentRing, Operation, OperationType, RingConfig
from blackbox.core.v2.hooks import (
    HookManager, HookDefinition, HookContext, HookResult,
    HookType, AttachPoint, HookAction
)
from blackbox.core.v2.context_tiering import ContextTiering, TieringConfig
from blackbox.core.v2.runtime_v2 import BBXRuntimeV2, RuntimeV2Config


class MockShellAdapter:
    """Mock shell adapter for integration tests"""

    def __init__(self):
        self.calls = []

    async def execute(self, method: str, args: dict):
        self.calls.append((method, args))

        if method == "run":
            command = args.get("command", "")
            return {
                "stdout": f"Executed: {command}",
                "stderr": "",
                "exit_code": 0,
            }
        elif method == "echo":
            return {"output": args.get("message", "")}

        return {"status": "ok"}


class MockHttpAdapter:
    """Mock HTTP adapter for integration tests"""

    def __init__(self):
        self.calls = []

    async def execute(self, method: str, args: dict):
        self.calls.append((method, args))

        if method == "get":
            return {
                "status_code": 200,
                "body": {"data": "mock response"},
            }
        elif method == "post":
            return {
                "status_code": 201,
                "body": {"id": 123},
            }

        return {"status": "ok"}


@pytest.fixture
def mock_adapters():
    """Create mock adapters for testing"""
    return {
        "shell": MockShellAdapter(),
        "http": MockHttpAdapter(),
        "bbx.shell": MockShellAdapter(),
        "bbx.http": MockHttpAdapter(),
    }


@pytest.fixture
def runtime_config():
    """Create runtime configuration for tests"""
    return RuntimeV2Config(
        ring_enabled=True,
        ring_config=RingConfig(
            worker_pool_size=2,
            max_batch_size=10,
        ),
        hooks_enabled=True,
        tiering_enabled=True,
        tiering_config=TieringConfig(
            hot_max_items=10,
            warm_max_items=20,
        ),
        parallel_execution=True,
    )


class TestRuntimeIntegration:
    """Runtime integration tests"""

    @pytest.mark.asyncio
    async def test_runtime_lifecycle(self, runtime_config):
        """Test runtime start/stop lifecycle"""
        runtime = BBXRuntimeV2(runtime_config)

        # Should start successfully
        await runtime.start()
        assert runtime._started
        assert runtime.ring is not None
        assert runtime.tiering is not None

        # Should stop successfully
        await runtime.stop()
        assert not runtime._started

    @pytest.mark.asyncio
    async def test_runtime_with_hooks(self, runtime_config):
        """Test runtime with custom hooks"""
        runtime = BBXRuntimeV2(runtime_config)
        await runtime.start()

        try:
            # Register a custom hook
            hook_triggered = False

            def custom_hook_handler(ctx: HookContext) -> HookResult:
                nonlocal hook_triggered
                hook_triggered = True
                return HookResult(action=HookAction.CONTINUE)

            custom_hook = HookDefinition(
                id="test_hook",
                name="Test Hook",
                type=HookType.PROBE,
                attach_points=[AttachPoint.WORKFLOW_START],
                handler=custom_hook_handler,
            )
            runtime.hooks.register(custom_hook)

            # Trigger the hook
            ctx = HookContext(workflow_id="test")
            await runtime.hooks.trigger(AttachPoint.WORKFLOW_START, ctx)

            assert hook_triggered
        finally:
            await runtime.stop()


class TestRingHooksIntegration:
    """AgentRing + Hooks integration tests"""

    @pytest.mark.asyncio
    async def test_ring_operations_with_hooks(self, mock_adapters):
        """Test that hooks are triggered during ring operations"""
        # Create ring
        ring_config = RingConfig(worker_pool_size=2)
        ring = AgentRing(ring_config)
        await ring.start(mock_adapters)

        # Create hook manager
        hook_manager = HookManager()
        operation_count = 0

        def operation_hook(ctx: HookContext) -> HookResult:
            nonlocal operation_count
            operation_count += 1
            return HookResult(action=HookAction.CONTINUE)

        hook = HookDefinition(
            id="op_counter",
            name="Operation Counter",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_POST_EXECUTE],
            handler=operation_hook,
        )
        hook_manager.register(hook)

        try:
            # Submit operations
            ops = [
                Operation(
                    op_type=OperationType.ADAPTER_CALL,
                    adapter="shell",
                    method="echo",
                    args={"message": f"test_{i}"},
                )
                for i in range(5)
            ]

            op_ids = await ring.submit_batch(ops)
            completions = await ring.wait_batch(op_ids, timeout=10.0)

            # Trigger hooks for each completion
            for c in completions:
                ctx = HookContext(
                    workflow_id="test",
                    step_id=c.operation_id,
                )
                await hook_manager.trigger(AttachPoint.STEP_POST_EXECUTE, ctx)

            assert operation_count == 5
        finally:
            await ring.stop()


class TestContextTieringIntegration:
    """ContextTiering integration with runtime tests"""

    @pytest.mark.asyncio
    async def test_tiering_stores_workflow_data(self):
        """Test that workflow data is stored in tiered context"""
        config = TieringConfig(hot_max_items=10)
        tiering = ContextTiering(config)
        await tiering.start()

        try:
            # Store workflow inputs
            await tiering.set("workflow_inputs", {"param1": "value1"}, pinned=True)
            await tiering.set("workflow_id", "test_workflow", pinned=True)

            # Store step outputs
            for i in range(5):
                await tiering.set(f"step_{i}_output", {"result": i})

            # Verify retrieval
            inputs = await tiering.get("workflow_inputs")
            assert inputs == {"param1": "value1"}

            step_0 = await tiering.get("step_0_output")
            assert step_0 == {"result": 0}

            # Check stats
            stats = tiering.get_stats()
            assert stats.pinned_items >= 2
        finally:
            await tiering.stop()


class TestEndToEndWorkflow:
    """End-to-end workflow tests"""

    @pytest.fixture
    def sample_workflow(self, tmp_path):
        """Create a sample workflow file"""
        workflow_content = """
workflow:
  id: integration_test
  name: Integration Test Workflow
  version: "1.0"

  inputs:
    message:
      type: string
      default: "Hello"

  steps:
    - id: step1
      mcp: bbx.shell
      method: echo
      inputs:
        message: "${inputs.message}"

    - id: step2
      mcp: bbx.shell
      method: echo
      inputs:
        message: "Step 2"
      depends_on:
        - step1

    - id: step3
      mcp: bbx.http
      method: get
      inputs:
        url: "https://example.com"
      depends_on:
        - step1
"""
        workflow_file = tmp_path / "test_workflow.bbx"
        workflow_file.write_text(workflow_content)
        return str(workflow_file)


class TestHookChaining:
    """Hook chaining and ordering tests"""

    @pytest.mark.asyncio
    async def test_multiple_hooks_chain(self):
        """Test that multiple hooks can chain their effects"""
        hook_manager = HookManager()
        execution_log = []

        # First hook: logs start
        def hook1_handler(ctx: HookContext) -> HookResult:
            execution_log.append("hook1_start")
            return HookResult(action=HookAction.CONTINUE)

        hook1 = HookDefinition(
            id="hook1",
            name="Hook 1",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            priority=-100,  # Runs first
            handler=hook1_handler,
        )

        # Second hook: can transform
        def hook2_handler(ctx: HookContext) -> HookResult:
            execution_log.append("hook2_transform")
            return HookResult(
                action=HookAction.TRANSFORM,
                data={"transformed": True}
            )

        hook2 = HookDefinition(
            id="hook2",
            name="Hook 2",
            type=HookType.TRANSFORM,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            priority=0,  # Runs second
            handler=hook2_handler,
        )

        # Third hook: logs end
        def hook3_handler(ctx: HookContext) -> HookResult:
            execution_log.append("hook3_end")
            return HookResult(action=HookAction.CONTINUE)

        hook3 = HookDefinition(
            id="hook3",
            name="Hook 3",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            priority=100,  # Runs last
            handler=hook3_handler,
        )

        hook_manager.register(hook1)
        hook_manager.register(hook2)
        hook_manager.register(hook3)

        ctx = HookContext(workflow_id="test", step_id="step1")
        result = await hook_manager.trigger(AttachPoint.STEP_PRE_EXECUTE, ctx)

        # Should execute in priority order
        assert execution_log == ["hook1_start", "hook2_transform", "hook3_end"]

        # Final result should be from last transform hook
        assert result.action == HookAction.CONTINUE  # Last hook's action


class TestRingBatchPerformance:
    """Performance tests for AgentRing batch operations"""

    @pytest.mark.asyncio
    async def test_batch_throughput(self, mock_adapters):
        """Test batch operation throughput"""
        ring_config = RingConfig(
            worker_pool_size=4,
            max_batch_size=100,
        )
        ring = AgentRing(ring_config)
        await ring.start(mock_adapters)

        try:
            import time

            # Submit 100 operations
            ops = [
                Operation(
                    op_type=OperationType.ADAPTER_CALL,
                    adapter="shell",
                    method="echo",
                    args={"message": f"test_{i}"},
                )
                for i in range(100)
            ]

            start = time.perf_counter()
            op_ids = await ring.submit_batch(ops)
            completions = await ring.wait_batch(op_ids, timeout=30.0)
            duration = time.perf_counter() - start

            # Verify all completed
            assert len(completions) == 100
            completed = sum(1 for c in completions if c.status.name == "COMPLETED")
            assert completed == 100

            # Should complete reasonably fast (< 5 seconds for 100 ops)
            assert duration < 5.0

            # Calculate throughput
            throughput = len(completions) / duration
            print(f"Throughput: {throughput:.2f} ops/sec")
        finally:
            await ring.stop()


class TestGuardHooksBlocking:
    """Guard hook blocking tests"""

    @pytest.mark.asyncio
    async def test_security_hook_blocks(self):
        """Test that security hooks can block execution"""
        hook_manager = HookManager()

        # Security hook that blocks certain steps
        def security_handler(ctx: HookContext) -> HookResult:
            if ctx.step_id and "dangerous" in ctx.step_id:
                return HookResult(
                    action=HookAction.BLOCK,
                    error="Blocked by security policy"
                )
            return HookResult(action=HookAction.CONTINUE)

        security_hook = HookDefinition(
            id="security",
            name="Security Hook",
            type=HookType.GUARD,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            priority=-1000,  # Runs first
            handler=security_handler,
        )
        hook_manager.register(security_hook)

        # Normal step should pass
        ctx1 = HookContext(workflow_id="test", step_id="normal_step")
        result1 = await hook_manager.trigger(AttachPoint.STEP_PRE_EXECUTE, ctx1)
        assert result1.action == HookAction.CONTINUE

        # Dangerous step should be blocked
        ctx2 = HookContext(workflow_id="test", step_id="dangerous_step")
        result2 = await hook_manager.trigger(AttachPoint.STEP_PRE_EXECUTE, ctx2)
        assert result2.action == HookAction.BLOCK
        assert "security" in result2.error.lower()
