# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Tests for BBX Hooks - eBPF-inspired dynamic workflow programming

These tests verify:
- Hook registration and unregistration
- Hook triggering at attach points
- Hook actions (continue, skip, block, transform)
- Hook priority ordering
- Hook verification
"""

import asyncio
import pytest
from unittest.mock import MagicMock

from blackbox.core.v2.hooks import (
    HookManager,
    HookDefinition,
    HookContext,
    HookResult,
    HookType,
    AttachPoint,
    HookAction,
    HookVerifier,
    get_hook_manager,
)


@pytest.fixture
def hook_manager():
    """Create a fresh HookManager for each test"""
    return HookManager()


@pytest.fixture
def sample_hook():
    """Create a sample hook definition"""
    def handler(ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.CONTINUE)

    return HookDefinition(
        id="test_hook",
        name="Test Hook",
        type=HookType.PROBE,
        attach_points=[AttachPoint.STEP_POST_EXECUTE],
        handler=handler,
    )


class TestHookRegistration:
    """Hook registration tests"""

    def test_register_hook(self, hook_manager, sample_hook):
        """Test registering a hook"""
        success = hook_manager.register(sample_hook)
        assert success

        hooks = hook_manager.list_hooks()
        assert len(hooks) == 1
        assert hooks[0].id == "test_hook"

    def test_register_duplicate_hook(self, hook_manager, sample_hook):
        """Test that duplicate hooks overwrite existing"""
        hook_manager.register(sample_hook)

        # Register again - will overwrite
        success = hook_manager.register(sample_hook)
        assert success  # Succeeds (overwrites)

        # Should still only have one hook
        hooks = hook_manager.list_hooks()
        assert len(hooks) == 1

    def test_unregister_hook(self, hook_manager, sample_hook):
        """Test unregistering a hook"""
        hook_manager.register(sample_hook)
        success = hook_manager.unregister("test_hook")
        assert success

        hooks = hook_manager.list_hooks()
        assert len(hooks) == 0

    def test_unregister_nonexistent_hook(self, hook_manager):
        """Test unregistering a hook that doesn't exist"""
        success = hook_manager.unregister("nonexistent")
        assert not success


class TestHookEnableDisable:
    """Hook enable/disable tests"""

    def test_disable_hook(self, hook_manager, sample_hook):
        """Test disabling a hook by modifying its enabled property"""
        hook_manager.register(sample_hook)
        hook = hook_manager.get_hook("test_hook")
        hook.enabled = False

        hooks = hook_manager.list_hooks()
        assert not hooks[0].enabled

    def test_enable_hook(self, hook_manager, sample_hook):
        """Test enabling a disabled hook"""
        hook_manager.register(sample_hook)
        hook = hook_manager.get_hook("test_hook")
        hook.enabled = False

        # Re-enable
        hook.enabled = True

        hooks = hook_manager.list_hooks()
        assert hooks[0].enabled


class TestHookTriggering:
    """Hook triggering tests"""

    @pytest.mark.asyncio
    async def test_trigger_probe_hook(self, hook_manager):
        """Test triggering a probe hook"""
        called = False

        def handler(ctx: HookContext) -> HookResult:
            nonlocal called
            called = True
            return HookResult(action=HookAction.CONTINUE)

        hook = HookDefinition(
            id="probe_hook",
            name="Probe Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_POST_EXECUTE],
            handler=handler,
        )
        hook_manager.register(hook)

        ctx = HookContext(
            workflow_id="test_workflow",
            step_id="test_step",
        )
        result = await hook_manager.trigger(AttachPoint.STEP_POST_EXECUTE, ctx)

        assert called
        assert result.action == HookAction.CONTINUE

    @pytest.mark.asyncio
    async def test_disabled_hook_not_triggered(self, hook_manager):
        """Test that disabled hooks are not triggered"""
        called = False

        def handler(ctx: HookContext) -> HookResult:
            nonlocal called
            called = True
            return HookResult(action=HookAction.CONTINUE)

        hook = HookDefinition(
            id="disabled_hook",
            name="Disabled Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_POST_EXECUTE],
            handler=handler,
        )
        hook_manager.register(hook)
        # Disable by setting enabled property
        hook_manager.get_hook("disabled_hook").enabled = False

        ctx = HookContext(workflow_id="test")
        await hook_manager.trigger(AttachPoint.STEP_POST_EXECUTE, ctx)

        assert not called

    @pytest.mark.asyncio
    async def test_wrong_attach_point_not_triggered(self, hook_manager):
        """Test that hooks with different attach points are not triggered"""
        called = False

        def handler(ctx: HookContext) -> HookResult:
            nonlocal called
            called = True
            return HookResult(action=HookAction.CONTINUE)

        hook = HookDefinition(
            id="wrong_point_hook",
            name="Wrong Point Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],  # Pre, not post
            handler=handler,
        )
        hook_manager.register(hook)

        ctx = HookContext(workflow_id="test")
        await hook_manager.trigger(AttachPoint.STEP_POST_EXECUTE, ctx)

        assert not called


class TestHookActions:
    """Hook action tests"""

    @pytest.mark.asyncio
    async def test_skip_action(self, hook_manager):
        """Test SKIP action from guard hook"""
        def handler(ctx: HookContext) -> HookResult:
            return HookResult(action=HookAction.SKIP, error="Skipped by hook")

        hook = HookDefinition(
            id="skip_hook",
            name="Skip Hook",
            type=HookType.GUARD,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            handler=handler,
        )
        hook_manager.register(hook)

        ctx = HookContext(workflow_id="test", step_id="test_step")
        result = await hook_manager.trigger(AttachPoint.STEP_PRE_EXECUTE, ctx)

        assert result.action == HookAction.SKIP

    @pytest.mark.asyncio
    async def test_block_action(self, hook_manager):
        """Test BLOCK action from guard hook"""
        def handler(ctx: HookContext) -> HookResult:
            return HookResult(action=HookAction.BLOCK, error="Blocked by security hook")

        hook = HookDefinition(
            id="security_hook",
            name="Security Hook",
            type=HookType.GUARD,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            handler=handler,
        )
        hook_manager.register(hook)

        ctx = HookContext(workflow_id="test", step_id="test_step")
        result = await hook_manager.trigger(AttachPoint.STEP_PRE_EXECUTE, ctx)

        assert result.action == HookAction.BLOCK
        assert "Blocked" in result.error

    @pytest.mark.asyncio
    async def test_transform_action(self, hook_manager):
        """Test TRANSFORM action from transform hook"""
        def handler(ctx: HookContext) -> HookResult:
            # Transform the inputs
            transformed = {"transformed": True, "original": ctx.step_inputs}
            return HookResult(action=HookAction.TRANSFORM, data=transformed)

        hook = HookDefinition(
            id="transform_hook",
            name="Transform Hook",
            type=HookType.TRANSFORM,
            attach_points=[AttachPoint.STEP_PRE_EXECUTE],
            handler=handler,
        )
        hook_manager.register(hook)

        ctx = HookContext(
            workflow_id="test",
            step_id="test_step",
            step_inputs={"key": "value"},
        )
        result = await hook_manager.trigger(AttachPoint.STEP_PRE_EXECUTE, ctx)

        assert result.action == HookAction.TRANSFORM
        assert result.data["transformed"] is True


class TestHookPriority:
    """Hook priority ordering tests"""

    @pytest.mark.asyncio
    async def test_priority_ordering(self, hook_manager):
        """Test that hooks execute in priority order"""
        execution_order = []

        def make_handler(name, priority):
            def handler(ctx: HookContext) -> HookResult:
                execution_order.append((name, priority))
                return HookResult(action=HookAction.CONTINUE)
            return handler

        # Register in reverse priority order
        for i, priority in enumerate([100, 0, -100]):
            hook = HookDefinition(
                id=f"hook_{i}",
                name=f"Hook {i}",
                type=HookType.PROBE,
                attach_points=[AttachPoint.STEP_POST_EXECUTE],
                priority=priority,
                handler=make_handler(f"hook_{i}", priority),
            )
            hook_manager.register(hook)

        ctx = HookContext(workflow_id="test")
        await hook_manager.trigger(AttachPoint.STEP_POST_EXECUTE, ctx)

        # Should execute in priority order (highest priority number first)
        # HookManager sorts by -priority, so higher numbers run first
        priorities = [p for _, p in execution_order]
        assert priorities == sorted(priorities, reverse=True)


class TestHookVerifier:
    """Hook verification tests"""

    def test_verify_valid_hook(self):
        """Test verification of valid hook"""
        def handler(ctx: HookContext) -> HookResult:
            return HookResult(action=HookAction.CONTINUE)

        hook = HookDefinition(
            id="valid_hook",
            name="Valid Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_POST_EXECUTE],
            handler=handler,
        )

        verifier = HookVerifier()
        result = verifier.verify(hook)

        assert result.passed
        assert len(result.errors) == 0

    def test_verify_hook_missing_id(self):
        """Test verification of hook without ID"""
        hook = HookDefinition(
            id="",  # Empty ID
            name="Invalid Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_POST_EXECUTE],
        )

        verifier = HookVerifier()
        result = verifier.verify(hook)

        assert not result.passed
        assert any("id" in e.lower() for e in result.errors)

    def test_verify_hook_no_attach_points(self):
        """Test verification of hook without attach points"""
        hook = HookDefinition(
            id="no_attach",
            name="No Attach",
            type=HookType.PROBE,
            attach_points=[],  # No attach points
        )

        verifier = HookVerifier()
        result = verifier.verify(hook)

        assert not result.passed
        assert any("attach" in e.lower() for e in result.errors)


class TestHookStatistics:
    """Hook statistics tests"""

    @pytest.mark.asyncio
    async def test_stats_collection(self, hook_manager):
        """Test that hook statistics are collected"""
        def handler(ctx: HookContext) -> HookResult:
            return HookResult(action=HookAction.CONTINUE)

        hook = HookDefinition(
            id="stats_hook",
            name="Stats Hook",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_POST_EXECUTE],
            handler=handler,
        )
        hook_manager.register(hook)

        # Trigger multiple times
        for _ in range(5):
            ctx = HookContext(workflow_id="test")
            await hook_manager.trigger(AttachPoint.STEP_POST_EXECUTE, ctx)

        stats = hook_manager.get_stats()
        # Stats is a dict per hook_id
        assert "stats_hook" in stats
        assert stats["stats_hook"].get("execution_count", 0) >= 5
