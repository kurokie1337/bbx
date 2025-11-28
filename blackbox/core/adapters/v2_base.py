# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 Adapter Base Classes

Provides base classes and mixins for BBX 2.0 compatible adapters:
- V2AdapterMixin: Adds BBX 2.0 features to existing adapters
- V2BaseAdapter: Base class for new v2-compatible adapters
- Hook integration points
- Context tiering support
- Operation batching support
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger("bbx.adapters.v2")


# =============================================================================
# Adapter Metadata
# =============================================================================

@dataclass
class AdapterMetadata:
    """Metadata for BBX adapters"""

    name: str
    version: str = "1.0"
    description: str = ""
    author: str = ""

    # V2 features
    supports_batching: bool = True
    supports_hooks: bool = True
    supports_tiering: bool = True

    # Methods
    methods: Dict[str, MethodMetadata] = field(default_factory=dict)


@dataclass
class MethodMetadata:
    """Metadata for adapter methods"""

    name: str
    description: str = ""
    async_only: bool = True

    # Parameters
    params: Dict[str, ParamMetadata] = field(default_factory=dict)

    # V2 features
    supports_timeout: bool = True
    supports_retry: bool = True
    idempotent: bool = False


@dataclass
class ParamMetadata:
    """Metadata for method parameters"""

    name: str
    type: str = "any"
    required: bool = False
    default: Any = None
    description: str = ""


# =============================================================================
# V2 Adapter Mixin
# =============================================================================

class V2AdapterMixin:
    """
    Mixin that adds BBX 2.0 features to existing adapters.

    Usage:
        class MyAdapter(V2AdapterMixin, BaseAdapter):
            ...

    Features:
        - Hook integration points
        - Context tiering support
        - Operation metadata tracking
        - Batch operation support
    """

    # V2 metadata
    _v2_metadata: Optional[AdapterMetadata] = None
    _v2_context: Optional[Any] = None  # ContextTiering instance
    _v2_hooks: Optional[Any] = None  # HookManager instance

    def v2_init(
        self,
        metadata: Optional[AdapterMetadata] = None,
        context: Optional[Any] = None,
        hooks: Optional[Any] = None,
    ):
        """Initialize V2 features"""
        self._v2_metadata = metadata
        self._v2_context = context
        self._v2_hooks = hooks

    def v2_set_context(self, context: Any):
        """Set context tiering instance"""
        self._v2_context = context

    def v2_set_hooks(self, hooks: Any):
        """Set hook manager instance"""
        self._v2_hooks = hooks

    async def v2_get_from_context(self, key: str) -> Optional[Any]:
        """Get value from tiered context"""
        if self._v2_context:
            return await self._v2_context.get(key)
        return None

    async def v2_set_in_context(self, key: str, value: Any, pinned: bool = False):
        """Set value in tiered context"""
        if self._v2_context:
            await self._v2_context.set(key, value, pinned=pinned)

    async def v2_trigger_hook(
        self,
        attach_point: str,
        step_id: Optional[str] = None,
        inputs: Optional[Dict] = None,
        outputs: Optional[Any] = None,
        error: Optional[str] = None,
    ):
        """Trigger hooks at specified attach point"""
        if self._v2_hooks:
            from blackbox.core.v2.hooks import HookContext, AttachPoint

            try:
                ap = AttachPoint(attach_point)
            except ValueError:
                logger.warning(f"Unknown attach point: {attach_point}")
                return

            ctx = HookContext(
                workflow_id=getattr(self, '_workflow_id', None),
                step_id=step_id,
                adapter_name=getattr(self._v2_metadata, 'name', None) if self._v2_metadata else None,
                step_inputs=inputs,
                step_outputs=outputs,
                step_error=error,
            )
            await self._v2_hooks.trigger(ap, ctx)

    def v2_get_method_metadata(self, method: str) -> Optional[MethodMetadata]:
        """Get metadata for a method"""
        if self._v2_metadata and method in self._v2_metadata.methods:
            return self._v2_metadata.methods[method]
        return None


# =============================================================================
# V2 Base Adapter
# =============================================================================

class V2BaseAdapter(V2AdapterMixin, ABC):
    """
    Base class for BBX 2.0 compatible adapters.

    Provides:
        - Standard execute() interface
        - Automatic hook triggering
        - Context tiering integration
        - Method registration
        - Error handling

    Usage:
        class MyV2Adapter(V2BaseAdapter):
            def __init__(self):
                super().__init__()
                self.register_method("my_method", self._my_method)

            async def _my_method(self, args: Dict) -> Any:
                return {"result": "ok"}
    """

    def __init__(self, metadata: Optional[AdapterMetadata] = None):
        self._v2_metadata = metadata
        self._methods: Dict[str, Callable] = {}
        self._workflow_id: Optional[str] = None
        self._execution_stats: Dict[str, Any] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0,
        }

    def register_method(
        self,
        name: str,
        handler: Callable,
        metadata: Optional[MethodMetadata] = None,
    ):
        """Register a method handler"""
        self._methods[name] = handler
        if metadata and self._v2_metadata:
            self._v2_metadata.methods[name] = metadata

    def set_context(self, context: Any):
        """Set workflow context (for compatibility with v1)"""
        # Extract workflow ID if available
        if hasattr(context, 'workflow_id'):
            self._workflow_id = context.workflow_id
        self._context = context

    async def execute(self, method: str, args: Dict[str, Any]) -> Any:
        """
        Execute a method on this adapter.

        This is the main entry point called by the runtime.
        Handles:
            - Method dispatch
            - Hook triggering
            - Error handling
            - Stats collection
        """
        start_time = datetime.now()
        self._execution_stats["total_calls"] += 1

        # Trigger pre-execute hook
        await self.v2_trigger_hook(
            "adapter.call",
            inputs=args,
        )

        try:
            # Find handler
            handler = self._methods.get(method)
            if not handler:
                # Try to find method on self
                handler = getattr(self, f"_{method}", None)
                if not handler:
                    handler = getattr(self, method, None)

            if not handler:
                raise ValueError(f"Unknown method: {method}")

            # Execute
            if asyncio.iscoroutinefunction(handler):
                result = await handler(args)
            else:
                result = handler(args)

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._execution_stats["successful_calls"] += 1
            self._execution_stats["total_duration_ms"] += duration_ms

            # Trigger post-execute hook
            await self.v2_trigger_hook(
                "adapter.result",
                outputs=result,
            )

            return result

        except Exception as e:
            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._execution_stats["failed_calls"] += 1
            self._execution_stats["total_duration_ms"] += duration_ms

            # Trigger error hook
            await self.v2_trigger_hook(
                "step.error",
                error=str(e),
            )

            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter execution statistics"""
        stats = self._execution_stats.copy()
        total = stats["total_calls"]
        if total > 0:
            stats["success_rate"] = stats["successful_calls"] / total
            stats["avg_duration_ms"] = stats["total_duration_ms"] / total
        else:
            stats["success_rate"] = 0
            stats["avg_duration_ms"] = 0
        return stats

    def list_methods(self) -> List[str]:
        """List available methods"""
        methods = list(self._methods.keys())

        # Also include methods starting with underscore that match
        for name in dir(self):
            if name.startswith('_') and not name.startswith('__'):
                method_name = name[1:]  # Remove underscore
                if method_name not in methods and callable(getattr(self, name)):
                    methods.append(method_name)

        return methods


# =============================================================================
# Adapter Factory for V2
# =============================================================================

class V2AdapterFactory:
    """
    Factory for creating V2-compatible adapter instances.

    Automatically wraps V1 adapters with V2 features.
    """

    _registered_adapters: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, adapter_class: Type):
        """Register an adapter class"""
        cls._registered_adapters[name] = adapter_class

    @classmethod
    def create(
        cls,
        name: str,
        context: Optional[Any] = None,
        hooks: Optional[Any] = None,
    ) -> Optional[Any]:
        """Create an adapter instance with V2 features"""
        adapter_class = cls._registered_adapters.get(name)
        if not adapter_class:
            return None

        adapter = adapter_class()

        # Add V2 features if supported
        if isinstance(adapter, V2AdapterMixin):
            adapter.v2_set_context(context)
            adapter.v2_set_hooks(hooks)
        elif hasattr(adapter, 'v2_init'):
            adapter.v2_init(context=context, hooks=hooks)

        return adapter

    @classmethod
    def wrap_v1_adapter(cls, adapter: Any) -> Any:
        """
        Wrap a V1 adapter to add V2 features.

        Creates a wrapper that delegates to the V1 adapter while
        adding V2 capabilities.
        """

        class V2Wrapper(V2AdapterMixin):
            def __init__(self, wrapped):
                self._wrapped = wrapped
                self._v2_metadata = None

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

            async def execute(self, method: str, args: Dict[str, Any]) -> Any:
                # Trigger pre-hook
                await self.v2_trigger_hook("adapter.call", inputs=args)

                try:
                    # Call wrapped adapter
                    if hasattr(self._wrapped, 'execute'):
                        if asyncio.iscoroutinefunction(self._wrapped.execute):
                            result = await self._wrapped.execute(method, args)
                        else:
                            result = self._wrapped.execute(method, args)
                    else:
                        # Try to call method directly
                        handler = getattr(self._wrapped, method, None)
                        if handler:
                            if asyncio.iscoroutinefunction(handler):
                                result = await handler(**args)
                            else:
                                result = handler(**args)
                        else:
                            raise ValueError(f"Unknown method: {method}")

                    # Trigger post-hook
                    await self.v2_trigger_hook("adapter.result", outputs=result)
                    return result

                except Exception as e:
                    await self.v2_trigger_hook("step.error", error=str(e))
                    raise

        return V2Wrapper(adapter)


# =============================================================================
# Decorator for V2 Methods
# =============================================================================

def v2_method(
    name: Optional[str] = None,
    description: str = "",
    idempotent: bool = False,
    supports_timeout: bool = True,
    supports_retry: bool = True,
):
    """
    Decorator to mark a method as V2-compatible.

    Usage:
        class MyAdapter(V2BaseAdapter):
            @v2_method(name="my_operation", description="Does something")
            async def _my_operation(self, args: Dict) -> Any:
                return {"result": "ok"}
    """
    def decorator(func):
        method_name = name or func.__name__.lstrip('_')

        # Store metadata on the function
        func._v2_metadata = MethodMetadata(
            name=method_name,
            description=description,
            idempotent=idempotent,
            supports_timeout=supports_timeout,
            supports_retry=supports_retry,
        )

        return func

    return decorator


# =============================================================================
# Compatibility helpers
# =============================================================================

def ensure_v2_compatible(adapter: Any) -> Any:
    """
    Ensure an adapter is V2 compatible.

    If already V2 compatible, returns as-is.
    Otherwise, wraps with V2 features.
    """
    if isinstance(adapter, V2AdapterMixin):
        return adapter
    if isinstance(adapter, V2BaseAdapter):
        return adapter
    return V2AdapterFactory.wrap_v1_adapter(adapter)
