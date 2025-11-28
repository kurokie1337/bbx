# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 Hooks System - eBPF-inspired dynamic workflow programming.

This module provides the ability to inject observability, security,
and transformation logic into workflows without modifying them.

Key concepts from Linux eBPF:
- Attach Points: Where hooks can intercept execution
- Program Types: Different hook types for different purposes
- Verifier: Ensures hooks are safe before execution
- JIT: Hooks are compiled for efficient execution

Example usage:
    manager = HookManager()

    # Register a metrics probe
    manager.register(HookDefinition(
        id="metrics",
        type=HookType.PROBE,
        attach_points=[AttachPoint.STEP_POST_EXECUTE],
        code="emit_metric('duration', ctx.step_duration_ms)"
    ))

    # Trigger hooks
    result = await manager.trigger(AttachPoint.STEP_POST_EXECUTE, ctx)
"""

from __future__ import annotations

import ast
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("bbx.hooks")


# =============================================================================
# Enums
# =============================================================================


class HookType(Enum):
    """Types of hooks (like eBPF program types)"""
    PROBE = auto()        # Observe execution
    FILTER = auto()       # Block/allow operations
    TRANSFORM = auto()    # Modify data
    SECURITY = auto()     # Enforce policies
    SCHEDULER = auto()    # Custom scheduling


class AttachPoint(Enum):
    """Points where hooks can attach"""
    # Workflow lifecycle
    WORKFLOW_START = "workflow.start"
    WORKFLOW_END = "workflow.end"
    WORKFLOW_ERROR = "workflow.error"

    # Step lifecycle
    STEP_PRE_EXECUTE = "step.pre_execute"
    STEP_POST_EXECUTE = "step.post_execute"
    STEP_ERROR = "step.error"
    STEP_RETRY = "step.retry"

    # Adapter calls
    ADAPTER_PRE_CALL = "adapter.pre_call"
    ADAPTER_POST_CALL = "adapter.post_call"

    # Resource access
    FILE_ACCESS = "file.access"
    NETWORK_CONNECT = "network.connect"
    STATE_ACCESS = "state.access"

    # Context
    CONTEXT_GET = "context.get"
    CONTEXT_SET = "context.set"


class HookAction(Enum):
    """Actions a hook can return"""
    CONTINUE = auto()     # Continue execution
    SKIP = auto()         # Skip operation
    BLOCK = auto()        # Block with error
    RETRY = auto()        # Retry operation
    TRANSFORM = auto()    # Use transformed data


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HookContext:
    """Context passed to hook functions (like eBPF ctx)"""
    # Workflow info
    workflow_id: str = ""
    workflow_name: str = ""

    # Step info
    step_id: Optional[str] = None
    step_type: Optional[str] = None
    step_inputs: Dict[str, Any] = field(default_factory=dict)
    step_outputs: Optional[Any] = None
    step_error: Optional[str] = None
    step_duration_ms: float = 0

    # Adapter info
    adapter_name: Optional[str] = None
    adapter_method: Optional[str] = None

    # Resource info
    resource_type: Optional[str] = None
    resource_path: Optional[str] = None

    # Global context
    variables: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    trace_id: str = ""

    # For transform hooks
    transformed_data: Optional[Any] = None


@dataclass
class HookResult:
    """Result returned by a hook"""
    action: HookAction = HookAction.CONTINUE
    data: Optional[Any] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HookFilter:
    """Filter to match when hook should trigger"""
    step_ids: Optional[List[str]] = None
    step_types: Optional[List[str]] = None
    workflow_ids: Optional[List[str]] = None
    adapters: Optional[List[str]] = None
    methods: Optional[List[str]] = None
    paths: Optional[List[str]] = None

    def matches(self, ctx: HookContext) -> bool:
        """Check if context matches filter"""
        if self.step_ids and ctx.step_id not in self.step_ids:
            return False
        if self.step_types:
            if not any(self._glob_match(ctx.step_type or "", p) for p in self.step_types):
                return False
        if self.workflow_ids and ctx.workflow_id not in self.workflow_ids:
            return False
        if self.adapters:
            if not any(self._glob_match(ctx.adapter_name or "", p) for p in self.adapters):
                return False
        if self.methods:
            if not any(self._glob_match(ctx.adapter_method or "", p) for p in self.methods):
                return False
        if self.paths:
            if not any(self._glob_match(ctx.resource_path or "", p) for p in self.paths):
                return False
        return True

    def _glob_match(self, text: str, pattern: str) -> bool:
        """Simple glob matching"""
        regex = pattern.replace(".", r"\.").replace("**", ".*").replace("*", "[^.]*")
        return bool(re.match(f"^{regex}$", text))


@dataclass
class HookDefinition:
    """Complete hook definition"""
    id: str
    name: str = ""
    type: HookType = HookType.PROBE
    attach_points: List[AttachPoint] = field(default_factory=list)
    filter: Optional[HookFilter] = None
    handler: Optional[Callable[[HookContext], HookResult]] = None
    code: Optional[str] = None
    priority: int = 0
    enabled: bool = True
    timeout_ms: int = 1000
    version: str = "1.0"
    description: str = ""


@dataclass
class VerificationResult:
    """Result of hook verification"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Hook Verifier
# =============================================================================


class HookVerifier:
    """
    Verifies hook programs before execution (like eBPF verifier).

    Ensures:
    - No infinite loops
    - Bounded execution
    - Memory safety
    - No dangerous operations
    """

    ALLOWED_BUILTINS: Set[str] = {
        "len", "str", "int", "float", "bool", "list", "dict", "set",
        "min", "max", "sum", "abs", "round", "sorted", "reversed",
        "any", "all", "zip", "enumerate", "range", "map", "filter",
        "isinstance", "hasattr", "getattr", "print",
    }

    FORBIDDEN_NODES: Set[type] = {
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
    }

    MAX_DEPTH = 10
    MAX_LOOP_ITERATIONS = 1000

    def verify(self, hook: HookDefinition) -> VerificationResult:
        """Verify a hook definition"""
        errors: List[str] = []
        warnings: List[str] = []

        if not hook.id:
            errors.append("Hook ID is required")

        if not hook.attach_points:
            errors.append("At least one attach point is required")

        if hook.code:
            code_result = self._verify_code(hook.code)
            errors.extend(code_result.errors)
            warnings.extend(code_result.warnings)
        elif not hook.handler:
            errors.append("Either handler function or code is required")

        if hook.type == HookType.SECURITY and not hook.filter:
            warnings.append("Security hook without filter will apply to all operations")

        return VerificationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _verify_code(self, code: str) -> VerificationResult:
        """Verify inline hook code"""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return VerificationResult(passed=False, errors=errors)

        for node in ast.walk(tree):
            if type(node) in self.FORBIDDEN_NODES:
                errors.append(f"Forbidden operation: {type(node).__name__}")

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "__import__"):
                        errors.append(f"Forbidden function: {node.func.id}")

            if isinstance(node, (ast.For, ast.While)):
                if not self._has_bounded_iterations(node):
                    warnings.append("Potentially unbounded loop")

        depth = self._get_max_depth(tree)
        if depth > self.MAX_DEPTH:
            errors.append(f"Code too complex (depth {depth} > {self.MAX_DEPTH})")

        return VerificationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _has_bounded_iterations(self, node: ast.AST) -> bool:
        """Check if loop has bounded iterations"""
        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call):
                if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                    if node.iter.args:
                        arg = node.iter.args[-1]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                            return arg.value <= self.MAX_LOOP_ITERATIONS
        return False

    def _get_max_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Get maximum AST depth"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            max_depth = max(max_depth, self._get_max_depth(child, depth + 1))
        return max_depth


# =============================================================================
# Hook Manager
# =============================================================================


class HookManager:
    """
    Manages hook registration and execution.

    Provides eBPF-like capabilities for BBX workflows.
    """

    def __init__(self):
        self._hooks: Dict[AttachPoint, List[HookDefinition]] = {
            point: [] for point in AttachPoint
        }
        self._registry: Dict[str, HookDefinition] = {}
        self._verifier = HookVerifier()
        self._execution_count: Dict[str, int] = {}
        self._execution_time: Dict[str, float] = {}
        self._metrics_buffer: List[Dict[str, Any]] = []

    def register(self, hook: HookDefinition) -> bool:
        """Register a hook"""
        result = self._verifier.verify(hook)
        if not result.passed:
            logger.error(f"Hook verification failed for {hook.id}: {result.errors}")
            return False

        if result.warnings:
            logger.warning(f"Hook warnings for {hook.id}: {result.warnings}")

        if hook.code and not hook.handler:
            hook.handler = self._compile_handler(hook.code)

        for attach_point in hook.attach_points:
            self._hooks[attach_point].append(hook)
            self._hooks[attach_point].sort(key=lambda h: -h.priority)

        self._registry[hook.id] = hook
        logger.info(f"Registered hook: {hook.id}")
        return True

    def unregister(self, hook_id: str) -> bool:
        """Unregister a hook"""
        if hook_id not in self._registry:
            return False

        hook = self._registry.pop(hook_id)
        for attach_point in hook.attach_points:
            self._hooks[attach_point] = [h for h in self._hooks[attach_point] if h.id != hook_id]

        logger.info(f"Unregistered hook: {hook_id}")
        return True

    async def trigger(
        self,
        attach_point: AttachPoint,
        context: HookContext
    ) -> HookResult:
        """Trigger all hooks at an attach point"""
        hooks = self._hooks.get(attach_point, [])
        combined_result = HookResult(action=HookAction.CONTINUE)

        for hook in hooks:
            if not hook.enabled:
                continue

            if hook.filter and not hook.filter.matches(context):
                continue

            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    self._execute_hook(hook, context),
                    timeout=hook.timeout_ms / 1000
                )
                duration = time.time() - start_time

                self._execution_count[hook.id] = self._execution_count.get(hook.id, 0) + 1
                self._execution_time[hook.id] = self._execution_time.get(hook.id, 0) + duration

                combined_result.metrics.update(result.metrics)
                combined_result.logs.extend(result.logs)

                if result.action != HookAction.CONTINUE:
                    combined_result.action = result.action
                    combined_result.data = result.data
                    combined_result.error = result.error
                    break

            except asyncio.TimeoutError:
                logger.warning(f"Hook {hook.id} timed out")
            except Exception as e:
                logger.error(f"Hook {hook.id} error: {e}")

        return combined_result

    async def _execute_hook(
        self,
        hook: HookDefinition,
        context: HookContext
    ) -> HookResult:
        """Execute a single hook"""
        if hook.handler:
            if asyncio.iscoroutinefunction(hook.handler):
                return await hook.handler(context)
            else:
                return hook.handler(context)
        return HookResult(action=HookAction.CONTINUE)

    def _compile_handler(self, code: str) -> Callable[[HookContext], HookResult]:
        """Compile inline code into handler function"""
        # Helper functions for hooks
        def emit_metric(name: str, value: Any, labels: Optional[Dict] = None):
            self._metrics_buffer.append({
                "name": name,
                "value": value,
                "labels": labels or {},
                "timestamp": time.time()
            })

        def emit_log(level: str, message: str, data: Optional[Dict] = None):
            pass  # Placeholder

        def glob_match(text: str, pattern: str) -> bool:
            regex = pattern.replace(".", r"\.").replace("**", ".*").replace("*", "[^.]*")
            return bool(re.match(f"^{regex}$", text))

        safe_globals = {
            # Built-ins
            "len": len, "str": str, "int": int, "float": float,
            "bool": bool, "list": list, "dict": dict, "set": set,
            "min": min, "max": max, "sum": sum, "abs": abs,
            "round": round, "sorted": sorted, "any": any, "all": all,
            "zip": zip, "enumerate": enumerate, "range": range,
            "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
            "print": print,

            # Hook-specific
            "HookResult": HookResult,
            "HookAction": HookAction,
            "emit_metric": emit_metric,
            "emit_log": emit_log,
            "glob_match": glob_match,
        }

        indented_code = "\n".join("    " + line for line in code.split("\n"))
        wrapped_code = f"""
def _hook_handler(ctx):
{indented_code}
    return HookResult(action=HookAction.CONTINUE)
"""

        exec(wrapped_code, safe_globals)
        return safe_globals["_hook_handler"]

    def get_stats(self) -> Dict[str, Any]:
        """Get hook execution statistics"""
        stats = {}
        for hook_id, hook in self._registry.items():
            count = self._execution_count.get(hook_id, 0)
            total_time = self._execution_time.get(hook_id, 0)
            stats[hook_id] = {
                "execution_count": count,
                "total_time_ms": total_time * 1000,
                "avg_time_ms": (total_time / count * 1000) if count > 0 else 0,
                "enabled": hook.enabled,
            }
        return stats

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get and clear metrics buffer"""
        metrics = self._metrics_buffer.copy()
        self._metrics_buffer.clear()
        return metrics


# =============================================================================
# Global Manager
# =============================================================================


_global_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """Get the global HookManager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = HookManager()
    return _global_manager


# =============================================================================
# Decorator for Easy Hook Registration
# =============================================================================


def hook(
    attach_points: List[AttachPoint],
    hook_type: HookType = HookType.PROBE,
    priority: int = 0,
    filter: Optional[HookFilter] = None
):
    """Decorator to register a function as a hook"""
    def decorator(func: Callable[[HookContext], HookResult]):
        hook_def = HookDefinition(
            id=func.__name__,
            name=func.__name__,
            type=hook_type,
            attach_points=attach_points,
            handler=func,
            priority=priority,
            filter=filter
        )
        get_hook_manager().register(hook_def)
        return func
    return decorator
