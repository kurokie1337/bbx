# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0


import asyncio
import logging
import traceback
from typing import Any, Dict, Optional

from .cache import get_cache
from .context import WorkflowContext
from .dag import DAGError, WorkflowDAG, should_use_dag
from .events import Event, EventBus, EventType
from .expressions import ExpressionError, SafeExpr
from .parsers.v6 import BBXv6Parser
from .registry import MCPRegistry

logger = logging.getLogger("bbx.runtime")


async def run_file(
    file_path: str,
    event_bus: Optional[EventBus] = None,
    use_cache: bool = True,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a local .bbx file using the Blackbox Core Runtime.

    Supports BBX v5.0 and v6.0 formats (auto-detected).

    Args:
        file_path: Path to the .bbx workflow file
        event_bus: Optional event bus for real-time tracking
        use_cache: Whether to use workflow parsing cache
        inputs: Optional dictionary of workflow inputs (accessible via ${inputs.key})

    Returns:
        Dictionary of step results
    """
    if event_bus is None:
        event_bus = EventBus()

    # 1. Load YAML with BBX version support (with caching)
    if use_cache:
        cache = get_cache()
        cached = cache.get(file_path)
        if cached:
            data = cached
        else:
            data = BBXv6Parser.load_yaml(file_path)
            cache.put(file_path, data)
    else:
        data = BBXv6Parser.load_yaml(file_path)

    # Support both v6 format {"workflow": {...}} and simple format
    if "workflow" in data:
        workflow = data.get("workflow", {})
        steps = workflow.get("steps", [])
        workflow_id = workflow.get("id", "unknown")
    else:
        # Simple format - entire file is workflow
        workflow = data
        steps = data.get("steps", [])
        workflow_id = data.get("id", "unknown")

    # 2. Initialize Runtime
    context = WorkflowContext()

    # Load default inputs from workflow definition
    workflow_inputs_def = workflow.get("inputs", {})
    default_inputs = {}
    for input_name, input_config in workflow_inputs_def.items():
        if isinstance(input_config, dict) and "default" in input_config:
            default_inputs[input_name] = input_config["default"]
        elif not isinstance(input_config, dict):
            # Simple value, treat as default
            default_inputs[input_name] = input_config

    # Merge: defaults + provided inputs (provided takes priority)
    merged_inputs = {**default_inputs, **(inputs or {})}
    context.variables["inputs"] = merged_inputs

    # Load state from current workspace if available
    try:
        from .workspace_manager import get_current_workspace
        workspace = get_current_workspace()
        if workspace:
            context.variables["state"] = workspace.get_all_state()
    except Exception:
        context.variables["state"] = {}

    # Get global registry with all adapters already registered
    from .registry import get_registry
    registry = get_registry()

    results: Dict[str, Any] = {}

    # 3. Execute Workflow
    if should_use_dag(steps):
        logger.info("📊 Using DAG parallel execution")
        try:
            dag = WorkflowDAG(steps)
            await _execute_dag(dag, context, registry, event_bus, workflow_id, results)
        except DAGError as e:
            logger.error(f"❌ DAG error: {e}")
            raise
    else:
        logger.info("📝 Using sequential execution")
        await _execute_sequential(
            steps, context, registry, event_bus, workflow_id, results
        )

    await event_bus.emit(
        Event(EventType.WORKFLOW_END, {"id": workflow_id, "results": results})
    )

    return results


async def _execute_dag(
    dag: WorkflowDAG,
    context: WorkflowContext,
    registry: MCPRegistry,
    event_bus: EventBus,
    workflow_id: str,
    results: Dict[str, Any],
):
    """Execute workflow using DAG with parallel execution"""
    levels = dag.get_execution_levels()

    for level_idx, level in enumerate(levels):
        logger.info(f"  📍 Level {level_idx + 1}: {len(level)} step(s)")

        # Execute all steps in this level in parallel
        tasks = []
        for step_id in level:
            step = dag.get_step(step_id)
            task = _execute_step(
                step, context, registry, event_bus, workflow_id, results
            )
            tasks.append(task)

        # Wait for all steps in this level to complete
        await asyncio.gather(*tasks, return_exceptions=True)


async def _execute_sequential(
    steps: list,
    context: WorkflowContext,
    registry: MCPRegistry,
    event_bus: EventBus,
    workflow_id: str,
    results: Dict[str, Any],
):
    """Execute workflow steps sequentially"""
    for step in steps:
        await _execute_step(step, context, registry, event_bus, workflow_id, results)


async def _execute_step(
    step: Dict[str, Any],
    context: WorkflowContext,
    registry: MCPRegistry,
    event_bus: EventBus,
    workflow_id: str,
    results: Dict[str, Any],
):
    """Execute a single workflow step with inline auto-fix support"""
    step_id = step.get("id")
    mcp_type = step.get("mcp")
    method = step.get("method")
    inputs = step.get("inputs", {})

    # Validate required fields
    if not step_id or not isinstance(step_id, str):
        raise ValueError("Step 'id' is required and must be a string")
    if not mcp_type or not isinstance(mcp_type, str):
        raise ValueError(f"Step '{step_id}': 'mcp' is required and must be a string")
    if not method or not isinstance(method, str):
        raise ValueError(f"Step '{step_id}': 'method' is required and must be a string")

    await event_bus.emit(
        Event(EventType.STEP_START, {"step_id": step_id, "workflow_id": workflow_id})
    )

    try:
        # Resolve inputs
        # Resolve inputs
        resolved_inputs = context.resolve_recursive(inputs)

        # Check 'when' condition
        when_condition = step.get("when")
        if when_condition:
            try:
                resolved_condition = context.resolve(when_condition)
                # Use safe expression parser instead of eval()
                if not SafeExpr.evaluate(resolved_condition, context.variables):
                    logger.info(f"⏭️  Skipping {step_id} (condition false)")
                    results[step_id] = {"status": "skipped"}
                    return
            except ExpressionError as e:
                logger.error(f"⚠️  Condition error in {step_id}: {e}")
                logger.info(f"⏭️  Skipping {step_id}")
                results[step_id] = {"status": "skipped", "error": str(e)}
                return

        # === PRE-CHECK ===
        pre_check = step.get("pre_check")
        if pre_check:
            logger.info(f"🔍 Running pre-check for {step_id}")
            try:
                pre_check_adapter = registry.get_adapter(pre_check.get("adapter", mcp_type))
                pre_check_method = pre_check.get("action") or pre_check.get("method")
                pre_check_inputs = context.resolve_recursive(pre_check.get("params", {}))

                await pre_check_adapter.execute(pre_check_method, pre_check_inputs)
                logger.info(f"✅ Pre-check passed for {step_id}")

                # Track metadata
                if not hasattr(context, 'meta'):
                    context.meta = {}
                if 'pre_checks_run' not in context.meta:
                    context.meta['pre_checks_run'] = {}
                context.meta['pre_checks_run'][step_id] = True

            except Exception as pre_error:
                logger.warning(f"⚠️  Pre-check failed for {step_id}: {pre_error}")

                # Execute on_failure actions for pre_check
                on_failure = pre_check.get("on_failure", [])
                if on_failure:
                    logger.info(f"🔧 Executing {len(on_failure)} on_failure actions")
                    for failure_action in on_failure:
                        try:
                            fa_adapter = registry.get_adapter(failure_action.get("adapter", mcp_type))
                            fa_method = failure_action.get("action") or failure_action.get("method")
                            fa_inputs = context.resolve_recursive(failure_action.get("params", {}))
                            await fa_adapter.execute(fa_method, fa_inputs)
                        except Exception as fa_error:
                            logger.error(f"❌ Failure action error: {fa_error}")

                # If pre-check failed and no recovery, skip or fail
                if not step.get("continue_on_pre_check_fail", False):
                    raise Exception(f"Pre-check failed: {pre_error}")

        # Get adapter
        adapter = registry.get_adapter(mcp_type)
        if not adapter:
            raise ValueError(f"Unknown MCP type: {mcp_type}")

        # Inject context if supported (e.g. PythonAdapter)
        if hasattr(adapter, "set_context"):
            adapter.set_context(context)

        # Get timeout and retry settings
        timeout_ms = step.get("timeout", 30000)
        timeout_sec = timeout_ms / 1000
        retry_count = step.get("retry", 0)

        # Parse retry_delay (support "2s", "500ms", or plain number)
        retry_delay_raw = step.get("retry_delay", "1s")
        if isinstance(retry_delay_raw, str):
            if retry_delay_raw.endswith("ms"):
                retry_delay = float(retry_delay_raw[:-2]) / 1000
            elif retry_delay_raw.endswith("s"):
                retry_delay = float(retry_delay_raw[:-1])
            else:
                retry_delay = float(retry_delay_raw)
        else:
            retry_delay = float(retry_delay_raw) / 1000

        retry_backoff = step.get("retry_backoff", 2)

        logger.info(f"▶️  Executing {step_id} ({mcp_type}.{method})")

        # Execute step with timeout and retry
        output = None
        last_error = None
        retry_attempts = 0

        for attempt in range(retry_count + 1):
            try:
                output = await asyncio.wait_for(
                    adapter.execute(method, resolved_inputs), timeout=timeout_sec
                )

                # Track retry count in metadata
                if not hasattr(context, 'meta'):
                    context.meta = {}
                if 'retry_counts' not in context.meta:
                    context.meta['retry_counts'] = {}
                context.meta['retry_counts'][step_id] = retry_attempts

                break  # Success, exit retry loop

            except asyncio.TimeoutError:
                last_error = Exception(f"Step timeout after {timeout_ms}ms")
                retry_attempts += 1
                if attempt < retry_count:
                    delay = retry_delay * (retry_backoff**attempt)
                    logger.info(
                        f"⏳ Timeout - Retry {attempt + 1}/{retry_count} after {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise last_error

            except Exception as e:
                last_error = e
                retry_attempts += 1
                if attempt < retry_count:
                    delay = retry_delay * (retry_backoff**attempt)
                    logger.error(
                        f"⏳ Error - Retry {attempt + 1}/{retry_count} after {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # === FALLBACK ===
                    fallback = step.get("fallback")
                    if fallback:
                        logger.info(f"🔄 Trying fallback for {step_id}")
                        try:
                            fb_adapter = registry.get_adapter(fallback.get("adapter", mcp_type))
                            fb_method = fallback.get("action") or fallback.get("method")
                            fb_inputs = context.resolve_recursive(fallback.get("params", resolved_inputs))

                            output = await fb_adapter.execute(fb_method, fb_inputs)

                            # Track fallback usage
                            if not hasattr(context, 'meta'):
                                context.meta = {}
                            if 'fallback_used' not in context.meta:
                                context.meta['fallback_used'] = {}
                            context.meta['fallback_used'][step_id] = True

                            logger.info(f"✅ Fallback succeeded for {step_id}")
                            break  # Success via fallback
                        except Exception as fb_error:
                            logger.error(f"❌ Fallback failed for {step_id}: {fb_error}")

                    # === ON_FAILURE ===
                    on_failure = step.get("on_failure", [])
                    if on_failure:
                        logger.info(f"🔧 Executing {len(on_failure)} on_failure actions")
                        for failure_action in on_failure:
                            try:
                                fa_adapter = registry.get_adapter(failure_action.get("adapter", mcp_type))
                                fa_method = failure_action.get("action") or failure_action.get("method")
                                fa_inputs = context.resolve_recursive(failure_action.get("params", {}))
                                await fa_adapter.execute(fa_method, fa_inputs)

                                # Check if retry is requested in failure action
                                if failure_action.get("retry"):
                                    logger.info("🔁 Retrying after on_failure action...")
                                    continue  # Continue retry loop
                            except Exception as fa_error:
                                logger.error(f"❌ Failure action error: {fa_error}")

                    raise

        # Store output - save the actual output for interpolation
        # This allows ${steps.id.outputs.field} syntax to work
        context.set_step_output(step_id, output)
        results[step_id] = {"status": "success", "output": output}

        await event_bus.emit(
            Event(EventType.STEP_END, {"step_id": step_id, "output": output})
        )
        logger.info(f"✅ {step_id} completed")

    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ {step_id} failed: {e}")

        error_data = {"status": "error", "error": str(e)}
        results[step_id] = error_data
        context.set_step_output(step_id, error_data)

        await event_bus.emit(
            Event(EventType.STEP_ERROR, {"step_id": step_id, "error": str(e)})
        )


class BBXRuntime:
    """BBX Runtime wrapper for bundled workflows"""

    def __init__(self):
        self.workflow: Optional[Dict[str, Any]] = None
        self.inputs: Dict[str, Any] = {}
        self.event_bus = EventBus()

    def load_workflow(self, workflow: Dict[str, Any]):
        """Load workflow definition"""
        self.workflow = workflow

    def set_input(self, key: str, value: Any):
        """Set workflow input"""
        self.inputs[key] = value

    async def execute(self) -> Dict[str, Any]:
        """Execute the workflow"""
        if not self.workflow:
            raise ValueError("No workflow loaded")

        # Create context
        context = WorkflowContext(inputs=self.inputs)

        # Get registry
        registry = MCPRegistry()
        registry.register("http", LocalHttpAdapter())

        # Get steps
        workflow_def = self.workflow.get("workflow", {})
        steps = workflow_def.get("steps", [])

        # Execute
        results: Dict[str, Any] = {}

        if should_use_dag(steps):
            dag = WorkflowDAG(steps)
            await _execute_dag(
                dag, context, registry, self.event_bus, "bundled", results
            )
        else:
            await _execute_sequential(
                steps, context, registry, self.event_bus, "bundled", results
            )

        return results
