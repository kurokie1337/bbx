# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging


import asyncio
import traceback
from typing import Dict, Any, Optional
from .context import WorkflowContext
from .registry import MCPRegistry
from .adapters.http import LocalHttpAdapter
from .events import EventBus, Event, EventType
from .expressions import SafeExpr, ExpressionError
from .dag import WorkflowDAG, should_use_dag, DAGError
from .cache import get_cache
from .parsers.v6 import BBXv6Parser

logger = logging.getLogger("bbx.runtime")


async def run_file(file_path: str, event_bus: Optional[EventBus] = None, use_cache: bool = True, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        workflow_name = workflow.get("name", workflow_id)
    else:
        # Simple format - entire file is workflow
        workflow = data
        steps = data.get("steps", [])
        workflow_id = data.get("id", "unknown")
        workflow_name = data.get("name", workflow_id)

    # 2. Initialize Runtime
    context = WorkflowContext()

    # Add inputs to context if provided
    if inputs:
        context.variables['inputs'] = inputs

    registry = MCPRegistry()

    # Register default adapters
    from .adapters.logger import LoggerAdapter
    from .adapters.telegram import TelegramAdapter
    from .adapters.mcp_bridge import MCPBridgeAdapter
    from .adapters.transform import TransformAdapter

    registry.register("http", LocalHttpAdapter())
    registry.register("http-client", LocalHttpAdapter())  # Alias
    registry.register("logger", LoggerAdapter())
    registry.register("telegram", TelegramAdapter())
    registry.register("transform", TransformAdapter())

    # Add MCP Bridge if available
    try:
        registry.register("mcp", MCPBridgeAdapter())
    except Exception:
        pass  # MCP bridge optional

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
        await _execute_sequential(steps, context, registry, event_bus, workflow_id, results)

    await event_bus.emit(Event(EventType.WORKFLOW_END, {
        "id": workflow_id,
        "results": results
    }))

    return results


async def _execute_dag(dag: WorkflowDAG, context: WorkflowContext, registry: MCPRegistry,
                       event_bus: EventBus, workflow_id: str, results: Dict[str, Any]):
    """Execute workflow using DAG with parallel execution"""
    levels = dag.get_execution_levels()

    for level_idx, level in enumerate(levels):
        logger.info(f"  📍 Level {level_idx + 1}: {len(level)} step(s)")

        # Execute all steps in this level in parallel
        tasks = []
        for step_id in level:
            step = dag.get_step(step_id)
            task = _execute_step(step, context, registry, event_bus, workflow_id, results)
            tasks.append(task)

        # Wait for all steps in this level to complete
        await asyncio.gather(*tasks, return_exceptions=True)


async def _execute_sequential(steps: list, context: WorkflowContext, registry: MCPRegistry,
                              event_bus: EventBus, workflow_id: str, results: Dict[str, Any]):
    """Execute workflow steps sequentially"""
    for step in steps:
        await _execute_step(step, context, registry, event_bus, workflow_id, results)


async def _execute_step(step: Dict[str, Any], context: WorkflowContext, registry: MCPRegistry,
                       event_bus: EventBus, workflow_id: str, results: Dict[str, Any]):
    """Execute a single workflow step"""
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

    await event_bus.emit(Event(EventType.STEP_START, {
        "step_id": step_id,
        "workflow_id": workflow_id
    }))

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

        # Get adapter
        adapter = registry.get_adapter(mcp_type)
        if not adapter:
            raise ValueError(f"Unknown MCP type: {mcp_type}")

        # Get timeout and retry settings
        timeout_ms = step.get("timeout", 30000)
        timeout_sec = timeout_ms / 1000
        retry_count = step.get("retry", 0)
        retry_delay = step.get("retry_delay", 1000) / 1000  # Convert to seconds
        retry_backoff = step.get("retry_backoff", 2)

        logger.info(f"▶️  Executing {step_id} ({mcp_type}.{method})")

        # Execute step with timeout and retry
        output = None
        last_error = None

        for attempt in range(retry_count + 1):
            try:
                output = await asyncio.wait_for(
                    adapter.execute(method, resolved_inputs),
                    timeout=timeout_sec
                )
                break  # Success, exit retry loop

            except asyncio.TimeoutError:
                last_error = Exception(f"Step timeout after {timeout_ms}ms")
                if attempt < retry_count:
                    delay = retry_delay * (retry_backoff ** attempt)
                    logger.info(f"⏳ Timeout - Retry {attempt + 1}/{retry_count} after {delay:.1f}s")
                    await asyncio.sleep(delay)
                else:
                    raise last_error

            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    delay = retry_delay * (retry_backoff ** attempt)
                    logger.error(f"⏳ Error - Retry {attempt + 1}/{retry_count} after {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise

        # Store output
        output_key = step.get("outputs", "result")
        step_context_data = {
            "status": "success",
            "output": output
        }

        if isinstance(output_key, str):
            step_context_data[output_key] = output

        context.set_step_output(step_id, step_context_data)
        results[step_id] = {"status": "success", "output": output}

        await event_bus.emit(Event(EventType.STEP_END, {
            "step_id": step_id,
            "output": output
        }))
        logger.info(f"✅ {step_id} completed")

    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ {step_id} failed: {e}")

        error_data = {"status": "error", "error": str(e)}
        results[step_id] = error_data
        context.set_step_output(step_id, error_data)

        await event_bus.emit(Event(EventType.STEP_ERROR, {
            "step_id": step_id,
            "error": str(e)
        }))


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
            await _execute_dag(dag, context, registry, self.event_bus, "bundled", results)
        else:
            await _execute_sequential(steps, context, registry, self.event_bus, "bundled", results)
        
        return results
