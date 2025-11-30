"""
BBX Bridge - Integration layer between Console and BBX Core

This module provides a clean interface to BBX Core components:
- Runtime (workflow execution)
- ContextTiering (memory management)
- AgentRing (operation scheduling)
- Agents (Claude subagents)
- A2A (agent-to-agent protocol)
"""

import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Add BBX path to sys.path
bbx_path = Path(settings.bbx_path)
if str(bbx_path) not in sys.path:
    sys.path.insert(0, str(bbx_path))


class BBXBridge:
    """
    Bridge between BBX Console and BBX Core.

    Provides a clean API for all BBX functionality.
    """

    def __init__(self):
        self._initialized = False
        self._context_tiering = None
        self._agent_ring = None
        self._event_bus = None
        self._registry = None

        # Execution tracking
        self._active_executions: Dict[str, Dict] = {}

        # Agent status tracking
        self._agent_status: Dict[str, str] = {}

    async def initialize(self):
        """Initialize all BBX components"""
        if self._initialized:
            return

        try:
            # Import BBX modules
            from blackbox.core.events import EventBus
            from blackbox.core.registry import get_registry
            from blackbox.core.v2.context_tiering import ContextTiering
            from blackbox.core.v2.ring import AgentRing

            # Initialize components
            self._event_bus = EventBus()
            self._registry = get_registry()
            self._context_tiering = ContextTiering()
            self._agent_ring = AgentRing()

            # Start background tasks
            await self._context_tiering.start()

            # Get adapters from registry
            adapters = {}
            for name in self._registry.list_adapters():
                try:
                    adapters[name] = self._registry.get_adapter(name)
                except Exception:
                    pass

            await self._agent_ring.start(adapters)

            self._initialized = True
            logger.info("BBX Bridge initialized")

        except ImportError as e:
            logger.error(f"Failed to import BBX modules: {e}")
            logger.warning("Running in mock mode - BBX Core not available")
            self._initialized = True  # Allow running in mock mode

    async def shutdown(self):
        """Shutdown all BBX components"""
        if self._context_tiering:
            await self._context_tiering.stop()
        if self._agent_ring:
            await self._agent_ring.stop()

        self._initialized = False
        logger.info("BBX Bridge shutdown complete")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # =========================================================================
    # Workflow Operations
    # =========================================================================

    async def list_workflows(self, directory: str = ".") -> List[Dict]:
        """List all workflows in directory"""
        workflows = []
        base_path = Path(directory)

        for bbx_file in base_path.glob("**/*.bbx"):
            try:
                import yaml
                content = bbx_file.read_text(encoding="utf-8")
                data = yaml.safe_load(content)

                wf = data.get("workflow", data)
                workflows.append({
                    "id": wf.get("id", bbx_file.stem),
                    "name": wf.get("name", bbx_file.stem),
                    "description": wf.get("description", ""),
                    "file_path": str(bbx_file),
                    "step_count": len(wf.get("steps", [])),
                })
            except Exception as e:
                logger.warning(f"Failed to parse {bbx_file}: {e}")

        return workflows

    async def get_workflow(self, file_path: str) -> Optional[Dict]:
        """Get workflow details"""
        try:
            import yaml
            from blackbox.core.dag import WorkflowDAG, should_use_dag

            path = Path(file_path)
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)

            wf = data.get("workflow", data)
            raw_steps = wf.get("steps", [])

            # Normalize steps: convert dict format to list format
            if isinstance(raw_steps, dict):
                # Dict format: {step_id: {use: ..., args: ...}}
                steps = []
                for step_id, step_config in raw_steps.items():
                    if isinstance(step_config, dict):
                        step = {"id": step_id, **step_config}
                        # Parse 'use' into mcp and method
                        if "use" in step_config:
                            use_parts = step_config["use"].split(".", 1)
                            step["mcp"] = use_parts[0] if len(use_parts) > 0 else ""
                            step["method"] = use_parts[1] if len(use_parts) > 1 else step_config["use"]
                        steps.append(step)
                    else:
                        steps.append({"id": step_id, "value": step_config})
            else:
                steps = raw_steps if isinstance(raw_steps, list) else []

            # Build DAG visualization
            dag_data = {"nodes": [], "edges": [], "levels": []}

            if should_use_dag(steps):
                dag = WorkflowDAG(steps)
                levels = dag.get_execution_levels()
                dag_data["levels"] = levels

                # Build nodes
                for level_idx, level in enumerate(levels):
                    for step_id in level:
                        step = dag.get_step(step_id)
                        dag_data["nodes"].append({
                            "id": step_id,
                            "label": step_id,
                            "level": level_idx,
                            "mcp": step.get("mcp", ""),
                            "method": step.get("method", ""),
                        })

                        # Build edges
                        for dep in dag.get_dependencies(step_id):
                            dag_data["edges"].append({"source": dep, "target": step_id})
            else:
                # Sequential workflow
                for idx, step in enumerate(steps):
                    step_id = step.get("id", f"step_{idx}")
                    dag_data["nodes"].append({
                        "id": step_id,
                        "label": step_id,
                        "level": idx,
                        "mcp": step.get("mcp", ""),
                        "method": step.get("method", ""),
                    })
                    if idx > 0:
                        prev_id = steps[idx-1].get("id", f"step_{idx-1}")
                        dag_data["edges"].append({"source": prev_id, "target": step_id})
                    dag_data["levels"].append([step_id])

            # Parse inputs
            inputs_def = wf.get("inputs", {})
            inputs = []
            for name, config in inputs_def.items():
                if isinstance(config, dict):
                    inputs.append({
                        "name": name,
                        "type": config.get("type", "string"),
                        "required": config.get("required", True),
                        "default": config.get("default"),
                        "description": config.get("description"),
                    })
                else:
                    inputs.append({
                        "name": name,
                        "type": "string",
                        "required": False,
                        "default": config,
                    })

            return {
                "id": wf.get("id", path.stem),
                "name": wf.get("name", path.stem),
                "description": wf.get("description", ""),
                "file_path": str(path),
                "bbx_version": data.get("bbx", "6.0"),
                "inputs": inputs,
                "steps": steps,
                "dag": dag_data,
            }

        except Exception as e:
            logger.error(f"Failed to get workflow {file_path}: {e}")
            return None

    async def run_workflow(
        self,
        file_path: str,
        inputs: Dict[str, Any],
        ws_callback=None,
    ) -> str:
        """
        Run a workflow and return execution ID.

        Args:
            file_path: Path to .bbx file
            inputs: Workflow inputs
            ws_callback: Optional callback for WebSocket updates
        """
        from blackbox.core.runtime import run_file
        from blackbox.core.events import EventBus, EventType

        execution_id = str(uuid.uuid4())

        # Create event bus with handlers
        event_bus = EventBus()

        async def on_step_start(event):
            self._active_executions[execution_id]["current_step"] = event.data.get("step_id")
            if ws_callback:
                await ws_callback("step:started", {
                    "execution_id": execution_id,
                    "step_id": event.data.get("step_id"),
                })

        async def on_step_end(event):
            if ws_callback:
                await ws_callback("step:completed", {
                    "execution_id": execution_id,
                    "step_id": event.data.get("step_id"),
                    "output": event.data.get("output"),
                })

        async def on_step_error(event):
            if ws_callback:
                await ws_callback("step:failed", {
                    "execution_id": execution_id,
                    "step_id": event.data.get("step_id"),
                    "error": event.data.get("error"),
                })

        event_bus.subscribe(EventType.STEP_START, on_step_start)
        event_bus.subscribe(EventType.STEP_END, on_step_end)
        event_bus.subscribe(EventType.STEP_ERROR, on_step_error)

        # Track execution
        self._active_executions[execution_id] = {
            "workflow_id": file_path,
            "inputs": inputs,
            "status": "running",
            "started_at": datetime.utcnow(),
            "current_step": None,
        }

        async def execute():
            try:
                results = await run_file(
                    file_path,
                    event_bus=event_bus,
                    inputs=inputs,
                )
                self._active_executions[execution_id]["status"] = "completed"
                self._active_executions[execution_id]["results"] = results
                self._active_executions[execution_id]["completed_at"] = datetime.utcnow()

                if ws_callback:
                    await ws_callback("execution:completed", {
                        "execution_id": execution_id,
                        "status": "completed",
                        "results": results,
                    })

            except Exception as e:
                self._active_executions[execution_id]["status"] = "failed"
                self._active_executions[execution_id]["error"] = str(e)
                self._active_executions[execution_id]["completed_at"] = datetime.utcnow()

                if ws_callback:
                    await ws_callback("execution:failed", {
                        "execution_id": execution_id,
                        "status": "failed",
                        "error": str(e),
                    })

        # Run in background
        asyncio.create_task(execute())

        return execution_id

    async def get_execution(self, execution_id: str) -> Optional[Dict]:
        """Get execution status"""
        return self._active_executions.get(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id in self._active_executions:
            self._active_executions[execution_id]["status"] = "cancelled"
            return True
        return False

    # =========================================================================
    # Agent Operations
    # =========================================================================

    async def list_agents(self) -> List[Dict]:
        """List all available agents"""
        agents = []
        agents_path = Path(settings.bbx_path) / ".claude" / "agents"

        if not agents_path.exists():
            return agents

        for md_file in agents_path.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                agent = self._parse_agent_file(md_file.stem, content)
                agent["status"] = self._agent_status.get(agent["id"], "idle")
                agents.append(agent)
            except Exception as e:
                logger.warning(f"Failed to parse agent {md_file}: {e}")

        return agents

    async def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get agent details"""
        agents_path = Path(settings.bbx_path) / ".claude" / "agents"
        agent_file = agents_path / f"{agent_id}.md"

        if not agent_file.exists():
            return None

        content = agent_file.read_text(encoding="utf-8")
        agent = self._parse_agent_file(agent_id, content)
        agent["status"] = self._agent_status.get(agent_id, "idle")
        agent["file_path"] = str(agent_file)

        return agent

    def _parse_agent_file(self, agent_id: str, content: str) -> Dict:
        """Parse agent .md file"""
        import yaml

        name = agent_id
        description = ""
        tools = []
        model = "sonnet"
        system_prompt = ""

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1])
                    name = frontmatter.get("name", agent_id)
                    description = frontmatter.get("description", "")
                    tools_raw = frontmatter.get("tools", [])
                    if isinstance(tools_raw, str):
                        tools = [t.strip() for t in tools_raw.split(",")]
                    else:
                        tools = tools_raw
                    model = frontmatter.get("model", "sonnet")
                    system_prompt = parts[2].strip()
                except:
                    pass
        else:
            system_prompt = content

        return {
            "id": agent_id,
            "name": name,
            "description": description,
            "tools": tools,
            "model": model,
            "system_prompt": system_prompt,
            "metrics": {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "avg_duration_ms": 0,
                "success_rate": 0,
            },
        }

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def get_memory_stats(self) -> Dict:
        """Get memory tiering statistics"""
        if self._context_tiering:
            return self._context_tiering.get_stats()

        # Mock data
        return {
            "generations": [
                {"tier": "HOT", "items": 5, "size_bytes": 1024, "max_size_bytes": 102400, "utilization": 0.01},
                {"tier": "WARM", "items": 10, "size_bytes": 4096, "max_size_bytes": 1048576, "utilization": 0.004},
                {"tier": "COOL", "items": 20, "size_bytes": 16384, "max_size_bytes": 104857600, "utilization": 0.0002},
                {"tier": "COLD", "items": 50, "size_bytes": 65536, "max_size_bytes": 1000000000000, "utilization": 0.0},
            ],
            "total_items": 85,
            "total_size_bytes": 87040,
            "promotions": 15,
            "demotions": 30,
            "cache_hits": 500,
            "cache_misses": 50,
            "hit_rate": 0.909,
        }

    async def get_memory_items(self, tier: str) -> List[Dict]:
        """Get items in a specific memory tier"""
        # In real implementation, would query ContextTiering
        return []

    async def pin_memory_item(self, key: str) -> bool:
        """Pin a memory item"""
        if self._context_tiering:
            try:
                await self._context_tiering.pin(key)
                return True
            except:
                return False
        return False

    async def unpin_memory_item(self, key: str) -> bool:
        """Unpin a memory item"""
        if self._context_tiering:
            try:
                await self._context_tiering.unpin(key)
                return True
            except:
                return False
        return False

    # =========================================================================
    # Ring Operations
    # =========================================================================

    async def get_ring_stats(self) -> Dict:
        """Get AgentRing statistics"""
        if self._agent_ring:
            stats = self._agent_ring.get_stats()
            return {
                "operations_submitted": stats.operations_submitted,
                "operations_completed": stats.operations_completed,
                "operations_failed": stats.operations_failed,
                "operations_cancelled": stats.operations_cancelled,
                "operations_timeout": stats.operations_timeout,
                "pending_count": stats.pending_count,
                "processing_count": stats.processing_count,
                "active_workers": stats.active_workers,
                "worker_pool_size": stats.worker_pool_size,
                "submission_queue_size": stats.submission_queue_size,
                "completion_queue_size": stats.completion_queue_size,
                "throughput_ops_sec": stats.throughput_ops_sec,
                "avg_latency_ms": stats.avg_latency_ms,
                "p50_latency_ms": stats.p50_latency_ms,
                "p95_latency_ms": stats.p95_latency_ms,
                "p99_latency_ms": stats.p99_latency_ms,
                "worker_utilization": stats.worker_utilization,
            }

        # Mock data
        return {
            "operations_submitted": 1000,
            "operations_completed": 950,
            "operations_failed": 30,
            "operations_cancelled": 10,
            "operations_timeout": 10,
            "pending_count": 5,
            "processing_count": 3,
            "active_workers": 4,
            "worker_pool_size": 32,
            "submission_queue_size": 5,
            "completion_queue_size": 2,
            "throughput_ops_sec": 15.5,
            "avg_latency_ms": 125.3,
            "p50_latency_ms": 85.0,
            "p95_latency_ms": 350.0,
            "p99_latency_ms": 800.0,
            "worker_utilization": 12.5,
        }


# Global BBX bridge instance
bbx_bridge = BBXBridge()
