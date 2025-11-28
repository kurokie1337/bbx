# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 Runtime - Operating System for AI Agents

This module integrates ALL BBX 2.0 components:

KERNEL LEVEL (Linux-inspired):
- AgentRing: io_uring-inspired batch operations
- BBX Hooks: eBPF-inspired dynamic programming
- ContextTiering: MGLRU-inspired memory management
- FlowIntegrity: CET-inspired control flow protection
- AgentQuotas: Cgroups v2-inspired resource limits
- StateSnapshots: XFS Reflink-inspired CoW state

NT KERNEL LEVEL (Windows-inspired):
- BBX Executive: Hybrid kernel architecture
- ObjectManager: Windows Object Manager-inspired namespace
- FilterStack: Windows Filter Drivers-inspired pipeline
- WorkingSet: Windows memory compression
- ConfigRegistry: Windows Registry-inspired config
- AAL: Windows HAL-inspired abstraction layer

DISTRIBUTION LEVEL:
- PolicyEngine: OPA/SELinux-inspired access control
- NetworkFabric: Istio-inspired service mesh
- AgentSandbox: Flatpak-inspired isolation
- AgentBundles: Kali-inspired toolkits
- AgentRegistry: AUR-inspired marketplace

Example usage:
    # Run workflow with all BBX 2.0 features
    results = await run_file_v2("workflow.bbx")

    # Or with explicit configuration
    runtime = BBXRuntimeV2(config)
    await runtime.start()
    results = await runtime.execute_file("workflow.bbx")
    await runtime.stop()
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..cache import get_cache
from ..context import WorkflowContext
from ..dag import DAGError, WorkflowDAG, should_use_dag
from ..events import Event, EventBus, EventType
from ..expressions import ExpressionError, SafeExpr
from ..parsers.v6 import BBXv6Parser
from ..registry import get_registry

# Core kernel components
from .ring import AgentRing, Operation, OperationType, OperationPriority, OperationStatus, RingConfig
from .hooks import (
    HookManager, HookDefinition, HookContext, HookResult,
    HookType, AttachPoint, HookAction, get_hook_manager
)
from .context_tiering import ContextTiering, TieringConfig, get_context_tiering
from .flow_integrity import FlowIntegrityEngine, FlowState, FlowConfig, get_flow_integrity
from .agent_quotas import QuotaManager, ResourceType, QuotaConfig, get_quota_manager
from .state_snapshots import StateSnapshotEngine, SnapshotType, get_snapshot_engine

# Distribution level components
from .policy_engine import (
    PolicyEngine, PolicyRequest, Subject, Resource, Action, Context,
    Decision, get_policy_engine
)
from .network_fabric import NetworkMesh, get_mesh
from .agent_sandbox import SandboxManager, SandboxConfig, SandboxTemplates
from .agent_bundles import BundleManager, get_bundle_manager

# NT Kernel components (Windows-inspired)
from .executive import (
    BBXExecutive, ExecutiveConfig, ExecutiveState,
    SystemCall, SystemCallType, SecurityToken,
    get_executive, start_executive, stop_executive
)
from .object_manager import (
    ObjectManager, BBXObject, DirectoryObject, Handle,
    SecurityDescriptor, ACL, ACE, ACEType, AccessMask, ObjectType,
    get_object_manager
)
from .filter_stack import (
    FilterManager, FilterContext, FilterResult, OperationClass,
    Filter, AuditFilter, SecurityFilter, QuotaFilter, RateLimitFilter,
    MetricsFilter, CacheFilter, FilterStackBuilder, get_filter_manager
)
from .working_set import (
    WorkingSetManager, AgentWorkingSet, AllocationRequest,
    PageState, MemoryPressure, WorkingSetLimits, get_working_set_manager
)
from .config_registry import (
    ConfigRegistry, RegistryHelper, RegistryKey, RegistryValue,
    ValueType, Hive, AccessRights, get_config_registry
)
from .aal import (
    AAL, LLMProvider, VectorProvider, LLMRequest, LLMResponse,
    VectorQuery, VectorResult, Capability, RoutingStrategy, get_aal
)

logger = logging.getLogger("bbx.runtime.v2")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RuntimeV2Config:
    """Configuration for BBX 2.0 Runtime"""

    # AgentRing configuration
    ring_enabled: bool = True
    ring_config: RingConfig = field(default_factory=RingConfig)

    # Hooks configuration
    hooks_enabled: bool = True
    hooks_dir: Optional[Path] = None  # Directory for hook files

    # Context tiering configuration
    tiering_enabled: bool = True
    tiering_config: TieringConfig = field(default_factory=TieringConfig)

    # Flow integrity (CET-inspired)
    flow_integrity_enabled: bool = True
    flow_config: FlowConfig = field(default_factory=FlowConfig)

    # Agent quotas (Cgroups v2-inspired)
    quotas_enabled: bool = True
    quota_config: QuotaConfig = field(default_factory=QuotaConfig)

    # State snapshots (XFS Reflink-inspired)
    snapshots_enabled: bool = True
    auto_snapshot_interval: int = 0  # 0 = disabled

    # Policy engine (OPA/SELinux-inspired)
    policy_enabled: bool = True
    policy_mode: str = "enforcing"  # enforcing, permissive, disabled

    # Network fabric (Istio-inspired)
    mesh_enabled: bool = True

    # Sandbox (Flatpak-inspired)
    sandbox_enabled: bool = False  # Disabled by default for performance
    sandbox_template: str = "standard"

    # NT Kernel components (Windows-inspired)
    executive_enabled: bool = True
    executive_config: ExecutiveConfig = field(default_factory=ExecutiveConfig)

    object_manager_enabled: bool = True
    filter_stack_enabled: bool = True
    working_set_enabled: bool = True
    working_set_limits: WorkingSetLimits = field(default_factory=WorkingSetLimits)

    registry_enabled: bool = True
    aal_enabled: bool = True

    # Execution settings
    use_cache: bool = True
    parallel_execution: bool = True
    max_concurrent_steps: int = 100

    # Observability
    emit_metrics: bool = True
    trace_enabled: bool = False


# =============================================================================
# BBX 2.0 Runtime
# =============================================================================


class BBXRuntimeV2:
    """
    BBX 2.0 Runtime - The Operating System for AI Agents.

    Integrates all kernel and distribution level components:

    LINUX KERNEL:
    - AgentRing: Efficient batch operations (io_uring)
    - Hooks: Dynamic observability and security (eBPF)
    - ContextTiering: Smart memory management (MGLRU)
    - FlowIntegrity: Control flow protection (CET)
    - AgentQuotas: Resource limits (Cgroups v2)
    - StateSnapshots: CoW state management (XFS Reflink)

    NT KERNEL:
    - BBXExecutive: Hybrid kernel (ntoskrnl)
    - ObjectManager: Unified namespace (ObMgr)
    - FilterStack: Extensible pipeline (Filter Drivers)
    - WorkingSet: Memory compression (Mm)
    - ConfigRegistry: Hierarchical config (Registry)
    - AAL: Backend abstraction (HAL)

    DISTRIBUTION:
    - PolicyEngine: Access control (OPA/SELinux)
    - NetworkFabric: Service mesh (Istio)
    - AgentSandbox: Isolation (Flatpak)
    - AgentBundles: Tool collections (Kali)
    """

    def __init__(self, config: Optional[RuntimeV2Config] = None):
        self.config = config or RuntimeV2Config()

        # Core Linux kernel components
        self.ring: Optional[AgentRing] = None
        self.hooks: HookManager = get_hook_manager()
        self.tiering: Optional[ContextTiering] = None
        self.flow_integrity: Optional[FlowIntegrityEngine] = None
        self.quotas: Optional[QuotaManager] = None
        self.snapshots: Optional[StateSnapshotEngine] = None

        # NT Kernel components
        self.executive: Optional[BBXExecutive] = None
        self.object_manager: Optional[ObjectManager] = None
        self.filter_manager: Optional[FilterManager] = None
        self.working_set: Optional[WorkingSetManager] = None
        self.config_registry: Optional[ConfigRegistry] = None
        self.aal: Optional[AAL] = None

        # Distribution components
        self.policy: Optional[PolicyEngine] = None
        self.mesh: Optional[NetworkMesh] = None
        self.sandbox_manager: Optional[SandboxManager] = None
        self.bundles: Optional[BundleManager] = None

        # Adapter Registry
        self.registry = get_registry()

        # Event bus
        self.event_bus = EventBus()

        # State
        self._started = False
        self._adapters_map: Dict[str, Any] = {}
        self._active_agents: Dict[str, str] = {}  # agent_id -> workflow_id

        # Metrics
        self._metrics = {
            "workflows_executed": 0,
            "steps_executed": 0,
            "operations_batched": 0,
            "policy_checks": 0,
            "policy_denials": 0,
            "snapshots_created": 0,
            "filter_operations": 0,
            "working_set_faults": 0,
            "registry_ops": 0,
        }

    async def start(self):
        """Start all BBX 2.0 components"""
        if self._started:
            return

        logger.info("Starting BBX 2.0 Runtime (Operating System for AI Agents)...")

        # Build adapters map
        self._build_adapters_map()

        # Start Linux kernel components
        await self._start_kernel_components()

        # Start NT kernel components
        await self._start_nt_kernel_components()

        # Start distribution components
        await self._start_distribution_components()

        # Register default hooks
        self._register_default_hooks()

        self._started = True
        logger.info("BBX 2.0 Runtime started successfully")

    async def _start_kernel_components(self):
        """Start kernel-level components"""

        # AgentRing (io_uring-inspired)
        if self.config.ring_enabled:
            self.ring = AgentRing(self.config.ring_config)
            await self.ring.start(self._adapters_map)
            logger.info("  [KERNEL] AgentRing started")

        # Load hooks from directory
        if self.config.hooks_enabled and self.config.hooks_dir:
            await self._load_hooks_from_dir(self.config.hooks_dir)
        logger.info("  [KERNEL] BBX Hooks started")

        # Context tiering (MGLRU-inspired)
        if self.config.tiering_enabled:
            self.tiering = get_context_tiering()
            await self.tiering.start()
            logger.info("  [KERNEL] ContextTiering started")

        # Flow integrity (CET-inspired)
        if self.config.flow_integrity_enabled:
            self.flow_integrity = get_flow_integrity()
            logger.info("  [KERNEL] FlowIntegrity started")

        # Agent quotas (Cgroups v2-inspired)
        if self.config.quotas_enabled:
            self.quotas = get_quota_manager()
            logger.info("  [KERNEL] AgentQuotas started")

        # State snapshots (XFS Reflink-inspired)
        if self.config.snapshots_enabled:
            self.snapshots = get_snapshot_engine()
            logger.info("  [KERNEL] StateSnapshots started")

    async def _start_nt_kernel_components(self):
        """Start NT kernel-level components (Windows-inspired)"""

        # BBX Executive (Hybrid Kernel)
        if self.config.executive_enabled:
            self.executive = BBXExecutive(self.config.executive_config)
            await self.executive.start()
            logger.info("  [NT KERNEL] BBX Executive started")

            # Use Executive's components if available
            if self.executive.state == ExecutiveState.RUNNING:
                self.object_manager = self.executive.object_manager
                self.filter_manager = self.executive.filter_manager
                self.working_set = self.executive.working_set
                self.config_registry = self.executive.registry
                self.aal = self.executive.aal
                logger.info("  [NT KERNEL] Using Executive-managed components")
                return

        # Start individual NT components if Executive not used
        # Object Manager (Windows Object Manager)
        if self.config.object_manager_enabled:
            self.object_manager = get_object_manager()
            logger.info("  [NT KERNEL] ObjectManager started")

        # Filter Stack (Windows Filter Drivers)
        if self.config.filter_stack_enabled:
            self.filter_manager = get_filter_manager()
            await self.filter_manager.start()
            await self._setup_filter_stack()
            logger.info("  [NT KERNEL] FilterStack started")

        # Working Set Manager (Windows Memory Management)
        if self.config.working_set_enabled:
            self.working_set = WorkingSetManager(
                limits=self.config.working_set_limits
            )
            await self.working_set.start()
            logger.info("  [NT KERNEL] WorkingSet started")

        # Config Registry (Windows Registry)
        if self.config.registry_enabled:
            self.config_registry = get_config_registry()
            await self._init_registry_defaults()
            logger.info("  [NT KERNEL] ConfigRegistry started")

        # AAL (Windows HAL)
        if self.config.aal_enabled:
            self.aal = get_aal()
            await self.aal.start()
            logger.info("  [NT KERNEL] AAL started")

    async def _setup_filter_stack(self):
        """Set up default filter stack"""
        if not self.filter_manager:
            return

        builder = FilterStackBuilder(self.filter_manager)

        # Add audit filter
        builder.add_audit(logger.info)
        builder.add_metrics()

        # Add rate limiting
        builder.add_rate_limit(rate_per_second=100.0, burst_size=200)

        # Add cache
        builder.add_cache(max_size=10000, ttl_seconds=300.0)

        await builder.build()

    async def _init_registry_defaults(self):
        """Initialize registry with runtime defaults"""
        if not self.config_registry:
            return

        # Set runtime info
        await self.config_registry.set_value(
            f"{Hive.HKBX_SYSTEM.value}\\Runtime",
            "version", "2.0.0", ValueType.REG_SZ
        )
        await self.config_registry.set_value(
            f"{Hive.HKBX_SYSTEM.value}\\Runtime",
            "started_at", datetime.now().isoformat(), ValueType.REG_SZ
        )

    async def _start_distribution_components(self):
        """Start distribution-level components"""

        # Policy engine (OPA/SELinux-inspired)
        if self.config.policy_enabled:
            self.policy = get_policy_engine()
            logger.info("  [DISTRO] PolicyEngine started")

        # Network fabric (Istio-inspired)
        if self.config.mesh_enabled:
            self.mesh = get_mesh()
            logger.info("  [DISTRO] NetworkFabric started")

        # Sandbox manager (Flatpak-inspired)
        if self.config.sandbox_enabled:
            self.sandbox_manager = SandboxManager(quota_manager=self.quotas)
            logger.info("  [DISTRO] AgentSandbox started")

        # Bundle manager (Kali-inspired)
        self.bundles = get_bundle_manager()
        logger.info("  [DISTRO] AgentBundles started")

    async def stop(self):
        """Stop all BBX 2.0 components"""
        if not self._started:
            return

        logger.info("Stopping BBX 2.0 Runtime...")

        # Stop Linux kernel components
        if self.ring:
            await self.ring.stop()

        if self.tiering:
            await self.tiering.stop()

        # Create final snapshot
        if self.snapshots:
            for agent_id in list(self._active_agents.keys()):
                self.snapshots.create_snapshot(
                    agent_id,
                    snapshot_type=SnapshotType.SHUTDOWN,
                    description="Runtime shutdown"
                )

        # Stop NT kernel components
        if self.executive:
            await self.executive.stop()
            logger.info("  [NT KERNEL] BBX Executive stopped")
        else:
            # Stop individual components if not using Executive
            if self.filter_manager:
                await self.filter_manager.stop()
            if self.working_set:
                await self.working_set.stop()
            if self.aal:
                await self.aal.stop()

        self._started = False
        logger.info("BBX 2.0 Runtime stopped")

    def _build_adapters_map(self):
        """Build adapters map for AgentRing"""
        adapter_names = self.registry.list_adapters()
        for name in adapter_names:
            adapter = self.registry.get_adapter(name)
            if adapter:
                # Use base name without prefix
                base_name = name.replace("bbx.", "")
                self._adapters_map[base_name] = adapter
                self._adapters_map[name] = adapter

    def _register_default_hooks(self):
        """Register built-in observability and security hooks"""

        # Metrics probe hook
        metrics_hook = HookDefinition(
            id="builtin_metrics",
            name="Built-in Metrics Collector",
            type=HookType.PROBE,
            attach_points=[AttachPoint.STEP_POST_EXECUTE],
            priority=-100,  # Low priority, runs last
        )

        def metrics_handler(ctx: HookContext) -> HookResult:
            self._metrics["steps_executed"] += 1
            return HookResult(
                action=HookAction.CONTINUE,
                metrics={
                    "step_duration_ms": ctx.step_duration_ms,
                    "step_id": ctx.step_id,
                    "workflow_id": ctx.workflow_id,
                }
            )

        metrics_hook.handler = metrics_handler
        self.hooks.register(metrics_hook)

        # Policy enforcement hook (if enabled)
        if self.config.policy_enabled and self.config.policy_mode == "enforcing":
            policy_hook = HookDefinition(
                id="builtin_policy",
                name="Policy Enforcement",
                type=HookType.GUARD,
                attach_points=[AttachPoint.STEP_PRE_EXECUTE],
                priority=1000,  # High priority, runs first
            )

            async def policy_handler(ctx: HookContext) -> HookResult:
                if not self.policy:
                    return HookResult(action=HookAction.CONTINUE)

                # Check policy
                request = PolicyRequest(
                    subject=Subject(
                        id=ctx.workflow_id or "unknown",
                        type="agent",
                    ),
                    resource=Resource(
                        id=ctx.adapter_name or "unknown",
                        type="adapter",
                    ),
                    action=Action(
                        name=ctx.adapter_method or "execute",
                        category="execute",
                    ),
                )

                self._metrics["policy_checks"] += 1
                response = await self.policy.evaluate(request)

                if response.decision == Decision.DENY:
                    self._metrics["policy_denials"] += 1
                    return HookResult(
                        action=HookAction.BLOCK,
                        error=f"Policy denied: {response.reason}"
                    )

                return HookResult(action=HookAction.CONTINUE)

            policy_hook.handler = policy_handler
            self.hooks.register(policy_hook)

        # Flow integrity hook
        if self.config.flow_integrity_enabled:
            flow_hook = HookDefinition(
                id="builtin_flow_integrity",
                name="Flow Integrity Check",
                type=HookType.GUARD,
                attach_points=[AttachPoint.STEP_PRE_EXECUTE],
                priority=900,
            )

            async def flow_handler(ctx: HookContext) -> HookResult:
                if not self.flow_integrity:
                    return HookResult(action=HookAction.CONTINUE)

                agent_id = ctx.workflow_id or "unknown"

                # Validate transition
                success, violation = await self.flow_integrity.transition(
                    agent_id,
                    FlowState.EXECUTING,
                    workflow_id=ctx.workflow_id,
                    step_id=ctx.step_id,
                )

                if not success and violation:
                    return HookResult(
                        action=HookAction.BLOCK,
                        error=f"Flow integrity violation: {violation.message}"
                    )

                return HookResult(action=HookAction.CONTINUE)

            flow_hook.handler = flow_handler
            self.hooks.register(flow_hook)

    async def _load_hooks_from_dir(self, hooks_dir: Path):
        """Load hook definitions from a directory"""
        if not hooks_dir.exists():
            return

        import yaml

        for hook_file in hooks_dir.glob("*.hook.yaml"):
            try:
                with open(hook_file, "r") as f:
                    hook_data = yaml.safe_load(f)

                hook_def = HookDefinition(
                    id=hook_data.get("hook", {}).get("id", hook_file.stem),
                    name=hook_data.get("hook", {}).get("name", hook_file.stem),
                    type=HookType[hook_data.get("hook", {}).get("type", "PROBE").upper()],
                    attach_points=[
                        AttachPoint(ap) for ap in hook_data.get("hook", {}).get("attach", [])
                    ],
                    code=hook_data.get("hook", {}).get("action", {}).get("code"),
                )

                self.hooks.register(hook_def)
                logger.info(f"Loaded hook from {hook_file}")

            except Exception as e:
                logger.error(f"Failed to load hook {hook_file}: {e}")

    # =========================================================================
    # Execution Methods
    # =========================================================================

    async def execute_file(
        self,
        file_path: str,
        inputs: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow file using all BBX 2.0 features.

        Args:
            file_path: Path to .bbx workflow file
            inputs: Workflow inputs
            agent_id: Optional agent ID for tracking

        Returns:
            Dictionary of step results
        """
        if not self._started:
            await self.start()

        agent_id = agent_id or str(uuid.uuid4())

        # Load workflow
        if self.config.use_cache:
            cache = get_cache()
            cached = cache.get(file_path)
            if cached:
                data = cached
            else:
                data = BBXv6Parser.load_yaml(file_path)
                cache.put(file_path, data)
        else:
            data = BBXv6Parser.load_yaml(file_path)

        # Parse workflow structure
        if "workflow" in data:
            workflow = data.get("workflow", {})
            steps = workflow.get("steps", [])
            workflow_id = workflow.get("id", "unknown")
        else:
            workflow = data
            steps = data.get("steps", [])
            workflow_id = data.get("id", "unknown")

        # Track active agent
        self._active_agents[agent_id] = workflow_id

        # Initialize agent resources
        await self._init_agent_resources(agent_id, workflow_id)

        # Create context
        context = WorkflowContext()

        # Load default inputs
        workflow_inputs_def = workflow.get("inputs", {})
        default_inputs = {}
        for input_name, input_config in workflow_inputs_def.items():
            if isinstance(input_config, dict) and "default" in input_config:
                default_inputs[input_name] = input_config["default"]
            elif not isinstance(input_config, dict):
                default_inputs[input_name] = input_config

        merged_inputs = {**default_inputs, **(inputs or {})}
        context.variables["inputs"] = merged_inputs

        # Load state from workspace if available
        try:
            from ..workspace_manager import get_current_workspace
            workspace = get_current_workspace()
            if workspace:
                context.variables["state"] = workspace.get_all_state()
        except Exception:
            context.variables["state"] = {}

        # Store in tiered context
        if self.tiering:
            await self.tiering.set("workflow_inputs", merged_inputs)
            await self.tiering.set("workflow_id", workflow_id, pinned=True)

        # Create initial snapshot
        if self.snapshots:
            self.snapshots.create_snapshot(
                agent_id,
                snapshot_type=SnapshotType.CHECKPOINT,
                description=f"Workflow start: {workflow_id}"
            )
            self._metrics["snapshots_created"] += 1

        # Trigger workflow start hooks
        hook_ctx = HookContext(
            workflow_id=workflow_id,
            workflow_name=workflow.get("name", ""),
            variables=context.variables,
        )
        await self.hooks.trigger(AttachPoint.WORKFLOW_START, hook_ctx)

        # Execute workflow
        results: Dict[str, Any] = {}
        self._metrics["workflows_executed"] += 1

        try:
            if self.config.parallel_execution and should_use_dag(steps):
                logger.info("Using DAG parallel execution with AgentRing")
                await self._execute_dag_v2(steps, context, workflow_id, agent_id, results)
            else:
                logger.info("Using sequential execution")
                await self._execute_sequential_v2(steps, context, workflow_id, agent_id, results)

            # Trigger workflow end hooks
            hook_ctx.variables["results"] = results
            await self.hooks.trigger(AttachPoint.WORKFLOW_END, hook_ctx)

            # Create success snapshot
            if self.snapshots:
                self.snapshots.create_snapshot(
                    agent_id,
                    snapshot_type=SnapshotType.CHECKPOINT,
                    description=f"Workflow completed: {workflow_id}"
                )

        except Exception as e:
            # Trigger workflow error hooks
            hook_ctx.step_error = str(e)
            await self.hooks.trigger(AttachPoint.WORKFLOW_ERROR, hook_ctx)

            # Create error snapshot
            if self.snapshots:
                self.snapshots.create_snapshot(
                    agent_id,
                    snapshot_type=SnapshotType.ERROR,
                    description=f"Workflow error: {str(e)}"
                )

            raise

        finally:
            # Cleanup agent resources
            await self._cleanup_agent_resources(agent_id)

        await self.event_bus.emit(
            Event(EventType.WORKFLOW_END, {"id": workflow_id, "results": results})
        )

        return results

    async def _init_agent_resources(self, agent_id: str, workflow_id: str):
        """Initialize resources for an agent"""

        # Set up quota group
        if self.quotas:
            await self.quotas.create_group(
                agent_id,
                parent=None,
                limits={
                    ResourceType.MEMORY: 512 * 1024 * 1024,  # 512MB
                    ResourceType.CPU: 100.0,  # 100%
                    ResourceType.API_CALLS: 1000,
                }
            )

        # Set up flow integrity tracking
        if self.flow_integrity:
            await self.flow_integrity.transition(
                agent_id,
                FlowState.INITIALIZING,
                workflow_id=workflow_id,
            )

        # Register in network fabric
        if self.mesh:
            self.mesh.create_sidecar(agent_id)

    async def _cleanup_agent_resources(self, agent_id: str):
        """Cleanup resources for an agent"""

        # Remove from active agents
        self._active_agents.pop(agent_id, None)

        # Cleanup quota group
        if self.quotas:
            await self.quotas.delete_group(agent_id)

        # Transition to terminated state
        if self.flow_integrity:
            await self.flow_integrity.transition(
                agent_id,
                FlowState.TERMINATED,
                force=True,
            )

    async def _execute_dag_v2(
        self,
        steps: list,
        context: WorkflowContext,
        workflow_id: str,
        agent_id: str,
        results: Dict[str, Any],
    ):
        """Execute workflow using DAG with AgentRing"""
        dag = WorkflowDAG(steps)
        levels = dag.get_execution_levels()

        for level_idx, level in enumerate(levels):
            logger.info(f"Level {level_idx + 1}: {len(level)} step(s)")

            if self.ring and len(level) > 1:
                # Use AgentRing for batch execution
                await self._execute_level_with_ring(
                    level, dag, context, workflow_id, agent_id, results
                )
            else:
                # Single step or ring disabled, execute directly
                for step_id in level:
                    step = dag.get_step(step_id)
                    await self._execute_step_v2(
                        step, context, workflow_id, agent_id, results
                    )

    async def _execute_level_with_ring(
        self,
        level: List[str],
        dag: WorkflowDAG,
        context: WorkflowContext,
        workflow_id: str,
        agent_id: str,
        results: Dict[str, Any],
    ):
        """Execute a level of steps using AgentRing batch submission"""

        # Prepare operations
        operations = []
        step_map = {}  # op_id -> step

        for step_id in level:
            step = dag.get_step(step_id)

            # Resolve inputs before submission
            resolved_inputs = context.resolve_recursive(step.get("inputs", {}))

            # Check 'when' condition
            when_condition = step.get("when")
            if when_condition:
                try:
                    resolved_condition = context.resolve(when_condition)
                    if not SafeExpr.evaluate(resolved_condition, context.variables):
                        logger.info(f"Skipping {step_id} (condition false)")
                        results[step_id] = {"status": "skipped"}
                        continue
                except ExpressionError as e:
                    logger.error(f"Condition error in {step_id}: {e}")
                    results[step_id] = {"status": "skipped", "error": str(e)}
                    continue

            # Check quota before submission
            if self.quotas:
                allowed, delay = await self.quotas.check_and_record(
                    agent_id, ResourceType.API_CALLS, 1
                )
                if not allowed:
                    results[step_id] = {"status": "error", "error": "Quota exceeded"}
                    continue
                if delay:
                    await asyncio.sleep(delay)

            # Get adapter info
            mcp_type = step.get("mcp", step.get("adapter", ""))
            method = step.get("method", "")

            # Create operation
            op = Operation(
                op_type=OperationType.ADAPTER_CALL,
                adapter=mcp_type.replace("bbx.", ""),
                method=method,
                args=resolved_inputs,
                timeout_ms=step.get("timeout", 30000),
                retry_count=step.get("retry", 0),
                user_data={"step_id": step_id, "step": step},
            )

            operations.append(op)
            step_map[op.id] = step

        if not operations:
            return

        # Submit batch to ring
        op_ids = await self.ring.submit_batch(operations)
        self._metrics["operations_batched"] += len(operations)

        # Wait for all completions
        completions = await self.ring.wait_batch(op_ids, timeout=300.0)

        # Process results
        for completion in completions:
            step_data = completion.user_data if hasattr(completion, 'user_data') else {}
            step_id = step_data.get("step_id") if step_data else None

            if not step_id:
                # Find step_id from operation
                for op in operations:
                    if op.id == completion.operation_id:
                        step_id = op.user_data.get("step_id")
                        break

            if completion.status == OperationStatus.COMPLETED:
                context.set_step_output(step_id, completion.result)
                results[step_id] = {"status": "success", "output": completion.result}
                logger.info(f"[RING] {step_id} completed in {completion.duration_ms:.1f}ms")
            else:
                error_msg = completion.error or f"Operation {completion.status.name}"
                results[step_id] = {"status": "error", "error": error_msg}
                context.set_step_output(step_id, {"status": "error", "error": error_msg})
                logger.error(f"[RING] {step_id} failed: {error_msg}")

            # Trigger hooks
            hook_ctx = HookContext(
                workflow_id=workflow_id,
                step_id=step_id,
                step_outputs=completion.result if completion.status == OperationStatus.COMPLETED else None,
                step_error=completion.error if completion.status != OperationStatus.COMPLETED else None,
                step_duration_ms=completion.duration_ms,
            )
            await self.hooks.trigger(AttachPoint.STEP_POST_EXECUTE, hook_ctx)

    async def _execute_sequential_v2(
        self,
        steps: list,
        context: WorkflowContext,
        workflow_id: str,
        agent_id: str,
        results: Dict[str, Any],
    ):
        """Execute workflow steps sequentially"""
        for step in steps:
            await self._execute_step_v2(step, context, workflow_id, agent_id, results)

    async def _execute_step_v2(
        self,
        step: Dict[str, Any],
        context: WorkflowContext,
        workflow_id: str,
        agent_id: str,
        results: Dict[str, Any],
    ):
        """Execute a single step with all BBX 2.0 integrations"""
        step_id = step.get("id")
        mcp_type = step.get("mcp", step.get("adapter", ""))
        method = step.get("method", "")
        inputs = step.get("inputs", {})

        # Validate required fields
        if not step_id:
            raise ValueError("Step 'id' is required")
        if not mcp_type:
            raise ValueError(f"Step '{step_id}': 'mcp' or 'adapter' is required")
        if not method:
            raise ValueError(f"Step '{step_id}': 'method' is required")

        # Create hook context
        hook_ctx = HookContext(
            workflow_id=workflow_id,
            step_id=step_id,
            step_type=f"{mcp_type}.{method}",
            step_inputs=inputs,
            adapter_name=mcp_type,
            adapter_method=method,
            variables=context.variables,
        )

        # Trigger pre-execute hooks (includes policy and flow integrity checks)
        hook_result = await self.hooks.trigger(AttachPoint.STEP_PRE_EXECUTE, hook_ctx)

        if hook_result.action == HookAction.SKIP:
            logger.info(f"Step {step_id} skipped by hook")
            results[step_id] = {"status": "skipped", "reason": "hook"}
            return

        if hook_result.action == HookAction.BLOCK:
            logger.error(f"Step {step_id} blocked by hook: {hook_result.error}")
            results[step_id] = {"status": "error", "error": hook_result.error}
            return

        # Check quotas
        if self.quotas:
            allowed, delay = await self.quotas.check_and_record(
                agent_id, ResourceType.API_CALLS, 1
            )
            if not allowed:
                results[step_id] = {"status": "error", "error": "Quota exceeded"}
                return
            if delay:
                await asyncio.sleep(delay)

        start_time = datetime.now()

        try:
            # Resolve inputs
            resolved_inputs = context.resolve_recursive(inputs)

            # Use transformed inputs if provided by hook
            if hook_result.action == HookAction.TRANSFORM and hook_result.data:
                resolved_inputs = hook_result.data

            # Check 'when' condition
            when_condition = step.get("when")
            if when_condition:
                try:
                    resolved_condition = context.resolve(when_condition)
                    if not SafeExpr.evaluate(resolved_condition, context.variables):
                        logger.info(f"Skipping {step_id} (condition false)")
                        results[step_id] = {"status": "skipped"}
                        return
                except ExpressionError as e:
                    logger.error(f"Condition error in {step_id}: {e}")
                    results[step_id] = {"status": "skipped", "error": str(e)}
                    return

            # Get adapter
            adapter = self.registry.get_adapter(mcp_type)
            if not adapter:
                raise ValueError(f"Unknown adapter: {mcp_type}")

            # Inject context if supported
            if hasattr(adapter, "set_context"):
                adapter.set_context(context)

            # Execute with timeout and retry
            timeout_ms = step.get("timeout", 30000)
            timeout_sec = timeout_ms / 1000
            retry_count = step.get("retry", 0)

            logger.info(f"Executing {step_id} ({mcp_type}.{method})")

            output = None
            for attempt in range(retry_count + 1):
                try:
                    output = await asyncio.wait_for(
                        adapter.execute(method, resolved_inputs),
                        timeout=timeout_sec
                    )
                    break
                except asyncio.TimeoutError:
                    if attempt < retry_count:
                        logger.warning(f"Timeout - Retry {attempt + 1}/{retry_count}")
                        await asyncio.sleep(1.0 * (2 ** attempt))
                    else:
                        raise
                except Exception as e:
                    if attempt < retry_count:
                        logger.warning(f"Error - Retry {attempt + 1}/{retry_count}: {e}")
                        await asyncio.sleep(1.0 * (2 ** attempt))
                    else:
                        raise

            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Update hook context with results
            hook_ctx.step_outputs = output
            hook_ctx.step_duration_ms = duration_ms

            # Trigger post-execute hooks
            post_result = await self.hooks.trigger(AttachPoint.STEP_POST_EXECUTE, hook_ctx)

            # Use transformed output if provided
            if post_result.action == HookAction.TRANSFORM and post_result.data:
                output = post_result.data

            # Store output
            context.set_step_output(step_id, output)
            results[step_id] = {"status": "success", "output": output}

            # Store in tiered context
            if self.tiering:
                await self.tiering.set(f"step.{step_id}.output", output)

            # Store in state snapshots
            if self.snapshots:
                self.snapshots.set_state(agent_id, f"step.{step_id}", output)

            await self.event_bus.emit(
                Event(EventType.STEP_END, {"step_id": step_id, "output": output})
            )
            logger.info(f"{step_id} completed in {duration_ms:.1f}ms")

        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Update hook context with error
            hook_ctx.step_error = str(e)
            hook_ctx.step_duration_ms = duration_ms

            # Trigger error hooks
            await self.hooks.trigger(AttachPoint.STEP_ERROR, hook_ctx)

            traceback.print_exc()
            logger.error(f"{step_id} failed: {e}")

            error_data = {"status": "error", "error": str(e)}
            results[step_id] = error_data
            context.set_step_output(step_id, error_data)

            await self.event_bus.emit(
                Event(EventType.STEP_ERROR, {"step_id": step_id, "error": str(e)})
            )

    # =========================================================================
    # Management Methods
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics"""
        metrics = {**self._metrics}

        # Add Linux kernel component metrics
        if self.ring:
            metrics["ring"] = self.ring.get_stats()

        if self.tiering:
            metrics["tiering"] = self.tiering.get_stats()

        if self.quotas:
            metrics["quotas"] = self.quotas.get_stats()

        if self.snapshots:
            metrics["snapshots"] = self.snapshots.get_stats()

        # Add NT kernel component metrics
        if self.executive:
            metrics["executive"] = self.executive.get_stats()

        if self.filter_manager:
            metrics["filter_stack"] = self.filter_manager.get_all_stats()

        if self.working_set:
            metrics["working_set"] = self.working_set.get_stats()

        # Add distribution component metrics
        if self.policy:
            metrics["policy"] = self.policy.get_stats()

        if self.mesh:
            metrics["mesh"] = self.mesh.get_metrics()

        if self.bundles:
            metrics["bundles"] = self.bundles.get_stats()

        return metrics

    def get_active_agents(self) -> Dict[str, str]:
        """Get currently active agents"""
        return self._active_agents.copy()


# =============================================================================
# Global Runtime Instance
# =============================================================================


_global_runtime: Optional[BBXRuntimeV2] = None


def get_runtime_v2() -> BBXRuntimeV2:
    """Get global BBX 2.0 Runtime instance"""
    global _global_runtime
    if _global_runtime is None:
        _global_runtime = BBXRuntimeV2()
    return _global_runtime


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_file_v2(
    file_path: str,
    inputs: Optional[Dict[str, Any]] = None,
    config: Optional[RuntimeV2Config] = None,
) -> Dict[str, Any]:
    """
    Execute a workflow file using BBX 2.0 Runtime.

    This is the main entry point for BBX 2.0 execution.

    Args:
        file_path: Path to .bbx workflow file
        inputs: Workflow inputs
        config: Optional runtime configuration

    Returns:
        Dictionary of step results
    """
    if config:
        runtime = BBXRuntimeV2(config)
        await runtime.start()
        try:
            return await runtime.execute_file(file_path, inputs)
        finally:
            await runtime.stop()
    else:
        runtime = get_runtime_v2()
        return await runtime.execute_file(file_path, inputs)


async def execute_batch(
    operations: List[Dict[str, Any]],
    timeout: float = 300.0,
) -> List[Dict[str, Any]]:
    """
    Execute a batch of operations using AgentRing.

    Args:
        operations: List of operation dicts with keys:
            - adapter: Adapter name
            - method: Method name
            - args: Method arguments
        timeout: Timeout in seconds

    Returns:
        List of results
    """
    runtime = get_runtime_v2()
    if not runtime._started:
        await runtime.start()

    if not runtime.ring:
        raise RuntimeError("AgentRing not enabled")

    # Convert to Operation objects
    ops = [
        Operation(
            adapter=op.get("adapter", ""),
            method=op.get("method", ""),
            args=op.get("args", {}),
        )
        for op in operations
    ]

    # Submit batch
    op_ids = await runtime.ring.submit_batch(ops)

    # Wait for completions
    completions = await runtime.ring.wait_batch(op_ids, timeout=timeout)

    # Convert to results
    return [
        {
            "status": "success" if c.status == OperationStatus.COMPLETED else "error",
            "result": c.result if c.status == OperationStatus.COMPLETED else None,
            "error": c.error if c.status != OperationStatus.COMPLETED else None,
            "duration_ms": c.duration_ms,
        }
        for c in completions
    ]
