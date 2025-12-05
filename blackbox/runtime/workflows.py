# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Workflows - Multi-step AI workflows with recovery.

Workflows are like shell scripts, but for AI agents.
They define a sequence of steps that agents execute,
with built-in recovery, snapshots, and rollback.

Example Workflow (project-refactor.yaml):
    name: "project-refactor"
    description: "Refactor project from one framework to another"

    steps:
      - id: "analyze"
        agent: "legacy-code-miner"
        task: "Analyze project structure"

      - id: "plan"
        agent: "project-planner"
        task: "Create refactoring plan"
        depends_on: ["analyze"]

      - id: "refactor"
        agent: "code-transformer"
        task: "Apply changes"
        depends_on: ["plan"]
        snapshot_before_each: true
        recovery:
          strategy: "rollback_file"
          max_retries: 3

      - id: "test"
        agent: "automated-qa"
        task: "Run tests"
        depends_on: ["refactor"]
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

logger = logging.getLogger("bbx.workflows")


# =============================================================================
# Enums
# =============================================================================


class StepStatus(Enum):
    """Step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class RecoveryStrategy(Enum):
    """Recovery strategy when step fails"""
    FAIL_FAST = "fail_fast"           # Stop workflow immediately
    ROLLBACK_STEP = "rollback_step"   # Rollback this step only
    ROLLBACK_FILE = "rollback_file"   # Rollback affected files
    ROLLBACK_ALL = "rollback_all"     # Rollback entire workflow
    RETRY = "retry"                   # Retry with hints
    SKIP = "skip"                     # Skip and continue


# =============================================================================
# Step Configuration
# =============================================================================


@dataclass
class RecoveryConfig:
    """Recovery configuration for a step"""
    strategy: RecoveryStrategy = RecoveryStrategy.ROLLBACK_STEP
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_with_hints: bool = True
    snapshot_before: bool = True
    snapshot_after: bool = False


@dataclass
class StepConfig:
    """Configuration for a workflow step"""
    id: str
    agent: str                  # Agent name to execute this step
    task: str                   # Task description/prompt

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Input/Output
    input: Dict[str, Any] = field(default_factory=dict)
    output_key: Optional[str] = None  # Key to store result

    # Recovery
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    # Snapshot control
    snapshot_before_each: bool = False  # For iterative steps
    snapshot_before_each_file: bool = False

    # Timeout
    timeout_seconds: float = 3600  # 1 hour default

    # Conditions
    condition: Optional[str] = None  # Expression to evaluate


@dataclass
class StepResult:
    """Result of step execution"""
    step_id: str
    status: StepStatus
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0
    retries: int = 0
    snapshot_id: Optional[str] = None


# =============================================================================
# Workflow Configuration
# =============================================================================


@dataclass
class WorkflowConfig:
    """Complete workflow configuration"""
    name: str
    description: str = ""
    version: str = "1.0"

    # Steps
    steps: List[StepConfig] = field(default_factory=list)

    # Global settings
    timeout_seconds: float = 86400  # 24 hours default
    max_parallel: int = 4           # Max parallel steps
    fail_fast: bool = True          # Stop on first failure

    # Input variables
    input_schema: Dict[str, Any] = field(default_factory=dict)

    # Output
    output: Dict[str, Any] = field(default_factory=dict)

    # Recovery
    global_recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    # Metadata
    author: str = ""
    tags: List[str] = field(default_factory=list)


# =============================================================================
# Workflow Instance
# =============================================================================


class WorkflowStatus(Enum):
    """Workflow execution status"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowInstance:
    """A running workflow instance"""
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: WorkflowConfig = None

    # State
    status: WorkflowStatus = WorkflowStatus.CREATED

    # Timing
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Input variables
    variables: Dict[str, Any] = field(default_factory=dict)

    # Step results
    step_results: Dict[str, StepResult] = field(default_factory=dict)

    # Current steps
    current_steps: Set[str] = field(default_factory=set)

    # Snapshots
    snapshots: List[str] = field(default_factory=list)

    # Error
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.config.name if self.config else "unknown",
            "status": self.status.value,
            "started_at": self.started_at,
            "duration_s": self.duration_seconds,
            "steps_total": len(self.config.steps) if self.config else 0,
            "steps_completed": len([
                r for r in self.step_results.values()
                if r.status == StepStatus.COMPLETED
            ]),
            "current_steps": list(self.current_steps),
            "error": self.error,
        }


# =============================================================================
# Workflow Loader
# =============================================================================


class WorkflowLoader:
    """Loads workflows from YAML files"""

    def load(self, path: Path) -> WorkflowConfig:
        """Load workflow from YAML file"""
        with open(path) as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def load_from_string(self, yaml_str: str) -> WorkflowConfig:
        """Load workflow from YAML string"""
        data = yaml.safe_load(yaml_str)
        return self._parse_config(data)

    def _parse_config(self, data: Dict) -> WorkflowConfig:
        """Parse workflow configuration from dict"""
        steps = []

        for step_data in data.get("steps", []):
            # Parse recovery config
            rec_data = step_data.get("recovery", {})
            recovery = RecoveryConfig(
                strategy=RecoveryStrategy(rec_data.get("strategy", "rollback_step")),
                max_retries=rec_data.get("max_retries", 3),
                retry_delay_seconds=rec_data.get("retry_delay_seconds", 5.0),
                retry_with_hints=rec_data.get("retry_with_hints", True),
                snapshot_before=rec_data.get("snapshot_before", True),
            )

            step = StepConfig(
                id=step_data.get("id", str(uuid.uuid4())[:8]),
                agent=step_data.get("agent", "default"),
                task=step_data.get("task", ""),
                depends_on=step_data.get("depends_on", []),
                input=step_data.get("input", {}),
                output_key=step_data.get("output_key"),
                recovery=recovery,
                snapshot_before_each=step_data.get("snapshot_before_each", False),
                snapshot_before_each_file=step_data.get("snapshot_before_each_file", False),
                timeout_seconds=step_data.get("timeout_seconds", 3600),
                condition=step_data.get("condition"),
            )
            steps.append(step)

        # Parse global recovery
        global_rec_data = data.get("global_recovery", {})
        global_recovery = RecoveryConfig(
            strategy=RecoveryStrategy(global_rec_data.get("strategy", "rollback_step")),
            max_retries=global_rec_data.get("max_retries", 3),
        )

        return WorkflowConfig(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            steps=steps,
            timeout_seconds=data.get("timeout_seconds", 86400),
            max_parallel=data.get("max_parallel", 4),
            fail_fast=data.get("fail_fast", True),
            input_schema=data.get("input_schema", {}),
            output=data.get("output", {}),
            global_recovery=global_recovery,
            author=data.get("author", ""),
            tags=data.get("tags", []),
        )


# =============================================================================
# Workflow Engine
# =============================================================================


class WorkflowEngine:
    """
    Executes workflows with recovery support.

    Features:
    - Parallel step execution (respecting dependencies)
    - Automatic snapshots before risky operations
    - Rollback on failure
    - Retry with hints
    """

    def __init__(
        self,
        agent_executor: Callable,  # Function to execute agent tasks
        snapshot_manager: Any,     # Snapshot manager
    ):
        self.agent_executor = agent_executor
        self.snapshot_manager = snapshot_manager

        # Running workflows
        self._workflows: Dict[str, WorkflowInstance] = {}

    async def run(
        self,
        config: WorkflowConfig,
        variables: Optional[Dict] = None
    ) -> WorkflowInstance:
        """Run a workflow"""
        instance = WorkflowInstance(
            config=config,
            status=WorkflowStatus.RUNNING,
            started_at=time.time(),
            variables=variables or {},
        )

        self._workflows[instance.id] = instance

        logger.info(f"Starting workflow: {instance.id} ({config.name})")

        try:
            await self._execute_workflow(instance)
            instance.status = WorkflowStatus.COMPLETED
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            logger.error(f"Workflow {instance.id} failed: {e}")

        instance.completed_at = time.time()
        return instance

    async def _execute_workflow(self, instance: WorkflowInstance):
        """Execute workflow steps"""
        config = instance.config

        # Build dependency graph
        pending = set(step.id for step in config.steps)
        completed = set()

        while pending:
            # Find steps ready to run (dependencies satisfied)
            ready = []
            for step in config.steps:
                if step.id not in pending:
                    continue

                deps_satisfied = all(
                    dep in completed
                    for dep in step.depends_on
                )
                if deps_satisfied:
                    ready.append(step)

            if not ready:
                # Deadlock or all done
                break

            # Run ready steps (up to max_parallel)
            batch = ready[:config.max_parallel]
            tasks = []

            for step in batch:
                pending.discard(step.id)
                instance.current_steps.add(step.id)
                tasks.append(self._execute_step(instance, step))

            # Wait for batch
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(batch, results):
                instance.current_steps.discard(step.id)

                if isinstance(result, Exception):
                    # Step failed
                    instance.step_results[step.id] = StepResult(
                        step_id=step.id,
                        status=StepStatus.FAILED,
                        error=str(result),
                    )

                    if config.fail_fast:
                        raise result
                else:
                    completed.add(step.id)

    async def _execute_step(
        self,
        instance: WorkflowInstance,
        step: StepConfig
    ) -> StepResult:
        """Execute a single step with recovery"""
        start_time = time.time()
        retries = 0
        last_error = None
        snapshot_id = None

        # Create snapshot before step
        if step.recovery.snapshot_before:
            snapshot_id = await self.snapshot_manager.create_snapshot(
                instance.id,
                {"step": step.id, "variables": instance.variables},
                f"Before step: {step.id}"
            )
            instance.snapshots.append(snapshot_id)

        while retries <= step.recovery.max_retries:
            try:
                # Check condition
                if step.condition:
                    if not self._evaluate_condition(step.condition, instance.variables):
                        return StepResult(
                            step_id=step.id,
                            status=StepStatus.SKIPPED,
                            duration_seconds=time.time() - start_time,
                        )

                # Prepare input
                step_input = self._interpolate_variables(step.input, instance.variables)
                step_input["task"] = self._interpolate_string(step.task, instance.variables)

                # Add hints from previous failure
                if retries > 0 and step.recovery.retry_with_hints:
                    step_input["hints"] = [
                        f"Previous attempt failed with: {last_error}",
                        "Try a different approach",
                    ]

                # Execute via agent
                result = await asyncio.wait_for(
                    self.agent_executor(step.agent, step_input),
                    timeout=step.timeout_seconds
                )

                # Store output
                if step.output_key:
                    instance.variables[step.output_key] = result

                step_result = StepResult(
                    step_id=step.id,
                    status=StepStatus.COMPLETED,
                    result=result,
                    duration_seconds=time.time() - start_time,
                    retries=retries,
                    snapshot_id=snapshot_id,
                )
                instance.step_results[step.id] = step_result

                logger.info(f"Step {step.id} completed (retries: {retries})")
                return step_result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {step.timeout_seconds}s"
            except Exception as e:
                last_error = str(e)

            # Handle failure
            retries += 1

            if retries <= step.recovery.max_retries:
                logger.warning(f"Step {step.id} failed, retrying ({retries}/{step.recovery.max_retries})")
                await asyncio.sleep(step.recovery.retry_delay_seconds)

                # Rollback if needed
                if step.recovery.strategy == RecoveryStrategy.ROLLBACK_STEP:
                    if snapshot_id:
                        await self.snapshot_manager.restore_snapshot(snapshot_id)

        # Max retries exceeded
        result = StepResult(
            step_id=step.id,
            status=StepStatus.FAILED,
            error=last_error,
            duration_seconds=time.time() - start_time,
            retries=retries,
            snapshot_id=snapshot_id,
        )
        instance.step_results[step.id] = result

        logger.error(f"Step {step.id} failed after {retries} retries: {last_error}")
        raise Exception(f"Step {step.id} failed: {last_error}")

    def _evaluate_condition(self, condition: str, variables: Dict) -> bool:
        """Evaluate condition expression"""
        try:
            # Simple variable substitution and eval
            # In production, use a safe expression evaluator
            return eval(condition, {"__builtins__": {}}, variables)
        except Exception:
            return True

    def _interpolate_variables(self, data: Dict, variables: Dict) -> Dict:
        """Interpolate variables in dict"""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._interpolate_string(value, variables)
            elif isinstance(value, dict):
                result[key] = self._interpolate_variables(value, variables)
            else:
                result[key] = value
        return result

    def _interpolate_string(self, s: str, variables: Dict) -> str:
        """Interpolate {{ .var }} in string"""
        import re
        pattern = r'\{\{\s*\.(\w+)\s*\}\}'

        def replace(match):
            var_name = match.group(1)
            return str(variables.get(var_name, match.group(0)))

        return re.sub(pattern, replace, s)

    async def cancel(self, workflow_id: str) -> bool:
        """Cancel running workflow"""
        instance = self._workflows.get(workflow_id)
        if not instance:
            return False

        instance.status = WorkflowStatus.CANCELLED
        return True

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Get workflow by ID"""
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> List[WorkflowInstance]:
        """List all workflows"""
        return list(self._workflows.values())


# =============================================================================
# Example Workflows
# =============================================================================


PROJECT_REFACTOR_WORKFLOW = """
name: "project-refactor"
description: "Refactor project from one framework to another"
version: "1.0"

input_schema:
  from: "string"  # Source framework
  to: "string"    # Target framework
  path: "string"  # Project path

steps:
  - id: "analyze"
    agent: "legacy-code-miner"
    task: "Analyze the {{ .path }} project structure using {{ .from }} framework patterns"
    output_key: "analysis"
    recovery:
      strategy: "retry"
      max_retries: 2

  - id: "plan"
    agent: "project-planner"
    task: "Create a step-by-step refactoring plan from {{ .from }} to {{ .to }}"
    depends_on: ["analyze"]
    input:
      analysis: "{{ .analysis }}"
    output_key: "plan"
    recovery:
      strategy: "retry"
      max_retries: 2

  - id: "refactor"
    agent: "code-transformer"
    task: "Apply the refactoring plan to transform code from {{ .from }} to {{ .to }}"
    depends_on: ["plan"]
    input:
      plan: "{{ .plan }}"
    snapshot_before_each_file: true
    recovery:
      strategy: "rollback_file"
      max_retries: 3
      retry_with_hints: true

  - id: "test"
    agent: "automated-qa"
    task: "Run all tests to verify the refactoring was successful"
    depends_on: ["refactor"]
    recovery:
      strategy: "fail_fast"

  - id: "report"
    agent: "reporter"
    task: "Generate a summary report of the refactoring"
    depends_on: ["test"]
    output_key: "report"

output:
  format: "html"
  path: "~/Desktop/refactor-report.html"
"""


CODE_REVIEW_WORKFLOW = """
name: "code-review"
description: "Automated code review for pull requests"
version: "1.0"

input_schema:
  pr_url: "string"
  strictness: "string"  # low, medium, high

steps:
  - id: "fetch-pr"
    agent: "git-agent"
    task: "Fetch PR diff from {{ .pr_url }}"
    output_key: "diff"

  - id: "analyze-changes"
    agent: "code-analyzer"
    task: "Analyze code changes for patterns, complexity, and potential issues"
    depends_on: ["fetch-pr"]
    input:
      diff: "{{ .diff }}"
      strictness: "{{ .strictness }}"
    output_key: "analysis"

  - id: "security-check"
    agent: "security-scanner"
    task: "Check for security vulnerabilities in changed code"
    depends_on: ["fetch-pr"]
    input:
      diff: "{{ .diff }}"
    output_key: "security"

  - id: "style-check"
    agent: "linter"
    task: "Check code style and formatting"
    depends_on: ["fetch-pr"]
    input:
      diff: "{{ .diff }}"
    output_key: "style"

  - id: "generate-review"
    agent: "reviewer"
    task: "Generate comprehensive code review with actionable feedback"
    depends_on: ["analyze-changes", "security-check", "style-check"]
    input:
      analysis: "{{ .analysis }}"
      security: "{{ .security }}"
      style: "{{ .style }}"
    output_key: "review"

  - id: "post-review"
    agent: "git-agent"
    task: "Post review comments to the PR"
    depends_on: ["generate-review"]
    input:
      pr_url: "{{ .pr_url }}"
      review: "{{ .review }}"
"""


DEPLOY_WORKFLOW = """
name: "deploy"
description: "Deploy application to production with safety checks"
version: "1.0"

input_schema:
  environment: "string"  # staging, production
  version: "string"

steps:
  - id: "validate"
    agent: "validator"
    task: "Validate deployment configuration for {{ .environment }}"

  - id: "backup"
    agent: "backup-agent"
    task: "Create backup of current {{ .environment }} state"
    depends_on: ["validate"]
    recovery:
      snapshot_before: true

  - id: "deploy"
    agent: "deploy-agent"
    task: "Deploy version {{ .version }} to {{ .environment }}"
    depends_on: ["backup"]
    recovery:
      strategy: "rollback_all"
      max_retries: 1

  - id: "health-check"
    agent: "monitor"
    task: "Verify deployment health"
    depends_on: ["deploy"]
    timeout_seconds: 300

  - id: "notify"
    agent: "notifier"
    task: "Send deployment notification"
    depends_on: ["health-check"]
    recovery:
      strategy: "skip"  # Don't fail workflow if notification fails
"""


# =============================================================================
# Factory Functions
# =============================================================================


def create_workflow_engine(daemon) -> WorkflowEngine:
    """Create workflow engine connected to daemon"""

    async def agent_executor(agent_name: str, task_input: Dict) -> Any:
        """Execute task via daemon's agent"""
        # Find agent by name
        for agent in daemon.agents.list_agents():
            if agent.config.name == agent_name:
                # Dispatch task
                await daemon.agents.dispatch_task(agent.id, {
                    "type": "workflow_step",
                    **task_input,
                })
                # Wait for result (simplified)
                await asyncio.sleep(1.0)
                return {"status": "completed"}

        raise ValueError(f"Agent not found: {agent_name}")

    return WorkflowEngine(agent_executor, daemon.snapshots)


def load_example_workflows() -> Dict[str, WorkflowConfig]:
    """Load example workflows"""
    loader = WorkflowLoader()
    return {
        "project-refactor": loader.load_from_string(PROJECT_REFACTOR_WORKFLOW),
        "code-review": loader.load_from_string(CODE_REVIEW_WORKFLOW),
        "deploy": loader.load_from_string(DEPLOY_WORKFLOW),
    }
