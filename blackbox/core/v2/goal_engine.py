# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 GoalEngine - LLM-based planning and goal execution.

Features:
- LLM-based task decomposition
- DAG execution with dependencies
- Hierarchical planning (goals -> milestones -> tasks -> steps)
- Cost-aware planning
- Automatic replanning on failure
- Multi-agent delegation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("bbx.goal_engine")


class GoalStatus(Enum):
    PENDING = auto()
    PLANNING = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """A single executable task"""
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    status: GoalStatus = GoalStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    depends_on: List[str] = field(default_factory=list)
    estimated_cost: float = 0.0  # Time in seconds
    actual_cost: float = 0.0
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    executor: Optional[str] = None  # Tool or agent to execute
    args: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class Milestone:
    """A milestone containing multiple tasks"""
    id: str = field(default_factory=lambda: f"milestone_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    tasks: List[Task] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Goal:
    """A high-level goal with milestones"""
    id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    milestones: List[Milestone] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    agent_id: Optional[str] = None


class PlannerBackend(ABC):
    """Abstract planner backend"""

    @abstractmethod
    async def decompose_goal(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Decompose a goal into tasks"""
        pass

    @abstractmethod
    async def replan(
        self,
        goal: Goal,
        failed_task: Task,
        error: str
    ) -> List[Task]:
        """Generate new plan after failure"""
        pass


class LLMPlanner(PlannerBackend):
    """LLM-based planner using OpenAI/Anthropic"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        provider: str = "openai"
    ):
        self._api_key = api_key
        self._model = model
        self._provider = provider
        self._client = None

    async def _call_llm(self, prompt: str) -> str:
        if self._provider == "openai":
            try:
                from openai import AsyncOpenAI
                if self._client is None:
                    self._client = AsyncOpenAI(api_key=self._api_key)
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return response.choices[0].message.content
            except ImportError:
                raise ImportError("openai required: pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self._provider}")

    async def decompose_goal(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        prompt = f"""You are a task planning assistant. Decompose this goal into concrete, executable tasks.

Goal: {goal}

Context: {json.dumps(context, indent=2)}

Return a JSON array of tasks with this structure:
[
  {{
    "name": "Task name",
    "description": "What needs to be done",
    "executor": "tool_name or agent_name",
    "args": {{}},
    "depends_on": ["task_id"],
    "estimated_cost": 10.0
  }}
]

Only return valid JSON, no other text."""

        response = await self._call_llm(prompt)

        try:
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response: {response}")

        return []

    async def replan(
        self,
        goal: Goal,
        failed_task: Task,
        error: str
    ) -> List[Task]:
        prompt = f"""A task in a plan failed. Generate alternative tasks to achieve the goal.

Goal: {goal.name} - {goal.description}

Failed Task: {failed_task.name}
Error: {error}

Completed Tasks: {[t.name for m in goal.milestones for t in m.tasks if t.status == GoalStatus.COMPLETED]}

Generate alternative tasks as JSON array:
[{{"name": "...", "description": "...", "executor": "...", "args": {{}}}}]"""

        response = await self._call_llm(prompt)

        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                tasks_data = json.loads(response[start:end])
                return [
                    Task(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        executor=t.get("executor"),
                        args=t.get("args", {})
                    )
                    for t in tasks_data
                ]
        except json.JSONDecodeError:
            pass

        return []


class SimplePlanner(PlannerBackend):
    """Simple rule-based planner (no LLM)"""

    def __init__(self, task_templates: Optional[Dict[str, List[Dict]]] = None):
        self._templates = task_templates or {}

    async def decompose_goal(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Match goal to templates
        goal_lower = goal.lower()
        for pattern, tasks in self._templates.items():
            if pattern in goal_lower:
                return tasks
        return []

    async def replan(
        self,
        goal: Goal,
        failed_task: Task,
        error: str
    ) -> List[Task]:
        # Simple retry logic
        if failed_task.retries < failed_task.max_retries:
            retry_task = Task(
                name=f"Retry: {failed_task.name}",
                description=failed_task.description,
                executor=failed_task.executor,
                args=failed_task.args,
                retries=failed_task.retries + 1
            )
            return [retry_task]
        return []


class DAGExecutor:
    """Executes tasks in DAG order with parallelization"""

    def __init__(self, max_parallel: int = 4):
        self._max_parallel = max_parallel
        self._executors: Dict[str, Callable] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None

    def register_executor(self, name: str, func: Callable):
        """Register a task executor"""
        self._executors[name] = func

    async def execute(
        self,
        tasks: List[Task],
        on_task_complete: Optional[Callable[[Task], None]] = None
    ) -> List[Task]:
        """Execute tasks in DAG order"""
        self._semaphore = asyncio.Semaphore(self._max_parallel)

        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        completed: Set[str] = set()
        pending = list(tasks)

        while pending:
            # Find tasks ready to execute
            ready = [
                t for t in pending
                if all(dep in completed for dep in t.depends_on)
            ]

            if not ready:
                # Check for deadlock
                logger.error("DAG deadlock detected")
                break

            # Execute ready tasks in parallel
            await asyncio.gather(*[
                self._execute_task(t, on_task_complete)
                for t in ready
            ])

            for t in ready:
                completed.add(t.id)
                pending.remove(t)

        return tasks

    async def _execute_task(
        self,
        task: Task,
        on_complete: Optional[Callable[[Task], None]]
    ):
        async with self._semaphore:
            task.status = GoalStatus.IN_PROGRESS
            task.started_at = time.time()

            try:
                executor = self._executors.get(task.executor)
                if executor:
                    task.result = await executor(**task.args)
                    task.status = GoalStatus.COMPLETED
                else:
                    task.error = f"Unknown executor: {task.executor}"
                    task.status = GoalStatus.FAILED

            except Exception as e:
                task.error = str(e)
                task.status = GoalStatus.FAILED
                logger.error(f"Task {task.name} failed: {e}")

            task.completed_at = time.time()
            task.actual_cost = task.completed_at - task.started_at

            if on_complete:
                on_complete(task)


@dataclass
class GoalEngineConfig:
    """Configuration for goal engine"""
    planner_type: str = "simple"  # 'simple', 'llm'
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    llm_provider: str = "openai"
    max_parallel_tasks: int = 4
    enable_replanning: bool = True
    max_replanning_attempts: int = 3


class GoalEngine:
    """
    LLM-powered goal planning and execution engine.

    Features:
    - Goal decomposition into executable tasks
    - DAG-based parallel execution
    - Automatic replanning on failure
    - Cost tracking
    """

    def __init__(self, config: Optional[GoalEngineConfig] = None):
        self.config = config or GoalEngineConfig()

        # Initialize planner
        if self.config.planner_type == "llm" and self.config.llm_api_key:
            self._planner = LLMPlanner(
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
                provider=self.config.llm_provider
            )
        else:
            self._planner = SimplePlanner()

        self._executor = DAGExecutor(self.config.max_parallel_tasks)

        # Track goals
        self._goals: Dict[str, Goal] = {}
        self._callbacks: List[Callable[[Goal, GoalStatus], None]] = []

    def register_executor(self, name: str, func: Callable):
        """Register a task executor"""
        self._executor.register_executor(name, func)

    async def create_goal(
        self,
        name: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        deadline: Optional[float] = None
    ) -> Goal:
        """Create and plan a new goal"""
        goal = Goal(
            name=name,
            description=description,
            context=context or {},
            agent_id=agent_id,
            deadline=deadline,
            status=GoalStatus.PLANNING
        )

        self._goals[goal.id] = goal

        # Decompose into tasks
        tasks_data = await self._planner.decompose_goal(
            f"{name}: {description}",
            goal.context
        )

        # Create milestone with tasks
        tasks = []
        for i, td in enumerate(tasks_data):
            task = Task(
                name=td.get("name", f"Task {i+1}"),
                description=td.get("description", ""),
                executor=td.get("executor"),
                args=td.get("args", {}),
                depends_on=td.get("depends_on", []),
                estimated_cost=td.get("estimated_cost", 0)
            )
            tasks.append(task)

        milestone = Milestone(
            name="Main",
            description="Primary milestone",
            tasks=tasks
        )
        goal.milestones = [milestone]
        goal.status = GoalStatus.PENDING

        return goal

    async def execute_goal(self, goal_id: str) -> Goal:
        """Execute a goal"""
        goal = self._goals.get(goal_id)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        goal.status = GoalStatus.IN_PROGRESS
        self._notify_status(goal)

        replanning_attempts = 0

        for milestone in goal.milestones:
            milestone.status = GoalStatus.IN_PROGRESS

            # Execute tasks
            await self._executor.execute(
                milestone.tasks,
                on_task_complete=lambda t: self._on_task_complete(goal, t)
            )

            # Check for failures
            failed_tasks = [t for t in milestone.tasks if t.status == GoalStatus.FAILED]

            if failed_tasks and self.config.enable_replanning:
                while failed_tasks and replanning_attempts < self.config.max_replanning_attempts:
                    replanning_attempts += 1
                    logger.info(f"Replanning attempt {replanning_attempts}")

                    for failed in failed_tasks:
                        new_tasks = await self._planner.replan(goal, failed, failed.error or "")
                        if new_tasks:
                            milestone.tasks.extend(new_tasks)

                    # Re-execute new tasks
                    pending = [t for t in milestone.tasks if t.status == GoalStatus.PENDING]
                    await self._executor.execute(pending)

                    failed_tasks = [t for t in milestone.tasks if t.status == GoalStatus.FAILED]

            # Update milestone status
            if all(t.status == GoalStatus.COMPLETED for t in milestone.tasks):
                milestone.status = GoalStatus.COMPLETED
            elif any(t.status == GoalStatus.FAILED for t in milestone.tasks):
                milestone.status = GoalStatus.FAILED

        # Update goal status
        if all(m.status == GoalStatus.COMPLETED for m in goal.milestones):
            goal.status = GoalStatus.COMPLETED
        elif any(m.status == GoalStatus.FAILED for m in goal.milestones):
            goal.status = GoalStatus.FAILED

        self._notify_status(goal)
        return goal

    def _on_task_complete(self, goal: Goal, task: Task):
        """Called when a task completes"""
        logger.info(f"Task {task.name} completed with status {task.status.name}")

    def _notify_status(self, goal: Goal):
        """Notify callbacks of status change"""
        for callback in self._callbacks:
            try:
                callback(goal, goal.status)
            except Exception:
                pass

    def on_goal_status(self, callback: Callable[[Goal, GoalStatus], None]):
        """Register status callback"""
        self._callbacks.append(callback)

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID"""
        return self._goals.get(goal_id)

    def list_goals(self, agent_id: Optional[str] = None) -> List[Goal]:
        """List all goals"""
        goals = list(self._goals.values())
        if agent_id:
            goals = [g for g in goals if g.agent_id == agent_id]
        return goals


# Factory
_global_engine: Optional[GoalEngine] = None


def get_goal_engine() -> GoalEngine:
    global _global_engine
    if _global_engine is None:
        _global_engine = GoalEngine()
    return _global_engine


def create_goal_engine(config: Optional[GoalEngineConfig] = None) -> GoalEngine:
    return GoalEngine(config)
