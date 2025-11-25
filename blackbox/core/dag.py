# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
DAG (Directed Acyclic Graph) execution engine for Blackbox workflows.
Enables parallel execution of independent steps.
"""

from collections import deque
from typing import Any, Dict, List, Set


class DAGError(Exception):
    """Raised when DAG construction or execution fails"""


class WorkflowDAG:
    """
    Directed Acyclic Graph for workflow execution.

    Analyzes step dependencies and executes steps in parallel when possible.
    """

    def __init__(self, steps: List[Dict[str, Any]]):
        """
        Build DAG from workflow steps.

        Args:
            steps: List of workflow steps
        """
        self.steps_map = {step["id"]: step for step in steps}
        self.graph = self._build_graph(steps)
        self._validate_acyclic()

    def _build_graph(self, steps: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """Build adjacency list representation of DAG"""
        graph = {}

        for step in steps:
            step_id = step["id"]
            depends_on = step.get("depends_on", [])
            parallel = step.get("parallel", False)

            graph[step_id] = {
                "step": step,
                "dependencies": (
                    set(depends_on)
                    if isinstance(depends_on, list)
                    else {depends_on} if depends_on else set()
                ),
                "parallel": parallel,
                "dependents": set(),
            }

        # Build reverse dependencies (who depends on me)
        for step_id, node in graph.items():
            for dep in node["dependencies"]:
                if dep not in graph:
                    raise DAGError(f"Step '{step_id}' depends on unknown step '{dep}'")
                graph[dep]["dependents"].add(step_id)

        return graph

    def _validate_acyclic(self):
        """Ensure no cycles in the graph using DFS"""
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for dep in self.graph[node_id]["dependencies"]:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.graph:
            if node_id not in visited:
                if has_cycle(node_id):
                    raise DAGError("Cycle detected in workflow dependencies")

    def get_execution_levels(self) -> List[List[str]]:
        """
        Get execution levels for parallel execution.

        Returns:
            List of levels, where each level contains step IDs that can run in parallel
        """
        levels = []
        in_degree = {
            node_id: len(node["dependencies"]) for node_id, node in self.graph.items()
        }
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])

        while queue:
            # Current level: all nodes with in-degree 0
            current_level = []
            level_size = len(queue)

            for _ in range(level_size):
                node_id = queue.popleft()
                current_level.append(node_id)

                # Decrease in-degree for dependents
                for dependent in self.graph[node_id]["dependents"]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            levels.append(current_level)

        # Verify all nodes were processed
        if sum(len(level) for level in levels) != len(self.graph):
            raise DAGError("Failed to create execution levels - possible cycle")

        return levels

    def can_parallelize(self, step_id: str) -> bool:
        """Check if step is marked for parallel execution"""
        return self.graph[step_id]["parallel"]

    def get_step(self, step_id: str) -> Dict[str, Any]:
        """Get step definition by ID"""
        return self.graph[step_id]["step"]

    def get_dependencies(self, step_id: str) -> Set[str]:
        """Get direct dependencies of a step"""
        return self.graph[step_id]["dependencies"]


def should_use_dag(steps: List[Dict[str, Any]]) -> bool:
    """
    Determine if DAG execution should be used.

    Returns True if any step has dependencies or parallel flag.
    """
    for step in steps:
        if step.get("depends_on") or step.get("parallel"):
            return True
    return False


def create_dag(steps: List[Dict[str, Any]]) -> WorkflowDAG:
    """
    Create DAG from workflow steps.

    Args:
        steps: List of workflow steps

    Returns:
        WorkflowDAG instance

    Raises:
        DAGError: If DAG construction fails
    """
    return WorkflowDAG(steps)
