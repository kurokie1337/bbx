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

"""
Workflow validation layer.
Validates workflow structure, steps, and dependencies before execution.
"""

from typing import Any, Dict, List

from .exceptions import WorkflowValidationError


class WorkflowValidator:
    """Validates BBX workflow structure."""

    RESERVED_KEYWORDS = {"workflow", "steps", "version", "id", "name"}
    REQUIRED_FIELDS = {"id", "steps"}
    STEP_REQUIRED_FIELDS = {"id", "use"}

    def validate(self, workflow: Dict[str, Any]) -> None:
        """
        Validate workflow structure.

        Args:
            workflow: Workflow data to validate

        Raises:
            WorkflowValidationError: If validation fails
        """
        errors = []

        # Check required top-level fields
        for field in self.REQUIRED_FIELDS:
            if field not in workflow:
                errors.append(f"Missing required field: '{field}'")

        if errors:
            raise WorkflowValidationError("Workflow validation failed", errors)

        # Validate workflow ID
        workflow_id = workflow.get("id")
        if not workflow_id or not isinstance(workflow_id, str):
            errors.append("Workflow 'id' must be a non-empty string")

        # Validate steps
        steps = workflow.get("steps")
        if not steps:
            errors.append("Workflow must have at least one step")
        elif isinstance(steps, dict):
            # BBX v6.0 format
            self._validate_steps_v6(steps, errors)
        elif isinstance(steps, list):
            # BBX v5.0 format
            self._validate_steps_v5(steps, errors)
        else:
            errors.append("'steps' must be a dictionary or list")

        if errors:
            raise WorkflowValidationError("Workflow validation failed", errors)

    def _validate_steps_v6(self, steps: Dict[str, Any], errors: List[str]) -> None:
        """Validate BBX v6.0 format steps."""
        if not steps:
            errors.append("Steps dictionary cannot be empty")
            return

        step_ids = set()

        for step_id, step_config in steps.items():
            # Validate step ID
            if not step_id or not isinstance(step_id, str):
                errors.append(f"Invalid step ID: {step_id}")
                continue

            # Check for duplicate step IDs
            if step_id in step_ids:
                errors.append(f"Duplicate step ID: '{step_id}'")
            step_ids.add(step_id)

            # Validate step configuration
            if not isinstance(step_config, dict):
                errors.append(f"Step '{step_id}' configuration must be a dictionary")
                continue

            # Check required fields
            if "use" not in step_config:
                errors.append(f"Step '{step_id}' missing required field: 'use'")

            # Validate adapter name
            use = step_config.get("use")
            if use and not isinstance(use, str):
                errors.append(f"Step '{step_id}': 'use' must be a string")

            # Validate depends_on
            depends_on = step_config.get("depends_on")
            if depends_on is not None:
                if not isinstance(depends_on, list):
                    errors.append(f"Step '{step_id}': 'depends_on' must be a list")
                else:
                    for dep in depends_on:
                        if dep not in steps:
                            errors.append(
                                f"Step '{step_id}' depends on unknown step: '{dep}'"
                            )

        # Check for circular dependencies
        if not errors:
            self._check_circular_dependencies(steps, errors)

    def _validate_steps_v5(self, steps: List[Dict], errors: List[str]) -> None:
        """Validate BBX v5.0 format steps."""
        if not steps:
            errors.append("Steps list cannot be empty")
            return

        step_ids = set()

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                errors.append(f"Step at index {idx} must be a dictionary")
                continue

            # Check required fields
            step_id = step.get("id")
            if not step_id:
                errors.append(f"Step at index {idx} missing required field: 'id'")
                continue

            # Check for duplicate IDs
            if step_id in step_ids:
                errors.append(f"Duplicate step ID: '{step_id}'")
            step_ids.add(step_id)

            # Check mcp and method
            if "mcp" not in step:
                errors.append(f"Step '{step_id}' missing required field: 'mcp'")
            if "method" not in step:
                errors.append(f"Step '{step_id}' missing required field: 'method'")

    def _check_circular_dependencies(
        self, steps: Dict[str, Any], errors: List[str]
    ) -> None:
        """Check for circular dependencies in workflow."""
        visited = set()
        rec_stack = set()

        def has_cycle(step_id: str, path: List[str]) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            path.append(step_id)

            step = steps.get(step_id, {})
            depends_on = step.get("depends_on", [])

            for dep in depends_on:
                if dep not in visited:
                    if has_cycle(dep, path):
                        return True
                elif dep in rec_stack:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")
                    return True

            path.pop()
            rec_stack.remove(step_id)
            return False

        for step_id in steps.keys():
            if step_id not in visited:
                has_cycle(step_id, [])


def validate_workflow(workflow: Dict[str, Any]) -> None:
    """
    Validate workflow.

    Args:
        workflow: Workflow data to validate

    Raises:
        WorkflowValidationError: If validation fails
    """
    validator = WorkflowValidator()
    validator.validate(workflow)
