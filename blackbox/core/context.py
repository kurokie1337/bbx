# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

import os
import re
from typing import Any, Dict, Optional


class WorkflowContext:
    def __init__(self, inputs: Optional[Dict[str, Any]] = None):
        self.variables: Dict[str, Any] = {"env": {}, "step": {}, "steps": {}, "inputs": inputs or {}}

        # Workflow execution metadata
        self.meta: Dict[str, Any] = {
            "retry_counts": {},       # step_id -> retry count
            "fallback_used": {},      # step_id -> bool (was fallback used)
            "pre_checks_run": {},     # step_id -> bool (was pre-check run)
            "failures": [],           # list of all failures {step_id, error, timestamp}
            "auto_fix_applied": [],   # list of applied auto-fixes
        }

        # Add meta to variables for ${meta.retry_counts} access
        self.variables["meta"] = self.meta

    def set_step_output(self, step_id: str, output: Any):
        # Store in both 'step' and 'steps' for compatibility
        # Wrap output in 'outputs' to support ${steps.id.outputs.field} syntax
        wrapped = {"outputs": output, **output} if isinstance(output, dict) else {"outputs": output}
        self.variables["step"][step_id] = wrapped
        self.variables["steps"][step_id] = wrapped

    def resolve(self, expression: str) -> Any:
        """
        Simple variable resolution.
        Supports: ${step.id.field} or ${inputs.field}
        """
        if not isinstance(expression, str):
            return expression

        # Regex to find ${...} patterns
        pattern = r"\$\{(.+?)\}"
        matches = re.findall(pattern, expression)

        if not matches:
            return expression

        # For now, we only support simple single variable replacement or return the value directly
        # if the whole string is a variable.

        for match in matches:
            parts = match.split(".")
            first_part = parts[0]

            # Secrets Handling
            if first_part == "secrets":
                if len(parts) > 1:
                    secret_key = parts[1]
                    # Try to get from env, default to empty string or raise error?
                    # For safety, let's return a masked string if not found or the actual value
                    secret_value = os.environ.get(
                        secret_key, f"MISSING_SECRET_{secret_key}"
                    )

                    # Replace and continue (secrets are usually strings)
                    expression = expression.replace(f"${{{match}}}", secret_value)
                    continue
                else:
                    # ${secrets} usage is invalid
                    continue

            # Determine the starting point
            value: Any
            if first_part in self.variables:
                # First check if it's a top-level key like "inputs", "step", "secrets"
                value = self.variables
            elif first_part in self.variables.get("inputs", {}):
                # Check if it's a key in inputs
                value = self.variables.get("inputs", {})
            elif first_part in self.variables.get("step", {}):
                value = self.variables["step"]
            else:
                # If not found, leave it as is or return empty?
                # For now, let's try to resolve from variables, if fails, it will return {}
                value = self.variables

            try:
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, None)
                        if value is None:
                            break
                    else:
                        # Handle object attributes if needed, or just fail
                        value = None
                        break

                # If the expression is JUST the variable, return the raw value
                if expression == f"${{{match}}}":
                    return value if value is not None else ""

                # Otherwise replace in string
                # Convert empty dict {} or None to empty string
                if value is None or (isinstance(value, dict) and not value):
                    expression = expression.replace(f"${{{match}}}", "")
                else:
                    expression = expression.replace(f"${{{match}}}", str(value))
            except Exception:
                pass

        return expression

    def resolve_recursive(self, data: Any) -> Any:
        """
        Recursively resolve variables in a data structure (dict, list, or string).
        """
        if isinstance(data, dict):
            return {k: self.resolve_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.resolve_recursive(item) for item in data]
        elif isinstance(data, str):
            return self.resolve(data)
        else:
            return data
