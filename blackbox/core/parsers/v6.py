# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Format v6.0 Parser

Simplified BBX syntax with:
- Step IDs as YAML keys (not in 'id' field)
- Combined 'use' field (adapter.method)
- Shorter field names ('args' instead of 'inputs', 'save' instead of 'outputs')
- Human-readable durations ('5s', '1m', '10ms')
"""

import re
from typing import Any, Dict

import yaml


class BBXv6Parser:
    """Parser for BBX Format v6.0"""

    @staticmethod
    def parse_duration(duration_str: str) -> int:
        """
        Parse human-readable duration to milliseconds.

        Examples:
            '5s' -> 5000
            '1m' -> 60000
            '500ms' -> 500
            '30' -> 30000 (defaults to ms if no unit)

        Args:
            duration_str: Duration string

        Returns:
            Duration in milliseconds
        """
        if isinstance(duration_str, (int, float)):
            return int(duration_str)

        duration_str = str(duration_str).strip()

        # Parse with regex
        match = re.match(r"^(\d+(?:\.\d+)?)(ms|s|m|h)?$", duration_str)
        if not match:
            raise ValueError(f"Invalid duration format: {duration_str}")

        value = float(match.group(1))
        unit = match.group(2) or "ms"  # Default to milliseconds

        # Convert to milliseconds
        multipliers = {"ms": 1, "s": 1000, "m": 60000, "h": 3600000}

        return int(value * multipliers[unit])

    @staticmethod
    def parse_step(step_id: str, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert v6.0 step to v5.0 format.

        Args:
            step_id: Step identifier
            step_data: v6.0 step data

        Returns:
            v5.0 step dictionary
        """
        v5_step: Dict[str, Any] = {"id": step_id}

        # Parse 'use' field (adapter.method or path/to/file.bbx)
        if "use" in step_data:
            use_value = step_data["use"]

            # Check if it's a .bbx file (must end with .bbx)
            if use_value.endswith(".bbx"):
                # It's a file path - treat as a sub-workflow
                v5_step["mcp"] = "workflow"
                v5_step["method"] = "run_file"
                v5_step["inputs"] = v5_step.get("inputs", {})
                v5_step["inputs"]["file_path"] = use_value
            elif "." in use_value:
                # Standard adapter.method format
                parts = use_value.split(".", 1)
                v5_step["mcp"] = parts[0]
                v5_step["method"] = parts[1]
            else:
                # No dot and not a .bbx file - might be a single adapter name
                # In this case, we need to check if 'method' is provided in inputs
                # For now, treat it as invalid
                raise ValueError(
                    f"Invalid 'use' format: {use_value}. Expected 'adapter.method' or path to .bbx file"
                )
        else:
            # Fallback to explicit mcp/method
            mcp_value = step_data.get("mcp")
            method_value = step_data.get("method")
            if mcp_value:
                v5_step["mcp"] = mcp_value
            if method_value:
                v5_step["method"] = method_value

        # Convert field names with special handling for args
        if "args" in step_data:
            args_value = step_data["args"]
            # If args is a simple value (string, number, etc.), wrap it in a dict
            if isinstance(args_value, (str, int, float, bool)):
                v5_step["inputs"] = {"value": args_value}
            elif isinstance(args_value, list):
                # If it's a list, keep as is (for commands like os.spawn)
                v5_step["inputs"] = {"args": args_value}
            else:
                # If it's already a dict, use directly
                v5_step["inputs"] = args_value

        # Convert other fields
        field_mapping = {
            "inputs": "inputs",  # Allow explicit inputs too
            "save": "outputs",
            "when": "when",
            "parallel": "parallel",
            "depends_on": "depends_on",
        }

        for v6_name, v5_name in field_mapping.items():
            if v6_name in step_data and v6_name != "args":  # Skip args, already handled
                # Merge with existing inputs if needed
                if v6_name == "inputs" and v5_name in v5_step:
                    v5_step[v5_name].update(step_data[v6_name])
                else:
                    v5_step[v5_name] = step_data[v6_name]

        # Parse durations
        if "timeout" in step_data:
            v5_step["timeout"] = BBXv6Parser.parse_duration(step_data["timeout"])

        if "retry_delay" in step_data:
            v5_step["retry_delay"] = BBXv6Parser.parse_duration(
                step_data["retry_delay"]
            )

        # Copy numeric fields
        if "retry" in step_data:
            v5_step["retry"] = step_data["retry"]

        if "retry_backoff" in step_data:
            v5_step["retry_backoff"] = step_data["retry_backoff"]

        # Cache configuration
        if "cache" in step_data:
            cache_config = step_data["cache"]
            if isinstance(cache_config, dict):
                # Convert TTL if needed
                if "ttl" in cache_config:
                    cache_config["ttl"] = BBXv6Parser.parse_duration(
                        cache_config["ttl"]
                    )
            v5_step["cache"] = cache_config

        return v5_step

    @staticmethod
    def parse_v6(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert v6.0 BBX format to v5.0.

        Args:
            data: v6.0 workflow/app data

        Returns:
            v5.0 format data
        """
        workflow: Dict[str, Any] = {}

        v5_data: Dict[str, Any] = {
            "bbx_version": "5.0",
            "type": "workflow",
            "workflow": workflow,
        }

        # Support both direct format and wrapped format
        source_data = data
        if "workflow" in data:
            # Wrapped format: {workflow: {name: ..., steps: [...]}}
            source_data = data["workflow"]
        elif "app" in data:
            # App format: {app: {name: ..., install: {steps: [...]}}}
            app_data = data["app"]
            # Convert app.install.steps to workflow.steps
            if "install" in app_data:
                source_data = {
                    "name": app_data.get("name", "App Installation"),
                    "id": app_data.get("id", "app_install"),
                    "steps": app_data["install"].get("steps", [])
                }
            else:
                source_data = app_data

        # Copy workflow metadata
        if "id" in source_data:
            workflow["id"] = source_data["id"]
        if "name" in source_data:
            workflow["name"] = source_data["name"]
        if "version" in source_data:
            workflow["version"] = source_data["version"]
        if "description" in source_data:
            workflow["description"] = source_data["description"]

        # Copy inputs section (for default values)
        if "inputs" in source_data:
            workflow["inputs"] = source_data["inputs"]
        elif "inputs" in data:
            # Also check top-level for v6 simple format
            workflow["inputs"] = data["inputs"]

        # Parse steps
        if "steps" in source_data:
            steps_data = source_data["steps"]

            if isinstance(steps_data, dict):
                # v6 format: steps are a dictionary
                v5_steps = []
                for step_id, step_data in steps_data.items():
                    v5_step = BBXv6Parser.parse_step(step_id, step_data)
                    v5_steps.append(v5_step)
                workflow["steps"] = v5_steps
            elif isinstance(steps_data, list):
                # v6 format: steps are a list (more common)
                v5_steps = []
                for step in steps_data:
                    if isinstance(step, dict) and "id" in step:
                        step_id = step["id"]
                        v5_step = BBXv6Parser.parse_step(step_id, step)
                        v5_steps.append(v5_step)
                    else:
                        # Already in v5 format?
                        v5_steps.append(step)
                workflow["steps"] = v5_steps
            else:
                workflow["steps"] = []

        return v5_data

    @staticmethod
    def detect_version(data: Dict[str, Any]) -> str:
        """
        Detect BBX format version.

        Returns:
            Version string ('5.0' or '6.0')
        """
        # Check explicit version
        if "bbx_version" in data:
            version = data["bbx_version"]
            if version.startswith("6"):
                return "6.0"
            return "5.0"

        # Heuristic: v6.0 has top-level 'id' and dict-based 'steps'
        if "id" in data and "steps" in data and isinstance(data["steps"], dict):
            return "6.0"

        # Check if workflow/app steps use v6 'use' field (most reliable indicator)
        steps = []
        if "workflow" in data and isinstance(data.get("workflow"), dict):
            steps = data["workflow"].get("steps", [])
        elif "app" in data and isinstance(data.get("app"), dict):
            app_data = data["app"]
            if "install" in app_data and isinstance(app_data["install"], dict):
                steps = app_data["install"].get("steps", [])
        elif "steps" in data:
            steps = data.get("steps", [])

        # If any step has 'use' field instead of 'mcp', it's v6
        if isinstance(steps, list) and len(steps) > 0:
            for step in steps:
                if isinstance(step, dict) and "use" in step:
                    return "6.0"
                # If we see 'mcp' field, it's definitely v5
                if isinstance(step, dict) and "mcp" in step:
                    return "5.0"

        # Check for 'app' format (v6 specific)
        if "app" in data:
            return "6.0"

        # Heuristic: v5.0 has 'workflow' container (but only if no 'use' detected)
        if "workflow" in data:
            return "5.0"

        # Default to v5.0
        return "5.0"

    @staticmethod
    def load_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load and parse BBX file (auto-detects version).

        Args:
            file_path: Path to BBX file

        Returns:
            v5.0 format data (converted if needed)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        version = BBXv6Parser.detect_version(data)

        if version == "6.0":
            return BBXv6Parser.parse_v6(data)
        else:
            return data
