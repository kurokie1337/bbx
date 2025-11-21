
import logging

logger = logging.getLogger("bbx.flow")

from typing import Dict, Any
from blackbox.core.base_adapter import MCPAdapter
import os

class FlowAdapter(MCPAdapter):
    """
    Adapter for executing other Blackbox workflows (Subflows).
    Allows composing complex scenarios from smaller building blocks.
    """

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        if method == "run":
            path = inputs.get("path")
            variables = inputs.get("inputs", {})

            if not path:
                raise ValueError("Flow adapter requires 'path' input")

            # Import here to avoid circular dependency
            from blackbox.core.runtime import run_file

            # Resolve path relative to CWD if not absolute
            if not os.path.isabs(path):
                path = os.path.abspath(path)

            if not os.path.exists(path):
                raise FileNotFoundError(f"Workflow file not found: {path}")

            logger.info(f"  ↳ Entering subflow: {os.path.basename(path)}")

            # Run the subflow
            # We don't pass the parent event bus for now to keep logs cleaner,
            # or we could pass it if we want unified logging.
            # Let's pass None for now to keep it simple, or maybe we want to see sub-steps?
            # If we pass None, we won't see sub-steps in the main log unless we handle it.
            # But run_file prints to stdout anyway.

            # Check recursion depth
            depth = inputs.get("_depth", 0)
            if depth > 10:
                raise RuntimeError("Maximum recursion depth exceeded (10)")

            # Pass incremented depth to subflow
            variables["_depth"] = depth + 1

            results = await run_file(path, initial_variables=variables)

            logger.info(f"  ↵ Exiting subflow: {os.path.basename(path)}")

            # Check for errors in subflow
            failed_steps = []
            for step_id, result in results.items():
                if result.get("status") == "error":
                    failed_steps.append(f"{step_id}: {result.get('error')}")

            if failed_steps:
                error_msg = f"Subflow failed in steps: {', '.join(failed_steps)}"
                raise RuntimeError(error_msg)

            return results

        else:
            raise ValueError(f"Unknown flow method: {method}")
