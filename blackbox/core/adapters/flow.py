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
import os
from typing import Any, Dict

from blackbox.core.base_adapter import MCPAdapter

logger = logging.getLogger("bbx.flow")


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
            results = await run_file(path, inputs=variables)

            # Check for failures
            failed_steps = []
            for step_id, result in results.items():
                if isinstance(result, dict) and result.get("status") == "failed":
                    failed_steps.append(step_id)

            if failed_steps:
                error_msg = f"Subflow failed in steps: {', '.join(failed_steps)}"
                raise RuntimeError(error_msg)

            return results

        else:
            raise ValueError(f"Unknown flow method: {method}")
