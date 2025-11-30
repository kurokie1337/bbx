
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from blackbox.core.adapters.v2_base import V2BaseAdapter, v2_method
from blackbox.core.v2.runtime_v2 import BBXRuntimeV2

logger = logging.getLogger("bbx.adapters.claude_hooks")

class ClaudeHooksAdapter(V2BaseAdapter):
    """
    Adapter for integrating Claude Code hooks into BBX.
    
    This adapter acts as a bridge, allowing BBX workflows to handle Claude Code events
    like PreToolUse, PostToolUse, UserPromptSubmit, etc.
    """

    def __init__(self):
        super().__init__()
        self.runtime = BBXRuntimeV2()

    @v2_method(name="handle_event", description="Handle a Claude Code hook event")
    async def handle_event(self, event_data: Dict[str, Any], workflow_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a hook event from Claude Code.

        Args:
            event_data: The JSON payload from Claude Code.
            workflow_path: Optional path to a specific workflow to run. 
                           If not provided, tries to find a matching workflow based on event name.

        Returns:
            A dictionary representing the response to Claude Code (e.g. decision, reason).
        """
        event_name = event_data.get("hook_event_name")
        if not event_name:
            return {"error": "Missing hook_event_name in payload"}

        logger.info(f"Handling Claude hook event: {event_name}")

        # Determine workflow to run
        if not workflow_path:
            # Default convention: examples/claude_hooks/workflows/{event_name}.yaml
            # This is a bit rigid, maybe we can make it configurable later.
            # For now, we'll assume the CLI passes the workflow path or we look in a standard location.
            # Let's try to find it in the current directory or a standard hooks dir.
            
            # Check for local hooks directory first
            local_hook = Path(f"hooks/{event_name}.yaml")
            if local_hook.exists():
                workflow_path = str(local_hook)
            else:
                # Fallback to examples (for testing/default behavior)
                # Assuming we are running from project root
                example_hook = Path(f"examples/claude_hooks/workflows/{event_name}.yaml")
                if example_hook.exists():
                    workflow_path = str(example_hook)
        
        if not workflow_path or not Path(workflow_path).exists():
            logger.warning(f"No workflow found for event {event_name}")
            # If no workflow is found, we should probably allow the action by default
            # or maybe fail if strict mode is on. For now, allow.
            return {"decision": "allow", "reason": f"No BBX workflow found for {event_name}"}

        try:
            # Run the workflow
            # We pass the event data as inputs to the workflow
            inputs = {"event": event_data}
            
            # We need to run this in the event loop
            results = await self.runtime.execute_file(workflow_path, inputs=inputs)
            
            # Extract the decision from the workflow results
            # We expect the workflow to produce a "result" or "decision" output
            # If the workflow has a step named "output" or "decision", we use that.
            # Or we look for a variable "hook_response" in the results.
            
            # Extract the decision from the workflow results
            # The hook_response step outputs JSON to stdout
            hook_response = None
            
            # Check for hook_response step
            if "hook_response" in results:
                step_result = results["hook_response"]
                if isinstance(step_result, dict) and "output" in step_result:
                    output = step_result["output"]
                    # For python adapter, output structure is: {success: true, metadata: {stdout: ..., stderr: ...}}
                    stdout_val = None
                    if isinstance(output, dict):
                        if "metadata" in output and isinstance(output["metadata"], dict):
                            stdout_val = output["metadata"].get("stdout", "").strip()
                        elif "data" in output and isinstance(output["data"], dict):
                            stdout_val = output["data"].get("stdout", "").strip()
                        elif "stdout" in output:
                            stdout_val = output["stdout"].strip()
                    
                    if stdout_val:
                        try:
                            # Parse JSON from stdout
                            hook_response = json.loads(stdout_val)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse stdout as JSON: {stdout_val}")
                            hook_response = {"decision": "allow", "reason": "Failed to parse workflow response"}
            
            # If still empty, assume success/allow
            if not hook_response:
                 hook_response = {"decision": "allow", "reason": "Workflow completed successfully"}

            return hook_response

        except Exception as e:
            logger.error(f"Error executing hook workflow: {e}")
            return {
                "decision": "block", # Fail safe? Or allow?
                # Claude docs say exit code 2 for blocking error.
                # Here we return a dict, the CLI will handle the exit code.
                "error": str(e),
                "reason": f"BBX Workflow Error: {str(e)}"
            }

