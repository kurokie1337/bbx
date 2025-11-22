"""Integration of WebSocket events into workflow runtime."""

from blackbox.server.websocket.emitter import emitter


# Add to runtime.py execute_workflow method:
async def execute_workflow_with_websocket(workflow, inputs=None):
    """Execute workflow with WebSocket event emission."""
    workflow_id = workflow["workflow"]["id"]
    workflow_name = workflow["workflow"].get("name", workflow_id)

    # Emit start event
    await emitter.emit_workflow_start(workflow_id, workflow_name)

    try:
        # Execute workflow steps
        for step in workflow["workflow"]["steps"]:
            step_id = step["id"]

            # Emit step start
            await emitter.emit_step_start(
                workflow_id, step_id, step.get("name", step_id)
            )

            # Execute step (existing logic)
            # ... step execution code ...
            step_output = {}  # Replace with actual step execution result

            # Emit step complete
            await emitter.emit_step_complete(
                workflow_id, step_id, status="success", output=step_output
            )

        # Emit workflow complete
        workflow_outputs = {}  # Replace with actual workflow outputs
        await emitter.emit_workflow_complete(
            workflow_id, status="success", outputs=workflow_outputs
        )

    except Exception as e:
        # Emit error
        await emitter.emit_workflow_complete(
            workflow_id, status="error", outputs={"error": str(e)}
        )
        raise
