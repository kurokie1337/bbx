"""Workflow event sourcing integration."""

from blackbox.event_sourcing.store import Event, EventStore

store = EventStore("postgresql://localhost/bbx_events")


def emit_workflow_started(workflow_id, inputs):
    store.append(Event(workflow_id, "WorkflowStarted", {"inputs": inputs}))


def emit_step_completed(workflow_id, step_id, output):
    store.append(
        Event(workflow_id, "StepCompleted", {"step_id": step_id, "output": output})
    )


def emit_workflow_completed(workflow_id, outputs):
    store.append(Event(workflow_id, "WorkflowCompleted", {"outputs": outputs}))


def replay_workflow(workflow_id):
    events = store.get_events(workflow_id)
    state = {}
    for event in events:
        if event.event_type == "StepCompleted":
            state[event.data["step_id"]] = event.data["output"]
    return state
