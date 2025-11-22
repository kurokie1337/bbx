"""Celery tasks for workflow execution."""

from blackbox.core.runtime import execute_workflow
from blackbox.scheduler.celery_app import app


@app.task
def execute_scheduled_workflow(workflow_path, inputs=None):
    """Execute workflow on schedule."""
    result = execute_workflow(workflow_path, inputs or {})
    return result
