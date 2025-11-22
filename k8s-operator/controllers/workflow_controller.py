"""Kubernetes Workflow Controller."""
import kopf
import yaml
from blackbox.core.runtime import execute_workflow

@kopf.on.create('bbx.io', 'v1alpha1', 'workflows')
def create_workflow(spec, name, namespace, **kwargs):
    """Handle Workflow creation."""
    workflow_spec = spec.get('workflowSpec', {})
    inputs = spec.get('inputs', {})

    # Execute BBX workflow
    result = execute_workflow(workflow_spec, inputs)

    # Update status
    return {
        'phase': 'Completed' if result['status'] == 'success' else 'Failed',
        'outputs': result.get('outputs', {})
    }

@kopf.on.update('bbx.io', 'v1alpha1', 'workflows')
def update_workflow(spec, name, namespace, **kwargs):
    """Handle Workflow updates."""
    return create_workflow(spec, name, namespace, **kwargs)

@kopf.on.delete('bbx.io', 'v1alpha1', 'workflows')
def delete_workflow(name, namespace, **kwargs):
    """Handle Workflow deletion."""
    print(f"Deleting workflow {name} in {namespace}")
