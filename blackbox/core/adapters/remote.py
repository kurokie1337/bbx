"""Remote execution adapter."""

from typing import Any, Dict, Optional

from pydantic import BaseModel

from blackbox.core.base_adapter import BaseAdapter
from blackbox.remote.client import RemoteExecutor


class RemoteExecuteInputs(BaseModel):
    """Inputs for remote workflow execution."""

    remote_url: str
    workflow_path: str
    api_key: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    wait: bool = True


class RemoteAdapter(BaseAdapter):
    """Adapter for executing workflows on remote BBX instances."""

    def __init__(self):
        super().__init__("remote")

    def execute(self, inputs: RemoteExecuteInputs) -> Dict[str, Any]:
        """Execute workflow on remote instance."""
        client = RemoteExecutor(inputs.remote_url, inputs.api_key)
        result = client.execute_workflow(
            inputs.workflow_path, inputs.inputs, inputs.wait
        )
        return result

    def get_status(
        self, remote_url: str, execution_id: str, api_key: Optional[str] = None
    ):
        """Get execution status."""
        client = RemoteExecutor(remote_url, api_key)
        return client.get_status(execution_id)

    def get_output(
        self, remote_url: str, execution_id: str, api_key: Optional[str] = None
    ):
        """Get execution output."""
        client = RemoteExecutor(remote_url, api_key)
        return client.get_output(execution_id)
