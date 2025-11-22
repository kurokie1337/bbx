"""Remote workflow execution client."""

from typing import Any, Dict, Optional

import requests


class RemoteExecutor:
    """Client for executing workflows on remote BBX instances."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def execute_workflow(
        self,
        workflow_path: str,
        inputs: Optional[Dict[str, Any]] = None,
        wait: bool = True,
    ) -> Dict[str, Any]:
        """Execute workflow on remote instance."""
        with open(workflow_path, "rb") as f:
            workflow_content = f.read()

        response = requests.post(
            f"{self.base_url}/api/workflows/execute",
            headers=self.headers,
            files={"workflow": workflow_content},
            json={"inputs": inputs or {}, "wait": wait},
            timeout=300 if wait else 30,
        )
        response.raise_for_status()
        return response.json()

    def get_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution status."""
        response = requests.get(
            f"{self.base_url}/api/executions/{execution_id}", headers=self.headers
        )
        response.raise_for_status()
        return response.json()

    def get_output(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution output."""
        response = requests.get(
            f"{self.base_url}/api/executions/{execution_id}/output",
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def list_executions(self) -> list:
        """List all workflow executions."""
        response = requests.get(f"{self.base_url}/api/executions", headers=self.headers)
        response.raise_for_status()
        return response.json()["executions"]
