# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
A2A Adapter for BBX Workflows

Enables workflows to communicate with other A2A agents.

Usage in workflow:
    steps:
      # Discover agent capabilities
      discover:
        use: a2a.discover
        args:
          agent: https://other-agent.example.com

      # Call a skill on another agent
      analyze:
        use: a2a.call
        args:
          agent: https://analyst-agent.example.com
          skill: analyze_data
          input:
            data: ${steps.fetch_data.outputs.data}
          wait: true  # Wait for result (default)

      # Create task without waiting
      start_job:
        use: a2a.call
        args:
          agent: https://worker-agent.example.com
          skill: long_running_job
          wait: false

      # Check task status
      check:
        use: a2a.status
        args:
          agent: https://worker-agent.example.com
          task_id: ${steps.start_job.outputs.id}

      # Cancel a task
      cancel:
        use: a2a.cancel
        args:
          agent: https://worker-agent.example.com
          task_id: ${steps.start_job.outputs.id}

      # Register agent in local registry
      register:
        use: a2a.register
        args:
          url: https://new-agent.example.com
          name: New Agent
          tags: [data, analysis]

      # List known agents
      list:
        use: a2a.list
"""

import logging
from typing import Any, Dict, List, Optional

from blackbox.core.base_adapter import MCPAdapter

logger = logging.getLogger(__name__)


class A2AAdapter(MCPAdapter):
    """
    Adapter for A2A (Agent-to-Agent) protocol communication.

    Enables BBX workflows to discover and communicate with other A2A agents.
    """

    def __init__(self):
        super().__init__("a2a")
        self._client = None

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute A2A adapter method.

        Routes to appropriate method based on method name.

        Args:
            method: Method name (discover, call, status, cancel, wait, ping, register, list, rpc)
            inputs: Input parameters for the method

        Returns:
            Method result
        """
        self.log_execution(method, inputs)

        method_map = {
            "discover": self.discover,
            "call": self.call,
            "status": self.status,
            "cancel": self.cancel,
            "wait": self.wait,
            "ping": self.ping,
            "register": self.register,
            "list": self.list,
            "rpc": self.rpc,
        }

        handler = method_map.get(method)
        if not handler:
            raise ValueError(f"Unknown A2A method: {method}. Available: {list(method_map.keys())}")

        try:
            result = await handler(**inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            raise

    def _get_client(self):
        """Create new A2A client for each request (avoids shared state issues)."""
        from blackbox.a2a.client import A2AClient
        return A2AClient()

    async def discover(
        self,
        agent: str,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Discover an agent's capabilities.

        Args:
            agent: Agent URL
            force_refresh: Force refresh even if cached

        Returns:
            Agent Card data
        """
        client = self._get_client()

        try:
            async with client:
                card = await client.discover(agent, force_refresh=force_refresh)

            return {
                "status": "success",
                "agent": agent,
                "name": card.name,
                "description": card.description,
                "skills": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "description": s.description,
                    }
                    for s in card.skills
                ],
                "tags": card.tags,
                "endpoints": card.endpoints.model_dump(by_alias=True),
            }

        except Exception as e:
            logger.error(f"Failed to discover agent {agent}: {e}")
            return {
                "status": "error",
                "agent": agent,
                "error": str(e),
            }

    async def call(
        self,
        agent: str,
        skill: str,
        input: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call a skill on a remote A2A agent.

        Args:
            agent: Agent URL
            skill: Skill ID to invoke
            input: Input parameters for the skill
            wait: Whether to wait for task completion
            timeout: Wait timeout in seconds
            context: Additional context for the skill
            callback_url: URL for async notifications

        Returns:
            Task result (if wait=True) or task info (if wait=False)
        """
        client = self._get_client()

        try:
            async with client:
                task = await client.create_task(
                    agent_url=agent,
                    skill_id=skill,
                    input=input or {},
                    context=context,
                    callback_url=callback_url,
                )

                if wait:
                    result = await client.wait_for_task(
                        agent_url=agent,
                        task_id=task["id"],
                        timeout=timeout,
                    )
                    return {
                        "status": "completed",
                        "agent": agent,
                        "skill": skill,
                        "task_id": result["id"],
                        "output": result.get("output"),
                        "artifacts": result.get("artifacts", []),
                    }
                else:
                    return {
                        "status": "started",
                        "agent": agent,
                        "skill": skill,
                        "task_id": task["id"],
                        "task_status": task.get("status"),
                    }

        except Exception as e:
            logger.error(f"Failed to call {skill} on {agent}: {e}")
            return {
                "status": "error",
                "agent": agent,
                "skill": skill,
                "error": str(e),
            }

    async def status(
        self,
        agent: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Get task status from a remote agent.

        Args:
            agent: Agent URL
            task_id: Task ID

        Returns:
            Task status and info
        """
        client = self._get_client()

        try:
            async with client:
                task = await client.get_task(agent, task_id)

            return {
                "status": "success",
                "agent": agent,
                "task_id": task_id,
                "task_status": task.get("status"),
                "progress": task.get("progress"),
                "output": task.get("output"),
                "error": task.get("error"),
                "started_at": task.get("startedAt"),
                "completed_at": task.get("completedAt"),
            }

        except Exception as e:
            return {
                "status": "error",
                "agent": agent,
                "task_id": task_id,
                "error": str(e),
            }

    async def cancel(
        self,
        agent: str,
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Cancel a task on a remote agent.

        Args:
            agent: Agent URL
            task_id: Task ID to cancel

        Returns:
            Cancellation result
        """
        client = self._get_client()

        try:
            async with client:
                task = await client.cancel_task(agent, task_id)

            return {
                "status": "cancelled",
                "agent": agent,
                "task_id": task_id,
                "task_status": task.get("status"),
            }

        except Exception as e:
            return {
                "status": "error",
                "agent": agent,
                "task_id": task_id,
                "error": str(e),
            }

    async def wait(
        self,
        agent: str,
        task_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete.

        Args:
            agent: Agent URL
            task_id: Task ID
            timeout: Wait timeout in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Completed task data
        """
        client = self._get_client()

        try:
            async with client:
                task = await client.wait_for_task(
                    agent_url=agent,
                    task_id=task_id,
                    timeout=timeout,
                    poll_interval=poll_interval,
                )

            return {
                "status": task.get("status"),
                "agent": agent,
                "task_id": task_id,
                "output": task.get("output"),
                "artifacts": task.get("artifacts", []),
                "error": task.get("error"),
            }

        except Exception as e:
            return {
                "status": "error",
                "agent": agent,
                "task_id": task_id,
                "error": str(e),
            }

    async def ping(
        self,
        agent: str,
    ) -> Dict[str, Any]:
        """
        Check if an agent is healthy.

        Args:
            agent: Agent URL

        Returns:
            Health check result
        """
        client = self._get_client()

        try:
            async with client:
                healthy = await client.ping(agent)

            return {
                "agent": agent,
                "healthy": healthy,
                "status": "healthy" if healthy else "unhealthy",
            }

        except Exception as e:
            return {
                "agent": agent,
                "healthy": False,
                "status": "error",
                "error": str(e),
            }

    async def register(
        self,
        url: str,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Register an agent in the local registry.

        Args:
            url: Agent URL
            name: Agent name
            description: Agent description
            tags: Agent tags

        Returns:
            Registration result
        """
        client = self._get_client()
        client.register_agent(url, name, description, tags or [])

        return {
            "status": "registered",
            "url": url,
            "name": name,
        }

    async def list(self) -> Dict[str, Any]:
        """
        List all registered agents.

        Returns:
            List of registered agents
        """
        client = self._get_client()
        agents = client.list_agents()

        return {
            "status": "success",
            "agents": agents,
            "count": len(agents),
        }

    async def rpc(
        self,
        agent: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a JSON-RPC call to an agent.

        Args:
            agent: Agent URL
            method: RPC method name
            params: Method parameters

        Returns:
            RPC result
        """
        client = self._get_client()

        try:
            async with client:
                result = await client.rpc_call(agent, method, params)

            return {
                "status": "success",
                "agent": agent,
                "method": method,
                "result": result,
            }

        except Exception as e:
            return {
                "status": "error",
                "agent": agent,
                "method": method,
                "error": str(e),
            }
