# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
A2A Client

HTTP client for communicating with other A2A agents.
Supports task creation, status polling, SSE streaming, and discovery.

Usage:
    client = A2AClient()

    # Discover agent capabilities
    card = await client.discover("https://other-agent.example.com")

    # Create a task
    task = await client.create_task(
        agent_url="https://other-agent.example.com",
        skill_id="analyze_data",
        input={"data": [...]}
    )

    # Wait for completion
    result = await client.wait_for_task(agent_url, task["id"])

    # Or stream updates
    async for update in client.stream_task(agent_url, task["id"]):
        print(update)
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from datetime import datetime, timedelta

import httpx

from .models import (
    AgentCard,
    A2ATask,
    A2ATaskStatus,
    AgentDiscoveryEntry,
    AgentRegistry,
)

logger = logging.getLogger(__name__)


class A2AClientError(Exception):
    """Base exception for A2A client errors."""
    pass


class AgentNotFoundError(A2AClientError):
    """Agent not found or unreachable."""
    pass


class TaskError(A2AClientError):
    """Task execution error."""
    pass


class A2AClient:
    """
    Client for communicating with A2A agents.

    Handles:
    - Agent discovery via Agent Cards
    - Task creation and management
    - SSE streaming for real-time updates
    - Local agent registry
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        registry_path: Optional[str] = None,
    ):
        """
        Initialize A2A client.

        Args:
            timeout: Default timeout for HTTP requests
            max_retries: Maximum number of retries for failed requests
            registry_path: Path to persistent agent registry
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.registry_path = registry_path

        # Local cache of discovered agents
        self._agent_cache: Dict[str, AgentCard] = {}
        self._registry = AgentRegistry()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    # =========================================================================
    # Discovery
    # =========================================================================

    async def discover(self, agent_url: str, force_refresh: bool = False) -> AgentCard:
        """
        Discover agent capabilities by fetching its Agent Card.

        Args:
            agent_url: Base URL of the agent
            force_refresh: Force refresh even if cached

        Returns:
            AgentCard with agent capabilities

        Raises:
            AgentNotFoundError: If agent is unreachable
        """
        agent_url = agent_url.rstrip("/")

        # Check cache
        if not force_refresh and agent_url in self._agent_cache:
            return self._agent_cache[agent_url]

        client = await self._get_client()
        card_url = f"{agent_url}/.well-known/agent-card.json"

        try:
            response = await client.get(card_url)
            response.raise_for_status()
            data = response.json()
            card = AgentCard(**data)

            # Cache the card
            self._agent_cache[agent_url] = card

            # Update registry
            self._registry.agents[agent_url] = AgentDiscoveryEntry(
                url=agent_url,
                name=card.name,
                description=card.description,
                skills=[s.id for s in card.skills],
                tags=card.tags,
                healthy=True,
            )

            logger.info(f"Discovered agent: {card.name} at {agent_url}")
            return card

        except httpx.HTTPError as e:
            logger.error(f"Failed to discover agent at {agent_url}: {e}")
            raise AgentNotFoundError(f"Could not reach agent at {agent_url}: {e}")

    async def discover_skill(
        self,
        skill_id: str,
        agent_urls: Optional[List[str]] = None
    ) -> Optional[tuple]:
        """
        Find an agent that provides a specific skill.

        Args:
            skill_id: Skill ID to search for
            agent_urls: List of agent URLs to search (uses registry if None)

        Returns:
            Tuple of (agent_url, AgentCard) or None if not found
        """
        urls = agent_urls or list(self._registry.agents.keys())

        for url in urls:
            try:
                card = await self.discover(url)
                for skill in card.skills:
                    if skill.id == skill_id:
                        return (url, card)
            except AgentNotFoundError:
                continue

        return None

    # =========================================================================
    # Task Management
    # =========================================================================

    async def create_task(
        self,
        agent_url: str,
        skill_id: str,
        input: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new task on a remote agent.

        Args:
            agent_url: Base URL of the target agent
            skill_id: Skill to invoke
            input: Input parameters for the skill
            context: Additional context
            callback_url: URL for push notifications
            metadata: Task metadata

        Returns:
            Created task data

        Raises:
            TaskError: If task creation fails
        """
        agent_url = agent_url.rstrip("/")
        client = await self._get_client()

        payload = {
            "skillId": skill_id,
            "input": input or {},
        }
        if context:
            payload["context"] = context
        if callback_url:
            payload["callbackUrl"] = callback_url
        if metadata:
            payload["metadata"] = metadata

        try:
            response = await client.post(
                f"{agent_url}/a2a/tasks",
                json=payload
            )
            response.raise_for_status()
            task_data = response.json()

            logger.info(f"Created task {task_data.get('id')} on {agent_url}")
            return task_data

        except httpx.HTTPError as e:
            raise TaskError(f"Failed to create task: {e}")

    async def get_task(self, agent_url: str, task_id: str) -> Dict[str, Any]:
        """
        Get task status from a remote agent.

        Args:
            agent_url: Base URL of the agent
            task_id: Task ID

        Returns:
            Task data including status and results
        """
        agent_url = agent_url.rstrip("/")
        client = await self._get_client()

        try:
            response = await client.get(f"{agent_url}/a2a/tasks/{task_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise TaskError(f"Failed to get task {task_id}: {e}")

    async def cancel_task(self, agent_url: str, task_id: str) -> Dict[str, Any]:
        """
        Cancel a task on a remote agent.

        Args:
            agent_url: Base URL of the agent
            task_id: Task ID to cancel

        Returns:
            Updated task data
        """
        agent_url = agent_url.rstrip("/")
        client = await self._get_client()

        try:
            response = await client.post(f"{agent_url}/a2a/tasks/{task_id}/cancel")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise TaskError(f"Failed to cancel task {task_id}: {e}")

    async def wait_for_task(
        self,
        agent_url: str,
        task_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete.

        Args:
            agent_url: Base URL of the agent
            task_id: Task ID
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            Completed task data

        Raises:
            TaskError: If task fails or times out
        """
        start_time = asyncio.get_event_loop().time()
        timeout = timeout or self.timeout * 10  # Default 5 minutes

        while True:
            task = await self.get_task(agent_url, task_id)
            status = task.get("status")

            if status == "completed":
                return task
            elif status == "failed":
                raise TaskError(f"Task failed: {task.get('error')}")
            elif status == "cancelled":
                raise TaskError("Task was cancelled")

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TaskError(f"Timeout waiting for task {task_id}")

            await asyncio.sleep(poll_interval)

    async def stream_task(
        self,
        agent_url: str,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream task updates via SSE.

        Args:
            agent_url: Base URL of the agent
            task_id: Task ID

        Yields:
            Task update events
        """
        agent_url = agent_url.rstrip("/")
        stream_url = f"{agent_url}/a2a/tasks/{task_id}/stream"

        async with httpx.AsyncClient(timeout=timeout or 300) as client:
            async with client.stream("GET", stream_url) as response:
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk

                    # Parse SSE events
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        event = self._parse_sse_event(event_str)
                        if event:
                            yield event

                            # Check for terminal states
                            if event.get("type") == "complete":
                                return

    def _parse_sse_event(self, event_str: str) -> Optional[Dict[str, Any]]:
        """Parse SSE event string."""
        event_type = "message"
        data = None

        for line in event_str.split("\n"):
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    data = line[5:].strip()

        if data is not None:
            return {"type": event_type, "data": data}
        return None

    # =========================================================================
    # JSON-RPC Interface
    # =========================================================================

    async def rpc_call(
        self,
        agent_url: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make a JSON-RPC call to an agent.

        Args:
            agent_url: Base URL of the agent
            method: RPC method name
            params: Method parameters

        Returns:
            RPC result

        Raises:
            TaskError: If RPC call fails
        """
        agent_url = agent_url.rstrip("/")
        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }

        try:
            response = await client.post(
                f"{agent_url}/a2a/rpc",
                json=payload
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result and result["error"]:
                raise TaskError(f"RPC error: {result['error']}")

            return result.get("result")

        except httpx.HTTPError as e:
            raise TaskError(f"RPC call failed: {e}")

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def call(
        self,
        agent_url: str,
        skill_id: str,
        input: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Convenient method to call a skill and optionally wait for result.

        Args:
            agent_url: Base URL of the target agent
            skill_id: Skill to invoke
            input: Input parameters
            wait: Whether to wait for completion
            timeout: Wait timeout

        Returns:
            Task data (with results if wait=True)
        """
        task = await self.create_task(agent_url, skill_id, input)

        if wait:
            return await self.wait_for_task(agent_url, task["id"], timeout=timeout)

        return task

    async def ping(self, agent_url: str) -> bool:
        """
        Check if an agent is healthy.

        Args:
            agent_url: Base URL of the agent

        Returns:
            True if agent is healthy
        """
        agent_url = agent_url.rstrip("/")
        client = await self._get_client()

        try:
            response = await client.get(f"{agent_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    # =========================================================================
    # Registry Management
    # =========================================================================

    def register_agent(self, url: str, name: str, description: str = "", tags: List[str] = None):
        """Manually register an agent in the local registry."""
        self._registry.agents[url] = AgentDiscoveryEntry(
            url=url,
            name=name,
            description=description,
            tags=tags or [],
        )

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [
            entry.model_dump()
            for entry in self._registry.agents.values()
        ]

    def get_cached_card(self, agent_url: str) -> Optional[AgentCard]:
        """Get cached Agent Card."""
        return self._agent_cache.get(agent_url.rstrip("/"))


# =============================================================================
# Global Client Instance
# =============================================================================

_client: Optional[A2AClient] = None


def get_a2a_client() -> A2AClient:
    """Get global A2A client instance."""
    global _client
    if _client is None:
        _client = A2AClient()
    return _client


async def discover_agent(agent_url: str) -> AgentCard:
    """Discover an agent's capabilities."""
    client = get_a2a_client()
    return await client.discover(agent_url)


async def call_agent(
    agent_url: str,
    skill_id: str,
    input: Optional[Dict[str, Any]] = None,
    wait: bool = True,
) -> Dict[str, Any]:
    """
    Call a skill on a remote agent.

    Args:
        agent_url: Base URL of the target agent
        skill_id: Skill to invoke
        input: Input parameters
        wait: Whether to wait for completion

    Returns:
        Task result
    """
    client = get_a2a_client()
    async with client:
        return await client.call(agent_url, skill_id, input, wait=wait)
