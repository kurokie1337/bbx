# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Agent Card Generator

Automatically generates A2A Agent Card from BBX configuration:
- Discovers workflows and exposes them as skills
- Includes MCP tools as capabilities
- Generates proper JSON schema for inputs/outputs
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    AgentCard,
    AgentSkill,
    AgentEndpoints,
    AgentAuthentication,
)


class AgentCardGenerator:
    """
    Generates A2A Agent Card from BBX workspace.

    Usage:
        generator = AgentCardGenerator(
            name="my-bbx-agent",
            url="https://my-agent.example.com",
            workspace_path="~/.bbx/workspaces/my-project"
        )
        card = generator.generate()
        card_json = card.model_dump_json(by_alias=True, indent=2)
    """

    def __init__(
        self,
        name: str,
        url: str,
        description: Optional[str] = None,
        workspace_path: Optional[str] = None,
        version: str = "1.0.0",
        provider: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        self.name = name
        self.url = url.rstrip("/")
        self.description = description or f"BBX Agent: {name}"
        self.workspace_path = Path(workspace_path) if workspace_path else None
        self.version = version
        self.provider = provider
        self.tags = tags or []

    def generate(self) -> AgentCard:
        """Generate complete Agent Card."""
        skills = []

        # Add built-in BBX skills
        skills.extend(self._get_builtin_skills())

        # Discover workflows as skills
        if self.workspace_path:
            skills.extend(self._discover_workflow_skills())

        # Add MCP tools as skills
        skills.extend(self._get_mcp_skills())

        # Count MCP tools
        mcp_count = self._count_mcp_tools()

        return AgentCard(
            name=self.name,
            description=self.description,
            url=self.url,
            version=self.version,
            protocol_version="0.3",
            skills=skills,
            endpoints=AgentEndpoints(
                task=f"{self.url}/a2a/tasks",
                task_status=f"{self.url}/a2a/tasks/{{task_id}}",
                task_cancel=f"{self.url}/a2a/tasks/{{task_id}}/cancel",
                stream=f"{self.url}/a2a/tasks/{{task_id}}/stream",
            ),
            authentication=AgentAuthentication(
                schemes=["none", "bearer", "apiKey"]
            ),
            provider=self.provider,
            tags=self.tags + ["bbx", "workflow", "ai-agent"],
            bbx_version="1.0.0",
            mcp_tools_count=mcp_count,
        )

    def _get_builtin_skills(self) -> List[AgentSkill]:
        """Get built-in BBX skills."""
        return [
            AgentSkill(
                id="bbx.run_workflow",
                name="Run Workflow",
                description="Execute a BBX workflow with DAG parallel execution",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workflow": {
                            "type": "string",
                            "description": "Workflow file path or ID"
                        },
                        "inputs": {
                            "type": "object",
                            "description": "Workflow input parameters"
                        },
                        "background": {
                            "type": "boolean",
                            "default": False,
                            "description": "Run in background"
                        }
                    },
                    "required": ["workflow"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "execution_id": {"type": "string"},
                        "status": {"type": "string"},
                        "outputs": {"type": "object"}
                    }
                },
                tags=["workflow", "execution"],
                supports_streaming=True,
                estimated_duration="varies"
            ),
            AgentSkill(
                id="bbx.state_management",
                name="Persistent State",
                description="Get, set, and manage persistent state that survives between sessions",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["get", "set", "delete", "list", "increment", "append"]
                        },
                        "key": {"type": "string"},
                        "value": {},
                        "namespace": {"type": "string"}
                    },
                    "required": ["operation"]
                },
                tags=["state", "memory", "persistence"]
            ),
            AgentSkill(
                id="bbx.process_management",
                name="Process Management",
                description="Monitor, kill, and manage running workflow executions (like Linux ps/kill)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["ps", "kill", "wait", "logs"]
                        },
                        "execution_id": {"type": "string"},
                        "all": {"type": "boolean", "default": False}
                    },
                    "required": ["operation"]
                },
                tags=["process", "monitoring"]
            ),
            AgentSkill(
                id="bbx.mcp_bridge",
                name="MCP Tool Bridge",
                description="Access any configured MCP server tools through BBX",
                input_schema={
                    "type": "object",
                    "properties": {
                        "server": {
                            "type": "string",
                            "description": "MCP server name"
                        },
                        "tool": {
                            "type": "string",
                            "description": "Tool name"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Tool arguments"
                        }
                    },
                    "required": ["server", "tool"]
                },
                tags=["mcp", "tools", "integration"]
            ),
        ]

    def _discover_workflow_skills(self) -> List[AgentSkill]:
        """Discover workflows in workspace and convert to skills."""
        skills = []

        if not self.workspace_path or not self.workspace_path.exists():
            return skills

        # Search for .bbx files
        workflow_paths = list(self.workspace_path.glob("**/*.bbx"))

        for wf_path in workflow_paths[:20]:  # Limit to 20 workflows
            try:
                skill = self._workflow_to_skill(wf_path)
                if skill:
                    skills.append(skill)
            except Exception:
                continue

        return skills

    def _workflow_to_skill(self, workflow_path: Path) -> Optional[AgentSkill]:
        """Convert a BBX workflow to an A2A skill."""
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                return None

            # Handle nested workflow structure
            if "workflow" in data:
                workflow = data["workflow"]
            else:
                workflow = data

            # Extract workflow metadata
            wf_id = workflow.get("id", workflow_path.stem)
            wf_name = workflow.get("name", wf_id.replace("_", " ").title())
            wf_desc = workflow.get("description", f"Execute {wf_name} workflow")

            # Build input schema from workflow inputs
            input_schema: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": []
            }

            inputs = workflow.get("inputs", {})
            for input_name, input_def in inputs.items():
                if isinstance(input_def, dict):
                    prop = {
                        "type": input_def.get("type", "string"),
                        "description": input_def.get("description", "")
                    }
                    if "default" in input_def:
                        prop["default"] = input_def["default"]
                    else:
                        input_schema["required"].append(input_name)
                    input_schema["properties"][input_name] = prop
                else:
                    input_schema["properties"][input_name] = {"type": "string"}

            # Determine tags from workflow
            tags = workflow.get("tags", [])
            if not tags:
                # Infer tags from path
                rel_path = workflow_path.relative_to(self.workspace_path)
                if len(rel_path.parts) > 1:
                    tags = [rel_path.parts[0]]

            # Check if workflow has streaming steps
            steps = workflow.get("steps", {})
            has_streaming = any(
                step.get("stream", False)
                for step in (steps.values() if isinstance(steps, dict) else steps)
            )

            return AgentSkill(
                id=f"workflow.{wf_id}",
                name=wf_name,
                description=wf_desc,
                input_schema=input_schema if input_schema["properties"] else None,
                tags=["workflow"] + tags,
                supports_streaming=has_streaming,
            )

        except Exception:
            return None

    def _get_mcp_skills(self) -> List[AgentSkill]:
        """Get skills representing MCP tool categories."""
        # Instead of listing all tools, we expose MCP as a meta-skill
        return [
            AgentSkill(
                id="bbx.mcp_discover",
                name="Discover MCP Tools",
                description="Discover all available MCP tools from configured servers",
                tags=["mcp", "discovery"]
            ),
        ]

    def _count_mcp_tools(self) -> int:
        """Count available MCP tools."""
        try:
            from blackbox.mcp.client.config import load_mcp_config
            configs = load_mcp_config()
            # Estimate ~10 tools per server on average
            return len(configs) * 10
        except Exception:
            return 0

    def to_json(self, indent: int = 2) -> str:
        """Generate Agent Card as JSON string."""
        card = self.generate()
        return card.model_dump_json(by_alias=True, indent=indent)

    def save(self, path: str) -> str:
        """Save Agent Card to file."""
        json_content = self.to_json()
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_content)
        return path


def generate_agent_card(
    name: str,
    url: str,
    workspace_path: Optional[str] = None,
    **kwargs
) -> AgentCard:
    """
    Convenience function to generate an Agent Card.

    Args:
        name: Agent name
        url: Base URL of the agent
        workspace_path: Optional path to BBX workspace
        **kwargs: Additional AgentCardGenerator arguments

    Returns:
        Generated AgentCard
    """
    generator = AgentCardGenerator(
        name=name,
        url=url,
        workspace_path=workspace_path,
        **kwargs
    )
    return generator.generate()
