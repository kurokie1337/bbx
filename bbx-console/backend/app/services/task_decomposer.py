"""
Task Decomposer Service

Uses AI (architect agent) to break down complex tasks into subtasks.
"""

import logging
import re
from typing import List, Optional

import yaml

from app.api.schemas.task import DecomposeResponse, DecomposedSubtask

logger = logging.getLogger(__name__)


class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks using AI.

    Uses the architect agent to analyze tasks and suggest:
    - Subtasks with descriptions
    - Agent assignments
    - Dependencies
    - Generated workflow
    """

    # Agent role recommendations
    AGENT_ROLES = {
        "design": "architect",
        "plan": "architect",
        "architecture": "architect",
        "analyze": "architect",
        "implement": "coder",
        "code": "coder",
        "write": "coder",
        "fix": "coder",
        "refactor": "coder",
        "review": "reviewer",
        "audit": "reviewer",
        "check": "reviewer",
        "test": "tester",
        "validate": "tester",
        "verify": "tester",
    }

    async def decompose(
        self,
        description: str,
        context: Optional[str] = None,
    ) -> DecomposeResponse:
        """
        Decompose a task into subtasks.

        Args:
            description: Task description
            context: Optional additional context

        Returns:
            DecomposeResponse with subtasks and suggested workflow
        """
        try:
            # Try to use Agent SDK for AI decomposition
            subtasks = await self._ai_decompose(description, context)
        except Exception as e:
            logger.warning(f"AI decomposition failed, using rule-based: {e}")
            subtasks = self._rule_based_decompose(description)

        # Generate workflow
        workflow = self._generate_workflow(subtasks, description)

        return DecomposeResponse(
            original_task=description,
            subtasks=subtasks,
            suggested_workflow=workflow,
            confidence=0.8 if len(subtasks) > 1 else 0.5,
        )

    async def _ai_decompose(
        self,
        description: str,
        context: Optional[str] = None,
    ) -> List[DecomposedSubtask]:
        """Use AI to decompose task"""
        from blackbox.core.adapters.agent_sdk import AgentSDKAdapter

        adapter = AgentSDKAdapter()

        prompt = f"""Analyze this task and break it down into specific subtasks.
For each subtask, specify:
1. A clear title
2. A detailed description
3. Which agent should handle it (architect, coder, reviewer, or tester)
4. Dependencies on other subtasks (if any)

Task: {description}
"""
        if context:
            prompt += f"\nContext: {context}"

        prompt += """

Respond in this exact format for each subtask:
SUBTASK: <title>
DESCRIPTION: <description>
AGENT: <agent>
DEPENDS_ON: <comma-separated subtask titles or "none">
---
"""

        result = await adapter.invoke_subagent(
            name="architect",
            prompt=prompt,
            timeout=60,
        )

        # Parse response
        response_text = result.get("response", "")
        return self._parse_ai_response(response_text)

    def _parse_ai_response(self, response: str) -> List[DecomposedSubtask]:
        """Parse AI response into subtasks"""
        subtasks = []
        blocks = response.split("---")

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            title_match = re.search(r"SUBTASK:\s*(.+?)(?:\n|$)", block)
            desc_match = re.search(r"DESCRIPTION:\s*(.+?)(?:\n(?:AGENT|DEPENDS)|$)", block, re.DOTALL)
            agent_match = re.search(r"AGENT:\s*(\w+)", block)
            deps_match = re.search(r"DEPENDS_ON:\s*(.+?)(?:\n|$)", block)

            if title_match:
                title = title_match.group(1).strip()
                description = desc_match.group(1).strip() if desc_match else ""
                agent = agent_match.group(1).strip().lower() if agent_match else "coder"
                deps_str = deps_match.group(1).strip() if deps_match else "none"

                depends_on = []
                if deps_str.lower() != "none":
                    depends_on = [d.strip() for d in deps_str.split(",") if d.strip()]

                # Validate agent
                if agent not in ["architect", "coder", "reviewer", "tester"]:
                    agent = self._recommend_agent(title + " " + description)

                subtasks.append(DecomposedSubtask(
                    title=title,
                    description=description,
                    assigned_agent=agent,
                    depends_on=depends_on,
                    priority="medium",
                ))

        return subtasks if subtasks else self._rule_based_decompose(response)

    def _rule_based_decompose(self, description: str) -> List[DecomposedSubtask]:
        """Fallback rule-based decomposition"""
        subtasks = []
        desc_lower = description.lower()

        # Default workflow: analyze -> implement -> review -> test
        subtasks.append(DecomposedSubtask(
            title="Analyze and plan",
            description=f"Analyze requirements and create implementation plan for: {description}",
            assigned_agent="architect",
            depends_on=[],
            priority="high",
        ))

        subtasks.append(DecomposedSubtask(
            title="Implement solution",
            description=f"Implement the planned solution",
            assigned_agent="coder",
            depends_on=["Analyze and plan"],
            priority="high",
        ))

        subtasks.append(DecomposedSubtask(
            title="Review implementation",
            description="Review the implementation for quality and issues",
            assigned_agent="reviewer",
            depends_on=["Implement solution"],
            priority="medium",
        ))

        subtasks.append(DecomposedSubtask(
            title="Write and run tests",
            description="Write tests and validate the implementation",
            assigned_agent="tester",
            depends_on=["Implement solution"],
            priority="medium",
        ))

        return subtasks

    def _recommend_agent(self, text: str) -> str:
        """Recommend agent based on keywords"""
        text_lower = text.lower()

        for keyword, agent in self.AGENT_ROLES.items():
            if keyword in text_lower:
                return agent

        return "coder"  # Default

    def _generate_workflow(
        self,
        subtasks: List[DecomposedSubtask],
        original_task: str,
    ) -> str:
        """Generate BBX workflow YAML from subtasks"""
        steps = []
        id_map = {}  # title -> step_id

        for idx, subtask in enumerate(subtasks):
            step_id = f"step_{idx + 1}"
            id_map[subtask.title] = step_id

            step = {
                "id": step_id,
                "mcp": "agent",
                "method": "subagent",
                "inputs": {
                    "name": subtask.assigned_agent,
                    "prompt": subtask.description,
                },
            }

            # Add dependencies
            depends_on = []
            for dep_title in subtask.depends_on:
                if dep_title in id_map:
                    depends_on.append(id_map[dep_title])

            if depends_on:
                step["depends_on"] = depends_on

            steps.append(step)

        workflow = {
            "bbx": "6.0",
            "workflow": {
                "id": "generated_workflow",
                "name": "Generated Workflow",
                "description": f"Auto-generated workflow for: {original_task[:100]}",
                "steps": steps,
            },
        }

        return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
