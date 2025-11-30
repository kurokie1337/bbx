#!/usr/bin/env python3
"""
BBX Claude Code Agents

Real AI agents powered by Claude Code CLI.
Each agent is specialized for a specific task.

Usage:
    # Start architect agent on port 9001
    python claude_code_agents.py architect

    # Start coder agent on port 9002
    python claude_code_agents.py coder

    # Start reviewer agent on port 9003
    python claude_code_agents.py reviewer

    # Start tester agent on port 9004
    python claude_code_agents.py tester
"""

import asyncio
import subprocess
import sys
import json
import uuid
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn


# =============================================================================
# Claude Code Integration
# =============================================================================

async def call_claude_code(
    prompt: str,
    working_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Call Claude Code CLI and get response.

    Args:
        prompt: The prompt to send to Claude
        working_dir: Working directory for Claude Code
        system_prompt: Optional system prompt for specialization
        timeout: Timeout in seconds

    Returns:
        Dict with 'success', 'output', 'error'
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json"]

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    env = os.environ.copy()

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
            env=env
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )

        output = stdout.decode('utf-8', errors='replace')
        error = stderr.decode('utf-8', errors='replace')

        # Try to parse JSON output
        try:
            result = json.loads(output)
            return {
                "success": True,
                "output": result,
                "raw": output,
                "error": error if error else None
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "output": output,
                "raw": output,
                "error": error if error else None
            }

    except asyncio.TimeoutError:
        return {
            "success": False,
            "output": None,
            "error": f"Timeout after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "output": None,
            "error": str(e)
        }


# =============================================================================
# Agent Definitions
# =============================================================================

AGENTS = {
    "architect": {
        "name": "Architect Agent",
        "description": "Designs system architecture, plans implementation, breaks down tasks",
        "port": 9001,
        "system_prompt": """You are a senior software architect. Your role is to:
- Analyze requirements and design clean architectures
- Break down complex tasks into manageable steps
- Identify potential issues and edge cases
- Propose design patterns and best practices
- Create clear implementation plans

Always output structured, actionable plans. Be specific about files, functions, and data flow.""",
        "skills": [
            {
                "id": "design_architecture",
                "name": "Design Architecture",
                "description": "Design system architecture for a feature or project",
            },
            {
                "id": "plan_implementation",
                "name": "Plan Implementation",
                "description": "Create step-by-step implementation plan",
            },
            {
                "id": "analyze_codebase",
                "name": "Analyze Codebase",
                "description": "Analyze existing codebase structure and patterns",
            },
        ],
    },
    "coder": {
        "name": "Coder Agent",
        "description": "Writes clean, efficient code following best practices",
        "port": 9002,
        "system_prompt": """You are an expert software developer. Your role is to:
- Write clean, efficient, and well-documented code
- Follow existing code patterns and conventions
- Handle edge cases and errors properly
- Write code that is easy to test and maintain

Output working code with clear comments. Be precise and thorough.""",
        "skills": [
            {
                "id": "write_code",
                "name": "Write Code",
                "description": "Write implementation code based on design",
            },
            {
                "id": "fix_bug",
                "name": "Fix Bug",
                "description": "Analyze and fix bugs in code",
            },
            {
                "id": "refactor",
                "name": "Refactor",
                "description": "Refactor code for better quality",
            },
        ],
    },
    "reviewer": {
        "name": "Reviewer Agent",
        "description": "Reviews code for quality, security, and best practices",
        "port": 9003,
        "system_prompt": """You are a senior code reviewer. Your role is to:
- Review code for bugs, security issues, and anti-patterns
- Check for proper error handling and edge cases
- Verify code follows project conventions
- Suggest improvements for readability and performance
- Ensure code is testable and maintainable

Be constructive but thorough. Provide specific feedback with line references.""",
        "skills": [
            {
                "id": "review_code",
                "name": "Review Code",
                "description": "Perform comprehensive code review",
            },
            {
                "id": "security_audit",
                "name": "Security Audit",
                "description": "Check code for security vulnerabilities",
            },
            {
                "id": "check_quality",
                "name": "Check Quality",
                "description": "Assess code quality and maintainability",
            },
        ],
    },
    "tester": {
        "name": "Tester Agent",
        "description": "Writes tests and validates code correctness",
        "port": 9004,
        "system_prompt": """You are a QA engineer and test specialist. Your role is to:
- Write comprehensive unit and integration tests
- Identify edge cases and error conditions
- Create test fixtures and mocks
- Validate code behavior against requirements
- Ensure high test coverage

Write practical, maintainable tests. Cover both happy paths and error cases.""",
        "skills": [
            {
                "id": "write_tests",
                "name": "Write Tests",
                "description": "Write unit and integration tests",
            },
            {
                "id": "create_test_plan",
                "name": "Create Test Plan",
                "description": "Create comprehensive test plan",
            },
            {
                "id": "validate_implementation",
                "name": "Validate Implementation",
                "description": "Validate implementation against requirements",
            },
        ],
    },
}


# =============================================================================
# Skill Handlers
# =============================================================================

async def handle_skill(
    agent_type: str,
    skill_id: str,
    input_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle any skill by calling Claude Code with appropriate context.
    """
    config = AGENTS[agent_type]
    system_prompt = config["system_prompt"]

    # Build the prompt from input
    prompt_parts = []

    # Add skill context
    skill_info = next((s for s in config["skills"] if s["id"] == skill_id), None)
    if skill_info:
        prompt_parts.append(f"Task: {skill_info['description']}")

    # Add input data
    if "prompt" in input_data:
        prompt_parts.append(f"\n{input_data['prompt']}")
    if "code" in input_data:
        prompt_parts.append(f"\nCode to work with:\n```\n{input_data['code']}\n```")
    if "files" in input_data:
        prompt_parts.append(f"\nFiles: {', '.join(input_data['files'])}")
    if "requirements" in input_data:
        prompt_parts.append(f"\nRequirements:\n{input_data['requirements']}")
    if "context" in input_data:
        prompt_parts.append(f"\nContext:\n{input_data['context']}")
    if "previous_step" in input_data:
        prompt_parts.append(f"\nPrevious step output:\n{input_data['previous_step']}")

    # Add any extra data
    for key, value in input_data.items():
        if key not in ["prompt", "code", "files", "requirements", "context", "previous_step"]:
            prompt_parts.append(f"\n{key}: {value}")

    full_prompt = "\n".join(prompt_parts)
    working_dir = input_data.get("working_dir")

    # Call Claude Code
    result = await call_claude_code(
        prompt=full_prompt,
        working_dir=working_dir,
        system_prompt=system_prompt,
        timeout=input_data.get("timeout", 300)
    )

    return {
        "agent": config["name"],
        "skill": skill_id,
        "result": result,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Create Agent App
# =============================================================================

def create_claude_agent(agent_type: str) -> FastAPI:
    """Create a Claude Code powered agent FastAPI app."""
    config = AGENTS[agent_type]

    app = FastAPI(
        title=config["name"],
        description=config["description"],
    )

    # In-memory task store
    tasks: Dict[str, Dict[str, Any]] = {}

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card():
        """Return Agent Card (A2A discovery)."""
        return {
            "name": config["name"],
            "description": config["description"],
            "url": f"http://localhost:{config['port']}",
            "version": "1.0.0",
            "protocolVersion": "0.3",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
            },
            "skills": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "description": s["description"],
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Main task description"},
                            "code": {"type": "string", "description": "Code to work with"},
                            "files": {"type": "array", "items": {"type": "string"}},
                            "requirements": {"type": "string"},
                            "context": {"type": "string"},
                            "previous_step": {"type": "string"},
                            "working_dir": {"type": "string"},
                        }
                    }
                }
                for s in config["skills"]
            ],
            "endpoints": {
                "task": "/a2a/tasks",
                "taskStatus": "/a2a/tasks/{task_id}",
            },
            "authentication": {"schemes": ["none"]},
            "tags": ["claude-code", agent_type],
        }

    @app.post("/a2a/tasks")
    async def create_task(task_input: dict, background_tasks: BackgroundTasks):
        """Create and execute a task using Claude Code."""
        task_id = str(uuid.uuid4())
        skill_id = task_input.get("skillId")
        input_data = task_input.get("input", {})

        task = {
            "id": task_id,
            "skillId": skill_id,
            "input": input_data,
            "status": "pending",
            "output": None,
            "error": None,
            "createdAt": datetime.utcnow().isoformat(),
        }
        tasks[task_id] = task

        async def execute():
            task["status"] = "in_progress"
            task["startedAt"] = datetime.utcnow().isoformat()

            try:
                result = await handle_skill(agent_type, skill_id, input_data)

                if result["result"]["success"]:
                    task["status"] = "completed"
                    task["output"] = result
                else:
                    task["status"] = "failed"
                    task["error"] = result["result"]["error"]

            except Exception as e:
                task["status"] = "failed"
                task["error"] = str(e)

            task["completedAt"] = datetime.utcnow().isoformat()

        background_tasks.add_task(execute)

        return task

    @app.get("/a2a/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get task status."""
        task = tasks.get(task_id)
        if not task:
            return JSONResponse(
                status_code=404,
                content={"error": f"Task not found: {task_id}"}
            )
        return task

    @app.get("/a2a/tasks")
    async def list_tasks():
        """List all tasks."""
        return {"tasks": list(tasks.values()), "count": len(tasks)}

    @app.get("/health")
    async def health():
        """Health check."""
        return {
            "status": "healthy",
            "agent": config["name"],
            "type": agent_type,
            "powered_by": "Claude Code"
        }

    @app.post("/quick")
    async def quick_task(request: dict):
        """
        Quick synchronous task execution.
        For simple cases where you don't need async polling.
        """
        skill_id = request.get("skill")
        input_data = request.get("input", {})

        result = await handle_skill(agent_type, skill_id, input_data)
        return result

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    """Run Claude Code agent."""
    if len(sys.argv) < 2:
        print("BBX Claude Code Agents")
        print("=" * 50)
        print("\nUsage: python claude_code_agents.py <agent_type>")
        print("\nAvailable agents:")
        for name, config in AGENTS.items():
            print(f"  {name:12} - {config['description'][:50]}... (port {config['port']})")
        print("\nExample:")
        print("  python claude_code_agents.py architect")
        print("  python claude_code_agents.py coder")
        print("  python claude_code_agents.py reviewer")
        sys.exit(1)

    agent_type = sys.argv[1].lower()

    if agent_type not in AGENTS:
        print(f"Unknown agent type: {agent_type}")
        print(f"Available: {', '.join(AGENTS.keys())}")
        sys.exit(1)

    config = AGENTS[agent_type]
    app = create_claude_agent(agent_type)

    print(f"\n{'='*60}")
    print(f"  BBX Claude Code Agent: {config['name']}")
    print(f"{'='*60}")
    print(f"  Port: {config['port']}")
    print(f"  Agent Card: http://localhost:{config['port']}/.well-known/agent-card.json")
    print(f"  Health: http://localhost:{config['port']}/health")
    print(f"  Skills:")
    for skill in config['skills']:
        print(f"    - {skill['id']}: {skill['description']}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=config["port"])


if __name__ == "__main__":
    main()
