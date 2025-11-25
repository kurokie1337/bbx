# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX MCP Tools - BBX workflow operations exposed as MCP tools
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Windows encoding fix: replace emoji with ASCII equivalents
def _safe_output(text: str) -> str:
    """Replace emoji with ASCII for Windows compatibility."""
    if sys.platform != "win32":
        return text

    # Emoji to ASCII mapping
    replacements = {
        "🔧": "[*]",
        "📋": "[#]",
        "📡": "[~]",
        "📦": "[+]",
        "📁": "[/]",
        "📊": "[=]",
        "📜": "[-]",
        "🔍": "[?]",
        "🔷": "[>]",
        "🔶": "[>]",
        "🔗": "[>]",
        "🤖": "[AI]",
        "🧠": "[AI]",
        "✅": "[OK]",
        "❌": "[ERR]",
        "⚠️": "[WARN]",
        "⬇️": "[DL]",
        "⏱️": "[TIME]",
        "🚀": "[>>]",
        "🛑": "[STOP]",
    }

    for emoji, ascii_char in replacements.items():
        text = text.replace(emoji, ascii_char)

    return text

# Lazy imports to avoid circular dependencies
WorkflowGenerator = None
run_file = None


def _ensure_imports():
    """Lazy import to avoid circular dependencies"""
    global WorkflowGenerator, run_file
    if WorkflowGenerator is None:
        from blackbox.ai.generator import WorkflowGenerator as WG
        from blackbox.core.runtime import run_file as rf
        WorkflowGenerator = WG
        run_file = rf


def get_bbx_tools() -> List[Dict[str, Any]]:
    """
    Get list of BBX tools for MCP server.

    Returns:
        List of MCP tool definitions
    """
    return [
        # === Core Workflow Tools ===
        {
            "name": "bbx_generate",
            "description": "Generate a BBX workflow from natural language description using local AI",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language description of the workflow task",
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output file path (default: generated.bbx)",
                        "default": "generated.bbx",
                    },
                },
                "required": ["description"],
            },
        },
        {
            "name": "bbx_validate",
            "description": "Validate a BBX workflow file for syntax and structure errors",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_file": {
                        "type": "string",
                        "description": "Path to the .bbx workflow file to validate",
                    }
                },
                "required": ["workflow_file"],
            },
        },
        {
            "name": "bbx_run",
            "description": "Execute a BBX workflow (runs all steps in order with DAG parallelization)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_file": {
                        "type": "string",
                        "description": "Path to the .bbx workflow file to run",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, show what would run without executing",
                        "default": False,
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Optional inputs to pass to the workflow",
                    },
                },
                "required": ["workflow_file"],
            },
        },
        {
            "name": "bbx_info",
            "description": "Show detailed information about a BBX workflow file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_file": {
                        "type": "string",
                        "description": "Path to the .bbx workflow file",
                    }
                },
                "required": ["workflow_file"],
            },
        },
        {
            "name": "bbx_list_workflows",
            "description": "List all available BBX workflows in the project",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search for .bbx files (default: current)",
                        "default": ".",
                    }
                },
            },
        },
        # === MCP Integration Tools ===
        {
            "name": "bbx_mcp_discover",
            "description": "Discover all tools from all configured MCP servers (test, filesystem, puppeteer, etc.)",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_mcp_call",
            "description": "Call any MCP tool directly. Use bbx_mcp_discover to see available tools.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name (e.g., 'test', 'filesystem', 'puppeteer')",
                    },
                    "tool": {
                        "type": "string",
                        "description": "Tool name (e.g., 'echo', 'read_file', 'puppeteer_navigate')",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Tool arguments",
                    },
                },
                "required": ["server", "tool"],
            },
        },
        {
            "name": "bbx_mcp_servers",
            "description": "List all configured MCP servers with their status",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        # === Learn Tools ===
        {
            "name": "bbx_learn",
            "description": "Learn a CLI tool by parsing its --help output and generate BBX adapter",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the CLI tool to learn (e.g., 'docker', 'kubectl')",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: 'yaml' or 'python'",
                        "enum": ["yaml", "python"],
                        "default": "yaml",
                    },
                },
                "required": ["tool_name"],
            },
        },
        {
            "name": "bbx_learned_tools",
            "description": "List all CLI tools that BBX has learned",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        # === System Tools ===
        {
            "name": "bbx_system",
            "description": "Check BBX system health: Python version, Docker status, available adapters",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_adapters",
            "description": "List all available BBX adapters (core, optional, MCP)",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_schema",
            "description": "Generate JSON Schema for BBX workflow files (for VS Code autocomplete)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "output_file": {
                        "type": "string",
                        "description": "Output file path (default: bbx.schema.json)",
                        "default": "bbx.schema.json",
                    }
                },
            },
        },
        # === MCP Extended Tools ===
        {
            "name": "bbx_mcp_test",
            "description": "Test connection to an MCP server",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name to test",
                    }
                },
                "required": ["server"],
            },
        },
        {
            "name": "bbx_mcp_tools",
            "description": "List all tools available on a specific MCP server",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name",
                    }
                },
                "required": ["server"],
            },
        },
        {
            "name": "bbx_mcp_tool_schema",
            "description": "Show input schema for a specific MCP tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "server": {
                        "type": "string",
                        "description": "MCP server name",
                    },
                    "tool": {
                        "type": "string",
                        "description": "Tool name",
                    },
                },
                "required": ["server", "tool"],
            },
        },
        # === AI Model Tools ===
        {
            "name": "bbx_model_list",
            "description": "List available AI models for workflow generation",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_model_download",
            "description": "Download an AI model for workflow generation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name to download (e.g., 'qwen-0.5b', 'qwen-1.8b')",
                    }
                },
                "required": ["model_name"],
            },
        },
        {
            "name": "bbx_model_remove",
            "description": "Remove a downloaded AI model",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Model name to remove",
                    }
                },
                "required": ["model_name"],
            },
        },
        # === Version Control Tools ===
        {
            "name": "bbx_version_create",
            "description": "Create a new version of a workflow",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_file": {
                        "type": "string",
                        "description": "Path to workflow file",
                    },
                    "version": {
                        "type": "string",
                        "description": "Version number (e.g., '1.0.0')",
                    },
                    "message": {
                        "type": "string",
                        "description": "Version description",
                    },
                },
                "required": ["workflow_file", "version"],
            },
        },
        {
            "name": "bbx_version_list",
            "description": "List all versions of a workflow",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID to list versions for",
                    }
                },
                "required": ["workflow_id"],
            },
        },
        {
            "name": "bbx_version_rollback",
            "description": "Rollback workflow to a previous version",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "Workflow ID",
                    },
                    "target_version": {
                        "type": "string",
                        "description": "Version to rollback to",
                    },
                },
                "required": ["workflow_id", "target_version"],
            },
        },
        # === Workspace Tools (OS-like environments) ===
        {
            "name": "bbx_workspace_create",
            "description": "Create a new workspace (isolated environment like /home/user in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Workspace name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Workspace description",
                    },
                    "path": {
                        "type": "string",
                        "description": "Custom path (optional, default: ~/.bbx/workspaces/<name>)",
                    },
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_workspace_list",
            "description": "List all available workspaces",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to search (default: ~/.bbx/workspaces)",
                    },
                },
            },
        },
        {
            "name": "bbx_workspace_set",
            "description": "Set current workspace context (like cd in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to workspace",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "bbx_workspace_info",
            "description": "Show information about current or specified workspace",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace path (default: current workspace)",
                    },
                },
            },
        },
        {
            "name": "bbx_workspace_clear",
            "description": "Clear current workspace context",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        # === Process Management Tools (like ps, kill, wait in Linux) ===
        {
            "name": "bbx_ps",
            "description": "List workflow executions (like ps in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "all": {
                        "type": "boolean",
                        "description": "Show all executions, not just running (like ps aux)",
                        "default": False,
                    },
                },
            },
        },
        {
            "name": "bbx_kill",
            "description": "Kill a running workflow execution (like kill in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID to kill (can be partial)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force kill (like kill -9)",
                        "default": False,
                    },
                },
                "required": ["execution_id"],
            },
        },
        {
            "name": "bbx_wait",
            "description": "Wait for a workflow execution to complete (like wait in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID to wait for",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds",
                    },
                },
                "required": ["execution_id"],
            },
        },
        {
            "name": "bbx_logs",
            "description": "Get logs for a workflow execution (like tail in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of log entries to return",
                        "default": 50,
                    },
                    "level": {
                        "type": "string",
                        "description": "Filter by log level (INFO, ERROR, etc.)",
                    },
                },
                "required": ["execution_id"],
            },
        },
        {
            "name": "bbx_run_background",
            "description": "Run a workflow in background (like command & in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_file": {
                        "type": "string",
                        "description": "Path to .bbx workflow file",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Workflow inputs",
                    },
                },
                "required": ["workflow_file"],
            },
        },
        # === State Management Tools (like env vars in Linux) ===
        {
            "name": "bbx_state_get",
            "description": "Get a persistent state value (like echo $VAR in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "State key",
                    },
                    "default": {
                        "type": "string",
                        "description": "Default value if not found",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "State namespace (for workflow-scoped state)",
                    },
                },
                "required": ["key"],
            },
        },
        {
            "name": "bbx_state_set",
            "description": "Set a persistent state value (like export VAR=value in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "State key",
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "object", "array"],
                        "description": "Value to set",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "State namespace",
                    },
                },
                "required": ["key", "value"],
            },
        },
        {
            "name": "bbx_state_list",
            "description": "List all state keys (like env in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filter keys by pattern (e.g., user_*)",
                        "default": "*",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "State namespace",
                    },
                },
            },
        },
        {
            "name": "bbx_state_delete",
            "description": "Delete a state key (like unset VAR in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "State key to delete",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "State namespace",
                    },
                },
                "required": ["key"],
            },
        },
        {
            "name": "bbx_state_increment",
            "description": "Atomically increment a numeric state value",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "State key",
                    },
                    "by": {
                        "type": "number",
                        "description": "Amount to increment by",
                        "default": 1,
                    },
                    "namespace": {
                        "type": "string",
                        "description": "State namespace",
                    },
                },
                "required": ["key"],
            },
        },
        {
            "name": "bbx_state_append",
            "description": "Append to a list state value (useful for agent memory/history)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "State key",
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "object", "array"],
                        "description": "Value to append",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Max items to keep (trims oldest)",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "State namespace",
                    },
                },
                "required": ["key", "value"],
            },
        },
        # === Nested Workflow Tools ===
        {
            "name": "bbx_workflow_run",
            "description": "Run a nested workflow (sync or background, like fork() in Linux)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to workflow file",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Workflow inputs",
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run in background",
                        "default": False,
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (for sync execution)",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "bbx_workflow_status",
            "description": "Get status of a workflow execution",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "execution_id": {
                        "type": "string",
                        "description": "Execution ID",
                    },
                },
                "required": ["execution_id"],
            },
        },
    ]


async def handle_bbx_generate(arguments: Dict[str, Any]) -> str:
    """
    Handle bbx_generate tool call.

    Args:
        arguments: Tool arguments (description, output_file)

    Returns:
        Result message
    """
    _ensure_imports()
    description = arguments.get("description")
    output_file = arguments.get("output_file", "generated.bbx")

    try:
        generator = WorkflowGenerator()
        yaml_content = generator.generate(
            description=description, output_file=output_file
        )

        return f"""Workflow generated successfully!

File: {output_file}

Content:
{yaml_content}

Next steps:
1. Validate: bbx validate {output_file}
2. Run: bbx run {output_file}
"""
    except Exception as e:
        return f"Generation failed: {str(e)}"


async def handle_bbx_validate(arguments: Dict[str, Any]) -> str:
    """
    Handle bbx_validate tool call.

    Args:
        arguments: Tool arguments (workflow_file)

    Returns:
        Validation result
    """
    workflow_file = arguments.get("workflow_file")

    if not os.path.exists(workflow_file):
        return f"File not found: {workflow_file}"

    try:
        # Parse workflow
        with open(workflow_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Basic validation
        if "workflow" in data:
            workflow = data["workflow"]
        else:
            workflow = data

        errors = []
        if "steps" not in workflow:
            errors.append("Missing 'steps' field")
        elif not isinstance(workflow.get("steps"), (list, dict)):
            errors.append("'steps' must be a list or dict")

        if errors:
            error_list = "\n".join(f"  - {err}" for err in errors)
            return f"Validation failed:\n{error_list}"
        else:
            return f"Workflow is valid: {workflow_file}"

    except yaml.YAMLError as e:
        return f"YAML parse error: {str(e)}"
    except Exception as e:
        return f"Validation error: {str(e)}"


async def handle_bbx_run(arguments: Dict[str, Any]) -> str:
    """
    Handle bbx_run tool call.

    Args:
        arguments: Tool arguments (workflow_file, dry_run)

    Returns:
        Execution result
    """
    _ensure_imports()
    workflow_file = arguments.get("workflow_file")
    dry_run = arguments.get("dry_run", False)

    if not os.path.exists(workflow_file):
        return f"File not found: {workflow_file}"

    try:
        # Parse workflow
        with open(workflow_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "workflow" in data:
            workflow = data["workflow"]
        else:
            workflow = data

        steps = workflow.get("steps", [])
        if isinstance(steps, dict):
            steps = list(steps.values())

        if dry_run:
            # Show what would run
            steps_info = "\n".join(
                f"  {i+1}. {step.get('id', step.get('use', 'unnamed'))}"
                for i, step in enumerate(steps)
            )
            return f"""Dry run: {workflow_file}

Workflow: {workflow.get('id', 'unnamed')}
Steps ({len(steps)}):
{steps_info}

To execute: bbx run {workflow_file}
"""
        else:
            # Actually run using runtime
            import asyncio
            results = await run_file(workflow_file)

            # Check results
            failed = [k for k, v in results.items() if v.get("status") == "error"]
            if failed:
                return f"Workflow completed with errors in: {', '.join(failed)}"
            else:
                return f"Workflow executed successfully: {workflow_file}"

    except Exception as e:
        return f"Execution error: {str(e)}"


async def handle_bbx_list_workflows(arguments: Dict[str, Any]) -> str:
    """
    Handle bbx_list_workflows tool call.

    Args:
        arguments: Tool arguments (category)

    Returns:
        List of workflows
    """
    category = arguments.get("category", "all")

    workflows_dir = Path("workflows")
    if not workflows_dir.exists():
        return "❌ No workflows directory found"

    # Find workflows based on category
    workflow_files = []

    if category == "all" or category == "core":
        core_dir = workflows_dir / "core"
        if core_dir.exists():
            workflow_files.extend(core_dir.glob("**/*.bbx"))

    if category == "all" or category == "user":
        user_dir = workflows_dir / "user"
        if user_dir.exists():
            workflow_files.extend(user_dir.glob("**/*.bbx"))

    if category == "all" or category == "meta":
        meta_dir = workflows_dir / "meta"
        if meta_dir.exists():
            workflow_files.extend(meta_dir.glob("**/*.bbx"))

    if not workflow_files:
        return f"No workflows found (category: {category})"

    # Format output
    result = f"📋 BBX Workflows ({category}):\n\n"

    for wf in sorted(workflow_files):
        relative_path = wf.relative_to(workflows_dir)
        result += f"  - {relative_path}\n"

    result += f"\nTotal: {len(workflow_files)} workflows"

    return result


async def handle_bbx_info(arguments: Dict[str, Any]) -> str:
    """Show detailed information about a workflow file."""
    workflow_file = arguments.get("workflow_file")

    if not os.path.exists(workflow_file):
        return f"File not found: {workflow_file}"

    try:
        with open(workflow_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "workflow" in data:
            workflow = data["workflow"]
        else:
            workflow = data

        steps = workflow.get("steps", [])
        if isinstance(steps, dict):
            steps = list(steps.values())

        result = f"""📋 Workflow Information
{'=' * 60}
File: {workflow_file}
ID: {workflow.get('id', 'N/A')}
Name: {workflow.get('name', 'N/A')}
Version: {workflow.get('version', 'N/A')}
Description: {workflow.get('description', 'N/A')}

Steps ({len(steps)}):
"""
        for i, step in enumerate(steps):
            step_id = step.get("id", f"step_{i}")
            use = step.get("use", step.get("mcp", "unknown"))
            method = step.get("method", "")
            deps = step.get("depends_on", [])
            result += f"  {i+1}. {step_id} ({use}.{method})"
            if deps:
                result += f" [depends: {', '.join(deps)}]"
            result += "\n"

        return result

    except Exception as e:
        return f"Error reading workflow: {str(e)}"


async def handle_bbx_mcp_discover(arguments: Dict[str, Any]) -> str:
    """Discover all tools from all MCP servers."""
    try:
        from blackbox.mcp.client.manager import get_mcp_manager

        manager = get_mcp_manager()
        await manager.load_config()
        all_tools = await manager.list_all_tools()

        result = "🔍 MCP Tools Discovery\n" + "=" * 60 + "\n"
        total_tools = 0

        for server_name, tools in all_tools.items():
            if "error" in tools:
                result += f"\n❌ {server_name}: {tools['error']}\n"
            else:
                result += f"\n[{server_name}] ({len(tools)} tools)\n"
                for tool_name, tool_info in tools.items():
                    desc = ""
                    if hasattr(tool_info, "description"):
                        desc = f" - {tool_info.description[:50]}..."
                    result += f"    mcp.{server_name}.{tool_name}{desc}\n"
                total_tools += len(tools)

        result += f"\n{'=' * 60}\n"
        result += f"Total: {total_tools} tools from {len(all_tools)} servers\n"
        return result

    except Exception as e:
        return f"Error discovering MCP tools: {str(e)}"


async def handle_bbx_mcp_call(arguments: Dict[str, Any]) -> str:
    """Call any MCP tool directly."""
    server = arguments.get("server")
    tool = arguments.get("tool")
    tool_args = arguments.get("arguments", {})

    try:
        from blackbox.mcp.client.manager import get_mcp_manager

        manager = get_mcp_manager()
        await manager.load_config()
        result = await manager.call_tool(server, tool, tool_args)

        return f"""✅ MCP Call Successful
Server: {server}
Tool: {tool}
Result: {result}"""

    except Exception as e:
        return f"❌ MCP call failed: {str(e)}"


async def handle_bbx_mcp_servers(arguments: Dict[str, Any]) -> str:
    """List all configured MCP servers."""
    try:
        from blackbox.mcp.client.config import load_mcp_config

        configs = load_mcp_config()

        if not configs:
            return "No MCP servers configured. Create mcp_servers.yaml to add servers."

        result = "📡 Configured MCP Servers\n" + "=" * 60 + "\n"

        for name, config in configs.items():
            result += f"\n[{name}]\n"
            result += f"  Transport: {config.transport}\n"
            if config.command:
                result += f"  Command: {' '.join(config.command)}\n"
            if config.description:
                result += f"  Description: {config.description}\n"

        return result

    except Exception as e:
        return f"Error listing MCP servers: {str(e)}"


async def handle_bbx_learn(arguments: Dict[str, Any]) -> str:
    """Learn a CLI tool by parsing --help."""
    tool_name = arguments.get("tool_name")
    output_format = arguments.get("format", "yaml")

    try:
        from blackbox.ai.adapter_factory import AIAdapterFactory

        factory = AIAdapterFactory()
        result_path = factory.learn(tool_name, output_format=output_format)

        if result_path:
            return f"""✅ Successfully learned {tool_name}
Adapter saved to: {result_path}

You can now use this adapter in BBX workflows."""
        else:
            return f"❌ Failed to learn {tool_name}. Tool may not be installed or have no --help."

    except Exception as e:
        return f"❌ Learn failed: {str(e)}"


async def handle_bbx_learned_tools(arguments: Dict[str, Any]) -> str:
    """List all learned CLI tools."""
    try:
        from blackbox.ai.adapter_factory import AIAdapterFactory

        factory = AIAdapterFactory()
        tools = factory.list_learned_tools()

        if not tools:
            return "No tools learned yet.\n\nLearn a tool with: bbx_learn(tool_name='docker')"

        result = "🧠 Learned CLI Tools\n" + "=" * 60 + "\n"
        for tool in tools:
            result += f"  - {tool}\n"
        result += f"\nTotal: {len(tools)} tools"
        return result

    except Exception as e:
        return f"Error listing learned tools: {str(e)}"


async def handle_bbx_system(arguments: Dict[str, Any]) -> str:
    """Check BBX system health."""
    import subprocess
    import sys

    result = "🔧 BBX System Health Check\n" + "=" * 60 + "\n"

    # Python
    result += f"\nPython: {sys.version.split()[0]}\n"

    # Docker
    result += "\nDocker: "
    try:
        proc = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if proc.returncode == 0:
            result += f"✅ {proc.stdout.strip()}\n"

            daemon = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if daemon.returncode == 0:
                result += "  Daemon: ✅ Running\n"
            else:
                result += "  Daemon: ❌ Not running\n"
        else:
            result += "❌ Not installed\n"
    except FileNotFoundError:
        result += "❌ Not found\n"

    # Adapters
    try:
        from blackbox.core.registry import registry
        adapters = registry.list_adapters()
        result += f"\nAdapters: {len(adapters)} registered\n"
    except Exception:
        result += "\nAdapters: ⚠️ Could not load registry\n"

    # MCP
    try:
        from blackbox.mcp.client.config import load_mcp_config
        mcp_configs = load_mcp_config()
        result += f"MCP Servers: {len(mcp_configs)} configured\n"
    except Exception:
        result += "MCP Servers: ⚠️ Could not load config\n"

    return result


async def handle_bbx_adapters(arguments: Dict[str, Any]) -> str:
    """List all available BBX adapters."""
    try:
        from blackbox.core.registry import (
            CORE_ADAPTERS,
            OPTIONAL_ADAPTERS,
            registry,
        )

        result = "📦 BBX Adapters\n" + "=" * 60 + "\n"

        result += "\n🔷 Core Adapters (zero dependencies):\n"
        for name, desc in CORE_ADAPTERS.items():
            result += f"  bbx.{name} - {desc}\n"

        result += "\n🔶 Optional Adapters:\n"
        for name, desc in OPTIONAL_ADAPTERS.items():
            result += f"  bbx.{name} - {desc}\n"

        result += "\n🔗 MCP Adapter:\n"
        result += "  mcp - Connect to external MCP servers\n"

        all_adapters = registry.list_adapters()
        result += f"\n{'=' * 60}\n"
        result += f"Total registered: {len(all_adapters)}\n"

        return result

    except Exception as e:
        return f"Error listing adapters: {str(e)}"


async def handle_bbx_schema(arguments: Dict[str, Any]) -> str:
    """Generate JSON Schema for BBX workflow files."""
    import json
    output_file = arguments.get("output_file", "bbx.schema.json")

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "BBX Workflow",
        "description": "BBX Workflow Definition v6.0",
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Workflow identifier"},
            "name": {"type": "string", "description": "Workflow name"},
            "version": {"type": "string", "description": "Workflow version"},
            "description": {"type": "string"},
            "steps": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "use": {"type": "string", "description": "adapter.method (e.g., mcp.test.echo)"},
                        "args": {"type": "object", "description": "Step arguments"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "when": {"type": "string", "description": "Conditional expression"},
                        "timeout": {"type": "string", "description": "Timeout (e.g., '30s')"},
                        "retry": {"type": "integer", "description": "Retry count"},
                    },
                    "required": ["use"],
                },
            },
        },
        "required": ["steps"],
    }

    try:
        with open(output_file, "w") as f:
            json.dump(schema, f, indent=2)
        return f"✅ JSON Schema generated: {output_file}\n\nAdd to VS Code settings:\n\"yaml.schemas\": {{\"{output_file}\": \"*.bbx\"}}"
    except Exception as e:
        return f"❌ Failed to generate schema: {str(e)}"


async def handle_bbx_mcp_test(arguments: Dict[str, Any]) -> str:
    """Test connection to an MCP server."""
    server = arguments.get("server")

    try:
        from blackbox.mcp.client.manager import get_mcp_manager

        manager = get_mcp_manager()
        await manager.load_config()
        result = await manager.test_connection(server)

        if result.get("status") == "success":
            return f"""✅ Connection successful: {server}
Tools available: {result.get('tools_count', 0)}"""
        else:
            return f"❌ Connection failed: {result.get('error')}"

    except Exception as e:
        return f"❌ Test failed: {str(e)}"


async def handle_bbx_mcp_tools(arguments: Dict[str, Any]) -> str:
    """List tools on a specific MCP server."""
    server = arguments.get("server")

    try:
        from blackbox.mcp.client.manager import get_mcp_manager

        manager = get_mcp_manager()
        await manager.load_config()
        tools = await manager.list_tools(server)

        result = f"🔧 Tools on [{server}]\n" + "=" * 60 + "\n"
        for name, tool in tools.items():
            desc = getattr(tool, "description", "No description")[:60]
            result += f"  {name} - {desc}\n"
        result += f"\nTotal: {len(tools)} tools"
        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


async def handle_bbx_mcp_tool_schema(arguments: Dict[str, Any]) -> str:
    """Show input schema for an MCP tool."""
    import json
    server = arguments.get("server")
    tool_name = arguments.get("tool")

    try:
        from blackbox.mcp.client.manager import get_mcp_manager

        manager = get_mcp_manager()
        await manager.load_config()
        conn = await manager.get_connection(server)
        tool = conn.tools.get(tool_name)

        if not tool:
            return f"Tool '{tool_name}' not found on {server}"

        result = f"[{server}.{tool_name}]\n"
        if hasattr(tool, "description"):
            result += f"Description: {tool.description}\n"
        if hasattr(tool, "inputSchema"):
            result += f"\nInput Schema:\n{json.dumps(tool.inputSchema, indent=2)}"
        return result

    except Exception as e:
        return f"❌ Error: {str(e)}"


async def handle_bbx_model_list(arguments: Dict[str, Any]) -> str:
    """List available AI models."""
    try:
        from blackbox.ai.models import list_models

        models = list_models()

        result = "🤖 Available AI Models\n" + "=" * 60 + "\n"
        for model in models:
            status = "✅ Downloaded" if model.get("downloaded") else "⬇️ Available"
            result += f"\n{model['name']} ({status})\n"
            result += f"  Size: {model.get('size', 'N/A')}\n"
            result += f"  Description: {model.get('description', 'N/A')}\n"
        return result

    except ImportError:
        return "⚠️ AI models module not available. Install with: pip install transformers torch"
    except Exception as e:
        return f"❌ Error: {str(e)}"


async def handle_bbx_model_download(arguments: Dict[str, Any]) -> str:
    """Download an AI model."""
    model_name = arguments.get("model_name")

    try:
        from blackbox.ai.models import download_model

        result = download_model(model_name)
        if result:
            return f"✅ Model '{model_name}' downloaded successfully"
        else:
            return f"❌ Failed to download model '{model_name}'"

    except ImportError:
        return "⚠️ AI models module not available. Install with: pip install transformers torch"
    except Exception as e:
        return f"❌ Download failed: {str(e)}"


async def handle_bbx_model_remove(arguments: Dict[str, Any]) -> str:
    """Remove a downloaded AI model."""
    model_name = arguments.get("model_name")

    try:
        from blackbox.ai.models import remove_model

        result = remove_model(model_name)
        if result:
            return f"✅ Model '{model_name}' removed"
        else:
            return f"❌ Model '{model_name}' not found"

    except ImportError:
        return "⚠️ AI models module not available"
    except Exception as e:
        return f"❌ Remove failed: {str(e)}"


async def handle_bbx_version_create(arguments: Dict[str, Any]) -> str:
    """Create a new version of a workflow."""
    workflow_file = arguments.get("workflow_file")
    version = arguments.get("version")
    message = arguments.get("message", "")

    try:
        from blackbox.core.versioning import create_version

        result = create_version(workflow_file, version, message)
        if result:
            return f"✅ Version {version} created for {workflow_file}"
        else:
            return f"❌ Failed to create version"

    except ImportError:
        # Fallback: simple file copy
        import shutil
        base = Path(workflow_file).stem
        version_file = f"{base}_v{version}.bbx"
        shutil.copy(workflow_file, version_file)
        return f"✅ Version saved as: {version_file}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


async def handle_bbx_version_list(arguments: Dict[str, Any]) -> str:
    """List versions of a workflow."""
    workflow_id = arguments.get("workflow_id")

    try:
        from blackbox.core.versioning import list_versions

        versions = list_versions(workflow_id)

        if not versions:
            return f"No versions found for {workflow_id}"

        result = f"📋 Versions of {workflow_id}\n" + "=" * 60 + "\n"
        for v in versions:
            result += f"  {v['version']} - {v.get('message', 'No message')} ({v.get('date', 'N/A')})\n"
        return result

    except ImportError:
        # Fallback: search for version files
        files = list(Path(".").glob(f"*{workflow_id}*_v*.bbx"))
        if not files:
            return f"No version files found for {workflow_id}"
        result = f"Version files found:\n"
        for f in files:
            result += f"  - {f}\n"
        return result
    except Exception as e:
        return f"❌ Error: {str(e)}"


async def handle_bbx_version_rollback(arguments: Dict[str, Any]) -> str:
    """Rollback to a previous version."""
    workflow_id = arguments.get("workflow_id")
    target_version = arguments.get("target_version")

    try:
        from blackbox.core.versioning import rollback_version

        result = rollback_version(workflow_id, target_version)
        if result:
            return f"✅ Rolled back {workflow_id} to version {target_version}"
        else:
            return f"❌ Rollback failed"

    except ImportError:
        return "⚠️ Versioning module not available. Use bbx_version_list to find version files manually."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# === Workspace Handlers ===

async def handle_bbx_workspace_create(arguments: Dict[str, Any]) -> str:
    """Create a new workspace."""
    from blackbox.core.workspace_manager import get_workspace_manager

    name = arguments.get("name")
    description = arguments.get("description", "")
    path = arguments.get("path")

    try:
        manager = get_workspace_manager()
        ws = manager.create(
            name=name,
            description=description,
            path=Path(path) if path else None,
        )

        return f"""✅ Created workspace: {name}
Path: {ws.root}

Structure:
  main.bbx      - Entry point
  config.yaml   - Configuration
  state/        - Persistent state
  logs/         - Execution logs
  workflows/    - Sub-workflows

Activate: bbx_workspace_set(path="{ws.root}")"""

    except Exception as e:
        return f"❌ Error: {str(e)}"


async def handle_bbx_workspace_list(arguments: Dict[str, Any]) -> str:
    """List all workspaces."""
    from blackbox.core.workspace_manager import get_workspace_manager

    directory = arguments.get("directory")

    manager = get_workspace_manager()
    workspaces = manager.list(Path(directory) if directory else None)

    if not workspaces:
        return "No workspaces found.\n\nCreate one: bbx_workspace_create(name='my-project')"

    result = "📁 Workspaces\n" + "=" * 60 + "\n"

    current = manager.get_current_path()
    for ws in workspaces:
        is_current = current and Path(ws["path"]).resolve() == current.resolve()
        marker = " ← current" if is_current else ""
        result += f"\n{ws['name']}{marker}\n"
        result += f"   Path: {ws['path']}\n"
        result += f"   Updated: {ws['updated_at']}\n"

    return result


async def handle_bbx_workspace_set(arguments: Dict[str, Any]) -> str:
    """Set current workspace."""
    from blackbox.core.workspace_manager import get_workspace_manager

    path = arguments.get("path")

    try:
        manager = get_workspace_manager()
        ws = manager.set_current(Path(path))

        return f"""✅ Active workspace: {ws.metadata.name}
Path: {ws.root}

You can now run: bbx_run(workflow_file="main.bbx")"""

    except ValueError as e:
        return f"❌ {str(e)}"


async def handle_bbx_workspace_info(arguments: Dict[str, Any]) -> str:
    """Show workspace info."""
    from blackbox.core.workspace_manager import get_workspace_manager

    path = arguments.get("path")

    manager = get_workspace_manager()

    try:
        info = manager.info(Path(path) if path else None)

        result = f"""📋 Workspace: {info['name']}
{'=' * 60}
ID: {info['workspace_id']}
Path: {info['root']}
Description: {info['description'] or 'N/A'}
Created: {info['created_at']}
Updated: {info['updated_at']}

Workflows: {info['workflows_count']}
Runs: {info['runs_count']}
State keys: {len(info['state_keys'])}"""

        if not info['valid']:
            result += f"\n\n⚠️ Issues: {', '.join(info['issues'])}"

        return result

    except ValueError as e:
        return f"❌ {str(e)}"


async def handle_bbx_workspace_clear(arguments: Dict[str, Any]) -> str:
    """Clear current workspace."""
    from blackbox.core.workspace_manager import get_workspace_manager

    manager = get_workspace_manager()
    manager.clear_current()
    return "✅ Cleared current workspace context."


# === Process Management Handlers ===

async def handle_bbx_ps(arguments: Dict[str, Any]) -> str:
    """List executions."""
    from blackbox.core.execution_manager import get_execution_manager

    show_all = arguments.get("all", False)

    manager = get_execution_manager()
    executions = await manager.ps(all=show_all)

    if not executions:
        if show_all:
            return "No executions found."
        return "No running executions.\n\nUse all=True to see all executions."

    result = "📊 Executions\n" + "=" * 80 + "\n"
    result += f"{'EXEC_ID':<12} {'STATUS':<12} {'WORKFLOW':<30} {'STARTED':<20}\n"
    result += "=" * 80 + "\n"

    for exec in executions:
        exec_id = exec.execution_id[:8] + "..."
        status = exec.status.value
        workflow = exec.workflow_path[-30:] if len(exec.workflow_path) > 30 else exec.workflow_path
        started = exec.started_at.strftime("%Y-%m-%d %H:%M:%S") if exec.started_at else "pending"
        result += f"{exec_id:<12} {status:<12} {workflow:<30} {started:<20}\n"

    return result


async def handle_bbx_kill(arguments: Dict[str, Any]) -> str:
    """Kill an execution."""
    from blackbox.core.execution_manager import get_execution_manager
    from blackbox.core.execution_store import get_execution_store

    execution_id = arguments.get("execution_id")
    force = arguments.get("force", False)

    # Find by partial ID
    store = get_execution_store()
    executions = store.list(limit=100)
    matching = [e for e in executions if e.execution_id.startswith(execution_id)]

    if not matching:
        return f"❌ No execution found matching: {execution_id}"

    if len(matching) > 1:
        ids = "\n".join(f"  - {e.execution_id}" for e in matching)
        return f"Multiple matches:\n{ids}\n\nPlease be more specific."

    full_id = matching[0].execution_id

    manager = get_execution_manager()
    result = await manager.kill(full_id, force=force)

    if result:
        return f"✅ Killed execution: {full_id[:12]}..."
    return f"❌ Could not kill execution (may have already finished)"


async def handle_bbx_wait(arguments: Dict[str, Any]) -> str:
    """Wait for execution."""
    import json
    from blackbox.core.execution_manager import get_execution_manager
    from blackbox.core.execution_store import get_execution_store

    execution_id = arguments.get("execution_id")
    timeout = arguments.get("timeout")

    # Find by partial ID
    store = get_execution_store()
    executions = store.list(limit=100)
    matching = [e for e in executions if e.execution_id.startswith(execution_id)]

    if not matching:
        return f"❌ No execution found matching: {execution_id}"

    full_id = matching[0].execution_id

    manager = get_execution_manager()
    result = await manager.wait(full_id, timeout=float(timeout) if timeout else None)

    if result:
        output = f"✅ Execution {result.status.value}\n"
        if result.outputs:
            output += f"\nOutputs:\n{json.dumps(result.outputs, indent=2, default=str)}"
        if result.error:
            output += f"\n\n❌ Error: {result.error}"
        return output

    return f"⏱️ Timeout waiting for execution"


async def handle_bbx_logs(arguments: Dict[str, Any]) -> str:
    """Get execution logs."""
    from blackbox.core.execution_store import get_execution_store

    execution_id = arguments.get("execution_id")
    limit = arguments.get("limit", 50)
    level = arguments.get("level")

    # Find by partial ID
    store = get_execution_store()
    executions = store.list(limit=100)
    matching = [e for e in executions if e.execution_id.startswith(execution_id)]

    if not matching:
        return f"❌ No execution found matching: {execution_id}"

    full_id = matching[0].execution_id
    logs = store.get_logs(full_id, limit=limit, level=level)

    if not logs:
        return f"No logs for execution {full_id[:12]}..."

    result = f"📜 Logs for {full_id[:12]}...\n" + "-" * 60 + "\n"

    for log in logs:
        ts = log["timestamp"].split("T")[1].split(".")[0] if "T" in log["timestamp"] else log["timestamp"]
        lvl = log["level"]
        msg = log["message"]
        step = f"[{log['step_id']}] " if log.get("step_id") else ""
        result += f"{ts} | {lvl:<5} | {step}{msg}\n"

    return result


async def handle_bbx_run_background(arguments: Dict[str, Any]) -> str:
    """Run workflow in background."""
    from blackbox.core.execution_manager import get_execution_manager

    workflow_file = arguments.get("workflow_file")
    inputs = arguments.get("inputs", {})

    if not os.path.exists(workflow_file):
        return f"❌ File not found: {workflow_file}"

    manager = get_execution_manager()
    exec_id = await manager.run_background(workflow_file, inputs)

    return f"""✅ Started background execution
Execution ID: {exec_id}

Monitor:
  bbx_ps()
  bbx_logs(execution_id="{exec_id[:8]}...")
  bbx_wait(execution_id="{exec_id[:8]}...")
  bbx_kill(execution_id="{exec_id[:8]}...")"""


# === State Management Handlers ===

async def handle_bbx_state_get(arguments: Dict[str, Any]) -> str:
    """Get state value."""
    import json
    from blackbox.core.adapters.state import StateAdapter

    adapter = StateAdapter()
    result = await adapter._get(
        key=arguments.get("key"),
        default=arguments.get("default"),
        namespace=arguments.get("namespace"),
    )

    if result["found"]:
        value = result["value"]
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        return str(value)
    elif arguments.get("default"):
        return arguments.get("default")
    return f"❌ Key not found: {arguments.get('key')}"


async def handle_bbx_state_set(arguments: Dict[str, Any]) -> str:
    """Set state value."""
    from blackbox.core.adapters.state import StateAdapter

    adapter = StateAdapter()
    result = await adapter._set(
        key=arguments.get("key"),
        value=arguments.get("value"),
        namespace=arguments.get("namespace"),
    )

    return f"✅ Set {result['key']} = {result['value']}"


async def handle_bbx_state_list(arguments: Dict[str, Any]) -> str:
    """List state keys."""
    import json
    from blackbox.core.adapters.state import StateAdapter

    adapter = StateAdapter()
    result = await adapter._keys(
        pattern=arguments.get("pattern", "*"),
        namespace=arguments.get("namespace"),
    )

    if not result["keys"]:
        return f"No keys matching pattern: {result['pattern']}"

    output = "📋 State Variables\n" + "=" * 60 + "\n"

    # Get all state to show values
    all_state = await adapter._all(namespace=arguments.get("namespace"))

    for key in result["keys"]:
        value = all_state["state"].get(key)
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value)[:40] + "..." if len(json.dumps(value)) > 40 else json.dumps(value)
        else:
            value_str = str(value)[:40] + "..." if len(str(value)) > 40 else str(value)
        output += f"  {key} = {value_str}\n"

    output += f"\nTotal: {result['count']} keys"
    return output


async def handle_bbx_state_delete(arguments: Dict[str, Any]) -> str:
    """Delete state key."""
    from blackbox.core.adapters.state import StateAdapter

    adapter = StateAdapter()
    result = await adapter._delete(
        key=arguments.get("key"),
        namespace=arguments.get("namespace"),
    )

    if result["deleted"]:
        return f"✅ Deleted: {result['key']}"
    return f"❌ Key not found: {arguments.get('key')}"


async def handle_bbx_state_increment(arguments: Dict[str, Any]) -> str:
    """Increment state value."""
    from blackbox.core.adapters.state import StateAdapter

    adapter = StateAdapter()
    result = await adapter._increment(
        key=arguments.get("key"),
        by=arguments.get("by", 1),
        namespace=arguments.get("namespace"),
    )

    if "error" in result:
        return f"❌ {result['error']}"

    return f"✅ {result['key']}: {result['old_value']} → {result['new_value']}"


async def handle_bbx_state_append(arguments: Dict[str, Any]) -> str:
    """Append to state list."""
    from blackbox.core.adapters.state import StateAdapter

    adapter = StateAdapter()
    result = await adapter._append(
        key=arguments.get("key"),
        value=arguments.get("value"),
        max_items=arguments.get("max_items"),
        namespace=arguments.get("namespace"),
    )

    if "error" in result:
        return f"❌ {result['error']}"

    return f"✅ Appended to {result['key']} (length: {result['list_length']})"


# === Nested Workflow Handlers ===

async def handle_bbx_workflow_run(arguments: Dict[str, Any]) -> str:
    """Run nested workflow."""
    import json
    from blackbox.core.adapters.workflow import WorkflowAdapter

    adapter = WorkflowAdapter()
    result = await adapter._run(
        path=arguments.get("path"),
        workflow_inputs=arguments.get("inputs"),
        background=arguments.get("background", False),
        timeout=arguments.get("timeout"),
    )

    if result["status"] == "error":
        return f"❌ {result['error']}"

    if result["status"] == "started":
        return f"""✅ Started background workflow
Execution ID: {result['execution_id']}
Path: {result['workflow_path']}"""

    if result["status"] == "completed":
        output = f"✅ Workflow completed: {result['workflow_path']}"
        if result.get("outputs"):
            output += f"\n\nOutputs:\n{json.dumps(result['outputs'], indent=2, default=str)}"
        return output

    if result["status"] == "timeout":
        return f"⏱️ {result['error']}"

    if result["status"] == "failed":
        return f"❌ Workflow failed: {result['error']}"

    return f"Status: {result['status']}"


async def handle_bbx_workflow_status(arguments: Dict[str, Any]) -> str:
    """Get workflow status."""
    import json
    from blackbox.core.adapters.workflow import WorkflowAdapter

    adapter = WorkflowAdapter()
    result = await adapter._status(execution_id=arguments.get("execution_id"))

    if result.get("status") == "not_found":
        return f"❌ Execution not found: {arguments.get('execution_id')}"

    return json.dumps(result, indent=2, default=str)


# Map tool names to handlers
TOOL_HANDLERS = {
    # Core Workflow
    "bbx_generate": handle_bbx_generate,
    "bbx_validate": handle_bbx_validate,
    "bbx_run": handle_bbx_run,
    "bbx_info": handle_bbx_info,
    "bbx_list_workflows": handle_bbx_list_workflows,
    "bbx_schema": handle_bbx_schema,
    # MCP
    "bbx_mcp_discover": handle_bbx_mcp_discover,
    "bbx_mcp_call": handle_bbx_mcp_call,
    "bbx_mcp_servers": handle_bbx_mcp_servers,
    "bbx_mcp_test": handle_bbx_mcp_test,
    "bbx_mcp_tools": handle_bbx_mcp_tools,
    "bbx_mcp_tool_schema": handle_bbx_mcp_tool_schema,
    # Learn
    "bbx_learn": handle_bbx_learn,
    "bbx_learned_tools": handle_bbx_learned_tools,
    # System
    "bbx_system": handle_bbx_system,
    "bbx_adapters": handle_bbx_adapters,
    # Models
    "bbx_model_list": handle_bbx_model_list,
    "bbx_model_download": handle_bbx_model_download,
    "bbx_model_remove": handle_bbx_model_remove,
    # Versioning
    "bbx_version_create": handle_bbx_version_create,
    "bbx_version_list": handle_bbx_version_list,
    "bbx_version_rollback": handle_bbx_version_rollback,
    # Workspace (OS-like)
    "bbx_workspace_create": handle_bbx_workspace_create,
    "bbx_workspace_list": handle_bbx_workspace_list,
    "bbx_workspace_set": handle_bbx_workspace_set,
    "bbx_workspace_info": handle_bbx_workspace_info,
    "bbx_workspace_clear": handle_bbx_workspace_clear,
    # Process Management (Linux-like)
    "bbx_ps": handle_bbx_ps,
    "bbx_kill": handle_bbx_kill,
    "bbx_wait": handle_bbx_wait,
    "bbx_logs": handle_bbx_logs,
    "bbx_run_background": handle_bbx_run_background,
    # State Management (env vars)
    "bbx_state_get": handle_bbx_state_get,
    "bbx_state_set": handle_bbx_state_set,
    "bbx_state_list": handle_bbx_state_list,
    "bbx_state_delete": handle_bbx_state_delete,
    "bbx_state_increment": handle_bbx_state_increment,
    "bbx_state_append": handle_bbx_state_append,
    # Nested Workflows
    "bbx_workflow_run": handle_bbx_workflow_run,
    "bbx_workflow_status": handle_bbx_workflow_status,
}
