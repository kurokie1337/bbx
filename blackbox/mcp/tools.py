"""
BBX MCP Tools - BBX workflow operations exposed as MCP tools
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from blackbox.ai.generator import WorkflowGenerator
from blackbox.core.parser import parse_workflow
from blackbox.core.runner import WorkflowRunner
from blackbox.core.validator import validate_workflow


def get_bbx_tools() -> List[Dict[str, Any]]:
    """
    Get list of BBX tools for MCP server.

    Returns:
        List of MCP tool definitions
    """
    return [
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
            "description": "Execute a BBX workflow (runs all steps in order)",
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
                    "category": {
                        "type": "string",
                        "description": "Filter by category: 'core', 'user', 'meta', or 'all'",
                        "enum": ["core", "user", "meta", "all"],
                        "default": "all",
                    }
                },
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
    description = arguments.get("description")
    output_file = arguments.get("output_file", "generated.bbx")

    try:
        generator = WorkflowGenerator()
        yaml_content = generator.generate(
            description=description, output_file=output_file
        )

        return f"""✅ Workflow generated successfully!

File: {output_file}

Content:
{yaml_content}

Next steps:
1. Validate: bbx validate {output_file}
2. Run: bbx run {output_file}
"""
    except Exception as e:
        return f"❌ Generation failed: {str(e)}"


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
        return f"❌ File not found: {workflow_file}"

    try:
        # Parse workflow
        with open(workflow_file, "r", encoding="utf-8") as f:
            content = f.read()

        workflow = parse_workflow(content)

        # Validate
        errors = validate_workflow(workflow)

        if errors:
            error_list = "\n".join(f"  - {err}" for err in errors)
            return f"❌ Validation failed:\n{error_list}"
        else:
            return f"✅ Workflow is valid: {workflow_file}"

    except Exception as e:
        return f"❌ Validation error: {str(e)}"


async def handle_bbx_run(arguments: Dict[str, Any]) -> str:
    """
    Handle bbx_run tool call.

    Args:
        arguments: Tool arguments (workflow_file, dry_run)

    Returns:
        Execution result
    """
    workflow_file = arguments.get("workflow_file")
    dry_run = arguments.get("dry_run", False)

    if not os.path.exists(workflow_file):
        return f"❌ File not found: {workflow_file}"

    try:
        # Parse workflow
        with open(workflow_file, "r", encoding="utf-8") as f:
            content = f.read()

        workflow = parse_workflow(content)

        # Validate first
        errors = validate_workflow(workflow)
        if errors:
            error_list = "\n".join(f"  - {err}" for err in errors)
            return f"❌ Cannot run: workflow has validation errors:\n{error_list}"

        if dry_run:
            # Show what would run
            steps_info = "\n".join(
                f"  {i+1}. {step.get('id', 'unnamed')} - {step.get('mcp', 'unknown')} adapter"
                for i, step in enumerate(workflow.get("steps", []))
            )
            return f"""🔍 Dry run: {workflow_file}

Workflow: {workflow.get('id', 'unnamed')}
Steps ({len(workflow.get('steps', []))}):{steps_info}

To execute: bbx run {workflow_file}
"""
        else:
            # Actually run
            runner = WorkflowRunner(workflow)
            result = runner.run()

            if result.get("success"):
                return f"✅ Workflow executed successfully: {workflow_file}"
            else:
                return f"❌ Workflow failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"❌ Execution error: {str(e)}"


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


# Map tool names to handlers
TOOL_HANDLERS = {
    "bbx_generate": handle_bbx_generate,
    "bbx_validate": handle_bbx_validate,
    "bbx_run": handle_bbx_run,
    "bbx_list_workflows": handle_bbx_list_workflows,
}
