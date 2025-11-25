# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""BBX CLI - Command Line Interface for Blackbox Workflow Engine"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import yaml

# Force UTF-8 encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from blackbox.core.config import get_settings
from blackbox.core.runtime import run_file
from blackbox.core.schema import BBXSchemaGenerator

settings = get_settings()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, settings.observability.log_level, logging.INFO),
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """BBX - Blackbox Workflow Engine CLI"""
    pass


# =============================================================================
# Core Commands
# =============================================================================

@cli.command()
@click.argument("file_path", type=click.Path(exists=True), required=False)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.option("--input", "-i", "inputs", multiple=True, help="Inputs in key=value format")
@click.option("--background", "-b", is_flag=True, help="Run in background (like command &)")
def run(file_path: str, verbose: bool, output: str, inputs: tuple, background: bool):
    """Run a workflow from a .bbx file.

    If no file specified and in a workspace, runs main.bbx.

    With --background flag, runs like 'command &' in Linux:
      bbx run deploy.bbx --background
      > Execution ID: 550e8400-e29b...
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # If no file specified, try to use workspace main.bbx
    if not file_path:
        from blackbox.core.workspace_manager import get_current_workspace
        ws = get_current_workspace()
        if ws and ws.paths.main_bbx.exists():
            file_path = str(ws.paths.main_bbx)
            click.echo(f"Running workspace main.bbx: {file_path}")
        else:
            click.echo("No workflow file specified and no workspace active.", err=True)
            click.echo("Either specify a file or use 'bbx workspace set <path>'")
            raise click.Abort()

    # Parse inputs
    input_dict = {}
    for input_str in inputs:
        if "=" in input_str:
            key, value = input_str.split("=", 1)
            if value.lower() in ("true", "false"):
                input_dict[key] = value.lower() == "true"
            elif value.isdigit():
                input_dict[key] = int(value)
            else:
                input_dict[key] = value

    try:
        if background:
            # Background execution (like command &)
            from blackbox.core.execution_manager import get_execution_manager

            async def _run_bg():
                manager = get_execution_manager()
                return await manager.run_background(file_path, input_dict)

            exec_id = asyncio.run(_run_bg())
            click.echo(f"Started background execution")
            click.echo(f"Execution ID: {exec_id}")
            click.echo(f"\nCheck status: bbx ps")
            click.echo(f"View logs:    bbx logs {exec_id[:8]}... --follow")
            click.echo(f"Stop:         bbx kill {exec_id[:8]}...")
        else:
            # Normal foreground execution
            results = asyncio.run(run_file(file_path, inputs=input_dict))

            if output == "json":
                click.echo(json.dumps(results, indent=2, default=str))
            else:
                click.echo("\n" + "=" * 60)
                click.echo("Workflow Results")
                click.echo("=" * 60)

                for step_id, result in results.items():
                    status = result.get("status", "unknown")
                    icon = "+" if status == "success" else "-" if status == "error" else "?"
                    click.echo(f"\n[{icon}] Step: {step_id}")
                    click.echo(f"    Status: {status}")
                    if result.get("output"):
                        click.echo(f"    Output: {result['output']}")
                    if result.get("error"):
                        click.echo(f"    Error: {result['error']}", err=True)

                click.echo("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=verbose)
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def validate(file_path: str):
    """Validate a workflow file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Basic validation
        if "workflow" in data:
            workflow = data["workflow"]
        else:
            workflow = data

        errors = []

        if "steps" not in workflow:
            errors.append("Missing 'steps' field")
        elif not isinstance(workflow["steps"], list):
            errors.append("'steps' must be a list")
        else:
            for i, step in enumerate(workflow["steps"]):
                if "id" not in step:
                    errors.append(f"Step {i} missing 'id'")
                if "mcp" not in step and "adapter" not in step:
                    errors.append(f"Step {i} missing 'mcp' or 'adapter'")

        if errors:
            click.echo("Validation failed:")
            for error in errors:
                click.echo(f"  - {error}", err=True)
            raise click.Abort()

        click.echo(f"Workflow is valid: {file_path}")

    except yaml.YAMLError as e:
        click.echo(f"YAML parse error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def info(file_path: str):
    """Show information about a workflow."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if "workflow" in data:
            workflow = data["workflow"]
        else:
            workflow = data

        click.echo("\n" + "=" * 60)
        click.echo("Workflow Information")
        click.echo("=" * 60)
        click.echo(f"ID: {workflow.get('id', 'N/A')}")
        click.echo(f"Name: {workflow.get('name', 'N/A')}")
        click.echo(f"Version: {workflow.get('version', 'N/A')}")
        click.echo(f"Description: {workflow.get('description', 'N/A')}")

        steps = workflow.get("steps", [])
        click.echo(f"\nSteps: {len(steps)}")
        for step in steps:
            step_id = step.get("id", "unknown")
            adapter = step.get("mcp", step.get("adapter", "unknown"))
            method = step.get("method", "")
            click.echo(f"  - {step_id} ({adapter}.{method})")

        click.echo("=" * 60)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--output", "-o", default="bbx.schema.json", help="Output file")
def schema(output: str):
    """Generate JSON Schema for VS Code."""
    schema_data = BBXSchemaGenerator.generate()
    with open(output, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=2)
    click.echo(f"Schema generated: {output}")


@cli.command("version")
def show_version():
    """Show BBX version."""
    click.echo("BBX - Blackbox Workflow Engine v1.0.0")
    click.echo("License: BSL 1.1 (converts to Apache 2.0 on 2028-11-05)")


# =============================================================================
# AI Commands
# =============================================================================

@cli.group()
def model():
    """Manage AI models for workflow generation"""
    pass


@model.command("download")
@click.argument("model_name", default="qwen-0.5b")
def model_download(model_name: str):
    """Download an AI model."""
    from blackbox.ai.model_manager import ModelManager

    try:
        manager = ModelManager()
        manager.download(model_name)
    except Exception as e:
        click.echo(f"Download failed: {e}", err=True)
        sys.exit(1)


@model.command("list")
def model_list():
    """List available AI models."""
    from blackbox.ai.model_manager import ModelManager

    manager = ModelManager()
    models = manager.list_available()

    click.echo("\nAvailable AI Models:\n")
    for model_name, info in models.items():
        status = "[INSTALLED]" if info["installed"] else "[available]"
        click.echo(f"{status} {model_name} ({info['size']})")
        click.echo(f"   {info['description']}")


@model.command("remove")
@click.argument("model_name")
def model_remove(model_name: str):
    """Remove an AI model."""
    from blackbox.ai.model_manager import ModelManager

    try:
        manager = ModelManager()
        manager.remove(model_name)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("description")
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--output", "-o", default="generated.bbx", help="Output file")
def generate(description: str, model: str, output: str):
    """Generate workflow from natural language using local AI."""
    from blackbox.ai.generator import WorkflowGenerator, check_dependencies
    from blackbox.ai.model_manager import ModelManager

    if not check_dependencies():
        click.echo("\nInstall llama-cpp-python:")
        click.echo("   pip install llama-cpp-python")
        sys.exit(1)

    try:
        manager = ModelManager()

        if model is None:
            model = manager.load_default_model_from_config()

        try:
            manager.get_model_path(model)
        except FileNotFoundError:
            click.echo(f"Model '{model}' not downloaded")
            click.echo(f"Download with: bbx model download {model}")
            sys.exit(1)

        click.echo(f"\nUsing model: {model}")
        click.echo(f"Task: {description}\n")

        generator = WorkflowGenerator(model_name=model)
        yaml_content = generator.generate(description, output_file=output)

        click.echo(f"\nGenerated: {output}")
        click.echo("\nPreview:")
        click.echo("-" * 60)
        for line in yaml_content.split("\n")[:15]:
            click.echo(line)
        click.echo("-" * 60)

    except Exception as e:
        click.echo(f"Generation failed: {e}", err=True)
        sys.exit(1)


# =============================================================================
# MCP Server
# =============================================================================

@cli.command("mcp-serve")
def mcp_serve():
    """Start BBX MCP Server for AI agent integration."""
    from blackbox.mcp.server import main as run_mcp_server

    click.echo("Starting BBX MCP Server...")
    click.echo("Protocol: Model Context Protocol (stdio)")
    click.echo("\nAvailable tools:")
    click.echo("  - bbx_generate: Generate workflows")
    click.echo("  - bbx_validate: Validate workflows")
    click.echo("  - bbx_run: Execute workflows")
    click.echo("\nServer ready.\n")

    try:
        run_mcp_server()
    except KeyboardInterrupt:
        click.echo("\nMCP Server stopped")
    except ImportError as e:
        click.echo(f"\nMissing MCP dependencies: {e}", err=True)
        click.echo("Install with: pip install mcp")
        sys.exit(1)


# =============================================================================
# Versioning Commands
# =============================================================================

@cli.group("version-ctrl")
def version_ctrl():
    """Workflow versioning commands."""
    pass


@version_ctrl.command("create")
@click.argument("workflow_path")
@click.option("--version", "-v", required=True, help="Version number")
@click.option("--message", "-m", help="Version description")
def version_create(workflow_path: str, version: str, message: str):
    """Create a new workflow version."""
    from blackbox.core.versioning.manager import VersionManager

    with open(workflow_path, encoding="utf-8") as f:
        content = yaml.safe_load(f)

    workflow_id = content.get("workflow", content).get("id", "unknown")
    manager = VersionManager(Path("~/.bbx/versions").expanduser())

    version_obj = manager.create_version(
        workflow_id=workflow_id,
        content=content,
        version=version,
        created_by="cli",
        description=message,
    )

    click.echo(f"Created version {version_obj.version} for {workflow_id}")


@version_ctrl.command("list")
@click.argument("workflow_id")
def version_list(workflow_id: str):
    """List workflow versions."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    versions = manager.list_versions(workflow_id)

    click.echo(f"Versions for {workflow_id}:")
    for v in versions:
        click.echo(f"  {v.version} - {v.created_at} - {v.description or 'No description'}")


@version_ctrl.command("rollback")
@click.argument("workflow_id")
@click.argument("target_version")
def version_rollback(workflow_id: str, target_version: str):
    """Rollback to a previous version."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    rollback_version = manager.rollback(workflow_id, target_version)

    click.echo(f"Rolled back to {target_version}")
    click.echo(f"New version: {rollback_version.version}")


# =============================================================================
# MCP Client Commands
# =============================================================================

@cli.group("mcp-client")
def mcp_client():
    """Manage external MCP server connections."""
    pass


@mcp_client.command("list")
def mcp_list_servers():
    """List configured MCP servers."""
    from blackbox.mcp.client.config import load_mcp_config

    configs = load_mcp_config()

    if not configs:
        click.echo("No MCP servers configured.")
        click.echo("\nCreate mcp_servers.yaml to add servers:")
        click.echo("  bbx mcp-client init")
        return

    click.echo("\n" + "=" * 60)
    click.echo("Configured MCP Servers")
    click.echo("=" * 60)

    for name, config in configs.items():
        click.echo(f"\n[{name}]")
        click.echo(f"  Transport: {config.transport}")
        if config.command:
            click.echo(f"  Command: {' '.join(config.command)}")
        if config.url:
            click.echo(f"  URL: {config.url}")
        if config.description:
            click.echo(f"  Description: {config.description}")

    click.echo("\n" + "=" * 60)


@mcp_client.command("init")
@click.option("--path", "-p", default="mcp_servers.yaml", help="Config file path")
def mcp_init_config(path: str):
    """Create default mcp_servers.yaml configuration."""
    from blackbox.mcp.client.config import create_default_config

    if Path(path).exists():
        click.echo(f"Config already exists: {path}")
        return

    content = create_default_config()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    click.echo(f"Created: {path}")
    click.echo("Edit the file to configure your MCP servers.")


@mcp_client.command("tools")
@click.argument("server_name")
def mcp_list_tools(server_name: str):
    """List tools available on an MCP server."""
    from blackbox.mcp.client.manager import get_mcp_manager

    async def _list():
        manager = get_mcp_manager()
        await manager.load_config()
        return await manager.test_connection(server_name)

    try:
        result = asyncio.run(_list())

        if result["status"] == "ok":
            click.echo(f"\nTools on {server_name}:")
            for tool in result.get("tools", []):
                click.echo(f"  - {tool}")
            click.echo(f"\nTotal: {result.get('tools_count', 0)} tools")
        else:
            click.echo(f"Error: {result.get('error')}", err=True)
    except Exception as e:
        click.echo(f"Failed to connect: {e}", err=True)


@mcp_client.command("test")
@click.argument("server_name")
def mcp_test_server(server_name: str):
    """Test connection to an MCP server."""
    from blackbox.mcp.client.manager import get_mcp_manager

    async def _test():
        manager = get_mcp_manager()
        await manager.load_config()
        return await manager.test_connection(server_name)

    click.echo(f"Testing connection to {server_name}...")

    try:
        result = asyncio.run(_test())

        if result["status"] == "ok":
            click.echo(f"[+] Connected successfully!")
            click.echo(f"    Tools available: {result.get('tools_count', 0)}")
        else:
            click.echo(f"[-] Connection failed: {result.get('error')}", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


# =============================================================================
# Learn Commands (AIAdapterFactory)
# =============================================================================

@cli.group()
def learn():
    """Auto-learn CLI tools by parsing --help output."""
    pass


@learn.command("tool")
@click.argument("tool_name")
@click.option("--format", "-f", type=click.Choice(["yaml", "python"]), default="yaml")
def learn_tool(tool_name: str, format: str):
    """Learn a CLI tool and generate BBX adapter.

    Example: bbx learn tool kubectl
    """
    from blackbox.ai.adapter_factory import AIAdapterFactory

    factory = AIAdapterFactory()
    result = factory.learn(tool_name, output_format=format)

    if result:
        click.echo(f"\n[+] Successfully learned {tool_name}")
        click.echo(f"    Adapter saved to: {result}")
    else:
        click.echo(f"[-] Failed to learn {tool_name}", err=True)
        sys.exit(1)


@learn.command("list")
def learn_list():
    """List all learned CLI tools."""
    from blackbox.ai.adapter_factory import AIAdapterFactory

    factory = AIAdapterFactory()
    tools = factory.list_learned_tools()

    if not tools:
        click.echo("No tools learned yet.")
        click.echo("\nLearn a tool with: bbx learn tool <tool_name>")
        return

    click.echo("\n" + "=" * 60)
    click.echo("Learned Tools")
    click.echo("=" * 60)

    for tool in tools:
        click.echo(f"  - {tool}")

    click.echo(f"\nTotal: {len(tools)} tools")
    click.echo("=" * 60)


@mcp_client.command("discover")
def mcp_discover_all():
    """Discover all tools from all configured MCP servers."""
    from blackbox.mcp.client.manager import get_mcp_manager

    async def _discover():
        manager = get_mcp_manager()
        await manager.load_config()
        return await manager.list_all_tools()

    click.echo("\n" + "=" * 60)
    click.echo("MCP Tools Discovery")
    click.echo("=" * 60)

    try:
        all_tools = asyncio.run(_discover())

        total_tools = 0
        for server_name, tools in all_tools.items():
            if "error" in tools:
                click.echo(f"\n[-] {server_name}: {tools['error']}")
            else:
                click.echo(f"\n[{server_name}] ({len(tools)} tools)")
                for tool_name, tool_info in tools.items():
                    desc = ""
                    if hasattr(tool_info, "description"):
                        desc = f" - {tool_info.description[:50]}..."
                    click.echo(f"    mcp.{server_name}.{tool_name}{desc}")
                total_tools += len(tools)

        click.echo("\n" + "=" * 60)
        click.echo(f"Total: {total_tools} tools from {len(all_tools)} servers")
        click.echo("=" * 60)

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@mcp_client.command("schema")
@click.argument("server_name")
@click.argument("tool_name")
def mcp_tool_schema(server_name: str, tool_name: str):
    """Show input schema for an MCP tool.

    Example: bbx mcp-client schema test echo
    """
    from blackbox.mcp.client.manager import get_mcp_manager
    import json

    async def _get_schema():
        manager = get_mcp_manager()
        await manager.load_config()
        conn = await manager.get_connection(server_name)
        return conn.tools.get(tool_name)

    try:
        tool = asyncio.run(_get_schema())

        if not tool:
            click.echo(f"Tool '{tool_name}' not found on {server_name}", err=True)
            return

        click.echo(f"\n[{server_name}.{tool_name}]")
        if hasattr(tool, "description"):
            click.echo(f"Description: {tool.description}")
        if hasattr(tool, "inputSchema"):
            click.echo("\nInput Schema:")
            click.echo(json.dumps(tool.inputSchema, indent=2))

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


# =============================================================================
# System Commands
# =============================================================================

@cli.command()
def system():
    """Check system health and Docker status."""
    import subprocess

    click.echo("\n" + "=" * 60)
    click.echo("BBX - Operating System for AI Agents")
    click.echo("System Health Check")
    click.echo("=" * 60)

    # Python
    click.echo(f"\nPython: {sys.version.split()[0]}")

    # Docker
    click.echo("\nDocker:")
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            click.echo(f"   [+] {result.stdout.strip()}")

            daemon = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if daemon.returncode == 0:
                click.echo("   [+] Daemon running")
            else:
                click.echo("   [-] Daemon not running")
        else:
            click.echo("   [-] Not installed")
    except FileNotFoundError:
        click.echo("   [-] Not found")

    # Adapters
    from blackbox.core.registry import registry
    adapters = registry.list_adapters()
    click.echo(f"\nAdapters: {len(adapters)} registered")

    # Current workspace
    from blackbox.core.workspace_manager import get_current_workspace
    ws = get_current_workspace()
    if ws:
        click.echo(f"\nCurrent Workspace: {ws.metadata.name}")
        click.echo(f"   Path: {ws.root}")
    else:
        click.echo("\nCurrent Workspace: None")

    # Running executions
    from blackbox.core.execution_store import get_execution_store, ExecutionStatus
    store = get_execution_store()
    running = store.count(ExecutionStatus.RUNNING)
    click.echo(f"\nRunning Executions: {running}")

    click.echo("\n" + "=" * 60)


# =============================================================================
# Workspace Commands (OS-like environments)
# =============================================================================

@cli.group()
def workspace():
    """Manage workspaces - isolated environments for AI agents.

    Workspaces are like /home directories in Linux.
    Each workspace has its own workflows, state, and logs.

    Examples:
      bbx workspace create my-project
      bbx workspace list
      bbx workspace set ./my-project
      bbx workspace info
    """
    pass


@workspace.command("create")
@click.argument("name")
@click.option("--path", "-p", help="Custom path (default: ~/.bbx/workspaces/<name>)")
@click.option("--description", "-d", default="", help="Workspace description")
def workspace_create(name: str, path: str, description: str):
    """Create a new workspace.

    Creates an isolated environment with:
    - main.bbx: Entry point workflow (empty, for agent to fill)
    - config.yaml: Local configuration
    - state/: Persistent state storage
    - logs/: Execution logs
    - workflows/: Sub-workflows
    """
    from blackbox.core.workspace_manager import get_workspace_manager

    try:
        manager = get_workspace_manager()
        ws = manager.create(
            name=name,
            description=description,
            path=Path(path) if path else None,
        )

        click.echo(f"\n[+] Created workspace: {name}")
        click.echo(f"    Path: {ws.root}")
        click.echo(f"\nWorkspace structure:")
        click.echo(f"    main.bbx      - Entry point (edit this!)")
        click.echo(f"    config.yaml   - Configuration")
        click.echo(f"    state/        - Persistent state")
        click.echo(f"    logs/         - Execution logs")
        click.echo(f"    workflows/    - Sub-workflows")
        click.echo(f"\nActivate with: bbx workspace set {ws.root}")

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)
        raise click.Abort()


@workspace.command("list")
@click.option("--path", "-p", help="Directory to search")
def workspace_list(path: str):
    """List all workspaces."""
    from blackbox.core.workspace_manager import get_workspace_manager

    manager = get_workspace_manager()
    workspaces = manager.list(Path(path) if path else None)

    if not workspaces:
        click.echo("No workspaces found.")
        click.echo("\nCreate one with: bbx workspace create <name>")
        return

    click.echo("\n" + "=" * 60)
    click.echo("Workspaces")
    click.echo("=" * 60)

    current = manager.get_current_path()
    for ws in workspaces:
        is_current = current and Path(ws["path"]).resolve() == current.resolve()
        marker = " *" if is_current else ""
        click.echo(f"\n{ws['name']}{marker}")
        click.echo(f"   Path: {ws['path']}")
        click.echo(f"   Updated: {ws['updated_at']}")
        if ws.get("description"):
            click.echo(f"   {ws['description']}")

    click.echo("\n" + "=" * 60)
    click.echo("* = current workspace")


@workspace.command("set")
@click.argument("path", type=click.Path(exists=True))
def workspace_set(path: str):
    """Set current workspace context.

    After setting, you can run 'bbx run' without specifying a file
    to execute main.bbx in the workspace.
    """
    from blackbox.core.workspace_manager import get_workspace_manager

    try:
        manager = get_workspace_manager()
        ws = manager.set_current(Path(path))

        click.echo(f"[+] Active workspace: {ws.metadata.name}")
        click.echo(f"    Path: {ws.root}")
        click.echo(f"\nYou can now run: bbx run")

    except ValueError as e:
        click.echo(f"[-] {e}", err=True)
        raise click.Abort()


@workspace.command("info")
@click.option("--path", "-p", help="Workspace path (default: current)")
def workspace_info(path: str):
    """Show workspace information."""
    from blackbox.core.workspace_manager import get_workspace_manager

    manager = get_workspace_manager()

    try:
        info = manager.info(Path(path) if path else None)

        click.echo("\n" + "=" * 60)
        click.echo("Workspace Information")
        click.echo("=" * 60)

        click.echo(f"\nName: {info['name']}")
        click.echo(f"ID: {info['workspace_id']}")
        click.echo(f"Path: {info['root']}")
        click.echo(f"Description: {info['description'] or 'N/A'}")
        click.echo(f"Created: {info['created_at']}")
        click.echo(f"Updated: {info['updated_at']}")
        click.echo(f"\nWorkflows: {info['workflows_count']}")
        click.echo(f"Runs: {info['runs_count']}")
        click.echo(f"State keys: {len(info['state_keys'])}")

        if not info['valid']:
            click.echo(f"\n[!] Issues: {', '.join(info['issues'])}")

        click.echo("\n" + "=" * 60)

    except ValueError as e:
        click.echo(f"[-] {e}", err=True)
        raise click.Abort()


@workspace.command("clear")
def workspace_clear():
    """Clear current workspace (deactivate)."""
    from blackbox.core.workspace_manager import get_workspace_manager

    manager = get_workspace_manager()
    manager.clear_current()
    click.echo("Cleared current workspace.")


# =============================================================================
# Process Management Commands (ps, kill, wait, logs)
# =============================================================================

@cli.command("ps")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all (not just running)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def ps_cmd(show_all: bool, as_json: bool):
    """List workflow executions (like ps in Linux).

    Examples:
      bbx ps           - Show running executions
      bbx ps -a        - Show all executions
      bbx ps --json    - Output as JSON
    """
    from blackbox.core.execution_manager import get_execution_manager

    async def _ps():
        manager = get_execution_manager()
        return await manager.ps(all=show_all)

    executions = asyncio.run(_ps())

    if as_json:
        click.echo(json.dumps([e.to_dict() for e in executions], indent=2, default=str))
        return

    if not executions:
        if show_all:
            click.echo("No executions found.")
        else:
            click.echo("No running executions.")
            click.echo("\nUse 'bbx ps -a' to see all executions.")
        return

    click.echo("\n" + "=" * 80)
    click.echo(f"{'EXEC_ID':<12} {'STATUS':<12} {'WORKFLOW':<30} {'STARTED':<20}")
    click.echo("=" * 80)

    for exec in executions:
        exec_id = exec.execution_id[:8] + "..."
        status = exec.status.value
        workflow = exec.workflow_path[-30:] if len(exec.workflow_path) > 30 else exec.workflow_path
        started = exec.started_at.strftime("%Y-%m-%d %H:%M:%S") if exec.started_at else "pending"

        # Color status
        if exec.status.value == "running":
            status = f"{status}"
        elif exec.status.value == "completed":
            status = f"{status}"
        elif exec.status.value == "failed":
            status = f"{status}"

        click.echo(f"{exec_id:<12} {status:<12} {workflow:<30} {started:<20}")

    click.echo("=" * 80)


@cli.command("kill")
@click.argument("execution_id")
@click.option("--force", "-f", is_flag=True, help="Force kill")
def kill_cmd(execution_id: str, force: bool):
    """Kill a running execution (like kill in Linux).

    Example:
      bbx kill 550e8400
    """
    from blackbox.core.execution_manager import get_execution_manager
    from blackbox.core.execution_store import get_execution_store

    # Find execution by partial ID
    store = get_execution_store()
    executions = store.list(limit=100)
    matching = [e for e in executions if e.execution_id.startswith(execution_id)]

    if not matching:
        click.echo(f"No execution found matching: {execution_id}", err=True)
        raise click.Abort()

    if len(matching) > 1:
        click.echo(f"Multiple executions match '{execution_id}':")
        for e in matching:
            click.echo(f"  - {e.execution_id}")
        click.echo("\nPlease be more specific.")
        raise click.Abort()

    full_id = matching[0].execution_id

    async def _kill():
        manager = get_execution_manager()
        return await manager.kill(full_id, force=force)

    result = asyncio.run(_kill())

    if result:
        click.echo(f"[+] Killed execution: {full_id[:12]}...")
    else:
        click.echo(f"[-] Could not kill execution (may have already finished)", err=True)


@cli.command("wait")
@click.argument("execution_id")
@click.option("--timeout", "-t", type=int, help="Timeout in seconds")
def wait_cmd(execution_id: str, timeout: int):
    """Wait for execution to complete (like wait in Linux).

    Example:
      bbx wait 550e8400 --timeout 300
    """
    from blackbox.core.execution_manager import get_execution_manager
    from blackbox.core.execution_store import get_execution_store

    # Find execution by partial ID
    store = get_execution_store()
    executions = store.list(limit=100)
    matching = [e for e in executions if e.execution_id.startswith(execution_id)]

    if not matching:
        click.echo(f"No execution found matching: {execution_id}", err=True)
        raise click.Abort()

    full_id = matching[0].execution_id
    click.echo(f"Waiting for {full_id[:12]}...")

    async def _wait():
        manager = get_execution_manager()
        return await manager.wait(full_id, timeout=float(timeout) if timeout else None)

    result = asyncio.run(_wait())

    if result:
        click.echo(f"\nExecution {result.status.value}")
        if result.outputs:
            click.echo(f"\nOutputs:")
            click.echo(json.dumps(result.outputs, indent=2, default=str))
        if result.error:
            click.echo(f"\nError: {result.error}", err=True)
    else:
        click.echo(f"\nTimeout waiting for execution", err=True)


@cli.command("logs")
@click.argument("execution_id")
@click.option("--follow", "-f", is_flag=True, help="Follow logs (like tail -f)")
@click.option("--limit", "-n", type=int, default=50, help="Number of lines")
def logs_cmd(execution_id: str, follow: bool, limit: int):
    """Show execution logs (like tail -f in Linux).

    Examples:
      bbx logs 550e8400           - Show last 50 logs
      bbx logs 550e8400 -f        - Follow logs in real-time
      bbx logs 550e8400 -n 100    - Show last 100 logs
    """
    from blackbox.core.execution_manager import get_execution_manager
    from blackbox.core.execution_store import get_execution_store

    # Find execution by partial ID
    store = get_execution_store()
    executions = store.list(limit=100)
    matching = [e for e in executions if e.execution_id.startswith(execution_id)]

    if not matching:
        click.echo(f"No execution found matching: {execution_id}", err=True)
        raise click.Abort()

    full_id = matching[0].execution_id

    if follow:
        click.echo(f"Following logs for {full_id[:12]}... (Ctrl+C to stop)")
        click.echo("-" * 60)

        async def _follow():
            manager = get_execution_manager()
            try:
                async for log in manager.logs(full_id, follow=True):
                    ts = log["timestamp"].split("T")[1].split(".")[0] if "T" in log["timestamp"] else log["timestamp"]
                    level = log["level"]
                    msg = log["message"]
                    step = f"[{log['step_id']}] " if log.get("step_id") else ""
                    click.echo(f"{ts} | {level:<5} | {step}{msg}")
            except asyncio.CancelledError:
                pass

        try:
            asyncio.run(_follow())
        except KeyboardInterrupt:
            click.echo("\n\nStopped following logs.")
    else:
        # Just show recent logs
        logs = store.get_logs(full_id, limit=limit)

        if not logs:
            click.echo(f"No logs for execution {full_id[:12]}...")
            return

        click.echo(f"\nLogs for {full_id[:12]}...")
        click.echo("-" * 60)

        for log in logs:
            ts = log["timestamp"].split("T")[1].split(".")[0] if "T" in log["timestamp"] else log["timestamp"]
            level = log["level"]
            msg = log["message"]
            step = f"[{log['step_id']}] " if log.get("step_id") else ""
            click.echo(f"{ts} | {level:<5} | {step}{msg}")


# =============================================================================
# State Management Commands (like env vars in Linux)
# =============================================================================

@cli.group()
def state():
    """Manage persistent state (like environment variables).

    State is stored per-workspace in state/vars.json.
    Use namespaces for workflow-scoped state.

    Examples:
      bbx state get user_name
      bbx state set user_name "Alice"
      bbx state list
      bbx state list --pattern "user_*"
    """
    pass


@state.command("get")
@click.argument("key")
@click.option("--namespace", "-n", help="State namespace (for workflow-scoped state)")
@click.option("--default", "-d", help="Default value if not found")
def state_get(key: str, namespace: str, default: str):
    """Get a state value."""
    from blackbox.core.workspace_manager import get_current_workspace
    from blackbox.core.config import get_config

    # Get state file path
    workspace = get_current_workspace()
    if workspace:
        state_dir = workspace.paths.state_dir
    else:
        config = get_config()
        state_dir = config.paths.bbx_home / "state"

    if namespace:
        safe_namespace = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
        state_file = state_dir / f"{safe_namespace}.json"
    else:
        state_file = state_dir / "vars.json"

    if not state_file.exists():
        if default:
            click.echo(default)
        else:
            click.echo(f"Key not found: {key}", err=True)
        return

    try:
        state_data = json.loads(state_file.read_text())
        value = state_data.get(key, default)
        if value is not None:
            if isinstance(value, (dict, list)):
                click.echo(json.dumps(value, indent=2))
            else:
                click.echo(value)
        elif default:
            click.echo(default)
        else:
            click.echo(f"Key not found: {key}", err=True)
    except Exception as e:
        click.echo(f"Error reading state: {e}", err=True)


@state.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--namespace", "-n", help="State namespace")
def state_set(key: str, value: str, namespace: str):
    """Set a state value."""
    from blackbox.core.workspace_manager import get_current_workspace
    from blackbox.core.config import get_config

    # Parse value - try JSON first, then string
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    # Get state file path
    workspace = get_current_workspace()
    if workspace:
        state_dir = workspace.paths.state_dir
    else:
        config = get_config()
        state_dir = config.paths.bbx_home / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

    if namespace:
        safe_namespace = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
        state_file = state_dir / f"{safe_namespace}.json"
    else:
        state_file = state_dir / "vars.json"

    # Load existing state
    if state_file.exists():
        state_data = json.loads(state_file.read_text())
    else:
        state_data = {}

    # Set value
    state_data[key] = parsed_value
    state_file.write_text(json.dumps(state_data, indent=2, default=str))

    click.echo(f"Set {key} = {parsed_value}")


@state.command("list")
@click.option("--namespace", "-n", help="State namespace")
@click.option("--pattern", "-p", default="*", help="Filter keys by pattern (e.g., user_*)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def state_list(namespace: str, pattern: str, as_json: bool):
    """List all state keys."""
    import fnmatch
    from blackbox.core.workspace_manager import get_current_workspace
    from blackbox.core.config import get_config

    # Get state file path
    workspace = get_current_workspace()
    if workspace:
        state_dir = workspace.paths.state_dir
    else:
        config = get_config()
        state_dir = config.paths.bbx_home / "state"

    if namespace:
        safe_namespace = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
        state_file = state_dir / f"{safe_namespace}.json"
    else:
        state_file = state_dir / "vars.json"

    if not state_file.exists():
        click.echo("No state found.")
        return

    state_data = json.loads(state_file.read_text())
    matching_keys = [k for k in state_data.keys() if fnmatch.fnmatch(k, pattern)]

    if as_json:
        result = {k: state_data[k] for k in matching_keys}
        click.echo(json.dumps(result, indent=2, default=str))
        return

    if not matching_keys:
        click.echo(f"No keys matching pattern: {pattern}")
        return

    click.echo("\n" + "=" * 60)
    click.echo("State Variables")
    click.echo("=" * 60)

    for key in sorted(matching_keys):
        value = state_data[key]
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value)[:50] + "..." if len(json.dumps(value)) > 50 else json.dumps(value)
        else:
            value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
        click.echo(f"  {key} = {value_str}")

    click.echo("\n" + "=" * 60)
    click.echo(f"Total: {len(matching_keys)} keys")


@state.command("delete")
@click.argument("key")
@click.option("--namespace", "-n", help="State namespace")
def state_delete(key: str, namespace: str):
    """Delete a state key."""
    from blackbox.core.workspace_manager import get_current_workspace
    from blackbox.core.config import get_config

    # Get state file path
    workspace = get_current_workspace()
    if workspace:
        state_dir = workspace.paths.state_dir
    else:
        config = get_config()
        state_dir = config.paths.bbx_home / "state"

    if namespace:
        safe_namespace = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
        state_file = state_dir / f"{safe_namespace}.json"
    else:
        state_file = state_dir / "vars.json"

    if not state_file.exists():
        click.echo(f"Key not found: {key}", err=True)
        return

    state_data = json.loads(state_file.read_text())

    if key in state_data:
        del state_data[key]
        state_file.write_text(json.dumps(state_data, indent=2, default=str))
        click.echo(f"Deleted: {key}")
    else:
        click.echo(f"Key not found: {key}", err=True)


@state.command("clear")
@click.option("--namespace", "-n", help="State namespace")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def state_clear(namespace: str, yes: bool):
    """Clear all state."""
    from blackbox.core.workspace_manager import get_current_workspace
    from blackbox.core.config import get_config

    if not yes:
        if not click.confirm("Are you sure you want to clear all state?"):
            return

    # Get state file path
    workspace = get_current_workspace()
    if workspace:
        state_dir = workspace.paths.state_dir
    else:
        config = get_config()
        state_dir = config.paths.bbx_home / "state"

    if namespace:
        safe_namespace = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
        state_file = state_dir / f"{safe_namespace}.json"
    else:
        state_file = state_dir / "vars.json"

    if state_file.exists():
        state_file.write_text("{}")
        click.echo("State cleared.")
    else:
        click.echo("No state to clear.")


# =============================================================================
# A2A Commands (Agent-to-Agent Protocol)
# =============================================================================

@cli.group()
def a2a():
    """A2A Protocol commands - multi-agent communication."""
    pass


@a2a.command("serve")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind")
@click.option("--port", "-p", default=8000, type=int, help="Port to listen")
@click.option("--name", "-n", default="bbx-agent", help="Agent name")
@click.option("--workspace", "-w", help="Workspace path")
def a2a_serve(host: str, port: int, name: str, workspace: str):
    """Start A2A server to receive tasks from other agents.

    Example:
        bbx a2a serve --port 8000 --name my-agent
    """
    try:
        from blackbox.a2a.server import run_server

        click.echo("=" * 60)
        click.echo(f"Starting BBX A2A Server: {name}")
        click.echo(f"URL: http://{host}:{port}")
        click.echo(f"Agent Card: http://{host}:{port}/.well-known/agent-card.json")
        click.echo("=" * 60)

        run_server(host=host, port=port, name=name, workspace_path=workspace)

    except ImportError as e:
        click.echo(f"A2A server requires FastAPI: pip install fastapi uvicorn", err=True)
        click.echo(f"Error: {e}", err=True)


@a2a.command("card")
@click.option("--name", "-n", default="bbx-agent", help="Agent name")
@click.option("--url", "-u", default="http://localhost:8000", help="Agent URL")
@click.option("--workspace", "-w", help="Workspace path")
@click.option("--output", "-o", help="Output file (default: stdout)")
def a2a_card(name: str, url: str, workspace: str, output: str):
    """Generate Agent Card JSON for this BBX instance.

    Example:
        bbx a2a card --name my-agent --url https://my-agent.example.com
        bbx a2a card -o agent-card.json
    """
    from blackbox.a2a.agent_card import AgentCardGenerator

    generator = AgentCardGenerator(
        name=name,
        url=url,
        workspace_path=workspace,
    )

    json_content = generator.to_json(indent=2)

    if output:
        Path(output).write_text(json_content)
        click.echo(f"Agent Card saved to: {output}")
    else:
        click.echo(json_content)


@a2a.command("discover")
@click.argument("agent_url")
def a2a_discover(agent_url: str):
    """Discover capabilities of a remote A2A agent.

    Example:
        bbx a2a discover http://localhost:8001
    """
    async def _discover():
        from blackbox.a2a.client import A2AClient

        client = A2AClient()
        async with client:
            card = await client.discover(agent_url)

        click.echo("=" * 60)
        click.echo(f"Agent: {card.name}")
        click.echo(f"Description: {card.description}")
        click.echo(f"URL: {card.url}")
        click.echo(f"Protocol: A2A v{card.protocol_version}")
        click.echo("=" * 60)
        click.echo(f"\nSkills ({len(card.skills)}):")
        for skill in card.skills:
            click.echo(f"  - {skill.id}: {skill.description}")

    asyncio.run(_discover())


@a2a.command("call")
@click.argument("agent_url")
@click.argument("skill_id")
@click.option("--input", "-i", "inputs", multiple=True, help="Input in key=value format")
@click.option("--json-input", "-j", help="JSON input")
@click.option("--wait/--no-wait", default=True, help="Wait for result")
@click.option("--timeout", "-t", type=float, help="Timeout in seconds")
def a2a_call(agent_url: str, skill_id: str, inputs: tuple, json_input: str, wait: bool, timeout: float):
    """Call a skill on a remote A2A agent.

    Examples:
        bbx a2a call http://localhost:8001 analyze_text -i text="Hello world"
        bbx a2a call http://localhost:8001 analyze_text -j '{"text": "Hello"}'
    """
    async def _call():
        from blackbox.a2a.client import A2AClient

        # Parse inputs
        input_dict = {}
        for input_str in inputs:
            if "=" in input_str:
                key, value = input_str.split("=", 1)
                input_dict[key] = value

        if json_input:
            input_dict.update(json.loads(json_input))

        client = A2AClient()
        async with client:
            result = await client.call(
                agent_url=agent_url,
                skill_id=skill_id,
                input=input_dict,
                wait=wait,
                timeout=timeout,
            )

        click.echo(json.dumps(result, indent=2, default=str))

    asyncio.run(_call())


@a2a.command("ping")
@click.argument("agent_url")
def a2a_ping(agent_url: str):
    """Check if an A2A agent is healthy.

    Example:
        bbx a2a ping http://localhost:8001
    """
    async def _ping():
        from blackbox.a2a.client import A2AClient

        client = A2AClient()
        async with client:
            healthy = await client.ping(agent_url)

        if healthy:
            click.echo(f"[OK] Agent at {agent_url} is healthy")
        else:
            click.echo(f"[ERR] Agent at {agent_url} is not responding", err=True)

    asyncio.run(_ping())


# =============================================================================
# Pack/Unpack Commands (BBX Bundler)
# =============================================================================

@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output .bbxpkg file")
@click.option("--include-state", "-s", is_flag=True, help="Include state data")
@click.option("--no-deps", is_flag=True, help="Don't include dependent workflows")
def pack(workflow_file: str, output: str, include_state: bool, no_deps: bool):
    """Pack a workflow into a self-contained .bbxpkg package.

    Creates a portable package with the workflow and all its dependencies.

    Examples:
        bbx pack deploy.bbx
        bbx pack deploy.bbx -o deploy.bbxpkg
        bbx pack deploy.bbx --include-state
    """
    from blackbox.core.bundler import get_bundler

    try:
        bundler = get_bundler()
        result = bundler.pack(
            workflow_path=workflow_file,
            output_path=output,
            include_state=include_state,
            include_deps=not no_deps,
        )

        click.echo(f"[+] Package created: {result}")

        # Show package info
        info = bundler.info(result)
        click.echo(f"    Size: {info['size_human']}")
        if info.get('manifest'):
            m = info['manifest']
            click.echo(f"    Workflow: {m.get('workflow_name', 'N/A')}")
            click.echo(f"    Files: {m['files']['count']}")

    except Exception as e:
        click.echo(f"[-] Pack failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("package_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output directory")
@click.option("--restore-state", "-s", is_flag=True, help="Restore state data")
def unpack(package_file: str, output: str, restore_state: bool):
    """Unpack a .bbxpkg package.

    Extracts workflow and dependencies to specified directory.

    Examples:
        bbx unpack deploy.bbxpkg
        bbx unpack deploy.bbxpkg -o ./my-workflow
        bbx unpack deploy.bbxpkg --restore-state
    """
    from blackbox.core.bundler import get_bundler

    try:
        bundler = get_bundler()
        result = bundler.unpack(
            package_path=package_file,
            output_dir=output,
            restore_state=restore_state,
        )

        click.echo(f"[+] Package unpacked to: {result['output_dir']}")
        click.echo(f"    Workflows: {len(result['workflows'])}")
        for wf in result['workflows']:
            click.echo(f"      - {wf}")
        if result['state_restored']:
            click.echo("    State: restored")

    except Exception as e:
        click.echo(f"[-] Unpack failed: {e}", err=True)
        raise click.Abort()


@cli.command("pack-info")
@click.argument("package_file", type=click.Path(exists=True))
def pack_info(package_file: str):
    """Show information about a .bbxpkg package.

    Example:
        bbx pack-info deploy.bbxpkg
    """
    from blackbox.core.bundler import get_bundler

    try:
        bundler = get_bundler()
        info = bundler.info(package_file)

        click.echo("\n" + "=" * 60)
        click.echo("Package Information")
        click.echo("=" * 60)
        click.echo(f"\nFile: {info['package']}")
        click.echo(f"Size: {info['size_human']}")

        if info.get('manifest'):
            m = info['manifest']
            click.echo(f"\nWorkflow: {m.get('workflow_name', 'N/A')}")
            click.echo(f"ID: {m.get('workflow_id', 'N/A')}")
            click.echo(f"Version: {m.get('workflow_version', 'N/A')}")
            click.echo(f"Created: {m.get('created_at', 'N/A')}")
            click.echo(f"BBX Version: {m.get('bbx_version', 'N/A')}")
            click.echo(f"\nFiles ({m['files']['count']}):")
            for f in m['files']['list'][:10]:
                click.echo(f"  - {f}")
            if len(m['files']['list']) > 10:
                click.echo(f"  ... and {len(m['files']['list']) - 10} more")

        click.echo("=" * 60)

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)
        raise click.Abort()


# =============================================================================
# Secrets Management Commands
# =============================================================================

@cli.group()
def secret():
    """Manage secrets (API keys, passwords, tokens).

    Secrets are stored encrypted and can be used in workflows via ${secrets.KEY}.

    Examples:
        bbx secret set API_KEY "sk-xxx"
        bbx secret get API_KEY
        bbx secret list
    """
    pass


@secret.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--description", "-d", help="Secret description")
@click.option("--expires", "-e", help="Expiration date (ISO format)")
def secret_set(key: str, value: str, description: str, expires: str):
    """Set a secret.

    Example:
        bbx secret set API_KEY "sk-xxx" -d "OpenAI API key"
    """
    from blackbox.core.secrets import get_secrets_manager

    manager = get_secrets_manager()
    result = manager.set(key, value, description=description, expires_at=expires)

    if result.get("status") == "success":
        click.echo(f"[+] Secret set: {key}")
        if not result.get("encrypted"):
            click.echo("    [!] Warning: cryptography not installed, using obfuscation only")
    else:
        click.echo(f"[-] Error: {result.get('error')}", err=True)


@secret.command("get")
@click.argument("key")
@click.option("--default", "-d", help="Default value if not found")
@click.option("--no-env", is_flag=True, help="Don't check environment variables")
def secret_get(key: str, default: str, no_env: bool):
    """Get a secret value.

    Example:
        bbx secret get API_KEY
    """
    from blackbox.core.secrets import get_secrets_manager

    manager = get_secrets_manager()
    result = manager.get(key, default=default, check_env=not no_env)

    if result.get("status") == "success":
        click.echo(result["value"])
    else:
        click.echo(f"[-] {result.get('error')}", err=True)
        raise click.Abort()


@secret.command("list")
@click.option("--pattern", "-p", help="Filter by pattern (e.g., API_*)")
@click.option("--show-values", is_flag=True, help="Show actual values (not masked)")
def secret_list(pattern: str, show_values: bool):
    """List all secrets.

    Example:
        bbx secret list
        bbx secret list --pattern "API_*"
    """
    from blackbox.core.secrets import get_secrets_manager

    manager = get_secrets_manager()
    result = manager.list(pattern=pattern, show_values=show_values)

    if not result.get("secrets"):
        click.echo("No secrets found.")
        return

    click.echo("\n" + "=" * 60)
    click.echo("Secrets")
    click.echo("=" * 60)

    for s in result["secrets"]:
        click.echo(f"\n{s['key']}")
        click.echo(f"   Value: {s['value']}")
        if s.get('description'):
            click.echo(f"   Description: {s['description']}")
        if s.get('expires_at'):
            click.echo(f"   Expires: {s['expires_at']}")

    click.echo("\n" + "=" * 60)
    click.echo(f"Total: {result['count']} secrets")
    if not result.get("encrypted"):
        click.echo("[!] Warning: Secrets are NOT encrypted (install cryptography)")


@secret.command("delete")
@click.argument("key")
def secret_delete(key: str):
    """Delete a secret.

    Example:
        bbx secret delete API_KEY
    """
    from blackbox.core.secrets import get_secrets_manager

    manager = get_secrets_manager()
    result = manager.delete(key)

    if result.get("status") == "success":
        click.echo(f"[+] Deleted: {key}")
    else:
        click.echo(f"[-] {result.get('error')}", err=True)


@secret.command("rotate")
@click.argument("key")
@click.argument("new_value")
@click.option("--keep-history", is_flag=True, help="Keep old value in history")
def secret_rotate(key: str, new_value: str, keep_history: bool):
    """Rotate a secret (update with new value).

    Example:
        bbx secret rotate API_KEY "sk-new-xxx"
    """
    from blackbox.core.secrets import get_secrets_manager

    manager = get_secrets_manager()
    result = manager.rotate(key, new_value, keep_history=keep_history)

    if result.get("status") == "success":
        click.echo(f"[+] Rotated: {key}")
    else:
        click.echo(f"[-] {result.get('error')}", err=True)


@secret.command("export")
@click.option("--format", "-f", type=click.Choice(["env", "json"]), default="env")
def secret_export(format: str):
    """Export secrets.

    Examples:
        bbx secret export > .env
        bbx secret export --format json > secrets.json
    """
    from blackbox.core.secrets import get_secrets_manager

    manager = get_secrets_manager()
    result = manager.export(format=format)

    if result.get("status") == "success":
        click.echo(result["data"])
    else:
        click.echo(f"[-] {result.get('error')}", err=True)


# =============================================================================
# Agent-Friendly Help Command
# =============================================================================

@cli.command("help")
@click.argument("command_path", required=False, nargs=-1)
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text",
              help="Output format (text for humans, json for AI agents)")
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all commands including subcommands")
@click.pass_context
def help_cmd(ctx, command_path, format, show_all):
    """Show help for BBX commands.

    For AI agents, use --format json to get machine-readable output.

    Examples:
        bbx help
        bbx help run
        bbx help a2a discover
        bbx help --format json
        bbx help run --format json
    """
    from blackbox.cli.help import get_command_schema, get_group_schema, generate_full_cli_schema

    # Get root CLI
    root = ctx.parent.command if ctx.parent else cli

    if not command_path:
        # Show root help
        if format == "json":
            if show_all:
                schema = generate_full_cli_schema(root)
            else:
                from click import Context
                root_ctx = Context(root, info_name="bbx")
                schema = get_group_schema(root, root_ctx)
            click.echo(json.dumps(schema, indent=2))
        else:
            click.echo(ctx.parent.get_help() if ctx.parent else root.get_help(ctx))
    else:
        # Navigate to specific command
        from click import Context, Group
        cmd = root
        cmd_ctx = Context(root, info_name="bbx")

        for name in command_path:
            if isinstance(cmd, Group):
                sub_cmd = cmd.get_command(cmd_ctx, name)
                if sub_cmd:
                    cmd = sub_cmd
                    cmd_ctx = Context(cmd, parent=cmd_ctx, info_name=name)
                else:
                    click.echo(f"Unknown command: {' '.join(command_path)}", err=True)
                    raise click.Abort()
            else:
                click.echo(f"'{cmd.name}' is not a group, cannot have subcommands", err=True)
                raise click.Abort()

        # Show help for found command
        if format == "json":
            if isinstance(cmd, Group):
                schema = get_group_schema(cmd, cmd_ctx)
            else:
                schema = get_command_schema(cmd, cmd_ctx)
            click.echo(json.dumps(schema, indent=2))
        else:
            click.echo(cmd.get_help(cmd_ctx))


# =============================================================================
# Adapters List Command
# =============================================================================

@cli.command("adapters")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
def adapters_list(format: str):
    """List all available BBX adapters.

    Examples:
        bbx adapters
        bbx adapters --format json
    """
    from blackbox.core.registry import CORE_ADAPTERS, OPTIONAL_ADAPTERS, get_registry

    registry = get_registry()
    all_adapters = registry.list_adapters()

    if format == "json":
        result = {
            "core": CORE_ADAPTERS,
            "optional": OPTIONAL_ADAPTERS,
            "registered": all_adapters,
            "total": len(all_adapters),
        }
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("\n" + "=" * 60)
        click.echo("BBX Adapters")
        click.echo("=" * 60)

        click.echo("\n[Core Adapters]")
        for name, desc in CORE_ADAPTERS.items():
            click.echo(f"  {name}: {desc}")

        click.echo("\n[Optional Adapters]")
        for name, desc in OPTIONAL_ADAPTERS.items():
            click.echo(f"  {name}: {desc}")

        click.echo("\n" + "=" * 60)
        click.echo(f"Total registered: {len(all_adapters)}")


if __name__ == "__main__":
    cli()
