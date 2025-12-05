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
@click.version_option(version="2.0.0")
def cli():
    """BBX - The WinRAR for AI Development.

    Simple interface, powerful under the hood.
    Like a hammer - it just works.

    Core commands:
        bbx pack     - Compress project understanding
        bbx unpack   - Decompress intent into code
        bbx recover  - Restore after AI errors (killer feature!)

    That's it. Everything else is optional.
    """
    pass


# =============================================================================
# CORE COMMANDS - The "Hammer" Interface (Never Changes)
# =============================================================================
# These commands are like WinRAR - simple, stable, reliable.
# The interface stays the same, power grows under the hood.

@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--output", "-o", type=click.Path(), help="Output .bbx file")
@click.option("--recovery", "-r", default=5, help="Recovery data percent (1-10)")
@click.option("--no-recovery", is_flag=True, help="Skip recovery record")
def pack(path: str, output: str, recovery: int, no_recovery: bool):
    """Compress project understanding into .bbx package.

    Like WinRAR compress, but for project understanding.
    Creates a package with:
    - Project structure and relationships
    - Semantic embeddings for search
    - Recovery record for rollback (like WinRAR!)

    Examples:
        bbx pack                    # Pack current directory
        bbx pack ./my-project       # Pack specific project
        bbx pack . -o project.bbx   # Specify output
        bbx pack . -r 10            # 10% recovery data
    """
    async def _pack():
        from blackbox.core.v2.bbx_core import BBXCore
        core = BBXCore(path)
        return await core.pack(
            output=output,
            include_recovery=not no_recovery,
            recovery_percent=recovery,
        )

    try:
        click.echo(f"Packing: {path}")
        result = asyncio.run(_pack())
        click.echo(f"\n[+] Created: {result}")
        click.echo(f"    Recovery record: {'disabled' if no_recovery else f'{recovery}%'}")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@cli.command()
@click.argument("intent")
@click.option("--path", "-p", type=click.Path(exists=True), default=".", help="Project path")
@click.option("--dry-run", "-n", is_flag=True, help="Show plan without executing")
@click.option("--output", "-o", type=click.Path(), help="Save workflow to file")
def unpack(intent: str, path: str, dry_run: bool, output: str):
    """Decompress intent into executable workflow.

    Like WinRAR decompress, but for intentions.
    Takes what you want to achieve and generates concrete steps.

    Uses:
    - Project Genome (understanding)
    - Memory (similar successful changes)
    - Templates + LLM (generation)

    Examples:
        bbx unpack "Deploy to AWS"
        bbx unpack "Add authentication" --dry-run
        bbx unpack "Fix the login bug" -o fix.yaml
    """
    async def _unpack():
        from blackbox.core.v2.bbx_core import BBXCore
        core = BBXCore(path)
        return await core.unpack(intent, dry_run=True)  # Always show plan first

    try:
        result = asyncio.run(_unpack())

        click.echo("\n" + "=" * 60)
        click.echo(f"Intent: {intent}")
        click.echo("=" * 60)
        click.echo(f"Confidence: {result['confidence']:.0%}")
        click.echo(f"Sources: {', '.join(result['sources'])}")
        click.echo(f"Context from memory: {result.get('context_files', 0)} items")
        click.echo(f"Similar patterns: {result.get('similar_patterns', 0)}")

        click.echo(f"\nGenerated {len(result['steps'])} steps:")
        for i, step in enumerate(result['steps'], 1):
            click.echo(f"  {i}. {step.get('id', '?')} -> {step.get('use', '?')}")

        if output or dry_run:
            if 'yaml' in result:
                click.echo("\n" + "-" * 60)
                click.echo(result['yaml'])
                if output:
                    from pathlib import Path as PathLib
                    PathLib(output).write_text(result['yaml'])
                    click.echo(f"\n[+] Saved to: {output}")

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@cli.command()
@click.option("--path", "-p", type=click.Path(exists=True), default=".", help="Project path")
@click.option("--snapshot", "-s", help="Specific snapshot ID to restore")
@click.option("--file", "-f", "file_path", help="Restore specific file only")
@click.option("--smart", is_flag=True, help="Smart recovery - detect and fix")
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be restored")
def recover(path: str, snapshot: str, file_path: str, smart: bool, dry_run: bool):
    """Restore project after AI errors.

    THIS IS THE KILLER FEATURE - like WinRAR's Recovery Record.
    No other AI tool can do this.

    Modes:
        bbx recover              # Restore from latest snapshot
        bbx recover --smart      # Detect what's broken, fix it
        bbx recover -f src/main.py  # Restore specific file

    Examples:
        bbx recover                 # Restore everything
        bbx recover --dry-run       # See what would be restored
        bbx recover --smart         # Auto-detect and fix
    """
    async def _recover():
        from blackbox.core.v2.bbx_core import BBXCore
        core = BBXCore(path)

        if smart:
            return await core.smart_recover()
        else:
            return await core.recover(
                snapshot_id=snapshot,
                file_path=file_path,
                dry_run=dry_run,
            )

    try:
        result = asyncio.run(_recover())

        if result["status"] == "error":
            click.echo(f"[-] {result['message']}", err=True)
            return

        click.echo("\n" + "=" * 60)
        click.echo("Recovery Result")
        click.echo("=" * 60)

        if result["status"] == "clean":
            click.echo(click.style("[+] Project is clean - no recovery needed", fg="green"))

        elif result["status"] == "changes_detected":
            click.echo(f"Snapshot from: {result['snapshot_time']}")
            click.echo(f"\nChanged files ({len(result['changed_files'])}):")
            for f in result['changed_files']:
                color = "red" if f['change'] == 'deleted' else "yellow"
                click.echo(click.style(f"  {f['change']}: {f['file']}", fg=color))
            click.echo(f"\n{result['recommendation']}")

        else:
            status = "DRY RUN" if dry_run else "RECOVERED"
            click.echo(f"Status: {status}")
            click.echo(f"Snapshot: {result.get('snapshot', 'latest')}")
            click.echo(f"Time: {result.get('snapshot_time', 'unknown')}")
            click.echo(f"Files: {result.get('files_recovered', 0)}")

            if result.get('details'):
                click.echo("\nDetails:")
                for d in result['details'][:10]:
                    action = click.style(d['action'], fg="green")
                    click.echo(f"  {action}: {d['file']}")
                if len(result['details']) > 10:
                    click.echo(f"  ... and {len(result['details']) - 10} more")

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


# =============================================================================
# Workflow Commands (Extended functionality)
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
@click.argument("event_type", required=False)
@click.option("--workflow", "-w", help="Path to specific workflow to run")
def hook(event_type: str, workflow: str):
    """Handle a Claude Code hook event.

    Reads JSON payload from stdin.
    """
    import sys
    import json
    from blackbox.core.registry import registry

    # Read stdin
    try:
        if sys.stdin.isatty():
            # If no input, just show help or error
            click.echo("Error: No input provided on stdin", err=True)
            sys.exit(1)

        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON input: {e}", err=True)
        sys.exit(1)

    # If event_type is not provided, try to get it from payload
    if not event_type:
        event_type = payload.get("hook_event_name")

    if not event_type:
        click.echo("Error: event_type not specified and not found in payload", err=True)
        sys.exit(1)

    # Ensure payload has event name
    if "hook_event_name" not in payload:
        payload["hook_event_name"] = event_type

    try:
        adapter = registry.get_adapter("claude_hooks")
        if not adapter:
             click.echo("Error: claude_hooks adapter not found", err=True)
             sys.exit(1)

        # Run event handler
        async def _run():
            return await adapter.handle_event(payload, workflow_path=workflow)

        result = asyncio.run(_run())

        # Output result
        click.echo(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        click.echo(f"Error handling hook: {e}", err=True)
        sys.exit(1)


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


# =============================================================================
# Sandbox Commands
# =============================================================================

@cli.group()
def sandbox():
    """Manage workflow sandboxing and isolation.

    Sandboxes provide secure execution environments for workflows.
    Configure permissions, network access, and resource limits.

    Examples:
        bbx sandbox init --template standard
        bbx sandbox test workflow.bbx
        bbx sandbox allow-domain api.example.com
        bbx sandbox allow-path /data --mode ro
    """
    pass


@sandbox.command("init")
@click.option("--template", "-t", type=click.Choice(["minimal", "standard", "network", "developer", "untrusted"]), default="standard")
@click.option("--project", "-p", type=click.Path(), default=".", help="Project directory")
def sandbox_init(template: str, project: str):
    """Initialize sandbox configuration for project.

    Templates:
        minimal   - No network, no spawn, read-only
        standard  - Basic permissions, no network
        network   - Allows outbound HTTP/HTTPS
        developer - Full permissions for development
        untrusted - Maximum isolation for untrusted code
    """
    from blackbox.core.v2 import SandboxTemplates

    config_path = Path(project) / ".bbx" / "sandbox.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Get template config
    if template == "minimal":
        config = SandboxTemplates.minimal()
    elif template == "network":
        config = SandboxTemplates.network()
    elif template == "developer":
        config = SandboxTemplates.developer()
    elif template == "untrusted":
        config = SandboxTemplates.untrusted()
    else:
        config = SandboxTemplates.standard()

    # Convert to YAML
    config_dict = {
        "name": config.name,
        "permissions": [p.value for p in config.permissions],
        "allow_network": config.allow_network,
        "allow_spawn": config.allow_spawn,
        "max_memory_mb": config.max_memory_mb,
        "max_runtime_seconds": config.max_runtime_seconds,
    }

    config_path.write_text(yaml.dump(config_dict, default_flow_style=False))
    click.echo(f"[+] Created sandbox config: {config_path}")
    click.echo(f"    Template: {template}")
    click.echo(f"    Permissions: {config_dict['permissions']}")
    click.echo(f"    Network: {'allowed' if config_dict['allow_network'] else 'blocked'}")


@sandbox.command("test")
@click.argument("workflow", type=click.Path(exists=True))
@click.option("--template", "-t", default="standard")
@click.option("--verbose", "-v", is_flag=True)
def sandbox_test(workflow: str, template: str, verbose: bool):
    """Test workflow execution in sandbox.

    Runs workflow in isolated environment to verify it works
    within sandbox constraints.

    Example:
        bbx sandbox test deploy.bbx --template network
    """
    from blackbox.core.v2 import SandboxManager, SandboxTemplates

    async def _test():
        manager = SandboxManager()

        # Get template
        if template == "minimal":
            config = SandboxTemplates.minimal()
        elif template == "network":
            config = SandboxTemplates.network()
        elif template == "developer":
            config = SandboxTemplates.developer()
        elif template == "untrusted":
            config = SandboxTemplates.untrusted()
        else:
            config = SandboxTemplates.standard()

        click.echo(f"Testing workflow in '{template}' sandbox...")
        if verbose:
            click.echo(f"  Permissions: {[p.value for p in config.permissions]}")
            click.echo(f"  Network: {'allowed' if config.allow_network else 'blocked'}")
            click.echo(f"  Memory limit: {config.max_memory_mb}MB")
            click.echo(f"  Timeout: {config.max_runtime_seconds}s")

        async with manager.sandbox_context(config) as sb:
            click.echo(f"  Sandbox ID: {sb.id[:8]}...")

            # Run workflow in sandbox
            result = await run_file(workflow)

            click.echo(f"\n[+] Workflow completed in sandbox")
            if verbose and sb.logs:
                click.echo(f"    Logs: {len(sb.logs)} entries")

            return result

    try:
        result = asyncio.run(_test())
        if result:
            click.echo(json.dumps(result, indent=2, default=str))
    except Exception as e:
        click.echo(f"[-] Sandbox test failed: {e}", err=True)
        raise click.Abort()


@sandbox.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def sandbox_list(as_json: bool):
    """List active sandboxes."""
    from blackbox.core.v2 import SandboxManager

    async def _list():
        manager = SandboxManager()
        return manager.list_sandboxes()

    sandboxes = asyncio.run(_list())

    if as_json:
        data = []
        for sb in sandboxes:
            data.append({
                "id": sb.id,
                "state": sb.state.value,
                "config": sb.config.name,
                "started_at": str(sb.started_at) if sb.started_at else None,
                "memory_mb": sb.current_memory_mb,
            })
        click.echo(json.dumps(data, indent=2))
        return

    if not sandboxes:
        click.echo("No active sandboxes")
        return

    click.echo(f"\nActive sandboxes: {len(sandboxes)}\n")
    click.echo("=" * 60)
    for sb in sandboxes:
        click.echo(f"ID: {sb.id[:12]}...")
        click.echo(f"  State: {sb.state.value}")
        click.echo(f"  Config: {sb.config.name}")
        click.echo(f"  Started: {sb.started_at}")
        click.echo(f"  Memory: {sb.current_memory_mb:.1f}MB")
    click.echo("=" * 60)


@sandbox.command("config")
@click.option("--show", is_flag=True, help="Show current config")
@click.option("--project", "-p", type=click.Path(), default=".")
def sandbox_config(show: bool, project: str):
    """Show or manage sandbox configuration."""
    config_path = Path(project) / ".bbx" / "sandbox.yaml"

    if show or True:  # Default behavior is show
        if not config_path.exists():
            click.echo("No sandbox config found.")
            click.echo("\nCreate one with: bbx sandbox init")
            return

        config = yaml.safe_load(config_path.read_text())
        click.echo("\n" + "=" * 60)
        click.echo("Sandbox Configuration")
        click.echo("=" * 60)
        click.echo(yaml.dump(config, default_flow_style=False))
        click.echo("=" * 60)


@sandbox.command("allow-domain")
@click.argument("domain")
@click.option("--project", "-p", type=click.Path(), default=".")
def sandbox_allow_domain(domain: str, project: str):
    """Allow network access to a domain.

    Example:
        bbx sandbox allow-domain api.openai.com
        bbx sandbox allow-domain *.github.com
    """
    config_path = Path(project) / ".bbx" / "sandbox.yaml"

    if not config_path.exists():
        click.echo("No sandbox config found. Run 'bbx sandbox init' first.", err=True)
        raise click.Abort()

    config = yaml.safe_load(config_path.read_text())

    if "allowed_domains" not in config:
        config["allowed_domains"] = []

    if domain not in config["allowed_domains"]:
        config["allowed_domains"].append(domain)
        config_path.write_text(yaml.dump(config, default_flow_style=False))
        click.echo(f"[+] Allowed domain: {domain}")
    else:
        click.echo(f"Domain already allowed: {domain}")


@sandbox.command("allow-path")
@click.argument("path")
@click.option("--mode", "-m", type=click.Choice(["ro", "rw"]), default="ro", help="Access mode (ro=read-only, rw=read-write)")
@click.option("--project", "-p", type=click.Path(), default=".")
def sandbox_allow_path(path: str, mode: str, project: str):
    """Allow filesystem access to a path.

    Examples:
        bbx sandbox allow-path /data --mode ro
        bbx sandbox allow-path /tmp/output --mode rw
    """
    config_path = Path(project) / ".bbx" / "sandbox.yaml"

    if not config_path.exists():
        click.echo("No sandbox config found. Run 'bbx sandbox init' first.", err=True)
        raise click.Abort()

    config = yaml.safe_load(config_path.read_text())

    if "allowed_paths" not in config:
        config["allowed_paths"] = {}

    config["allowed_paths"][path] = mode
    config_path.write_text(yaml.dump(config, default_flow_style=False))
    click.echo(f"[+] Allowed path: {path} ({mode})")


@sandbox.command("verify")
@click.argument("workflow", type=click.Path(exists=True))
@click.option("--template", "-t", default="standard")
def sandbox_verify(workflow: str, template: str):
    """Verify workflow is sandbox-compatible.

    Analyzes workflow without executing to check if it will
    run successfully within sandbox constraints.

    Example:
        bbx sandbox verify deploy.bbx --template untrusted
    """
    from blackbox.core.v2 import SandboxTemplates

    click.echo(f"Verifying workflow: {workflow}")
    click.echo(f"Template: {template}\n")

    # Load workflow
    with open(workflow, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "workflow" in data:
        wf = data["workflow"]
    else:
        wf = data

    steps = wf.get("steps", [])

    # Get template permissions
    if template == "minimal":
        config = SandboxTemplates.minimal()
    elif template == "network":
        config = SandboxTemplates.network()
    elif template == "developer":
        config = SandboxTemplates.developer()
    elif template == "untrusted":
        config = SandboxTemplates.untrusted()
    else:
        config = SandboxTemplates.standard()

    issues = []
    warnings = []

    # Check each step
    for step in steps:
        adapter = step.get("mcp", step.get("adapter", ""))

        # Check network requirements
        if adapter in ["http", "fetch", "api"] and not config.allow_network:
            issues.append(f"Step '{step.get('id')}' requires network (blocked in {template})")

        # Check shell requirements
        if adapter in ["shell", "bash", "exec"] and not config.allow_spawn:
            issues.append(f"Step '{step.get('id')}' spawns processes (blocked in {template})")

        # Check file access
        if adapter in ["file", "fs"] and "write" in step.get("method", ""):
            from blackbox.core.v2 import SandboxPermission
            if SandboxPermission.FILE_WRITE not in config.permissions:
                warnings.append(f"Step '{step.get('id')}' writes files (may need FILE_WRITE permission)")

    click.echo("Results:")
    click.echo("-" * 40)

    if issues:
        click.echo("\n[ISSUES]")
        for issue in issues:
            click.echo(f"  [-] {issue}")

    if warnings:
        click.echo("\n[WARNINGS]")
        for warning in warnings:
            click.echo(f"  [!] {warning}")

    if not issues and not warnings:
        click.echo("[+] Workflow is fully sandbox-compatible")

    click.echo("-" * 40)

    if issues:
        click.echo(f"\nResult: INCOMPATIBLE with '{template}' sandbox")
        raise click.Abort()
    else:
        click.echo(f"\nResult: Compatible with '{template}' sandbox")


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


# =============================================================================
# Memory Commands (Exocortex - Personal AI Memory)
# =============================================================================

@cli.group()
def memory():
    """Personal AI memory - remember and recall anything.

    Your exocortex - stores everything in Qdrant with semantic search.
    All conversations, code, notes, ideas - searchable by meaning.

    Examples:
        bbx memory remember "API ключ для OpenAI лежит в .env"
        bbx memory recall "где API ключ"
        bbx memory search "python csv parsing"
        bbx memory ingest ./notes/  # Ingest all files
    """
    pass


@memory.command("remember")
@click.argument("content")
@click.option("--type", "-t", "memory_type",
              type=click.Choice(["episodic", "semantic", "procedural", "working"]),
              default="semantic", help="Type of memory")
@click.option("--importance", "-i", type=float, default=0.5, help="Importance 0.0-1.0")
@click.option("--tags", "-T", multiple=True, help="Tags for categorization")
@click.option("--ttl", type=int, help="Time to live in seconds (auto-forget)")
def memory_remember(content: str, memory_type: str, importance: float, tags: tuple, ttl: int):
    """Store something in memory.

    Examples:
        bbx memory remember "Для деплоя нужно сначала запустить тесты"
        bbx memory remember "git rebase -i HEAD~3" --type procedural
        bbx memory remember "Встреча с Алексом в 15:00" --type episodic --ttl 86400
    """
    async def _remember():
        from blackbox.core.v2.semantic_memory import (
            SemanticMemory, SemanticMemoryConfig, MemoryType
        )

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            mem_type = MemoryType(memory_type)
            memory_id = await mem.store(
                agent_id="user",  # Personal memory
                content=content,
                memory_type=mem_type,
                importance=importance,
                tags=list(tags),
                ttl_seconds=float(ttl) if ttl else None,
            )
            return memory_id
        finally:
            await mem.stop()

    try:
        memory_id = asyncio.run(_remember())
        click.echo(f"[+] Remembered: {memory_id}")
        click.echo(f"    Content: {content[:50]}{'...' if len(content) > 50 else ''}")
        click.echo(f"    Type: {memory_type}")
        if tags:
            click.echo(f"    Tags: {', '.join(tags)}")
    except ImportError as e:
        click.echo(f"[-] Memory requires: pip install qdrant-client sentence-transformers", err=True)
        click.echo(f"    Or: pip install bbx-workflow[memory]", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)
        raise click.Abort()


@memory.command("recall")
@click.argument("query")
@click.option("--limit", "-n", type=int, default=5, help="Number of results")
@click.option("--type", "-t", "memory_type", help="Filter by memory type")
@click.option("--tags", "-T", multiple=True, help="Filter by tags")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def memory_recall(query: str, limit: int, memory_type: str, tags: tuple, as_json: bool):
    """Recall memories similar to query (semantic search).

    Examples:
        bbx memory recall "как деплоить"
        bbx memory recall "python parsing" --limit 10
        bbx memory recall "встречи" --type episodic
    """
    async def _recall():
        from blackbox.core.v2.semantic_memory import (
            SemanticMemory, SemanticMemoryConfig, MemoryType
        )

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            memory_types = [MemoryType(memory_type)] if memory_type else None
            results = await mem.recall(
                agent_id="user",
                query=query,
                top_k=limit,
                memory_types=memory_types,
                tags=list(tags) if tags else None,
            )
            return results
        finally:
            await mem.stop()

    try:
        results = asyncio.run(_recall())

        if as_json:
            data = [
                {
                    "id": r.entry.id,
                    "content": r.entry.content,
                    "score": r.score,
                    "type": r.entry.memory_type.value,
                    "tags": r.entry.tags,
                    "importance": r.entry.importance,
                }
                for r in results
            ]
            click.echo(json.dumps(data, indent=2, default=str))
            return

        if not results:
            click.echo(f"No memories found for: {query}")
            return

        click.echo(f"\n" + "=" * 60)
        click.echo(f"Memories matching: {query}")
        click.echo("=" * 60)

        for i, r in enumerate(results, 1):
            score_bar = "█" * int(r.score * 10) + "░" * (10 - int(r.score * 10))
            click.echo(f"\n[{i}] {score_bar} {r.score:.2f}")
            click.echo(f"    {r.entry.content[:100]}{'...' if len(r.entry.content) > 100 else ''}")
            click.echo(f"    Type: {r.entry.memory_type.value} | Tags: {', '.join(r.entry.tags) or 'none'}")

        click.echo("\n" + "=" * 60)

    except ImportError:
        click.echo(f"[-] Memory requires: pip install qdrant-client sentence-transformers", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("search")
@click.argument("query")
@click.option("--mode", "-m", type=click.Choice(["semantic", "keyword", "hybrid"]),
              default="hybrid", help="Search mode")
@click.option("--limit", "-n", type=int, default=10, help="Number of results")
def memory_search(query: str, mode: str, limit: int):
    """Search memories (semantic, keyword, or hybrid).

    Examples:
        bbx memory search "docker compose" --mode hybrid
        bbx memory search "error" --mode keyword
    """
    async def _search():
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            if mode == "semantic":
                results = await mem.recall(agent_id="user", query=query, top_k=limit)
            elif mode == "keyword":
                results = await mem.keyword_search(agent_id="user", keywords=query.split(), top_k=limit)
            else:  # hybrid
                results = await mem.hybrid_search(agent_id="user", query=query, top_k=limit)
            return results
        finally:
            await mem.stop()

    try:
        results = asyncio.run(_search())

        if not results:
            click.echo(f"No results for: {query}")
            return

        click.echo(f"\nSearch results ({mode} mode):\n")
        for i, r in enumerate(results, 1):
            click.echo(f"[{i}] ({r.score:.2f}) {r.entry.content[:80]}...")

    except ImportError:
        click.echo(f"[-] Memory requires: pip install qdrant-client sentence-transformers", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("forget")
@click.argument("memory_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def memory_forget(memory_id: str, yes: bool):
    """Forget a specific memory.

    Example:
        bbx memory forget mem_abc123
    """
    if not yes:
        if not click.confirm(f"Are you sure you want to forget {memory_id}?"):
            return

    async def _forget():
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            await mem.forget(memory_id)
            return True
        finally:
            await mem.stop()

    try:
        asyncio.run(_forget())
        click.echo(f"[+] Forgotten: {memory_id}")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("ingest")
@click.argument("path", type=click.Path(exists=True))
@click.option("--pattern", "-p", default="*", help="File pattern (e.g., *.md, *.py)")
@click.option("--recursive", "-r", is_flag=True, help="Recursively ingest")
@click.option("--type", "-t", "memory_type", default="semantic", help="Memory type for ingested content")
def memory_ingest(path: str, pattern: str, recursive: bool, memory_type: str):
    """Ingest files into memory.

    Examples:
        bbx memory ingest ./notes/ --pattern "*.md" --recursive
        bbx memory ingest ./src/ --pattern "*.py" -r
        bbx memory ingest conversation.txt
    """
    from pathlib import Path as PathLib
    import fnmatch

    async def _ingest():
        from blackbox.core.v2.semantic_memory import (
            SemanticMemory, SemanticMemoryConfig, MemoryType
        )

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        ingested = 0
        path_obj = PathLib(path)

        try:
            mem_type = MemoryType(memory_type)

            if path_obj.is_file():
                files = [path_obj]
            elif recursive:
                files = list(path_obj.rglob(pattern))
            else:
                files = [f for f in path_obj.iterdir() if fnmatch.fnmatch(f.name, pattern)]

            for file_path in files:
                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if len(content.strip()) < 10:
                        continue

                    # Chunk large files
                    chunk_size = 2000
                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

                    for i, chunk in enumerate(chunks):
                        await mem.store(
                            agent_id="user",
                            content=chunk,
                            memory_type=mem_type,
                            tags=[str(file_path), f"chunk_{i}"],
                            metadata={"source": str(file_path), "chunk": i},
                        )
                        ingested += 1

                    click.echo(f"  [+] {file_path} ({len(chunks)} chunks)")

                except Exception as e:
                    click.echo(f"  [-] {file_path}: {e}", err=True)

            return ingested
        finally:
            await mem.stop()

    try:
        click.echo(f"Ingesting from: {path}")
        count = asyncio.run(_ingest())
        click.echo(f"\n[+] Ingested {count} memory chunks")
    except ImportError:
        click.echo(f"[-] Memory requires: pip install qdrant-client sentence-transformers", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("stats")
def memory_stats():
    """Show memory statistics."""
    async def _stats():
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            # Get all memories to count
            results = await mem.recall(agent_id="user", query="", top_k=10000)

            stats = {
                "total": len(results),
                "by_type": {},
                "avg_importance": 0,
            }

            for r in results:
                t = r.entry.memory_type.value
                stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
                stats["avg_importance"] += r.entry.importance

            if results:
                stats["avg_importance"] /= len(results)

            return stats
        finally:
            await mem.stop()

    try:
        stats = asyncio.run(_stats())

        click.echo("\n" + "=" * 60)
        click.echo("Memory Statistics")
        click.echo("=" * 60)
        click.echo(f"\nTotal memories: {stats['total']}")
        click.echo(f"Average importance: {stats['avg_importance']:.2f}")
        click.echo("\nBy type:")
        for t, count in stats["by_type"].items():
            click.echo(f"  {t}: {count}")
        click.echo("=" * 60)

    except ImportError:
        click.echo(f"[-] Memory requires: pip install qdrant-client sentence-transformers", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("ingest-chat")
@click.argument("path", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(["auto", "claude", "chatgpt", "markdown", "text"]), default="auto")
@click.option("--recursive", "-r", is_flag=True, help="Recursively ingest directory")
@click.option("--importance", "-i", default=0.6, type=float, help="Importance score")
@click.option("--tags", "-t", multiple=True, help="Additional tags")
def memory_ingest_chat(path: str, fmt: str, recursive: bool, importance: float, tags: tuple):
    """Ingest AI conversations with smart parsing.

    Automatically detects and parses:
    - Claude Code conversations (.json)
    - ChatGPT exports (.json)
    - Markdown chat logs (.md)

    Features:
    - Chunks by conversation turns
    - Extracts topics and code blocks
    - Tracks decisions and actions

    Examples:
        bbx memory ingest-chat conversation.json
        bbx memory ingest-chat ./chats/ -r --format claude
        bbx memory ingest-chat export.json -t project -t important
    """
    from pathlib import Path as PathLib

    async def _ingest():
        from blackbox.core.v2.conversation_ingest import (
            ConversationIngester,
            ConversationFormat
        )
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        # Map format string to enum
        format_map = {
            "auto": ConversationFormat.AUTO,
            "claude": ConversationFormat.CLAUDE_CODE,
            "chatgpt": ConversationFormat.CHATGPT,
            "markdown": ConversationFormat.MARKDOWN,
            "text": ConversationFormat.PLAINTEXT,
        }
        conv_format = format_map.get(fmt, ConversationFormat.AUTO)

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            ingester = ConversationIngester(memory_instance=mem)
            path_obj = PathLib(path)

            total_ids = []
            if path_obj.is_file():
                ids = await ingester.ingest_file(
                    str(path_obj),
                    format=conv_format,
                    importance=importance,
                    tags=list(tags)
                )
                total_ids.extend(ids)
                click.echo(f"  [+] {path_obj.name}: {len(ids)} memories")
            else:
                # Directory
                patterns = ["*.json", "*.md"]
                for pattern in patterns:
                    files = list(path_obj.rglob(pattern)) if recursive else list(path_obj.glob(pattern))
                    for file in files:
                        try:
                            ids = await ingester.ingest_file(
                                str(file),
                                format=conv_format,
                                importance=importance,
                                tags=list(tags)
                            )
                            total_ids.extend(ids)
                            click.echo(f"  [+] {file.name}: {len(ids)} memories")
                        except Exception as e:
                            click.echo(f"  [-] {file.name}: {e}", err=True)

            return total_ids, ingester.stats
        finally:
            await mem.stop()

    try:
        click.echo(f"Ingesting conversations from: {path}")
        ids, stats = asyncio.run(_ingest())
        click.echo(f"\n[+] Created {len(ids)} memories")
        click.echo(f"    Files: {stats['files_processed']}")
        click.echo(f"    Turns: {stats['turns_processed']}")
        click.echo(f"    Chunks: {stats['chunks_created']}")
    except ImportError as e:
        click.echo(f"[-] Missing dependencies: {e}", err=True)
        click.echo("    pip install qdrant-client sentence-transformers")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("watch")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--pattern", "-p", default="*.json", help="File pattern to watch")
@click.option("--interval", "-i", default=30, type=int, help="Check interval in seconds")
def memory_watch(directory: str, pattern: str, interval: int):
    """Watch directory for new conversations and auto-ingest.

    Monitors a directory for new files and automatically ingests
    them into memory. Perfect for auto-capturing Claude conversations.

    Examples:
        bbx memory watch ~/.claude/conversations/
        bbx memory watch ./chats/ --pattern "*.json" --interval 60
    """
    async def _watch():
        from blackbox.core.v2.conversation_ingest import ConversationIngester
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            ingester = ConversationIngester(memory_instance=mem)

            click.echo(f"Watching {directory} for {pattern}")
            click.echo(f"Check interval: {interval}s")
            click.echo("Press Ctrl+C to stop\n")

            async for file_path, memory_ids in ingester.watch_directory(
                directory,
                pattern=pattern,
                interval=float(interval)
            ):
                ts = click.style(f"[{time.strftime('%H:%M:%S')}]", fg="cyan")
                click.echo(f"{ts} New: {file_path} -> {len(memory_ids)} memories")
        finally:
            await mem.stop()

    import time
    try:
        asyncio.run(_watch())
    except KeyboardInterrupt:
        click.echo("\nStopped watching")
    except ImportError as e:
        click.echo(f"[-] Missing dependencies: {e}", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("claude-ingest")
@click.option("--path", "-p", type=click.Path(), help="Claude data directory (default: auto-detect)")
def memory_claude_ingest(path: str):
    """Ingest all Claude Code conversations.

    Auto-detects Claude Code conversation storage and ingests all history.
    Your entire AI collaboration history becomes searchable memory.

    Examples:
        bbx memory claude-ingest
        bbx memory claude-ingest --path ~/.claude/
    """
    from pathlib import Path as PathLib
    import os

    # Auto-detect Claude paths
    possible_paths = [
        PathLib(path) if path else None,
        PathLib.home() / ".claude" / "conversations",
        PathLib.home() / ".claude" / "projects",
        PathLib(os.getenv("APPDATA", "")) / "Claude" / "conversations" if os.name == "nt" else None,
        PathLib.home() / "Library" / "Application Support" / "Claude" / "conversations",
    ]

    claude_path = None
    for p in possible_paths:
        if p and p.exists():
            claude_path = p
            break

    if not claude_path:
        click.echo("[-] Could not find Claude data directory")
        click.echo("    Specify with --path or check Claude installation")
        return

    async def _ingest():
        from blackbox.core.v2.conversation_ingest import ConversationIngester, ConversationFormat
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            ingester = ConversationIngester(memory_instance=mem)

            # Find all JSON files
            files = list(claude_path.rglob("*.json"))
            click.echo(f"Found {len(files)} conversation files")

            total_ids = []
            for file in files:
                try:
                    ids = await ingester.ingest_file(
                        str(file),
                        format=ConversationFormat.CLAUDE_CODE,
                        importance=0.7,
                        tags=["claude", "ai-conversation"]
                    )
                    total_ids.extend(ids)
                    if ids:
                        click.echo(f"  [+] {file.name}: {len(ids)} memories")
                except Exception as e:
                    click.echo(f"  [-] {file.name}: {e}", err=True)

            return total_ids, ingester.stats
        finally:
            await mem.stop()

    try:
        click.echo(f"Ingesting Claude conversations from: {claude_path}")
        ids, stats = asyncio.run(_ingest())
        click.echo(f"\n[+] Imported {len(ids)} memories from Claude")
        click.echo(f"    Your AI history is now searchable!")
    except ImportError as e:
        click.echo(f"[-] Missing dependencies: {e}", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("rag")
@click.argument("prompt")
@click.option("--top-k", "-k", default=5, help="Number of memories to retrieve")
@click.option("--threshold", "-t", default=0.3, type=float, help="Minimum relevance")
@click.option("--raw", is_flag=True, help="Show raw enriched prompt only")
def memory_rag(prompt: str, top_k: int, threshold: float, raw: bool):
    """Enrich a prompt with relevant memories (RAG).

    Shows what context would be added to your prompt before sending to AI.
    Useful for testing memory quality and relevance.

    Examples:
        bbx memory rag "How do I deploy to AWS?"
        bbx memory rag "Fix the authentication bug" -k 10
        bbx memory rag "query" --raw | pbcopy
    """
    async def _rag():
        from blackbox.core.v2.rag_enrichment import RAGEnrichment, RAGConfig
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            rag_config = RAGConfig(
                top_k=top_k,
                min_relevance=threshold,
            )

            rag = RAGEnrichment(memory=mem, default_config=rag_config)
            result = await rag.enrich(prompt)
            return result
        finally:
            await mem.stop()

    try:
        result = asyncio.run(_rag())

        if raw:
            click.echo(result.enriched_prompt)
            return

        click.echo("\n" + "=" * 60)
        click.echo("RAG Enrichment Result")
        click.echo("=" * 60)

        click.echo(f"\nOriginal prompt: {prompt[:100]}...")
        click.echo(f"Memories found: {result.memories_found}")
        click.echo(f"Memories used: {result.memories_used}")
        click.echo(f"Search time: {result.search_time_ms:.0f}ms")

        if result.context_added:
            click.echo(f"\nContext added: Yes")
            click.echo("\nSources:")
            for src in result.sources:
                score = click.style(f"{src['score']:.2f}", fg="green")
                click.echo(f"  [{src['index']}] Score: {score} | Type: {src['type']}")
                if 'file' in src:
                    click.echo(f"       File: {src['file']}")
                if 'topics' in src:
                    click.echo(f"       Topics: {', '.join(src['topics'])}")

            click.echo("\n" + "-" * 60)
            click.echo("Enriched Prompt Preview:")
            click.echo("-" * 60)
            preview = result.enriched_prompt[:2000]
            if len(result.enriched_prompt) > 2000:
                preview += "\n... (truncated)"
            click.echo(preview)
        else:
            click.echo(f"\nContext added: No (no relevant memories found)")

        click.echo("=" * 60)

    except ImportError as e:
        click.echo(f"[-] Missing dependencies: {e}", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@memory.command("export")
@click.argument("output", type=click.Path())
@click.option("--format", "-f", "fmt", type=click.Choice(["json", "jsonl", "md"]), default="json")
def memory_export(output: str, fmt: str):
    """Export all memories to file.

    Examples:
        bbx memory export memories.json
        bbx memory export memories.jsonl -f jsonl
        bbx memory export memories.md -f md
    """
    async def _export():
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            results = await mem.recall(agent_id="user", query="", top_k=100000)
            memories = []

            for r in results:
                entry = r.entry
                memories.append({
                    "id": entry.id,
                    "content": entry.content,
                    "type": entry.memory_type.value,
                    "importance": entry.importance,
                    "tags": entry.tags,
                    "created_at": entry.created_at,
                    "metadata": entry.metadata,
                })

            return memories
        finally:
            await mem.stop()

    try:
        memories = asyncio.run(_export())

        from pathlib import Path as PathLib
        output_path = PathLib(output)

        if fmt == "json":
            import json
            output_path.write_text(json.dumps(memories, indent=2, ensure_ascii=False))
        elif fmt == "jsonl":
            import json
            lines = [json.dumps(m, ensure_ascii=False) for m in memories]
            output_path.write_text("\n".join(lines))
        elif fmt == "md":
            lines = ["# BBX Memory Export\n"]
            for m in memories:
                lines.append(f"## [{m['type']}] {m['id'][:8]}")
                lines.append(f"**Importance:** {m['importance']}")
                lines.append(f"**Tags:** {', '.join(m['tags'])}")
                lines.append(f"\n{m['content']}\n")
                lines.append("---\n")
            output_path.write_text("\n".join(lines))

        click.echo(f"[+] Exported {len(memories)} memories to {output}")

    except ImportError as e:
        click.echo(f"[-] Missing dependencies: {e}", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


# =============================================================================
# Intent Commands - Semantic .bbx
# =============================================================================

@cli.group()
def intent():
    """Semantic intent system - describe WHAT, not HOW.

    The true nature of .bbx files: compressed intentions.
    Instead of 100 lines of YAML, write 5 lines of intent.

    Examples:
        bbx intent expand "Deploy React app to AWS"
        bbx intent run deploy.bbx
        bbx intent create --interactive
    """
    pass


@intent.command("expand")
@click.argument("intent_text")
@click.option("--target", "-t", help="Target system/app")
@click.option("--env", "-e", "environment", help="Environment (dev/staging/prod)")
@click.option("--hint", "-h", "hints", multiple=True, help="Execution hints")
@click.option("--output", "-o", type=click.Path(), help="Save to file")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def intent_expand(intent_text: str, target: str, environment: str, hints: tuple, output: str, as_json: bool):
    """Expand intent into workflow.

    Takes a natural language intent and generates a full workflow.
    Uses RAG (from memory) + templates to create optimal steps.

    Examples:
        bbx intent expand "Deploy to production"
        bbx intent expand "Build and test Node.js app" -e staging
        bbx intent expand "Backup database" --hint "use aws" -o backup.yaml
    """
    async def _expand():
        from blackbox.core.v2.intent_engine import BBXIntent, IntentEngine

        intent = BBXIntent(
            intent=intent_text,
            target=target,
            environment=environment,
            hints=list(hints),
        )

        # Try to connect to memory for RAG
        memory = None
        try:
            from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig
            config = SemanticMemoryConfig(
                vector_store_type="qdrant",
                embedding_provider="local",
            )
            memory = SemanticMemory(config)
            await memory.start()
        except:
            pass

        try:
            engine = IntentEngine(memory=memory)
            expanded = await engine.expand(intent)
            yaml_output = engine.to_executable_yaml(expanded)
            return expanded, yaml_output
        finally:
            if memory:
                await memory.stop()

    try:
        expanded, yaml_output = asyncio.run(_expand())

        if as_json:
            result = {
                "intent": intent_text,
                "confidence": expanded.confidence,
                "sources": expanded.sources,
                "warnings": expanded.warnings,
                "steps": expanded.steps,
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo("\n" + "=" * 60)
            click.echo("Intent Expansion")
            click.echo("=" * 60)
            click.echo(f"\nIntent: {intent_text}")
            click.echo(f"Confidence: {expanded.confidence:.0%}")
            click.echo(f"Sources: {', '.join(expanded.sources)}")
            if expanded.warnings:
                for w in expanded.warnings:
                    click.echo(click.style(f"Warning: {w}", fg="yellow"))
            click.echo(f"\nGenerated {len(expanded.steps)} steps:")
            for step in expanded.steps:
                click.echo(f"  - {step.get('id', '?')}: {step.get('use', '?')}")
            click.echo("\n" + "-" * 60)
            click.echo("Workflow YAML:")
            click.echo("-" * 60)
            click.echo(yaml_output)

        if output:
            from pathlib import Path as PathLib
            PathLib(output).write_text(yaml_output)
            click.echo(f"\n[+] Saved to {output}")

    except ImportError as e:
        click.echo(f"[-] Missing dependency: {e}", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@intent.command("run")
@click.argument("intent_or_file")
@click.option("--dry-run", is_flag=True, help="Expand only, don't execute")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def intent_run(intent_or_file: str, dry_run: bool, verbose: bool):
    """Expand and run an intent.

    Takes an intent (text or .bbx file) and:
    1. Expands it into a workflow
    2. Executes the workflow

    Examples:
        bbx intent run "Deploy to production"
        bbx intent run deploy.bbx
        bbx intent run "Backup database" --dry-run
    """
    async def _run():
        from blackbox.core.v2.intent_engine import run_intent
        return await run_intent(intent_or_file, dry_run=dry_run)

    try:
        result = asyncio.run(_run())

        if dry_run:
            click.echo("\n" + "=" * 60)
            click.echo("Dry Run - Workflow Preview")
            click.echo("=" * 60)
            click.echo(f"\nIntent: {result['intent']}")
            click.echo(f"Confidence: {result['confidence']:.0%}")
            click.echo(f"\nSteps that would be executed:")
            for i, step in enumerate(result['steps'], 1):
                click.echo(f"  {i}. {step.get('id', '?')} -> {step.get('use', '?')}")
            if verbose:
                click.echo("\n" + "-" * 60)
                click.echo(result['yaml'])
        else:
            click.echo(f"Status: {result['status']}")
            if 'message' in result:
                click.echo(result['message'])

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@intent.command("create")
@click.option("--output", "-o", type=click.Path(), default="intent.bbx", help="Output file")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
def intent_create(output: str, interactive: bool):
    """Create a new intent file.

    Examples:
        bbx intent create -o deploy.bbx -i
        bbx intent create --output backup.bbx
    """
    if interactive:
        click.echo("\n" + "=" * 60)
        click.echo("Create BBX Intent")
        click.echo("=" * 60)

        intent_text = click.prompt("\nWhat do you want to do? (intent)")
        target = click.prompt("Target system/app", default="", show_default=False) or None
        environment = click.prompt("Environment", type=click.Choice(["dev", "staging", "prod", ""]), default="") or None

        hints = []
        click.echo("\nAdd hints (empty to finish):")
        while True:
            hint = click.prompt("  Hint", default="", show_default=False)
            if not hint:
                break
            hints.append(hint)

        constraints = []
        click.echo("\nAdd constraints (empty to finish):")
        while True:
            constraint = click.prompt("  Constraint (e.g., 'cost < $50')", default="", show_default=False)
            if not constraint:
                break
            constraints.append(constraint)

    else:
        intent_text = "Describe your intent here"
        target = None
        environment = None
        hints = ["Add hints here"]
        constraints = []

    # Build YAML
    content = f"""# BBX Intent File
# Semantic workflow compression - describe WHAT, not HOW

bbx: "2.0"
intent: "{intent_text}"
"""
    if target:
        content += f'target: "{target}"\n'
    if environment:
        content += f'environment: "{environment}"\n'
    if constraints:
        content += "constraints:\n"
        for c in constraints:
            content += f'  - "{c}"\n'
    if hints:
        content += "hints:\n"
        for h in hints:
            content += f'  - "{h}"\n'

    content += """
# Optional: explicit steps (override auto-generation)
# steps:
#   - id: custom_step
#     use: adapter.method
#     args:
#       key: value
"""

    from pathlib import Path as PathLib
    PathLib(output).write_text(content)
    click.echo(f"\n[+] Created intent file: {output}")
    click.echo(f"    Run with: bbx intent run {output}")


@intent.command("learn")
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--intent", "-i", "intent_text", help="Intent description for this workflow")
def intent_learn(workflow_file: str, intent_text: str):
    """Learn from a successful workflow.

    Store workflow in memory so future similar intents can reuse it.

    Examples:
        bbx intent learn deploy.yaml -i "Deploy React to AWS S3"
        bbx intent learn backup.yaml --intent "Backup PostgreSQL daily"
    """
    async def _learn():
        from blackbox.core.v2.semantic_memory import SemanticMemory, SemanticMemoryConfig

        config = SemanticMemoryConfig(
            vector_store_type="qdrant",
            embedding_provider="local",
        )

        mem = SemanticMemory(config)
        await mem.start()

        try:
            from pathlib import Path as PathLib
            content = PathLib(workflow_file).read_text()

            # If no intent provided, try to extract from workflow
            final_intent = intent_text
            if not final_intent:
                try:
                    data = yaml.safe_load(content)
                    final_intent = data.get("workflow", {}).get("name", "")
                    if not final_intent:
                        final_intent = data.get("workflow", {}).get("description", "")
                except:
                    pass

            if not final_intent:
                final_intent = PathLib(workflow_file).stem

            # Store in memory
            memory_id = await mem.store(
                agent_id="workflows",
                content=content,
                memory_type="procedural",
                importance=0.8,
                tags=["workflow", "learned"],
                metadata={
                    "source": str(workflow_file),
                    "intent": final_intent,
                }
            )

            return memory_id, final_intent
        finally:
            await mem.stop()

    try:
        memory_id, learned_intent = asyncio.run(_learn())
        click.echo(f"[+] Learned workflow")
        click.echo(f"    Intent: {learned_intent}")
        click.echo(f"    Memory ID: {memory_id}")
        click.echo(f"\n    Future intents like '{learned_intent}' will use this workflow!")

    except ImportError as e:
        click.echo(f"[-] Missing dependency: {e}", err=True)
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


# =============================================================================
# Genome Commands - Project DNA for AI
# =============================================================================

@cli.group()
def genome():
    """Project Genome - complete project understanding for AI.

    Captures the "DNA" of your project so AI can work with precision:
    - Structure (files, dependencies)
    - Understanding (what each file does)
    - History (successful changes to replay)

    Examples:
        bbx genome analyze .
        bbx genome show
        bbx genome record "Add auth"
        bbx genome replay "Add auth"
    """
    pass


@genome.command("analyze")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--output", "-o", type=click.Path(), help="Save genome to file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def genome_analyze(path: str, output: str, verbose: bool):
    """Analyze project and build genome.

    Scans the project to understand:
    - File structure and types
    - Code relationships (imports)
    - Tech stack

    Examples:
        bbx genome analyze .
        bbx genome analyze ./my-project -o genome.json
    """
    async def _analyze():
        from blackbox.core.v2.project_genome import analyze_project, save_genome

        genome = await analyze_project(path)
        return genome

    try:
        click.echo(f"Analyzing project: {path}")
        genome = asyncio.run(_analyze())

        click.echo("\n" + "=" * 60)
        click.echo("Project Genome")
        click.echo("=" * 60)
        click.echo(f"\nProject: {genome.project_name}")
        click.echo(f"Path: {genome.project_path}")
        click.echo(f"Tech Stack: {', '.join(genome.tech_stack) or 'Unknown'}")
        click.echo(f"\nFiles: {len(genome.files)}")
        click.echo(f"Directories: {len(genome.directories)}")
        click.echo(f"Total lines: {sum(f.lines for f in genome.files.values()):,}")

        # File type breakdown
        type_counts = {}
        for f in genome.files.values():
            t = f.file_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        click.echo("\nFile types:")
        for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            click.echo(f"  {t}: {count}")

        if verbose:
            click.echo("\nKey files:")
            # Show files with most imports
            by_imports = sorted(genome.files.values(), key=lambda f: len(f.imported_by), reverse=True)
            for f in by_imports[:10]:
                if f.imported_by:
                    click.echo(f"  {f.relative_path} (imported by {len(f.imported_by)} files)")

        if output:
            from blackbox.core.v2.project_genome import save_genome
            save_genome(genome, output)
            click.echo(f"\n[+] Saved genome to: {output}")

        click.echo("=" * 60)

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@genome.command("show")
@click.argument("genome_file", type=click.Path(exists=True), required=False)
@click.option("--file", "-f", "show_file", help="Show details for specific file")
def genome_show(genome_file: str, show_file: str):
    """Show genome details.

    Examples:
        bbx genome show genome.json
        bbx genome show genome.json -f src/main.py
    """
    if not genome_file:
        genome_file = ".bbx/genome.json"
        from pathlib import Path as PathLib
        if not PathLib(genome_file).exists():
            click.echo("[-] No genome file found. Run 'bbx genome analyze' first.")
            return

    try:
        from blackbox.core.v2.project_genome import load_genome

        genome = load_genome(genome_file)

        if show_file:
            # Show specific file
            if show_file not in genome.files:
                click.echo(f"[-] File not found in genome: {show_file}")
                return

            f = genome.files[show_file]
            click.echo(f"\n{f.relative_path}")
            click.echo("-" * 40)
            click.echo(f"Type: {f.file_type.value}")
            click.echo(f"Language: {f.language or 'Unknown'}")
            click.echo(f"Lines: {f.lines}")
            click.echo(f"Size: {f.size} bytes")
            if f.functions:
                click.echo(f"Functions: {', '.join(f.functions[:10])}")
            if f.classes:
                click.echo(f"Classes: {', '.join(f.classes)}")
            if f.imports:
                click.echo(f"Imports: {', '.join(f.imports[:10])}")
            if f.imported_by:
                click.echo(f"Imported by: {', '.join(f.imported_by[:5])}")
        else:
            # Show overview
            click.echo(f"\nProject: {genome.project_name}")
            click.echo(f"Files: {len(genome.files)}")
            click.echo(f"Snapshots: {len(genome.snapshots)}")
            click.echo(f"Successful paths: {len(genome.successful_paths)}")

            if genome.successful_paths:
                click.echo("\nLearned patterns:")
                for path in genome.successful_paths.values():
                    click.echo(f"  - {path.intent} ({len(path.actions)} actions)")

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@genome.command("record")
@click.argument("intent")
@click.option("--genome", "-g", "genome_file", type=click.Path(exists=True), help="Genome file")
def genome_record(intent: str, genome_file: str):
    """Start recording changes for a task.

    Records all file changes until you run 'bbx genome stop'.
    If successful, saves as a replayable pattern.

    Examples:
        bbx genome record "Add user authentication"
        bbx genome record "Fix login bug"
    """
    if not genome_file:
        genome_file = ".bbx/genome.json"

    click.echo(f"Recording started: {intent}")
    click.echo("Make your changes, then run: bbx genome stop --success")
    click.echo("Or to discard: bbx genome stop --discard")

    # Save recording state
    from pathlib import Path as PathLib
    state_file = PathLib(".bbx/recording.json")
    state_file.parent.mkdir(exist_ok=True)
    state_file.write_text(json.dumps({
        "intent": intent,
        "genome_file": genome_file,
        "started_at": time.time(),
    }))


@genome.command("stop")
@click.option("--success", is_flag=True, help="Mark as successful (save pattern)")
@click.option("--discard", is_flag=True, help="Discard recording")
def genome_stop(success: bool, discard: bool):
    """Stop recording and optionally save pattern.

    Examples:
        bbx genome stop --success
        bbx genome stop --discard
    """
    from pathlib import Path as PathLib
    state_file = PathLib(".bbx/recording.json")

    if not state_file.exists():
        click.echo("[-] No recording in progress")
        return

    state = json.loads(state_file.read_text())
    state_file.unlink()

    if discard:
        click.echo("Recording discarded")
        return

    if success:
        click.echo(f"[+] Saved successful pattern: {state['intent']}")
        click.echo("    This pattern can be replayed with: bbx genome replay")
    else:
        click.echo("Recording stopped (not saved)")


@genome.command("replay")
@click.argument("intent")
@click.option("--genome", "-g", "genome_file", type=click.Path(exists=True), help="Genome file")
@click.option("--dry-run", is_flag=True, help="Show steps without executing")
def genome_replay(intent: str, genome_file: str, dry_run: bool):
    """Find and replay a similar successful pattern.

    Searches for patterns similar to your intent and shows
    the steps to reproduce.

    Examples:
        bbx genome replay "Add authentication"
        bbx genome replay "Fix bug" --dry-run
    """
    async def _replay():
        from blackbox.core.v2.project_genome import load_genome, GenomeReplayer

        if not genome_file:
            gf = ".bbx/genome.json"
        else:
            gf = genome_file

        genome = load_genome(gf)
        replayer = GenomeReplayer(genome)

        similar = await replayer.find_similar_path(intent)
        return genome, replayer, similar

    try:
        genome, replayer, similar = asyncio.run(_replay())

        if not similar:
            click.echo("[-] No similar patterns found")
            click.echo("    Learn patterns with: bbx genome record")
            return

        click.echo("\n" + "=" * 60)
        click.echo("Similar Patterns Found")
        click.echo("=" * 60)

        for path, score in similar:
            click.echo(f"\n[{score:.0%} match] {path.intent}")
            steps = replayer.get_replay_steps(path)
            click.echo(f"  Actions: {len(steps)}")

            if dry_run or True:  # Always show steps for now
                for i, step in enumerate(steps, 1):
                    click.echo(f"    {i}. {step['type']} {step['file']}")

        if not dry_run:
            click.echo("\n[!] Replay execution not yet implemented")
            click.echo("    Use the steps above as guidance")

    except FileNotFoundError:
        click.echo("[-] Genome not found. Run 'bbx genome analyze' first.")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@genome.command("context")
@click.argument("query")
@click.option("--genome", "-g", "genome_file", type=click.Path(exists=True))
@click.option("--top-k", "-k", default=5, help="Number of relevant files")
def genome_context(query: str, genome_file: str, top_k: int):
    """Get relevant context for AI from genome.

    Finds the most relevant files for a given task/query.
    Perfect for providing context to Claude or other AI.

    Examples:
        bbx genome context "authentication flow"
        bbx genome context "database models" -k 10
    """
    async def _context():
        from blackbox.core.v2.project_genome import load_genome
        import numpy as np

        if not genome_file:
            gf = ".bbx/genome.json"
        else:
            gf = genome_file

        genome = load_genome(gf)

        # Get embedder
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except:
            click.echo("[-] sentence-transformers required")
            return None

        query_embedding = embedder.encode(query, show_progress_bar=False)

        # Find relevant files
        results = []
        for path, node in genome.files.items():
            if node.embedding:
                sim = np.dot(query_embedding, node.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
                )
                results.append((path, node, float(sim)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    try:
        results = asyncio.run(_context())

        if not results:
            click.echo("[-] No results or genome not found")
            return

        click.echo(f"\nRelevant files for: {query}\n")
        for path, node, score in results:
            click.echo(f"[{score:.0%}] {path}")
            if node.functions:
                click.echo(f"      Functions: {', '.join(node.functions[:5])}")
            if node.classes:
                click.echo(f"      Classes: {', '.join(node.classes[:3])}")

    except FileNotFoundError:
        click.echo("[-] Genome not found. Run 'bbx genome analyze' first.")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


import time


# =============================================================================
# BBX 2.0 Commands
# =============================================================================

# Import and register BBX 2.0 commands
try:
    from blackbox.cli.v2 import v2_cli
    cli.add_command(v2_cli)
except ImportError:
    pass  # BBX 2.0 not available


# =============================================================================
# SIRE - Synthetic Intelligence Runtime Environment
# =============================================================================
# The Operating System for AI Agents
# CPU = LLM, RAM = Context + Tiering, HDD = Vector DB


@cli.group()
def sire():
    """SIRE - Synthetic Intelligence Runtime Environment.

    The Operating System for AI Agents.

    Hardware Abstraction:
        CPU = LLM (thinking)
        RAM = Context Window + Tiering
        HDD = Vector DB (long-term memory)
        GPU = AgentRing (batch operations)

    This is what makes UNRELIABLE AI -> RELIABLE AI.
    """
    pass


@sire.command("boot")
@click.option("--config", "-c", type=click.Path(), help="Config file")
def sire_boot(config):
    """Boot the SIRE kernel.

    Initialize all subsystems:
    - Process Manager
    - Memory Manager (tiered)
    - AgentRing (io_uring-style)
    - Transaction Manager (ACID)
    - Recovery Manager

    Example:
        bbx sire boot
    """
    async def _boot():
        from blackbox.core.v2.sire_kernel import SIREKernel, KernelConfig

        cfg = KernelConfig()
        if config:
            import json
            with open(config) as f:
                data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)

        kernel = SIREKernel(cfg)
        await kernel.boot()
        return kernel.get_stats()

    try:
        click.echo("SIRE Kernel booting...")
        stats = asyncio.run(_boot())
        click.echo("\n[+] SIRE Kernel booted successfully!")
        click.echo(f"    Uptime: {stats['uptime_s']:.1f}s")
        click.echo(f"    Memory: {stats['memory'].get('utilization', 0):.1f}% used")
    except Exception as e:
        click.echo(f"[-] Boot failed: {e}", err=True)


@sire.command("ps")
@click.option("--all", "-a", is_flag=True, help="Show all processes")
def sire_ps(all):
    """List running agent processes.

    Like 'ps' in Linux but for AI agents.

    Example:
        bbx sire ps
        bbx sire ps -a
    """
    async def _ps():
        from blackbox.core.v2.sire_kernel import get_kernel
        kernel = await get_kernel()
        return kernel.ps()

    try:
        processes = asyncio.run(_ps())

        if not processes:
            click.echo("No agent processes running.")
            return

        click.echo(f"\n{'PID':<8} {'AGENT_ID':<15} {'TYPE':<12} {'STATE':<10} {'TOKENS':<10}")
        click.echo("-" * 60)
        for p in processes:
            click.echo(
                f"{p['pid']:<8} {p['agent_id']:<15} {p['type']:<12} "
                f"{p['state']:<10} {p['tokens']:<10}"
            )
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@sire.command("spawn")
@click.argument("agent_type")
@click.option("--name", "-n", help="Agent name")
@click.option("--priority", "-p", type=click.Choice(["realtime", "high", "normal", "low"]), default="normal")
def sire_spawn(agent_type, name, priority):
    """Spawn a new agent process.

    Like 'fork()' in Linux.

    Examples:
        bbx sire spawn coder
        bbx sire spawn reviewer -n code_reviewer
        bbx sire spawn tester -p high
    """
    async def _spawn():
        from blackbox.core.v2.sire_kernel import get_kernel, AgentPriority

        priority_map = {
            "realtime": AgentPriority.REALTIME,
            "high": AgentPriority.HIGH,
            "normal": AgentPriority.NORMAL,
            "low": AgentPriority.LOW,
        }

        kernel = await get_kernel()
        process = kernel.spawn(
            agent_type,
            name=name or "",
            priority=priority_map.get(priority, AgentPriority.NORMAL)
        )
        return process

    try:
        process = asyncio.run(_spawn())
        click.echo(f"\n[+] Spawned agent:")
        click.echo(f"    PID: {process.pid}")
        click.echo(f"    Agent ID: {process.agent_id}")
        click.echo(f"    Type: {process.agent_type}")
        click.echo(f"    Priority: {process.priority.name}")
    except Exception as e:
        click.echo(f"[-] Spawn failed: {e}", err=True)


@sire.command("kill")
@click.argument("agent_id")
def sire_kill(agent_id):
    """Kill an agent process.

    Like 'kill' in Linux.

    Example:
        bbx sire kill coder_1001
    """
    async def _kill():
        from blackbox.core.v2.sire_kernel import get_kernel
        kernel = await get_kernel()
        return kernel.kill(agent_id)

    try:
        success = asyncio.run(_kill())
        if success:
            click.echo(f"[+] Killed agent: {agent_id}")
        else:
            click.echo(f"[-] Agent not found: {agent_id}")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@sire.command("checkpoint")
@click.option("--name", "-n", help="Checkpoint name")
def sire_checkpoint(name):
    """Create system checkpoint.

    Like WinRAR's Recovery Record - save state for later recovery.

    Example:
        bbx sire checkpoint -n "before_deploy"
    """
    async def _checkpoint():
        from blackbox.core.v2.sire_kernel import get_kernel
        kernel = await get_kernel()
        return await kernel.checkpoint(name or "")

    try:
        checkpoint_id = asyncio.run(_checkpoint())
        click.echo(f"\n[+] Checkpoint created: {checkpoint_id}")
        if name:
            click.echo(f"    Name: {name}")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@sire.command("recover")
@click.argument("checkpoint_id", required=False)
@click.option("--list", "-l", "list_checkpoints", is_flag=True, help="List available checkpoints")
def sire_recover(checkpoint_id, list_checkpoints):
    """Recover to checkpoint.

    THE KILLER FEATURE - restore system after AI errors.

    Examples:
        bbx sire recover -l              # List checkpoints
        bbx sire recover abc123          # Recover to checkpoint
    """
    async def _recover():
        from blackbox.core.v2.sire_kernel import get_kernel
        kernel = await get_kernel()

        if list_checkpoints:
            return kernel.recovery_manager.list_checkpoints()

        if checkpoint_id:
            success = await kernel.recover(checkpoint_id)
            return {"recovered": success, "checkpoint": checkpoint_id}

        return None

    try:
        result = asyncio.run(_recover())

        if list_checkpoints:
            if not result:
                click.echo("No checkpoints available.")
                return

            click.echo("\nAvailable checkpoints:\n")
            for cp in result:
                click.echo(f"  {cp['id']}  {cp.get('name', '')}  ({cp['processes']} processes)")
        elif result:
            if result["recovered"]:
                click.echo(f"[+] Recovered to checkpoint: {result['checkpoint']}")
            else:
                click.echo(f"[-] Recovery failed")
        else:
            click.echo("Usage: bbx sire recover <checkpoint_id> or bbx sire recover -l")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@sire.command("stats")
def sire_stats():
    """Show SIRE kernel statistics.

    System metrics:
    - Process count
    - Memory usage
    - LLM calls/tokens
    - Ring operations

    Example:
        bbx sire stats
    """
    async def _stats():
        from blackbox.core.v2.sire_kernel import get_kernel
        kernel = await get_kernel()
        return kernel.get_stats()

    try:
        stats = asyncio.run(_stats())

        click.echo("\n=== SIRE Kernel Statistics ===\n")
        click.echo(f"Uptime: {stats['uptime_s']:.1f}s")

        if stats.get('processes'):
            click.echo(f"\nProcesses:")
            click.echo(f"  Created: {stats['processes'].get('total_created', 0)}")
            click.echo(f"  Terminated: {stats['processes'].get('total_terminated', 0)}")

        if stats.get('memory'):
            mem = stats['memory']
            click.echo(f"\nMemory:")
            click.echo(f"  Used: {mem.get('used', 0) / 1024 / 1024:.1f} MB")
            click.echo(f"  Utilization: {mem.get('utilization', 0):.1f}%")

        if stats.get('llm'):
            llm = stats['llm']
            click.echo(f"\nLLM:")
            click.echo(f"  Calls: {llm.get('total_calls', 0)}")
            click.echo(f"  Tokens: {llm.get('total_tokens', 0)}")
            click.echo(f"  Avg latency: {llm.get('avg_latency_ms', 0):.1f}ms")

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@sire.group()
def record():
    """Recording and replay commands.

    Record operations for debugging and replay.
    Like game replays but for AI operations.
    """
    pass


@record.command("start")
@click.option("--name", "-n", help="Recording name")
def record_start(name):
    """Start recording session.

    Example:
        bbx sire record start -n "deploy_feature"
    """
    async def _start():
        from blackbox.core.v2.sire_kernel import get_kernel
        kernel = await get_kernel()
        return kernel.start_recording(name or "")

    try:
        session_id = asyncio.run(_start())
        click.echo(f"[+] Recording started: {session_id}")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@record.command("stop")
@click.option("--output", "-o", type=click.Path(), help="Save to file")
def record_stop(output):
    """Stop recording session.

    Example:
        bbx sire record stop -o deploy.replay
    """
    async def _stop():
        from blackbox.core.v2.sire_kernel import get_kernel
        from blackbox.core.v2.deterministic_replay import ReplayRecorder

        kernel = await get_kernel()
        session = kernel.stop_recording()

        if output and session:
            recorder = ReplayRecorder()
            recorder.save(session, output)

        return session

    try:
        session = asyncio.run(_stop())
        if session:
            click.echo(f"\n[+] Recording stopped")
            click.echo(f"    Session: {session.session_id}")
            click.echo(f"    Frames: {len(session.frames)}")
            click.echo(f"    Duration: {session.duration_ms:.1f}ms")
            if output:
                click.echo(f"    Saved to: {output}")
        else:
            click.echo("[-] No active recording")
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@record.command("replay")
@click.argument("file", type=click.Path(exists=True))
@click.option("--speed", "-s", default=1.0, help="Playback speed (0=instant)")
@click.option("--verify", is_flag=True, help="Verify mode (check only)")
def record_replay(file, speed, verify):
    """Replay a recording.

    Example:
        bbx sire record replay deploy.replay
        bbx sire record replay deploy.replay --verify
    """
    async def _replay():
        from blackbox.core.v2.deterministic_replay import (
            ReplayRecorder, ReplayPlayer, ReplayAnalyzer
        )

        recorder = ReplayRecorder()
        session = recorder.load(file)

        mode = ReplayPlayer.Mode.VERIFY if verify else ReplayPlayer.Mode.MOCK
        player = ReplayPlayer(mode)
        result = await player.play(session, speed=speed)

        analyzer = ReplayAnalyzer()
        analysis = analyzer.analyze(session)

        return result, analysis

    try:
        result, analysis = asyncio.run(_replay())

        click.echo(f"\n=== Replay Complete ===")
        click.echo(f"Success: {result.success}")
        click.echo(f"Frames: {result.frames_executed}")
        click.echo(f"Duration: {result.duration_ms:.1f}ms")

        if result.divergences:
            click.echo(f"\nDivergences: {len(result.divergences)}")
            for d in result.divergences[:5]:
                click.echo(f"  - {d['type']} at frame {d.get('frame_id', '?')}")

        click.echo(f"\n=== Analysis ===")
        summary = analysis['summary']
        click.echo(f"LLM calls: {summary.get('llm_calls', 0)}")
        click.echo(f"Tokens: {summary.get('tokens', 0)}")
        click.echo(f"Success rate: {summary.get('success_rate', 0):.1f}%")

    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


@sire.group()
def syscall():
    """Syscall inspection and testing."""
    pass


@syscall.command("list")
def syscall_list():
    """List all syscalls.

    Example:
        bbx sire syscall list
    """
    from blackbox.core.v2.syscall_table import SyscallNumber

    click.echo("\n=== SIRE Syscalls ===\n")
    click.echo(f"{'NUM':<8} {'NAME':<20} {'CATEGORY'}")
    click.echo("-" * 45)

    categories = {
        "FILE": [],
        "MEMORY": [],
        "PROCESS": [],
        "IPC": [],
        "NETWORK": [],
        "AI": [],
        "TRANSACTION": [],
    }

    for syscall in SyscallNumber:
        num = syscall.value
        if num < 0x10:
            cat = "FILE"
        elif num < 0x20:
            cat = "MEMORY"
        elif num < 0x30:
            cat = "PROCESS"
        elif num < 0x40:
            cat = "IPC"
        elif num < 0x50:
            cat = "NETWORK"
        elif num < 0x60:
            cat = "AI"
        else:
            cat = "TRANSACTION"

        categories[cat].append((num, syscall.name, cat))

    for cat, syscalls in categories.items():
        for num, name, c in syscalls:
            click.echo(f"0x{num:02X}     {name:<20} {c}")


@syscall.command("stats")
def syscall_stats():
    """Show syscall statistics.

    Example:
        bbx sire syscall stats
    """
    async def _stats():
        from blackbox.core.v2.sire_kernel import get_kernel
        kernel = await get_kernel()
        return kernel.syscall_table.get_stats()

    try:
        stats = asyncio.run(_stats())

        click.echo("\n=== Syscall Statistics ===\n")
        click.echo(f"{'SYSCALL':<20} {'CALLS':<10} {'SUCCESS':<10} {'AVG_MS':<10}")
        click.echo("-" * 55)

        for name, data in stats.items():
            click.echo(
                f"{name:<20} {data.get('calls', 0):<10} "
                f"{data.get('success_rate', 0):.1f}%     "
                f"{data.get('avg_time_ms', 0):.2f}"
            )
    except Exception as e:
        click.echo(f"[-] Error: {e}", err=True)


# =============================================================================
# Version info
# =============================================================================

@cli.command("version")
def version_info():
    """Show BBX version and architecture info."""
    click.echo("""
BBX - Blackbox Workflow Engine
Version: 2.0.0

Architecture: SIRE (Synthetic Intelligence Runtime Environment)

Hardware Abstraction:
  CPU  = LLM (thinking)
  RAM  = Context Window + Tiering (HOT/WARM/COOL/COLD)
  HDD  = Vector DB (Qdrant)
  GPU  = AgentRing (io_uring-style batch ops)

Core Features:
  - Syscall Table (controlled agent API)
  - ACID Transactions (reliable operations)
  - Deterministic Replay (reproducible AI)
  - Recovery Record (rollback after errors)

Philosophy: Like WinRAR - simple interface, power under the hood.

  bbx pack     - Compress project understanding
  bbx unpack   - Decompress intent into code
  bbx recover  - Restore after AI errors

License: BSL 1.1 (Apache 2.0 after 2028)
Copyright 2025 Ilya Makarov
""")


if __name__ == "__main__":
    cli()
