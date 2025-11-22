# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
import yaml

# Force UTF-8 encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from blackbox.core.config import get_settings
from blackbox.core.runtime import run_file
from blackbox.core.schema import BBXSchemaGenerator
from blackbox.core.validation import WorkflowValidationError, validate_workflow

settings = get_settings()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, settings.observability.log_level, logging.INFO),
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🚀 Blackbox Workflow Engine CLI"""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option(
    "--input",
    "-i",
    "inputs",
    multiple=True,
    help="Workflow inputs in key=value format (can be used multiple times)",
)
def run(file_path: str, verbose: bool, output: str, inputs: tuple):
    """
    Run a workflow from a .bbx file.

    Example:
        blackbox run workflow.bbx
        blackbox run workflow.bbx --verbose
        blackbox run workflow.bbx --output json
        blackbox run workflow.bbx -i project_name=MyApp -i include_docker=true
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Running workflow: {file_path}")

    # Parse inputs
    input_dict = {}
    for input_str in inputs:
        if "=" not in input_str:
            logger.warning(f"Invalid input format (expected key=value): {input_str}")
            continue

        key, value = input_str.split("=", 1)

        # Try to parse as boolean
        if value.lower() in ("true", "false"):
            input_dict[key] = value.lower() == "true"
        # Try to parse as number
        elif value.isdigit():
            input_dict[key] = int(value)
        elif value.replace(".", "", 1).isdigit():
            input_dict[key] = float(value)
        else:
            input_dict[key] = value

    if verbose and input_dict:
        logger.debug(f"Workflow inputs: {input_dict}")

    try:
        results = asyncio.run(run_file(file_path, inputs=input_dict))

        if output == "json":
            click.echo(json.dumps(results, indent=2, default=str))
        else:
            click.echo("\n" + "=" * 60)
            click.echo("📊 Workflow Execution Results")
            click.echo("=" * 60)

            for step_id, result in results.items():
                status = result.get("status", "unknown")
                icon = (
                    "✅" if status == "success" else "❌" if status == "error" else "⏭️"
                )

                click.echo(f"\n{icon} Step: {step_id}")
                click.echo(f"   Status: {status}")

                if result.get("output"):
                    click.echo(f"   Output: {result['output']}")
                if result.get("error"):
                    click.echo(f"   Error: {result['error']}", err=True)

            click.echo("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=verbose)
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def validate(file_path: str):
    """
    Validate a workflow file without executing it.

    Example:
        blackbox validate workflow.bbx
    """
    try:
        with open(file_path, "r") as f:
            workflow_data = yaml.safe_load(f)

        # Support both v6 format {"workflow": {...}} and simple format
        if "workflow" in workflow_data:
            workflow = workflow_data["workflow"]
        else:
            workflow = workflow_data

        validate_workflow(workflow)

        click.echo(f"✅ Workflow is valid: {file_path}")

    except WorkflowValidationError as e:
        click.echo("❌ Workflow validation failed:", err=True)
        for error in e.errors:
            click.echo(f"   - {error}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def validate_all(directory: str):
    """
    Validate all .bbx files in a directory.

    Example:
        blackbox validate-all ./examples
    """
    path = Path(directory)
    bbx_files = list(path.glob("*.bbx"))

    if not bbx_files:
        click.echo(f"No .bbx files found in {directory}")
        return

    click.echo(f"Validating {len(bbx_files)} workflow(s)...\n")

    valid_count = 0
    invalid_count = 0

    for bbx_file in bbx_files:
        try:
            with open(bbx_file, "r") as f:
                workflow_data = yaml.safe_load(f)

            # Support both v6 format {"workflow": {...}} and simple format
            if "workflow" in workflow_data:
                workflow = workflow_data["workflow"]
            else:
                workflow = workflow_data

            validate_workflow(workflow)
            click.echo(f"✅ {bbx_file.name}")
            valid_count += 1

        except WorkflowValidationError as e:
            click.echo(f"❌ {bbx_file.name}:")
            for error in e.errors:
                click.echo(f"   - {error}")
            invalid_count += 1
        except Exception as e:
            click.echo(f"❌ {bbx_file.name}: {e}")
            invalid_count += 1

    click.echo(f"\nResults: {valid_count} valid, {invalid_count} invalid")

    if invalid_count > 0:
        raise click.Abort()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def info(file_path: str):
    """
    Show information about a workflow file.

    Example:
        blackbox info workflow.bbx
    """
    try:
        with open(file_path, "r") as f:
            workflow_data = yaml.safe_load(f)

        click.echo("\n" + "=" * 60)
        click.echo("📋 Workflow Information")
        click.echo("=" * 60)

        click.echo(f"\nID: {workflow_data.get('id', 'N/A')}")
        click.echo(f"Name: {workflow_data.get('name', 'N/A')}")
        click.echo(f"Version: {workflow_data.get('version', 'N/A')}")
        click.echo(f"Description: {workflow_data.get('description', 'N/A')}")

        steps = workflow_data.get("steps", {})
        if isinstance(steps, dict):
            click.echo(f"\nSteps: {len(steps)}")
            for step_id, step_config in steps.items():
                adapter = step_config.get("use", "unknown")
                click.echo(f"  - {step_id} ({adapter})")
        elif isinstance(steps, list):
            click.echo(f"\nSteps: {len(steps)}")
            for step in steps:
                step_id = step.get("id", "unknown")
                adapter = (
                    f"{step.get('mcp', 'unknown')}.{step.get('method', 'unknown')}"
                )
                click.echo(f"  - {step_id} ({adapter})")

        click.echo("\n" + "=" * 60)

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command("version")
def show_version():
    """Show Blackbox version."""
    click.echo("Blackbox Workflow Engine v1.0.0")
    click.echo("License: PolyForm Noncommercial")


@cli.command()
def test():
    """Run the Blackbox test suite (using Blackbox!)."""
    workflow_path = Path(__file__).parent / "workflows" / "system" / "test.bbx"
    if not workflow_path.exists():
        click.echo("❌ Test workflow not found!", err=True)
        raise click.Abort()

    click.echo("🧪 Running tests via Blackbox...")
    asyncio.run(run_file(str(workflow_path)))


@cli.command()
def cleanup():
    """Clean up temporary files (using Blackbox!)."""
    workflow_path = Path(__file__).parent / "workflows" / "system" / "cleanup.bbx"
    if not workflow_path.exists():
        click.echo("❌ Cleanup workflow not found!", err=True)
        raise click.Abort()

    click.echo("🧹 Cleaning up via Blackbox...")
    asyncio.run(run_file(str(workflow_path)))


@cli.command()
@click.option("--output", "-o", default="bbx.schema.json", help="Output file path")
def schema(output: str):
    """Generate JSON Schema for VS Code."""
    schema_data = BBXSchemaGenerator.generate()
    with open(output, "w") as f:
        json.dump(schema_data, f, indent=2)
    click.echo(f"✅ Schema generated: {output}")
    click.echo("Add this to your VS Code settings.json:")
    click.echo(f'  "json.schemas": [{{"fileMatch": ["*.bbx"], "url": "./{output}"}}]')


@cli.command()
def wizard():
    """🧙 Interactive workflow creator."""
    from blackbox.cli.wizard import WorkflowWizard

    WorkflowWizard.run()


@cli.command()
def init():
    """Alias for wizard."""
    from blackbox.cli.wizard import WorkflowWizard

    WorkflowWizard.run()


@cli.command()
def schedule():
    """📅 List scheduled workflows."""
    from blackbox.core.scheduler import Scheduler

    triggers = Scheduler.scan_triggers()

    if not triggers:
        click.echo("No scheduled workflows found.")
        return

    click.echo("\n" + "=" * 60)
    click.echo("📅 Scheduled Workflows")
    click.echo("=" * 60)

    for t in triggers:
        click.echo(f"\nWorkflow: {t['workflow']}")
        click.echo(f"  ID: {t['workflow_id']}")
        click.echo(f"  Type: {t['type']}")
        if t["type"] == "cron":
            click.echo(f"  Schedule: {t['schedule']}")
        elif t["type"] == "webhook":
            click.echo(f"  Path: {t['details'].get('path')}")

    click.echo("\n" + "=" * 60)


@cli.command()
@click.option("--pull", is_flag=True, help="Pull missing images")
def system(pull: bool):
    """🏥 Check system health and Docker images."""
    click.echo("\n" + "=" * 60)
    click.echo("🏥 System Health Check")
    click.echo("=" * 60)

    # 1. Check Docker
    click.echo("\n🐳 Docker Status:")
    docker_ok = False
    try:
        import subprocess

        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            click.echo(f"   ✅ Available: {result.stdout.strip()}")

            # Check daemon
            daemon_check = subprocess.run(
                ["docker", "info"], capture_output=True, text=True
            )
            if daemon_check.returncode == 0:
                click.echo("   ✅ Daemon: Running")
                docker_ok = True
            else:
                click.echo("   ❌ Daemon: Not running (Is Docker Desktop started?)")
        else:
            click.echo("   ❌ Docker CLI not found")
    except FileNotFoundError:
        click.echo("   ❌ Docker CLI not found")
    except Exception as e:
        click.echo(f"   ❌ Error checking Docker: {e}")

    if not docker_ok:
        click.echo("\n⚠️  Docker is required for infrastructure adapters.")
        return

    # 2. Check Images
    click.echo("\n📦 Adapter Images:")

    images = [
        "hashicorp/terraform:latest",
        "amazon/aws-cli:latest",
        "google/cloud-sdk:latest",
        "mcr.microsoft.com/azure-cli",
        "dtzar/helm-kubectl:latest",
        "willhallonline/ansible:latest",
    ]

    missing_images = []

    for image in images:
        try:
            # Check if image exists locally
            check = subprocess.run(
                ["docker", "images", "-q", image], capture_output=True, text=True
            )
            exists = check.returncode == 0 and bool(check.stdout.strip())

            if exists:
                click.echo(f"   ✅ {image}")
            else:
                click.echo(f"   ❌ {image} (Missing)")
                missing_images.append(image)
        except Exception:
            click.echo(f"   ❓ {image} (Check failed)")

    # 3. Pull missing
    if missing_images:
        if pull:
            click.echo("\n⬇️  Pulling missing images...")
            for image in missing_images:
                click.echo(f"   Pulling {image}...")
                try:
                    subprocess.run(["docker", "pull", image], check=True)
                    click.echo(f"   ✅ Pulled {image}")
                except subprocess.CalledProcessError:
                    click.echo(f"   ❌ Failed to pull {image}")
        else:
            click.echo(
                "\n💡 Tip: Run 'blackbox system --pull' to download missing images."
            )
    else:
        click.echo("\n✨ All systems go!")

    click.echo("\n" + "=" * 60)


# ============================================================================
# PHASE 3: UNIVERSAL ADAPTER CLI EXTENSIONS
# ============================================================================


@cli.group()
def package():
    """📦 Manage Universal Adapter packages"""
    pass


@package.command("install")
@click.argument("package_name")
@click.option("--version", default="latest", help="Package version")
def install_package(package_name: str, version: str):
    """Install a Universal Adapter package"""
    from blackbox.core.package_manager import get_package_manager

    pm = get_package_manager()
    click.echo(f"📦 Installing {package_name}@{version}...")

    if pm.install(package_name, version):
        click.echo(f"✅ Successfully installed {package_name}")
    else:
        click.echo(f"❌ Failed to install {package_name}", err=True)
        raise click.Abort()


@package.command("list")
def list_packages():
    """List all available packages"""
    from blackbox.core.package_manager import get_package_manager

    pm = get_package_manager()
    available = pm.list_available()
    installed = pm.list_installed()

    click.echo("\n📚 Available Packages:")
    click.echo("=" * 60)

    for pkg in available:
        status = "✅ installed" if pkg in installed else "⬜ available"
        click.echo(f"  {pkg:20} {status}")

    click.echo(f"\nTotal: {len(available)} available, {len(installed)} installed")


@package.command(name="info_pkg")
@click.argument("package_name")
def info_pkg(package_name: str):
    """Show package information"""
    from blackbox.core.package_manager import get_package_manager

    pm = get_package_manager()
    definition = pm.get(package_name)

    if not definition:
        click.echo(f"❌ Package not found: {package_name}", err=True)
        raise click.Abort()

    click.echo(f"\n📋 Package: {package_name}")
    click.echo("=" * 60)
    click.echo(f"ID:     {definition.get('id')}")
    click.echo(f"Image:  {definition.get('uses')}")

    if "auth" in definition:
        click.echo(f"Auth:   {definition['auth'].get('type')}")

    if "output_parser" in definition:
        click.echo(f"Output: {definition['output_parser'].get('type')}")

    click.echo("\nCommand Template:")
    cmd = definition.get("cmd", [])
    for line in (cmd if isinstance(cmd, list) else [cmd]):
        click.echo(f"  {line}")


@package.command(name="validate_pkg")
def validate_pkg():
    """Validate all package definitions"""
    from blackbox.core.package_manager import get_package_manager

    pm = get_package_manager()
    results = pm.validate_all()

    click.echo("\n🔍 Validating all packages...")
    click.echo("=" * 60)

    valid_count = 0
    invalid_count = 0

    for pkg, (is_valid, errors) in results.items():
        if is_valid:
            click.echo(f"✅ {pkg}")
            valid_count += 1
        else:
            click.echo(f"❌ {pkg}")
            for error in errors:
                click.echo(f"   - {error}", err=True)
            invalid_count += 1

    if invalid_count > 0:
        raise click.Abort()


@cli.command()
@click.option("--adapter", help="Filter by adapter ID")
@click.option("--days", default=7, help="Days to look back")
def audit(adapter: str, days: int):
    """📊 Show audit logs and statistics"""
    from datetime import datetime, timedelta

    from blackbox.core.audit import get_audit_logger

    logger = get_audit_logger()

    start_date = (datetime.now() - timedelta(days=days)).isoformat()
    entries = logger.query(adapter_id=adapter, start_date=start_date)

    click.echo(f"\n📊 Audit Log (last {days} days)")
    click.echo("=" * 80)

    for entry in entries[-20:]:  # Show last 20
        status = "✅" if entry.success else "❌"
        click.echo(
            f"{entry.timestamp[:19]} {status} {entry.adapter_id:15} ({entry.duration_ms:.0f}ms)"
        )

    # Statistics
    stats = logger.get_statistics()
    click.echo("\n📈 Statistics:")
    click.echo(f"  Total Executions: {stats.get('total', 0)}")
    click.echo(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
    click.echo(f"  Avg Duration: {stats.get('avg_duration_ms', 0):.0f}ms")

    if stats.get("most_used"):
        click.echo("\n🔥 Most Used:")
        for adapter_id, count in stats["most_used"][:5]:
            click.echo(f"  {adapter_id:15} {count:3} executions")


@cli.command()
@click.argument("image")
@click.option("--scanner", default="trivy", help="Scanner to use (trivy, grype)")
@click.option(
    "--severity", default="HIGH", help="Minimum severity (CRITICAL, HIGH, MEDIUM, LOW)"
)
def security_scan(image: str, scanner: str, severity: str):
    """🔍 Scan Docker image for vulnerabilities"""
    from blackbox.core.security import ImageSecurityScanner

    scanner_obj = ImageSecurityScanner(scanner)
    click.echo(f"🔍 Scanning {image} with {scanner}...")
    click.echo(f"   Severity threshold: {severity}\n")

    result = scanner_obj.scan_image(image, severity)

    if not result.get("success"):
        click.echo(f"❌ Scan failed: {result.get('error')}", err=True)
        return

    click.echo("✅ Scan complete")
    click.echo(f"  Scanner: {result['scanner']}")
    click.echo(f"  Vulnerabilities: {result['vulnerability_count']}")
    click.echo(f"  🔴 CRITICAL: {result.get('critical_count', 0)}")
    click.echo(f"  🟠 HIGH: {result.get('high_count', 0)}")

    if result.get("critical_count", 0) > 0:
        click.echo("\n⚠️  WARNING: Image has critical vulnerabilities!", err=True)


@cli.command()
@click.argument("registry")
@click.option("--username", help="Registry username")
@click.option("--password", help="Registry password", hide_input=True)
@click.option("--token", help="Access token")
def registry_login(registry: str, username: str, password: str, token: str):
    """🔐 Login to private Docker registry"""
    from blackbox.core.registry_auth import PrivateRegistryAuth

    auth = PrivateRegistryAuth()

    if auth.login(registry, username, password, token):
        click.echo(f"✅ Successfully logged in to {registry}")
    else:
        click.echo("❌ Login failed", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()


# Workflow Versioning Commands
@click.group()
def version():
    """Workflow versioning commands."""
    pass


@version.command()
@click.argument("workflow_path")
@click.option("--version", "-v", required=True, help="Version number (semver)")
@click.option("--message", "-m", help="Version description")
def create(workflow_path: str, version: str, message: str):
    """Create a new workflow version."""
    import yaml

    from blackbox.core.versioning.manager import VersionManager

    # Load workflow
    with open(workflow_path) as f:
        content = yaml.safe_load(f)

    workflow_id = content["workflow"]["id"]
    manager = VersionManager(Path("~/.bbx/versions").expanduser())

    version_obj = manager.create_version(
        workflow_id=workflow_id,
        content=content,
        version=version,
        created_by="cli",
        description=message,
    )

    click.echo(f"✅ Created version {version_obj.version} for {workflow_id}")


@version.command("list")
@click.argument("workflow_id")
def list_versions(workflow_id: str):
    """List all versions of a workflow."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    versions = manager.list_versions(workflow_id)

    click.echo(f"Versions for {workflow_id}:")
    for v in versions:
        click.echo(
            f"  {v.version} - {v.created_at} - {v.description or 'No description'}"
        )


@version.command()
@click.argument("workflow_id")
@click.argument("target_version")
def rollback(workflow_id: str, target_version: str):
    """Rollback workflow to a previous version."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    rollback_version = manager.rollback(workflow_id, target_version)

    click.echo(f"✅ Rolled back to {target_version}")
    click.echo(f"   New version: {rollback_version.version}")


@version.command()
@click.argument("workflow_id")
@click.argument("from_version")
@click.argument("to_version")
def diff(workflow_id: str, from_version: str, to_version: str):
    """Show differences between two versions."""
    from blackbox.core.versioning.manager import VersionManager

    manager = VersionManager(Path("~/.bbx/versions").expanduser())
    diff = manager.diff(workflow_id, from_version, to_version)

    click.echo(f"Diff: {from_version} → {to_version}")
    click.echo(f"  Added steps: {', '.join(diff.added_steps) or 'None'}")
    click.echo(f"  Removed steps: {', '.join(diff.removed_steps) or 'None'}")
    click.echo(f"  Modified steps: {', '.join(diff.modified_steps) or 'None'}")


cli.add_command(version)


@click.command()
def marketplace():
    """Browse workflow marketplace."""
    import requests

    resp = requests.get("http://localhost:8000/api/marketplace/templates")
    templates = resp.json()["templates"]
    for t in templates:
        click.echo(f"{t['name']} - {t['description']}")


cli.add_command(marketplace)


@click.group()
def plugin():
    """Plugin management commands."""
    pass


@plugin.command("list")
def list_plugins():
    """List installed plugins."""
    from blackbox.plugins.loader import PluginLoader

    loader = PluginLoader()
    loader.load_all_plugins()
    for name, plugin in loader.plugins.items():
        click.echo(f"{name} v{plugin.version}")


@plugin.command("install")
@click.argument("plugin_path")
def install_plugin(plugin_path):
    """Install plugin from file."""
    click.echo(f"Installing plugin from {plugin_path}")


cli.add_command(plugin)


# ============================================================================
# AI-FIRST: LOCAL LLM WORKFLOW GENERATION
# ============================================================================


@cli.group()
def model():
    """🤖 Manage AI models for workflow generation"""
    pass


@model.command("download")
@click.argument("model_name", default="qwen-0.5b")
def model_download(model_name: str):
    """
    Download an AI model for workflow generation.

    Example:
        bbx model download qwen-0.5b
    """
    from blackbox.ai.model_manager import ModelManager

    try:
        manager = ModelManager()
        manager.download(model_name)
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Download failed: {e}", err=True)
        sys.exit(1)


@model.command("list")
def model_list():
    """
    List available AI models.

    Example:
        bbx model list
    """
    from blackbox.ai.model_manager import ModelManager

    manager = ModelManager()
    models = manager.list_available()

    click.echo("\n📦 Available AI Models:\n")

    for model_name, info in models.items():
        status = "✅ INSTALLED" if info["installed"] else "📥 Available"
        click.echo(f"{status} {model_name} ({info['size']})")
        click.echo(f"   {info['description']}")
        if info["installed"]:
            click.echo(f"   Location: {info['path']}")
        click.echo()

    # Show how to download
    installed = manager.list_installed()
    if not installed:
        click.echo("💡 Tip: Download a model with: bbx model download qwen-0.5b\n")


@model.command("installed")
def model_installed():
    """
    List installed AI models.

    Example:
        bbx model installed
    """
    from blackbox.ai.model_manager import ModelManager

    manager = ModelManager()
    installed = manager.list_installed()

    if not installed:
        click.echo("❌ No models installed")
        click.echo("💡 Download one with: bbx model download qwen-0.5b")
        return

    click.echo("\n✅ Installed AI Models:\n")
    for model_name in installed:
        model_info = manager.MODELS[model_name]
        click.echo(f"  • {model_name} ({model_info['size']})")
    click.echo()


@model.command("remove")
@click.argument("model_name")
def model_remove(model_name: str):
    """
    Remove an installed AI model.

    Example:
        bbx model remove qwen-0.5b
    """
    from blackbox.ai.model_manager import ModelManager

    try:
        manager = ModelManager()
        manager.remove(model_name)
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


cli.add_command(model)


@cli.command()
@click.argument("description")
@click.option("--model", "-m", default=None, help="Model to use (default: qwen-0.5b)")
@click.option("--output", "-o", default="generated.bbx", help="Output file path")
def generate(description: str, model: str, output: str):
    """
    🤖 Generate BBX workflow from natural language using local AI.

    This command uses a local LLM (runs on your machine, no API keys needed)
    to generate BBX workflows from natural language descriptions.

    Examples:
        bbx generate "Deploy Next.js app to AWS S3"
        bbx generate "Run pytest tests" --output test.bbx
        bbx generate "CI/CD pipeline for Python" --model qwen-0.5b

    First time setup:
        bbx model download qwen-0.5b
    """
    from blackbox.ai.generator import WorkflowGenerator, check_dependencies
    from blackbox.ai.model_manager import ModelManager

    # Check dependencies
    if not check_dependencies():
        click.echo("\n💡 Install llama-cpp-python:")
        click.echo("   pip install llama-cpp-python")
        sys.exit(1)

    # Check if model is downloaded
    try:
        manager = ModelManager()

        # Use specified model or default
        if model is None:
            model = manager.load_default_model_from_config()

        # Check if model exists
        try:
            manager.get_model_path(model)
        except FileNotFoundError:
            click.echo(f"❌ Model '{model}' is not downloaded")
            click.echo("\n💡 Download it with:")
            click.echo(f"   bbx model download {model}")
            sys.exit(1)

        # Generate workflow
        click.echo(f"\n🤖 Using local AI: {model}")
        click.echo(f"📝 Task: {description}")
        click.echo()

        generator = WorkflowGenerator(model_name=model)
        yaml_content = generator.generate(description, output_file=output)

        click.echo(f"\n✅ Generated workflow saved to: {output}")
        click.echo("\n📋 Preview:")
        click.echo("─" * 60)

        # Show first 15 lines
        lines = yaml_content.split("\n")
        preview_lines = lines[:15]
        for line in preview_lines:
            click.echo(line)

        if len(lines) > 15:
            click.echo(f"... ({len(lines) - 15} more lines)")

        click.echo("─" * 60)

        click.echo("\n💡 Next steps:")
        click.echo(f"   1. Review the workflow: cat {output}")
        click.echo(f"   2. Validate it: bbx validate {output}")
        click.echo(f"   3. Run it: bbx run {output}")

    except ImportError as e:
        click.echo(f"❌ {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Generation failed: {e}", err=True)
        sys.exit(1)


@cli.command("mcp-serve")
@click.option("--host", default="localhost", help="Host to bind (default: localhost)")
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind (optional, MCP uses stdio by default)",
)
def mcp_serve(host: str, port: int):
    """
    🔌 Start BBX MCP Server for AI agent integration.

    This command starts a Model Context Protocol (MCP) server that exposes
    BBX workflow engine as tools that AI agents like Claude Code can call.

    The server runs in stdio mode (communicates via stdin/stdout) which is
    the standard way for Claude Code and other AI agents to integrate.

    Examples:
        bbx mcp-serve

    Claude Code Configuration:
        Add to your MCP client config:
        {
          "mcpServers": {
            "bbx": {
              "command": "python",
              "args": ["-m", "cli", "mcp-serve"]
            }
          }
        }

    Available MCP Tools:
        - bbx_generate: Generate workflow from natural language
        - bbx_validate: Validate workflow file
        - bbx_run: Execute workflow
        - bbx_list_workflows: List available workflows
    """
    from blackbox.mcp.server import main as run_mcp_server

    click.echo("🔌 Starting BBX MCP Server...")
    click.echo("📡 Protocol: Model Context Protocol (stdio)")
    click.echo()
    click.echo("Available tools:")
    click.echo("  - bbx_generate: Generate workflows from natural language")
    click.echo("  - bbx_validate: Validate workflow files")
    click.echo("  - bbx_run: Execute workflows")
    click.echo("  - bbx_list_workflows: List available workflows")
    click.echo()
    click.echo("✅ Server ready. Waiting for MCP client connections...")
    click.echo()

    try:
        run_mcp_server()
    except KeyboardInterrupt:
        click.echo("\n🛑 MCP Server stopped", err=True)
    except ImportError as e:
        click.echo(f"\n❌ Missing MCP dependencies: {e}", err=True)
        click.echo("\n💡 Install with:")
        click.echo("   pip install mcp")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Server error: {e}", err=True)
        sys.exit(1)
