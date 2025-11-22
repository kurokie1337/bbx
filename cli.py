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

import sys
import click
import asyncio
import json
import logging
import yaml
from pathlib import Path

# Force UTF-8 encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from blackbox.core.runtime import run_file
from blackbox.core.validation import validate_workflow, WorkflowValidationError
from blackbox.core.config import get_settings
from blackbox.core.schema import BBXSchemaGenerator

settings = get_settings()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, settings.observability.log_level, logging.INFO)
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🚀 Blackbox Workflow Engine CLI"""
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--output', '-o', type=click.Choice(['text', 'json']), default='text', help='Output format')
@click.option('--input', '-i', 'inputs', multiple=True, help='Workflow inputs in key=value format (can be used multiple times)')
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
        if '=' not in input_str:
            logger.warning(f"Invalid input format (expected key=value): {input_str}")
            continue
        
        key, value = input_str.split('=', 1)
        
        # Try to parse as boolean
        if value.lower() in ('true', 'false'):
            input_dict[key] = value.lower() == 'true'
        # Try to parse as number
        elif value.isdigit():
            input_dict[key] = int(value)
        elif value.replace('.', '', 1).isdigit():
            input_dict[key] = float(value)
        else:
            input_dict[key] = value
    
    if verbose and input_dict:
        logger.debug(f"Workflow inputs: {input_dict}")
    
    try:
        results = asyncio.run(run_file(file_path, inputs=input_dict))
        
        if output == 'json':
            click.echo(json.dumps(results, indent=2, default=str))
        else:
            click.echo("\n" + "="*60)
            click.echo("📊 Workflow Execution Results")
            click.echo("="*60)
            
            for step_id, result in results.items():
                status = result.get("status", "unknown")
                icon = "✅" if status == "success" else "❌" if status == "error" else "⏭️"
                
                click.echo(f"\n{icon} Step: {step_id}")
                click.echo(f"   Status: {status}")
                
                if result.get("output"):
                    click.echo(f"   Output: {result['output']}")
                if result.get("error"):
                    click.echo(f"   Error: {result['error']}", err=True)
            
            click.echo("\n" + "="*60)
            
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=verbose)
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def validate(file_path: str):
    """
    Validate a workflow file without executing it.
    
    Example:
        blackbox validate workflow.bbx
    """
    try:
        with open(file_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        validate_workflow(workflow_data)
        
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
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
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
            with open(bbx_file, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            validate_workflow(workflow_data)
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
@click.argument('file_path', type=click.Path(exists=True))
def info(file_path: str):
    """
    Show information about a workflow file.
    
    Example:
        blackbox info workflow.bbx
    """
    try:
        with open(file_path, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        click.echo("\n" + "="*60)
        click.echo("📋 Workflow Information")
        click.echo("="*60)
        
        click.echo(f"\nID: {workflow_data.get('id', 'N/A')}")
        click.echo(f"Name: {workflow_data.get('name', 'N/A')}")
        click.echo(f"Version: {workflow_data.get('version', 'N/A')}")
        click.echo(f"Description: {workflow_data.get('description', 'N/A')}")
        
        steps = workflow_data.get('steps', {})
        if isinstance(steps, dict):
            click.echo(f"\nSteps: {len(steps)}")
            for step_id, step_config in steps.items():
                adapter = step_config.get('use', 'unknown')
                click.echo(f"  - {step_id} ({adapter})")
        elif isinstance(steps, list):
            click.echo(f"\nSteps: {len(steps)}")
            for step in steps:
                step_id = step.get('id', 'unknown')
                adapter = f"{step.get('mcp', 'unknown')}.{step.get('method', 'unknown')}"
                click.echo(f"  - {step_id} ({adapter})")
        
        click.echo("\n" + "="*60)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        raise click.Abort()


@cli.command()
def version():
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
@click.option('--output', '-o', default='bbx.schema.json', help='Output file path')
def schema(output: str):
    """Generate JSON Schema for VS Code."""
    schema_data = BBXSchemaGenerator.generate()
    with open(output, 'w') as f:
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
        
    click.echo("\n" + "="*60)
    click.echo("📅 Scheduled Workflows")
    click.echo("="*60)
    
    for t in triggers:
        click.echo(f"\nWorkflow: {t['workflow']}")
        click.echo(f"  ID: {t['workflow_id']}")
        click.echo(f"  Type: {t['type']}")
        if t['type'] == 'cron':
            click.echo(f"  Schedule: {t['schedule']}")
        elif t['type'] == 'webhook':
            click.echo(f"  Path: {t['details'].get('path')}")
            
    click.echo("\n" + "="*60)


@cli.command()
@click.option('--pull', is_flag=True, help='Pull missing images')
def system(pull: bool):
    """🏥 Check system health and Docker images."""
    click.echo("\n" + "="*60)
    click.echo("🏥 System Health Check")
    click.echo("="*60)
    
    # 1. Check Docker
    click.echo("\n🐳 Docker Status:")
    docker_ok = False
    try:
        import subprocess
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            click.echo(f"   ✅ Available: {result.stdout.strip()}")
            
            # Check daemon
            daemon_check = subprocess.run(["docker", "info"], capture_output=True, text=True)
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
        "willhallonline/ansible:latest"
    ]
    
    missing_images = []
    
    for image in images:
        try:
            # Check if image exists locally
            check = subprocess.run(
                ["docker", "images", "-q", image],
                capture_output=True,
                text=True
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
            click.echo("\n💡 Tip: Run 'blackbox system --pull' to download missing images.")
    else:
        click.echo("\n✨ All systems go!")

    click.echo("\n" + "="*60)



# ============================================================================
# PHASE 3: UNIVERSAL ADAPTER CLI EXTENSIONS
# ============================================================================

@cli.group()
def package():
    """📦 Manage Universal Adapter packages"""
    pass

@package.command()
@click.argument('package_name')
@click.option('--version', default='latest', help='Package version')
def install(package_name: str, version: str):
    """Install a Universal Adapter package"""
    from blackbox.core.package_manager import get_package_manager
    
    pm = get_package_manager()
    click.echo(f"📦 Installing {package_name}@{version}...")
    
    if pm.install(package_name, version):
        click.echo(f"✅ Successfully installed {package_name}")
    else:
        click.echo(f"❌ Failed to install {package_name}", err=True)
        raise click.Abort()

@package.command('list')
def list_packages():
    """List all available packages"""
    from blackbox.core.package_manager import get_package_manager
    
    pm = get_package_manager()
    available = pm.list_available()
    installed = pm.list_installed()
    
    click.echo("\n📚 Available Packages:")
    click.echo("="*60)
    
    for pkg in available:
        status = "✅ installed" if pkg in installed else "⬜ available"
        click.echo(f"  {pkg:20} {status}")
    
    click.echo(f"\nTotal: {len(available)} available, {len(installed)} installed")

@package.command(name="info_pkg")
@click.argument('package_name')
def info_pkg(package_name: str):
    """Show package information"""
    from blackbox.core.package_manager import get_package_manager
    
    pm = get_package_manager()
    definition = pm.get(package_name)
    
    if not definition:
        click.echo(f"❌ Package not found: {package_name}", err=True)
        raise click.Abort()
    
    click.echo(f"\n📋 Package: {package_name}")
    click.echo("="*60)
    click.echo(f"ID:     {definition.get('id')}")
    click.echo(f"Image:  {definition.get('uses')}")
    
    if 'auth' in definition:
        click.echo(f"Auth:   {definition['auth'].get('type')}")
    
    if 'output_parser' in definition:
        click.echo(f"Output: {definition['output_parser'].get('type')}")
    
    click.echo("\nCommand Template:")
    cmd = definition.get('cmd', [])
    for line in (cmd if isinstance(cmd, list) else [cmd]):
        click.echo(f"  {line}")

@package.command(name="validate_pkg")
def validate_pkg():
    """Validate all package definitions"""
    from blackbox.core.package_manager import get_package_manager
    
    pm = get_package_manager()
    results = pm.validate_all()
    
    click.echo("\n🔍 Validating all packages...")
    click.echo("="*60)
    
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
@click.option('--adapter', help='Filter by adapter ID')
@click.option('--days', default=7, help='Days to look back')
def audit(adapter: str, days: int):
    """📊 Show audit logs and statistics"""
    from blackbox.core.audit import get_audit_logger
    from datetime import datetime, timedelta
    
    logger = get_audit_logger()
    
    start_date = (datetime.now() - timedelta(days=days)).isoformat()
    entries = logger.query(adapter_id=adapter, start_date=start_date)
    
    click.echo(f"\n📊 Audit Log (last {days} days)")
    click.echo("="*80)
    
    for entry in entries[-20:]:  # Show last 20
        status = "✅" if entry.success else "❌"
        click.echo(f"{entry.timestamp[:19]} {status} {entry.adapter_id:15} ({entry.duration_ms:.0f}ms)")
    
    # Statistics
    stats = logger.get_statistics()
    click.echo("\n📈 Statistics:")
    click.echo(f"  Total Executions: {stats.get('total', 0)}")
    click.echo(f"  Success Rate: {stats.get('success_rate', 0):.1f}%")
    click.echo(f"  Avg Duration: {stats.get('avg_duration_ms', 0):.0f}ms")
    
    if stats.get('most_used'):
        click.echo("\n🔥 Most Used:")
        for adapter_id, count in stats['most_used'][:5]:
            click.echo(f"  {adapter_id:15} {count:3} executions")

@cli.command()
@click.argument('image')
@click.option('--scanner', default='trivy', help='Scanner to use (trivy, grype)')
@click.option('--severity', default='HIGH', help='Minimum severity (CRITICAL, HIGH, MEDIUM, LOW)')
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
    
    if result.get('critical_count', 0) > 0:
        click.echo("\n⚠️  WARNING: Image has critical vulnerabilities!", err=True)

@cli.command()
@click.argument('registry')
@click.option('--username', help='Registry username')
@click.option('--password', help='Registry password', hide_input=True)
@click.option('--token', help='Access token')
def registry_login(registry: str, username: str, password: str, token: str):
    """🔐 Login to private Docker registry"""
    from blackbox.core.registry_auth import PrivateRegistryAuth
    
    auth = PrivateRegistryAuth()
    
    if auth.login(registry, username, password, token):
        click.echo(f"✅ Successfully logged in to {registry}")
    else:
        click.echo("❌ Login failed", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli()
