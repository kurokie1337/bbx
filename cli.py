import sys
import os
import click
import asyncio
import json
import logging
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
    level=settings.log_level
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

    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)
    
    server_process = subprocess.Popen(
        [sys.executable, str(api_script)],
        cwd=str(root_dir),
        env=env
    )
    
    try:
        # Wait for server
        click.echo("⏳ Waiting for server...")
        for _ in range(20):
            try:
                if requests.get("http://localhost:8000/health").status_code == 200:
                    break
            except:
                pass
            time.sleep(0.5)
        else:
            click.echo("❌ Server failed to start", err=True)
            raise click.Abort()
            
        # Run Workflow
        click.echo("🧪 Running E2E tests...")
        asyncio.run(run_file(str(workflow_path)))
        
    finally:
        click.echo("🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()


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


if __name__ == '__main__':
    cli()
