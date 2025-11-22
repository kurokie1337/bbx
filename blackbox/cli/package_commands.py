#!/usr/bin/env python3
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

# BBX CLI Extension - Package Manager Commands

import click
from blackbox.core.package_manager import get_package_manager
from blackbox.core.universal_schema import validate_definition
import yaml

@click.group()
def package():
    """📦 Manage Universal Adapter packages"""
    pass

@package.command()
@click.argument('package_name')
@click.option('--version', default='latest', help='Package version')
def install(package_name: str, version: str):
    """Install a Universal Adapter package"""
    pm = get_package_manager()
    
    click.echo(f"📦 Installing {package_name}@{version}...")
    
    if pm.install(package_name, version):
        click.echo(f"✅ Successfully installed {package_name}")
    else:
        click.echo(f"❌ Failed to install {package_name}", err=True)
        raise click.Abort()

@package.command()
def list_all():
    """List all available packages"""
    pm = get_package_manager()
    available = pm.list_available()
    installed = pm.list_installed()
    
    click.echo("\n📚 Available Packages:")
    click.echo("="*60)
    
    for pkg in available:
        status = "✅ installed" if pkg in installed else "⬜ available"
        click.echo(f"  {pkg:20} {status}")
    
    click.echo(f"\nTotal: {len(available)} available, {len(installed)} installed")

@package.command()
@click.argument('package_name')
def info(package_name: str):
    """Show package information"""
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

@package.command()
def validate():
    """Validate all package definitions"""
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
    
    click.echo(f"\nResults: {valid_count} valid, {invalid_count} invalid")
    
    if invalid_count > 0:
        raise click.Abort()

@package.command()
@click.argument('package_name')
def reload(package_name: str):
    """Hot-reload a package (update cache from file)"""
    pm = get_package_manager()
    
    click.echo(f"🔄 Reloading {package_name}...")
    
    if pm.reload(package_name):
        click.echo(f"✅ Successfully reloaded {package_name}")
    else:
        click.echo(f"❌ Failed to reload {package_name}", err=True)
        raise click.Abort()

@package.command()
@click.argument('definition_file', type=click.Path(exists=True))
def validate_file(definition_file: str):
    """Validate a single definition file"""
    with open(definition_file, 'r') as f:
        definition = yaml.safe_load(f)
    
    is_valid, errors = validate_definition(definition)
    
    if is_valid:
        click.echo(f"✅ {definition_file} is valid")
    else:
        click.echo(f"❌ {definition_file} has errors:")
        for error in errors:
            click.echo(f"   - {error}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    package()
