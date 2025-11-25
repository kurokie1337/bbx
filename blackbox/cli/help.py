# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0
# BBX Agent-Friendly Help System
#
# Provides machine-readable JSON help output for AI agents.

"""
BBX Agent-Friendly Help System

Enables AI agents to understand CLI commands better with JSON output.

Usage:
    bbx --help --format json
    bbx run --help --format json
"""

import json
import sys
from typing import Any, Dict, List, Optional

import click
from click import Command, Context, Group


def get_command_schema(cmd: Command, ctx: Context) -> Dict[str, Any]:
    """
    Generate JSON schema for a click command.

    Args:
        cmd: Click command
        ctx: Click context

    Returns:
        JSON-serializable dict with command schema
    """
    schema = {
        "command": ctx.info_name,
        "description": cmd.help or "",
        "usage": cmd.get_usage(ctx).replace("Usage: ", ""),
        "arguments": [],
        "options": [],
        "examples": [],
    }

    # Extract arguments
    for param in cmd.params:
        if isinstance(param, click.Argument):
            arg_info = {
                "name": param.name.upper(),
                "type": _get_param_type(param),
                "required": param.required,
                "multiple": param.multiple,
            }
            # Handle default (avoid Sentinel objects)
            if param.default is not None and not callable(param.default):
                try:
                    # Test if serializable
                    import json
                    json.dumps(param.default)
                    arg_info["default"] = param.default
                except (TypeError, ValueError):
                    pass  # Skip non-serializable defaults
            schema["arguments"].append(arg_info)
        elif isinstance(param, click.Option):
            opt_info = {
                "name": max(param.opts, key=len),  # Get long option
                "short": min(param.opts, key=len) if len(param.opts) > 1 else None,
                "type": _get_param_type(param),
                "required": param.required,
                "multiple": param.multiple,
                "is_flag": param.is_flag,
                "help": param.help or "",
            }
            # Handle default (avoid Sentinel objects)
            if param.default is not None and not param.is_flag and not callable(param.default):
                try:
                    import json
                    json.dumps(param.default)
                    opt_info["default"] = param.default
                except (TypeError, ValueError):
                    pass
            if param.type and hasattr(param.type, 'choices'):
                opt_info["choices"] = list(param.type.choices)
            schema["options"].append(opt_info)

    # Extract examples from docstring
    if cmd.help:
        examples = _extract_examples(cmd.help)
        schema["examples"] = examples

    return schema


def get_group_schema(group: Group, ctx: Context) -> Dict[str, Any]:
    """
    Generate JSON schema for a click group.

    Args:
        group: Click group
        ctx: Click context

    Returns:
        JSON-serializable dict with group schema
    """
    schema = {
        "command": ctx.info_name or "bbx",
        "description": group.help or "",
        "type": "group",
        "commands": {},
    }

    # Get all subcommands
    for name in group.list_commands(ctx):
        cmd = group.get_command(ctx, name)
        if cmd:
            if isinstance(cmd, Group):
                schema["commands"][name] = {
                    "type": "group",
                    "description": cmd.help or "",
                }
            else:
                schema["commands"][name] = {
                    "type": "command",
                    "description": cmd.help.split('\n')[0] if cmd.help else "",
                }

    return schema


def _get_param_type(param) -> str:
    """Get string representation of parameter type."""
    if hasattr(param, 'is_flag') and param.is_flag:
        return "boolean"
    if param.type:
        type_name = param.type.name
        type_map = {
            "TEXT": "string",
            "INT": "integer",
            "FLOAT": "number",
            "BOOL": "boolean",
            "PATH": "path",
            "FILE": "file",
            "CHOICE": "choice",
        }
        return type_map.get(type_name, type_name.lower())
    return "string"


def _extract_examples(help_text: str) -> List[Dict[str, str]]:
    """Extract examples from help text."""
    examples = []
    lines = help_text.split('\n')

    in_example = False
    for line in lines:
        stripped = line.strip()

        # Look for example markers
        if stripped.lower().startswith('example'):
            in_example = True
            continue

        # Look for command examples
        if in_example or stripped.startswith('bbx ') or stripped.startswith('$ bbx'):
            cmd = stripped.lstrip('$ ').strip()
            if cmd.startswith('bbx '):
                examples.append({
                    "command": cmd,
                    "description": "",
                })

    return examples


class AgentHelpMixin:
    """
    Mixin for Click commands to support --format json.
    """

    def format_help(self, ctx: Context, formatter) -> None:
        """Override to check for JSON format."""
        # Check if --format json was passed
        if '--format' in sys.argv and 'json' in sys.argv:
            self._print_json_help(ctx)
            ctx.exit(0)
        else:
            super().format_help(ctx, formatter)

    def _print_json_help(self, ctx: Context) -> None:
        """Print help in JSON format."""
        if isinstance(self, Group):
            schema = get_group_schema(self, ctx)
        else:
            schema = get_command_schema(self, ctx)

        click.echo(json.dumps(schema, indent=2))


class AgentCommand(AgentHelpMixin, click.Command):
    """Click Command with agent-friendly JSON help."""
    pass


class AgentGroup(AgentHelpMixin, click.Group):
    """Click Group with agent-friendly JSON help."""
    pass


# Helper decorator
def agent_command(*args, **kwargs):
    """Decorator for agent-friendly commands."""
    kwargs.setdefault('cls', AgentCommand)
    return click.command(*args, **kwargs)


def agent_group(*args, **kwargs):
    """Decorator for agent-friendly groups."""
    kwargs.setdefault('cls', AgentGroup)
    return click.group(*args, **kwargs)


# === BBX Help Command ===

def create_help_command():
    """Create the bbx help command with JSON support."""

    @click.command("help")
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
        # Get root CLI
        root = ctx.parent.command if ctx.parent else ctx.command

        if not command_path:
            # Show root help
            if format == "json":
                schema = get_group_schema(root, ctx)
                if show_all:
                    schema = _expand_all_commands(root, ctx, schema)
                click.echo(json.dumps(schema, indent=2))
            else:
                click.echo(ctx.parent.get_help() if ctx.parent else root.get_help(ctx))
        else:
            # Navigate to specific command
            cmd = root
            cmd_ctx = ctx.parent if ctx.parent else ctx

            for name in command_path:
                if isinstance(cmd, Group):
                    sub_cmd = cmd.get_command(cmd_ctx, name)
                    if sub_cmd:
                        cmd = sub_cmd
                        cmd_ctx = Context(cmd, parent=cmd_ctx, info_name=name)
                    else:
                        click.echo(f"Unknown command: {' '.join(command_path)}", err=True)
                        ctx.exit(1)
                else:
                    click.echo(f"'{cmd.name}' is not a group, cannot have subcommands", err=True)
                    ctx.exit(1)

            # Show help for found command
            if format == "json":
                if isinstance(cmd, Group):
                    schema = get_group_schema(cmd, cmd_ctx)
                else:
                    schema = get_command_schema(cmd, cmd_ctx)
                click.echo(json.dumps(schema, indent=2))
            else:
                click.echo(cmd.get_help(cmd_ctx))

    return help_cmd


def _expand_all_commands(group: Group, ctx: Context, schema: Dict) -> Dict:
    """Recursively expand all commands in schema."""
    for name in group.list_commands(ctx):
        cmd = group.get_command(ctx, name)
        if cmd:
            sub_ctx = Context(cmd, parent=ctx, info_name=name)
            if isinstance(cmd, Group):
                schema["commands"][name] = get_group_schema(cmd, sub_ctx)
                schema["commands"][name] = _expand_all_commands(cmd, sub_ctx, schema["commands"][name])
            else:
                schema["commands"][name] = get_command_schema(cmd, sub_ctx)
    return schema


# === Full CLI Schema Generator ===

def generate_full_cli_schema(cli_group: Group) -> Dict[str, Any]:
    """
    Generate complete JSON schema of entire CLI.

    Useful for AI agents to understand all available commands.

    Args:
        cli_group: Root CLI group

    Returns:
        Complete schema with all commands and subcommands
    """
    ctx = Context(cli_group, info_name="bbx")
    schema = {
        "name": "bbx",
        "version": "1.0.0",
        "description": "BBX - Operating System for AI Agents",
        "commands": {},
    }

    def process_group(group: Group, parent_ctx: Context) -> Dict:
        result = {}
        for name in group.list_commands(parent_ctx):
            cmd = group.get_command(parent_ctx, name)
            if cmd:
                cmd_ctx = Context(cmd, parent=parent_ctx, info_name=name)
                if isinstance(cmd, Group):
                    result[name] = {
                        "type": "group",
                        "description": cmd.help or "",
                        "commands": process_group(cmd, cmd_ctx),
                    }
                else:
                    result[name] = get_command_schema(cmd, cmd_ctx)
                    result[name]["type"] = "command"
        return result

    schema["commands"] = process_group(cli_group, ctx)
    return schema
