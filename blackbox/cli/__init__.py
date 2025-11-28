# Copyright 2025 Ilya Makarov, Krasnoyarsk
# BBX CLI Module

from blackbox.cli.help import (
    AgentCommand,
    AgentGroup,
    agent_command,
    agent_group,
    create_help_command,
    generate_full_cli_schema,
    get_command_schema,
    get_group_schema,
)

from blackbox.cli.v2 import (
    v2_cli,
    register_v2_commands,
)

__all__ = [
    # Help utilities
    "AgentCommand",
    "AgentGroup",
    "agent_command",
    "agent_group",
    "create_help_command",
    "generate_full_cli_schema",
    "get_command_schema",
    "get_group_schema",
    # BBX 2.0 CLI
    "v2_cli",
    "register_v2_commands",
]
