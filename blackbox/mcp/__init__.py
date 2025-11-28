# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX MCP Server - Model Context Protocol integration

Exposes BBX workflow engine as MCP tools for AI agents like Claude Code.

BBX 2.0 extends MCP with tools for:
- Linux kernel-inspired: AgentRing, Hooks, ContextTiering, FlowIntegrity, Quotas, Snapshots
- Distribution-inspired: Flakes, Registry, Bundles, Sandbox, Mesh, Policy
- NT kernel-inspired: Executive, ObjectManager, FilterStack, WorkingSet, ConfigRegistry, AAL
"""

from .server import create_server
from .tools import get_bbx_tools, get_all_bbx_tools, TOOL_HANDLERS
from .tools_v2 import get_bbx_v2_tools, V2_TOOL_HANDLERS

__all__ = [
    # Server
    "create_server",
    # BBX 1.x Tools
    "get_bbx_tools",
    "TOOL_HANDLERS",
    # BBX 2.0 Tools
    "get_bbx_v2_tools",
    "V2_TOOL_HANDLERS",
    # Combined
    "get_all_bbx_tools",
]
