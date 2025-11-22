"""
BBX MCP Server - Model Context Protocol integration

Exposes BBX workflow engine as MCP tools for AI agents like Claude Code.
"""

from .server import create_server
from .tools import get_bbx_tools

__all__ = ["create_server", "get_bbx_tools"]
