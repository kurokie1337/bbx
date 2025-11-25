# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX MCP Server - Model Context Protocol server for BBX workflow engine

Exposes BBX as MCP tools that AI agents like Claude Code can call directly.
"""

import asyncio
import sys
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .tools import TOOL_HANDLERS, get_bbx_tools, _safe_output


def create_server() -> Server:
    """
    Create and configure BBX MCP server.

    Returns:
        Configured MCP Server instance
    """
    server = Server("bbx-workflow-engine")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available BBX tools."""
        tools = get_bbx_tools()
        return [
            Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
            )
            for tool in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
        """
        Handle tool calls from MCP client.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        handler = TOOL_HANDLERS.get(name)

        if not handler:
            return [TextContent(type="text", text=_safe_output(f"❌ Unknown tool: {name}"))]

        try:
            result = await handler(arguments)
            # Apply Windows-safe encoding
            safe_result = _safe_output(result) if isinstance(result, str) else str(result)
            return [TextContent(type="text", text=safe_result)]
        except Exception as e:
            return [
                TextContent(type="text", text=_safe_output(f"❌ Tool execution failed: {str(e)}"))
            ]

    return server


async def run_server():
    """
    Run BBX MCP server using stdio transport.

    This is the main entry point for the MCP server.
    """
    server = create_server()

    # Run server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    """
    Main entry point for bbx mcp-serve command.
    """
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass  # Silent exit
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
