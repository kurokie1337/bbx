# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Test MCP Server - Simple test server for MCP integration testing

Provides basic tools: echo, timestamp, add, json_format
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


def create_test_server() -> Server:
    """Create test MCP server with basic tools."""
    server = Server("bbx-test-server")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available test tools."""
        return [
            Tool(
                name="echo",
                description="Echo back the provided message",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo back",
                        }
                    },
                    "required": ["message"],
                },
            ),
            Tool(
                name="timestamp",
                description="Get current timestamp in ISO format",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="add",
                description="Add two numbers together",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "number",
                            "description": "First number",
                        },
                        "b": {
                            "type": "number",
                            "description": "Second number",
                        },
                    },
                    "required": ["a", "b"],
                },
            ),
            Tool(
                name="json_format",
                description="Format data as pretty JSON",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "data": {
                            "description": "Data to format as JSON",
                        }
                    },
                    "required": ["data"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
        """Handle tool calls."""
        try:
            if name == "echo":
                message = arguments.get("message", "")
                return [TextContent(type="text", text=message)]

            elif name == "timestamp":
                ts = datetime.now().isoformat()
                return [TextContent(type="text", text=ts)]

            elif name == "add":
                a = arguments.get("a", 0)
                b = arguments.get("b", 0)
                result = a + b
                return [TextContent(type="text", text=str(result))]

            elif name == "json_format":
                data = arguments.get("data", {})
                formatted = json.dumps(data, indent=2, ensure_ascii=False)
                return [TextContent(type="text", text=formatted)]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def run_server():
    """Run the test MCP server."""
    server = create_test_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main():
    """Main entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
