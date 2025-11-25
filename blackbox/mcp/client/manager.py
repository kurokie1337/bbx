# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""MCP Server Manager - Manages lifecycle of MCP server connections"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from .config import MCPServerConfig, load_mcp_config

logger = logging.getLogger("bbx.mcp.client")

# Check if MCP SDK is available
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    stdio_client = None
    StdioServerParameters = None


class MCPConnection:
    """Wrapper for MCP client connection"""

    def __init__(self, name: str, config: MCPServerConfig):
        self.name = name
        self.config = config
        self.session: Optional[ClientSession] = None
        self.tools: Dict[str, Any] = {}
        self._connected = False
        self._context_stack = None

    @property
    def connected(self) -> bool:
        return self._connected and self.session is not None

    async def connect(self) -> bool:
        """Establish connection to MCP server"""
        if not MCP_AVAILABLE:
            logger.error("MCP SDK not installed. Run: pip install mcp")
            return False

        try:
            if self.config.transport == "stdio":
                return await self._connect_stdio()
            elif self.config.transport in ("sse", "http"):
                return await self._connect_sse()
            else:
                logger.error(f"Unknown transport: {self.config.transport}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _connect_stdio(self) -> bool:
        """Connect via stdio (subprocess)"""
        if not self.config.command:
            logger.error(f"No command specified for stdio server {self.name}")
            return False

        # Build environment
        env = {**os.environ, **self.config.resolve_env()}

        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.config.command[0],
                args=self.config.command[1:] if len(self.config.command) > 1 else [],
                env=env,
            )

            # Start the connection
            stdio_ctx = stdio_client(server_params)
            read_stream, write_stream = await stdio_ctx.__aenter__()

            # Create session
            session_ctx = ClientSession(read_stream, write_stream)
            self.session = await session_ctx.__aenter__()

            # Initialize
            await self.session.initialize()

            # Get tools
            tools_response = await self.session.list_tools()
            self.tools = {tool.name: tool for tool in tools_response.tools}

            self._connected = True
            self._context_stack = (stdio_ctx, session_ctx)

            logger.info(f"Connected to {self.name}: {len(self.tools)} tools available")
            return True

        except Exception as e:
            logger.error(f"Stdio connection failed for {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _connect_sse(self) -> bool:
        """Connect via HTTP/SSE"""
        logger.error("SSE transport not yet implemented")
        return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        if not self.connected:
            raise ConnectionError(f"Not connected to {self.name}")

        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            raise ValueError(
                f"Tool '{tool_name}' not found on {self.name}. Available: {available}"
            )

        try:
            result = await self.session.call_tool(tool_name, arguments)
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Tool call failed: {self.name}.{tool_name}: {e}")
            raise

    def _process_result(self, result: Any) -> Any:
        """Process MCP tool result into standard format"""
        if hasattr(result, "content"):
            contents = []
            for content in result.content:
                if hasattr(content, "text"):
                    contents.append(content.text)
                elif hasattr(content, "data"):
                    contents.append(content.data)
            return contents[0] if len(contents) == 1 else contents
        return result

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self._context_stack:
            stdio_ctx, session_ctx = self._context_stack
            try:
                await session_ctx.__aexit__(None, None, None)
                await stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._connected = False
        self.session = None
        self.tools = {}
        self._context_stack = None


class MCPServerManager:
    """
    Manages multiple MCP server connections.

    Usage:
        manager = MCPServerManager()
        await manager.load_config()

        # Call a tool
        result = await manager.call_tool("test", "echo", {"message": "hello"})
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.configs: Dict[str, MCPServerConfig] = {}
        self.connections: Dict[str, MCPConnection] = {}
        self._initialized = False

    async def load_config(self, config_path: Optional[str] = None):
        """Load MCP server configurations"""
        path = config_path or self.config_path
        self.configs = load_mcp_config(path)
        self._initialized = True
        logger.info(f"Loaded {len(self.configs)} MCP server configs")

    def list_servers(self) -> List[str]:
        """List configured server names"""
        return list(self.configs.keys())

    async def get_connection(self, name: str) -> MCPConnection:
        """Get or create connection to named MCP server"""
        if not self._initialized:
            await self.load_config()

        if name not in self.configs:
            raise ValueError(f"Unknown MCP server: {name}")

        if name not in self.connections:
            config = self.configs[name]
            conn = MCPConnection(name, config)
            self.connections[name] = conn

        conn = self.connections[name]
        if not conn.connected:
            success = await conn.connect()
            if not success:
                raise ConnectionError(f"Failed to connect to {name}")

        return conn

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on an MCP server.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        conn = await self.get_connection(server_name)
        return await conn.call_tool(tool_name, arguments)

    async def list_tools(self, server_name: str) -> Dict[str, Any]:
        """List tools available on a server"""
        conn = await self.get_connection(server_name)
        return conn.tools

    async def list_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """List tools from all configured servers"""
        all_tools = {}
        for name in self.configs.keys():
            try:
                conn = await self.get_connection(name)
                all_tools[name] = conn.tools
            except Exception as e:
                logger.warning(f"Could not get tools from {name}: {e}")
                all_tools[name] = {"error": str(e)}
        return all_tools

    async def disconnect_all(self):
        """Disconnect from all servers"""
        for conn in self.connections.values():
            await conn.disconnect()
        self.connections.clear()

    async def test_connection(self, name: str) -> Dict[str, Any]:
        """Test connection to a server"""
        try:
            conn = await self.get_connection(name)
            return {
                "status": "ok",
                "server": name,
                "tools_count": len(conn.tools),
                "tools": list(conn.tools.keys()),
            }
        except Exception as e:
            return {
                "status": "error",
                "server": name,
                "error": str(e),
            }


# Global manager instance
_manager: Optional[MCPServerManager] = None


def get_mcp_manager() -> MCPServerManager:
    """Get global MCP manager instance"""
    global _manager
    if _manager is None:
        _manager = MCPServerManager()
    return _manager
