# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
MCP Client Adapter for BBX

Allows using external MCP servers as adapters in BBX workflows.

Example usage in workflow:
    steps:
      create_issue:
        use: mcp.github.create_issue
        args:
          repo: "owner/repo"
          title: "Bug fix"
"""

import logging
from typing import Any, Dict, Optional

from blackbox.core.base_adapter import MCPAdapter

from .manager import MCPServerManager, get_mcp_manager

logger = logging.getLogger("bbx.mcp.client.adapter")


class MCPClientAdapter(MCPAdapter):
    """
    BBX Adapter that proxies calls to external MCP servers.

    Method format: "server_name.tool_name"
    Example: "github.create_issue", "kubernetes.apply", "slack.post_message"
    """

    def __init__(self, manager: Optional[MCPServerManager] = None):
        super().__init__("mcp")
        self.manager = manager or get_mcp_manager()
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure manager is initialized"""
        if not self._initialized:
            await self.manager.load_config()
            self._initialized = True

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute MCP tool call.

        Args:
            method: Format "server_name.tool_name" (e.g., "github.create_issue")
            inputs: Tool arguments

        Returns:
            Tool result
        """
        await self._ensure_initialized()

        # Parse method
        if "." not in method:
            raise ValueError(
                f"Invalid MCP method format: {method}. "
                f"Expected 'server_name.tool_name' (e.g., 'github.create_issue')"
            )

        parts = method.split(".", 1)
        server_name = parts[0]
        tool_name = parts[1]

        logger.info(f"MCP call: {server_name}.{tool_name}")

        try:
            result = await self.manager.call_tool(server_name, tool_name, inputs)

            return {
                "status": "success",
                "server": server_name,
                "tool": tool_name,
                "result": result,
            }
        except ConnectionError as e:
            logger.error(f"MCP connection error: {e}")
            return {
                "status": "error",
                "server": server_name,
                "tool": tool_name,
                "error": str(e),
                "error_type": "connection",
            }
        except ValueError as e:
            logger.error(f"MCP value error: {e}")
            return {
                "status": "error",
                "server": server_name,
                "tool": tool_name,
                "error": str(e),
                "error_type": "validation",
            }
        except Exception as e:
            logger.error(f"MCP call failed: {e}")
            return {
                "status": "error",
                "server": server_name,
                "tool": tool_name,
                "error": str(e),
                "error_type": "unknown",
            }

    async def list_servers(self) -> Dict[str, Any]:
        """List all configured MCP servers"""
        await self._ensure_initialized()

        servers = {}
        for name in self.manager.list_servers():
            config = self.manager.configs[name]
            servers[name] = {
                "transport": config.transport,
                "description": config.description,
                "auto_start": config.auto_start,
            }
        return servers

    async def list_tools(self, server_name: str) -> Dict[str, Any]:
        """List tools available on a specific server"""
        await self._ensure_initialized()
        return await self.manager.list_tools(server_name)

    async def test_server(self, server_name: str) -> Dict[str, Any]:
        """Test connection to a server"""
        await self._ensure_initialized()
        return await self.manager.test_connection(server_name)


class MCPProxyAdapter(MCPAdapter):
    """
    Simplified proxy adapter for a specific MCP server.

    Used when you want to expose a single MCP server directly.

    Example:
        github_adapter = MCPProxyAdapter("github")
        result = await github_adapter.execute("create_issue", {...})
    """

    def __init__(self, server_name: str, manager: Optional[MCPServerManager] = None):
        super().__init__(f"mcp.{server_name}")
        self.server_name = server_name
        self.manager = manager or get_mcp_manager()
        self._initialized = False

    async def _ensure_initialized(self):
        if not self._initialized:
            await self.manager.load_config()
            self._initialized = True

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute tool on the proxied server.

        Args:
            method: Tool name (e.g., "create_issue")
            inputs: Tool arguments

        Returns:
            Tool result
        """
        await self._ensure_initialized()

        logger.info(f"MCP proxy call: {self.server_name}.{method}")

        try:
            result = await self.manager.call_tool(self.server_name, method, inputs)
            return {
                "status": "success",
                "result": result,
            }
        except Exception as e:
            logger.error(f"MCP proxy call failed: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
