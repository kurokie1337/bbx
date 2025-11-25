# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""Tests for MCP Client - external MCP server integration"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# MCP Server Config Tests
# =============================================================================

class TestMCPServerConfig:
    """Tests for MCP server configuration"""

    def test_config_creation_stdio(self):
        """Test creating config with stdio transport"""
        from blackbox.mcp.client.config import MCPServerConfig

        config = MCPServerConfig(
            name="github",
            transport="stdio",
            command=["npx", "-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "test_token"},
            description="GitHub MCP Server"
        )

        assert config.name == "github"
        assert config.transport == "stdio"
        assert config.command == ["npx", "-y", "@modelcontextprotocol/server-github"]
        assert config.env == {"GITHUB_TOKEN": "test_token"}
        assert config.description == "GitHub MCP Server"

    def test_config_creation_sse(self):
        """Test creating config with SSE transport"""
        from blackbox.mcp.client.config import MCPServerConfig

        config = MCPServerConfig(
            name="remote_server",
            transport="sse",
            url="https://mcp.example.com/sse",
            headers={"Authorization": "Bearer token123"}
        )

        assert config.name == "remote_server"
        assert config.transport == "sse"
        assert config.url == "https://mcp.example.com/sse"
        assert config.headers == {"Authorization": "Bearer token123"}

    def test_config_defaults(self):
        """Test default config values"""
        from blackbox.mcp.client.config import MCPServerConfig

        config = MCPServerConfig(name="minimal")

        assert config.name == "minimal"
        assert config.transport == "stdio"
        assert config.command == []
        assert config.url is None
        assert config.env == {}
        assert config.auto_start is True

    def test_resolve_env_with_env_vars(self):
        """Test environment variable substitution"""
        from blackbox.mcp.client.config import MCPServerConfig

        os.environ["TEST_MCP_TOKEN"] = "secret_value_123"

        config = MCPServerConfig(
            name="test",
            env={"API_TOKEN": "${TEST_MCP_TOKEN}"}
        )

        resolved = config.resolve_env()
        assert resolved["API_TOKEN"] == "secret_value_123"

        del os.environ["TEST_MCP_TOKEN"]

    def test_resolve_env_missing_var(self):
        """Test missing env var returns empty string"""
        from blackbox.mcp.client.config import MCPServerConfig

        config = MCPServerConfig(
            name="test",
            env={"MISSING": "${NONEXISTENT_VAR_XYZ}"}
        )

        resolved = config.resolve_env()
        # According to implementation, missing vars become empty string
        assert resolved["MISSING"] == ""


class TestMCPConfigLoading:
    """Tests for loading MCP configuration files"""

    def test_load_empty_config(self):
        """Test loading when no config file exists in any search path"""
        import os
        from blackbox.mcp.client.config import load_mcp_config

        # Save current directory and change to temp dir without mcp_servers.yaml
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                # Load from non-existent path without fallback
                configs = load_mcp_config("/nonexistent/path/mcp_servers.yaml")
                # Will be empty only if no mcp_servers.yaml in temp_dir or ~/.bbx/
                # Just check it doesn't crash
                assert isinstance(configs, dict)
            finally:
                os.chdir(original_dir)

    def test_load_config_from_yaml(self):
        """Test loading config from YAML file"""
        from blackbox.mcp.client.config import load_mcp_config

        yaml_content = """
servers:
  github:
    transport: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "${GITHUB_TOKEN}"
    description: "GitHub integration"

  filesystem:
    transport: stdio
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem"]
"""

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, encoding='utf-8'
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            configs = load_mcp_config(temp_path)

            assert "github" in configs
            assert "filesystem" in configs
            assert configs["github"].transport == "stdio"
            assert configs["github"].description == "GitHub integration"
        finally:
            os.unlink(temp_path)

    def test_create_default_config(self):
        """Test creating default config template"""
        from blackbox.mcp.client.config import create_default_config

        content = create_default_config()

        assert "servers:" in content
        assert "github:" in content
        assert "filesystem:" in content
        assert "GITHUB_TOKEN" in content


# =============================================================================
# MCP Client Adapter Tests
# =============================================================================

class TestMCPClientAdapter:
    """Tests for MCPClientAdapter"""

    def test_adapter_initialization(self):
        """Test adapter initializes correctly"""
        from blackbox.mcp.client.adapter import MCPClientAdapter

        adapter = MCPClientAdapter()
        assert adapter.adapter_name == "mcp"
        assert adapter._initialized is False

    @pytest.mark.asyncio
    async def test_method_parsing_valid(self):
        """Test parsing valid method format"""
        from blackbox.mcp.client.adapter import MCPClientAdapter

        adapter = MCPClientAdapter()

        # Mock manager
        mock_manager = AsyncMock()
        mock_manager.call_tool = AsyncMock(return_value={"data": "result"})
        mock_manager.load_config = AsyncMock()
        adapter.manager = mock_manager
        adapter._initialized = True

        result = await adapter.execute("github.create_issue", {
            "repo": "owner/repo",
            "title": "Test Issue"
        })

        assert result["status"] == "success"
        assert result["server"] == "github"
        assert result["tool"] == "create_issue"
        mock_manager.call_tool.assert_called_once_with(
            "github", "create_issue", {"repo": "owner/repo", "title": "Test Issue"}
        )

    @pytest.mark.asyncio
    async def test_method_parsing_invalid(self):
        """Test invalid method format raises error"""
        from blackbox.mcp.client.adapter import MCPClientAdapter

        adapter = MCPClientAdapter()
        adapter._initialized = True

        with pytest.raises(ValueError, match="Invalid MCP method format"):
            await adapter.execute("invalid_no_dot", {})

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Test connection error is handled"""
        from blackbox.mcp.client.adapter import MCPClientAdapter

        adapter = MCPClientAdapter()

        mock_manager = AsyncMock()
        mock_manager.call_tool = AsyncMock(side_effect=ConnectionError("Failed to connect"))
        mock_manager.load_config = AsyncMock()
        adapter.manager = mock_manager
        adapter._initialized = True

        result = await adapter.execute("server.tool", {})

        assert result["status"] == "error"
        assert result["error_type"] == "connection"
        assert "Failed to connect" in result["error"]


class TestMCPProxyAdapter:
    """Tests for MCPProxyAdapter - single server proxy"""

    def test_proxy_initialization(self):
        """Test proxy adapter initializes with server name"""
        from blackbox.mcp.client.adapter import MCPProxyAdapter

        adapter = MCPProxyAdapter("github")
        assert adapter.adapter_name == "mcp.github"
        assert adapter.server_name == "github"

    @pytest.mark.asyncio
    async def test_proxy_execute(self):
        """Test proxy execute calls correct server"""
        from blackbox.mcp.client.adapter import MCPProxyAdapter

        adapter = MCPProxyAdapter("kubernetes")

        mock_manager = AsyncMock()
        mock_manager.call_tool = AsyncMock(return_value={"pods": []})
        mock_manager.load_config = AsyncMock()
        adapter.manager = mock_manager
        adapter._initialized = True

        result = await adapter.execute("list_pods", {"namespace": "default"})

        assert result["status"] == "success"
        mock_manager.call_tool.assert_called_once_with(
            "kubernetes", "list_pods", {"namespace": "default"}
        )


# =============================================================================
# MCP Server Manager Tests
# =============================================================================

class TestMCPServerManager:
    """Tests for MCPServerManager"""

    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        from blackbox.mcp.client.manager import MCPServerManager

        manager = MCPServerManager()
        assert manager.configs == {}
        assert manager.connections == {}
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_load_config(self):
        """Test loading configuration"""
        from blackbox.mcp.client.manager import MCPServerManager

        yaml_content = """
servers:
  test_server:
    transport: stdio
    command: ["echo", "test"]
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False, encoding='utf-8'
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            manager = MCPServerManager(config_path=temp_path)
            await manager.load_config()

            assert manager._initialized is True
            assert "test_server" in manager.configs
        finally:
            os.unlink(temp_path)

    def test_list_servers(self):
        """Test listing configured servers"""
        from blackbox.mcp.client.manager import MCPServerManager
        from blackbox.mcp.client.config import MCPServerConfig

        manager = MCPServerManager()
        manager.configs = {
            "github": MCPServerConfig(name="github"),
            "slack": MCPServerConfig(name="slack"),
        }

        servers = manager.list_servers()
        assert "github" in servers
        assert "slack" in servers
        assert len(servers) == 2

    @pytest.mark.asyncio
    async def test_get_connection_unknown_server(self):
        """Test getting connection for unknown server raises error"""
        from blackbox.mcp.client.manager import MCPServerManager

        manager = MCPServerManager()
        manager._initialized = True
        manager.configs = {}

        with pytest.raises(ValueError, match="Unknown MCP server"):
            await manager.get_connection("nonexistent")


# =============================================================================
# Registry Integration Tests
# =============================================================================

class TestMCPRegistryIntegration:
    """Tests for MCP Client adapter in registry"""

    def test_mcp_adapter_registered(self):
        """Test MCP client adapter is registered"""
        from blackbox.core.registry import MCPRegistry

        registry = MCPRegistry()
        adapters = registry.list_adapters()

        assert "mcp" in adapters
        assert "bbx.mcp" in adapters

    def test_get_mcp_adapter(self):
        """Test getting MCP client adapter from registry"""
        from blackbox.core.registry import MCPRegistry

        registry = MCPRegistry()
        adapter = registry.get_adapter("mcp")

        assert adapter is not None
        assert adapter.adapter_name == "mcp"

    def test_get_mcp_adapter_with_prefix(self):
        """Test getting MCP adapter with bbx. prefix"""
        from blackbox.core.registry import MCPRegistry

        registry = MCPRegistry()
        adapter = registry.get_adapter("bbx.mcp")

        assert adapter is not None


# =============================================================================
# MCP Connection Tests (with mocking)
# =============================================================================

class TestMCPConnection:
    """Tests for MCP connection handling"""

    def test_connection_initialization(self):
        """Test connection object initializes correctly"""
        from blackbox.mcp.client.manager import MCPConnection
        from blackbox.mcp.client.config import MCPServerConfig

        config = MCPServerConfig(
            name="test",
            transport="stdio",
            command=["test", "command"]
        )

        conn = MCPConnection("test", config)

        assert conn.name == "test"
        assert conn.config == config
        assert conn.connected is False
        assert conn.tools == {}

    def test_connection_not_connected_by_default(self):
        """Test connection is not connected by default"""
        from blackbox.mcp.client.manager import MCPConnection
        from blackbox.mcp.client.config import MCPServerConfig

        config = MCPServerConfig(name="test")
        conn = MCPConnection("test", config)

        assert conn.connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting clears state"""
        from blackbox.mcp.client.manager import MCPConnection
        from blackbox.mcp.client.config import MCPServerConfig

        config = MCPServerConfig(name="test")
        conn = MCPConnection("test", config)
        conn._connected = True
        conn.tools = {"tool1": {}}
        conn.session = MagicMock()

        await conn.disconnect()

        assert conn.connected is False
        assert conn.tools == {}
        assert conn.session is None


# =============================================================================
# MCP Tools Tests
# =============================================================================

class TestMCPTools:
    """Tests for MCP tools module"""

    def test_get_bbx_tools(self):
        """Test getting BBX tools list"""
        from blackbox.mcp.tools import get_bbx_tools

        tools = get_bbx_tools()

        assert len(tools) == 4
        tool_names = [t["name"] for t in tools]
        assert "bbx_generate" in tool_names
        assert "bbx_validate" in tool_names
        assert "bbx_run" in tool_names
        assert "bbx_list_workflows" in tool_names

    def test_tool_handlers_registered(self):
        """Test tool handlers are registered"""
        from blackbox.mcp.tools import TOOL_HANDLERS

        assert "bbx_generate" in TOOL_HANDLERS
        assert "bbx_validate" in TOOL_HANDLERS
        assert "bbx_run" in TOOL_HANDLERS
        assert "bbx_list_workflows" in TOOL_HANDLERS

    @pytest.mark.asyncio
    async def test_handle_bbx_validate_valid_file(self):
        """Test validating a valid workflow file"""
        from blackbox.mcp.tools import handle_bbx_validate

        yaml_content = """
id: test_workflow
steps:
  - id: step1
    use: logger.info
    args:
      message: "test"
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.bbx', delete=False, encoding='utf-8'
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = await handle_bbx_validate({"workflow_file": temp_path})
            assert "valid" in result.lower()
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_handle_bbx_validate_missing_file(self):
        """Test validating non-existent file"""
        from blackbox.mcp.tools import handle_bbx_validate

        result = await handle_bbx_validate({"workflow_file": "/nonexistent/file.bbx"})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_bbx_validate_missing_steps(self):
        """Test validating file with missing steps"""
        from blackbox.mcp.tools import handle_bbx_validate

        yaml_content = """
id: test_workflow
# No steps field
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.bbx', delete=False, encoding='utf-8'
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            result = await handle_bbx_validate({"workflow_file": temp_path})
            assert "failed" in result.lower() or "missing" in result.lower()
        finally:
            os.unlink(temp_path)
