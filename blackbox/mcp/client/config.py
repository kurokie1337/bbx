# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""MCP Server Configuration"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server"""

    name: str
    transport: str = "stdio"  # stdio, sse, websocket
    command: List[str] = field(default_factory=list)  # For stdio transport
    url: Optional[str] = None  # For sse/websocket transport
    env: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30000  # ms
    auto_start: bool = True
    description: str = ""

    def resolve_env(self) -> Dict[str, str]:
        """Resolve environment variables in env dict"""
        resolved = {}
        for key, value in self.env.items():
            resolved[key] = _resolve_env_vars(value)
        return resolved

    def resolve_headers(self) -> Dict[str, str]:
        """Resolve environment variables in headers"""
        resolved = {}
        for key, value in self.headers.items():
            resolved[key] = _resolve_env_vars(value)
        return resolved


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR} with environment variable value"""
    pattern = r"\$\{([^}]+)\}"

    def replacer(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return re.sub(pattern, replacer, value)


def load_mcp_config(config_path: Optional[str] = None) -> Dict[str, MCPServerConfig]:
    """
    Load MCP server configurations from YAML file.

    Search order:
    1. Provided config_path
    2. ./mcp_servers.yaml
    3. ~/.bbx/mcp_servers.yaml

    Returns:
        Dict mapping server name to config
    """
    search_paths = []

    if config_path:
        search_paths.append(Path(config_path))

    search_paths.extend(
        [
            Path("./mcp_servers.yaml"),
            Path("./mcp_servers.yml"),
            Path.home() / ".bbx" / "mcp_servers.yaml",
            Path.home() / ".bbx" / "mcp_servers.yml",
        ]
    )

    config_file = None
    for path in search_paths:
        if path.exists():
            config_file = path
            break

    if not config_file:
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    servers_data = data.get("servers", {})
    configs = {}

    for name, server_config in servers_data.items():
        if isinstance(server_config, dict):
            configs[name] = MCPServerConfig(
                name=name,
                transport=server_config.get("transport", "stdio"),
                command=server_config.get("command", []),
                url=server_config.get("url"),
                env=server_config.get("env", {}),
                headers=server_config.get("headers", {}),
                timeout=server_config.get("timeout", 30000),
                auto_start=server_config.get("auto_start", True),
                description=server_config.get("description", ""),
            )

    return configs


def create_default_config() -> str:
    """Generate default mcp_servers.yaml content"""
    return """# BBX MCP Servers Configuration
# Documentation: https://github.com/kurokie1337/bbx/docs/MCP_CLIENT_ARCHITECTURE.md

servers:
  # GitHub MCP Server
  # github:
  #   transport: stdio
  #   command: ["npx", "-y", "@modelcontextprotocol/server-github"]
  #   env:
  #     GITHUB_TOKEN: "${GITHUB_TOKEN}"
  #   description: "GitHub API integration"

  # Kubernetes MCP Server
  # kubernetes:
  #   transport: stdio
  #   command: ["python", "-m", "kubernetes_mcp_server"]
  #   env:
  #     KUBECONFIG: "${KUBECONFIG}"
  #   description: "Kubernetes cluster management"

  # Filesystem MCP Server (official)
  # filesystem:
  #   transport: stdio
  #   command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed"]
  #   description: "Local filesystem access"

  # Slack MCP Server
  # slack:
  #   transport: sse
  #   url: "http://localhost:8081/sse"
  #   headers:
  #     Authorization: "Bearer ${SLACK_TOKEN}"
  #   description: "Slack messaging"

  # PostgreSQL MCP Server
  # postgres:
  #   transport: stdio
  #   command: ["python", "-m", "postgres_mcp_server"]
  #   env:
  #     DATABASE_URL: "${DATABASE_URL}"
  #   description: "PostgreSQL database"
"""
