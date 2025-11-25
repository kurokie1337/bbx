# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Adapter Registry

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    BBX Engine                            │
    │                                                          │
    │  ┌──────────────────┐    ┌──────────────────────────┐  │
    │  │  CORE Adapters   │    │    MCP Plugins           │  │
    │  │  (zero deps)     │    │   (external servers)     │  │
    │  │                  │    │                          │  │
    │  │  - logger        │    │  → github MCP            │  │
    │  │  - transform     │    │  → kubernetes MCP        │  │
    │  │  - system        │    │  → postgres MCP          │  │
    │  │  - process       │    │  → filesystem MCP        │  │
    │  │  - python        │    │  → puppeteer MCP         │  │
    │  │  - http          │    │  → slack MCP             │  │
    │  │  - docker        │    │  → any MCP server...     │  │
    │  │  - universal     │    │                          │  │
    │  └──────────────────┘    └──────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘

Core adapters work out of the box with zero external dependencies.
MCP plugins extend BBX via external MCP servers (configured in mcp_servers.yaml).
"""

import logging
from typing import Callable, Dict, List, Optional

from blackbox.core.base_adapter import MCPAdapter

logger = logging.getLogger("bbx.registry")


# =============================================================================
# Adapter Categories
# =============================================================================

# Core adapters - always available, zero external dependencies
CORE_ADAPTERS = {
    "logger": "Essential logging functionality",
    "transform": "Data transformation (merge, filter, map, reduce)",
    "system": "OS/shell command execution",
    "process": "Process lifecycle management",
    "python": "Python script execution",
    "http": "HTTP client requests",
    "docker": "Docker container operations",
    "universal": "Run any CLI via Docker images",
    "workflow": "Nested workflow execution (run, wait, kill, status)",
    "state": "Persistent state management (get, set, increment, append)",
    "a2a": "Agent-to-Agent protocol (discover, call, status other A2A agents)",
    "file": "File operations (read, write, copy, delete, find, glob)",
    "string": "String manipulation (split, join, replace, regex, encode/decode)",
}

# Optional adapters - useful but have specific dependencies or use cases
OPTIONAL_ADAPTERS = {
    "database": "SQL migrations and database operations",
    "storage": "Key-value storage and file caching",
}

# Deprecated adapters - recommend MCP alternatives
DEPRECATED_ADAPTERS = {
    "browser": {
        "reason": "Use puppeteer/playwright MCP for full browser automation",
        "alternatives": [
            "mcp.puppeteer (@anthropic/puppeteer-mcp-server)",
            "mcp.playwright (Playwright MCP Server)",
        ],
    },
    "telegram": {
        "reason": "Use HTTP adapter with Telegram Bot API or dedicated MCP",
        "alternatives": [
            "http adapter with Telegram Bot API",
            "mcp.telegram (if available)",
        ],
    },
    "ai": {
        "reason": "Use provider-specific MCP servers for better integration",
        "alternatives": [
            "mcp.openai (OpenAI MCP Server)",
            "mcp.anthropic (Anthropic MCP Server)",
            "mcp.ollama (Ollama MCP Server)",
        ],
    },
}


class MCPRegistry:
    """
    Lazy-loading registry for BBX adapters.

    Separates adapters into:
    - Core: Always available, no external dependencies
    - Deprecated: Still work but recommend MCP alternatives
    - MCP: External MCP server integrations (plugins)
    """

    def __init__(self):
        self._factories: Dict[str, Callable[[], MCPAdapter]] = {}
        self._instances: Dict[str, MCPAdapter] = {}
        self._register_core_adapters()
        self._register_optional_adapters()
        self._register_deprecated_adapters()
        self._register_mcp_adapter()

    def _register_core_adapters(self):
        """Register core adapters - always available, zero dependencies"""

        # Logger - essential
        self.register_lazy(
            ["bbx.logger", "logger"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.logger", "LoggerAdapter"
            ),
        )

        # Transform - data operations
        self.register_lazy(
            ["bbx.transform", "transform"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.transform", "TransformAdapter"
            ),
        )

        # System/OS - shell execution
        self.register_lazy(
            ["bbx.system", "system", "bbx.os", "os", "shell", "bbx.shell"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.os_abstraction", "OSAbstractionAdapter"
            ),
        )

        # Process - process management
        self.register_lazy(
            ["bbx.process", "process"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.process", "ProcessAdapter"
            ),
        )

        # Python - script execution
        self.register_lazy(
            ["bbx.python", "python"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.python", "PythonAdapter"
            ),
        )

        # HTTP - web requests
        self.register_lazy(
            ["bbx.http", "http"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.http", "LocalHttpAdapter"
            ),
        )

        # Docker - container operations
        self.register_lazy(
            ["bbx.docker", "docker"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.docker", "DockerAdapter"
            ),
        )

        # Universal - Docker-based CLI execution
        self.register_lazy(
            ["universal", "bbx.universal"],
            lambda: self._import_adapter_universal()
        )

        # Workflow - nested workflow execution (like fork() in Linux)
        self.register_lazy(
            ["bbx.workflow", "workflow"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.workflow", "WorkflowAdapter"
            ),
        )

        # State - persistent state management (like env vars + config)
        self.register_lazy(
            ["bbx.state", "state"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.state", "StateAdapter"
            ),
        )

        # A2A - Agent-to-Agent protocol for multi-agent communication
        self.register_lazy(
            ["bbx.a2a", "a2a"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.a2a", "A2AAdapter"
            ),
        )

        # File - file operations (for BBX-only coding)
        self.register_lazy(
            ["bbx.file", "file"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.file", "FileAdapter"
            ),
        )

        # String - string manipulation (for BBX-only coding)
        self.register_lazy(
            ["bbx.string", "string", "str"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.string", "StringAdapter"
            ),
        )

    def _register_optional_adapters(self):
        """Register optional adapters - useful but with specific use cases"""

        # Database - SQL migrations
        self.register_lazy(
            ["bbx.db", "db", "database"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.database",
                "DatabaseMigrationAdapter",
            ),
        )

        # Storage - KV store
        self.register_lazy(
            ["bbx.storage", "storage"],
            lambda: self._import_adapter(
                "blackbox.core.adapters.storage",
                "StorageAdapter",
            ),
        )

    def _register_deprecated_adapters(self):
        """
        Deprecated adapters have been removed from the codebase.
        Use MCP alternatives instead:
        - browser → mcp.puppeteer or mcp.playwright
        - telegram → http adapter with Telegram Bot API
        - ai → mcp.openai, mcp.anthropic, or mcp.ollama
        """
        pass  # Deprecated adapters removed - use MCP plugins

    def _register_mcp_adapter(self):
        """Register MCP Client Adapter for external MCP servers"""
        self.register_lazy(
            ["mcp", "bbx.mcp"],
            lambda: self._import_mcp_client_adapter()
        )

    def _import_adapter(self, module_path: str, class_name: str) -> MCPAdapter:
        """Dynamically import and instantiate adapter."""
        try:
            import importlib
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            return adapter_class()
        except ImportError as e:
            logger.error(f"Failed to import {module_path}.{class_name}: {e}")
            raise
        except AttributeError as e:
            logger.error(f"Class {class_name} not found in {module_path}: {e}")
            raise

    def _import_adapter_universal(self) -> MCPAdapter:
        """Import UniversalAdapter."""
        from blackbox.core.universal import UniversalAdapter
        return UniversalAdapter({})

    def _import_mcp_client_adapter(self) -> MCPAdapter:
        """Import MCP Client Adapter for external servers."""
        from blackbox.mcp.client import MCPClientAdapter
        return MCPClientAdapter()

    def register(self, name: str, adapter: MCPAdapter):
        """Register an already-instantiated adapter."""
        self._instances[name] = adapter

    def register_lazy(self, names: list, factory: Callable[[], MCPAdapter]):
        """Register adapter with lazy loading."""
        for name in names:
            self._factories[name] = factory

    def get_adapter(self, name: str) -> Optional[MCPAdapter]:
        """Get adapter by name (with lazy loading)."""
        if name in self._instances:
            return self._instances[name]

        if name in self._factories:
            try:
                adapter = self._factories[name]()
                self._instances[name] = adapter
                return adapter
            except Exception as e:
                logger.error(f"Failed to load adapter {name}: {e}")
                return None

        return None

    def list_adapters(self) -> List[str]:
        """List all registered adapter names."""
        all_names = set(self._instances.keys()) | set(self._factories.keys())
        return sorted(all_names)

    def list_core_adapters(self) -> Dict[str, str]:
        """List core adapters with descriptions."""
        return CORE_ADAPTERS.copy()

    def list_optional_adapters(self) -> Dict[str, str]:
        """List optional adapters with descriptions."""
        return OPTIONAL_ADAPTERS.copy()

    def list_deprecated_adapters(self) -> Dict[str, dict]:
        """List deprecated adapters with alternatives."""
        return DEPRECATED_ADAPTERS.copy()

    def is_core(self, name: str) -> bool:
        """Check if adapter is a core adapter."""
        base_name = name.replace("bbx.", "")
        return base_name in CORE_ADAPTERS

    def is_optional(self, name: str) -> bool:
        """Check if adapter is optional."""
        base_name = name.replace("bbx.", "")
        return base_name in OPTIONAL_ADAPTERS

    def is_deprecated(self, name: str) -> bool:
        """Check if adapter is deprecated."""
        base_name = name.replace("bbx.", "")
        return base_name in DEPRECATED_ADAPTERS


# Global registry instance
_registry = None


def get_registry() -> MCPRegistry:
    """Get global registry instance (singleton)"""
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry


registry = get_registry()

__all__ = [
    "MCPRegistry",
    "get_registry",
    "registry",
    "CORE_ADAPTERS",
    "OPTIONAL_ADAPTERS",
    "DEPRECATED_ADAPTERS",
]
