# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BBX Adapter Registry

Provides lazy-loading adapter registry to avoid initializing
all adapters at startup (performance optimization).
"""

import logging
from typing import Dict, Optional, Callable
from blackbox.core.base_adapter import MCPAdapter


logger = logging.getLogger("bbx.registry")


class MCPRegistry:
    """
    Lazy-loading registry for BBX adapters.

    Adapters are registered as factory functions and only
    instantiated when first requested. This improves startup
    performance and memory usage.
    """

    def __init__(self):
        # Store adapter factories (callables that create adapters)
        self._factories: Dict[str, Callable[[], MCPAdapter]] = {}

        # Cache instantiated adapters
        self._instances: Dict[str, MCPAdapter] = {}

        # Register all built-in adapters
        self._register_builtin_adapters()

    def _register_builtin_adapters(self):
        """Register all built-in adapters with lazy loading"""

        # Phase 0: OS Abstraction
        self.register_lazy(
            ["bbx.os", "os"],
            lambda: self._import_adapter("blackbox.core.adapters.os_abstraction", "OSAbstractionAdapter")
        )

        # Phase 2: Code Generation
        self.register_lazy(
            ["codegen.template", "template"],
            lambda: self._import_adapter("blackbox.core.adapters.codegen.template", "TemplateAdapter")
        )
        self.register_lazy(
            ["codegen.fs", "fs"],
            lambda: self._import_adapter("blackbox.core.adapters.codegen.fs", "FileSystemGenAdapter")
        )

        # Phase 4: Process Management
        self.register_lazy(
            ["bbx.process", "process"],
            lambda: self._import_adapter("blackbox.core.adapters.process", "ProcessAdapter")
        )

        # Phase 4: Docker
        self.register_lazy(
            ["bbx.docker", "docker"],
            lambda: self._import_adapter("blackbox.core.adapters.docker", "DockerAdapter")
        )

        # Phase 5: AI Integration
        self.register_lazy(
            ["bbx.ai", "ai"],
            lambda: self._import_adapter("blackbox.core.adapters.ai", "AIAdapter")
        )

        # Phase 6: Infrastructure - Terraform
        self.register_lazy(
            ["bbx.terraform", "terraform"],
            lambda: self._import_adapter("blackbox.core.adapters.terraform", "TerraformAdapter")
        )

        # Phase 6: Infrastructure - Ansible
        self.register_lazy(
            ["bbx.ansible", "ansible"],
            lambda: self._import_adapter("blackbox.core.adapters.ansible", "AnsibleAdapter")
        )

        # Phase 6: Infrastructure - Kubernetes
        self.register_lazy(
            ["bbx.k8s", "k8s", "kubectl"],
            lambda: self._import_adapter("blackbox.core.adapters.kubernetes", "KubernetesAdapter")
        )

        # Phase 7: Sandbox
        self.register_lazy(
            ["bbx.sandbox", "sandbox"],
            lambda: self._import_adapter("blackbox.core.adapters.sandbox", "SandboxAdapter")
        )

        # Phase 8: HTTP Server
        self.register_lazy(
            ["bbx.http_server", "http_server"],
            lambda: self._import_adapter("blackbox.core.adapters.http_server", "HTTPServerAdapter")
        )

        # Universal Adapter (God Mode)
        self.register_lazy(
            ["universal", "bbx.universal"],
            lambda: self._import_adapter_universal()
        )

        # Cloud Providers - AWS
        self.register_lazy(
            ["bbx.aws", "aws"],
            lambda: self._import_adapter("blackbox.core.adapters.aws", "AWSAdapter")
        )

        # Cloud Providers - GCP
        self.register_lazy(
            ["bbx.gcp", "gcp"],
            lambda: self._import_adapter("blackbox.core.adapters.gcp", "GCPAdapter")
        )

        # Cloud Providers - Azure
        self.register_lazy(
            ["bbx.azure", "azure"],
            lambda: self._import_adapter("blackbox.core.adapters.azure", "AzureAdapter")
        )

        # Cloud Providers - DigitalOcean
        self.register_lazy(
            ["bbx.digitalocean", "digitalocean", "do"],
            lambda: self._import_adapter("blackbox.core.adapters.digitalocean", "DigitalOceanAdapter")
        )

        # Cloud Providers - Linode
        self.register_lazy(
            ["bbx.linode", "linode"],
            lambda: self._import_adapter("blackbox.core.adapters.linode", "LinodeAdapter")
        )

        # Database Migration
        self.register_lazy(
            ["bbx.db", "db", "database"],
            lambda: self._import_adapter("blackbox.core.adapters.database", "DatabaseMigrationAdapter")
        )

        # Storage
        self.register_lazy(
            ["bbx.storage", "storage"],
            lambda: self._import_adapter("blackbox.core.adapters.storage", "StorageAdapter")
        )

        # Queue
        self.register_lazy(
            ["bbx.queue", "queue"],
            lambda: self._import_adapter("blackbox.core.adapters.queue", "QueueAdapter")
        )

        # Mobile
        self.register_lazy(
            ["bbx.mobile", "mobile"],
            lambda: self._import_adapter("blackbox.core.adapters.mobile", "MobileAdapter")
        )

        # WebAssembly
        self.register_lazy(
            ["bbx.wasm", "wasm"],
            lambda: self._import_adapter("blackbox.core.adapters.wasm", "WasmAdapter")
        )

    def _import_adapter(self, module_path: str, class_name: str) -> MCPAdapter:
        """
        Dynamically import and instantiate adapter.

        Args:
            module_path: Python module path
            class_name: Adapter class name

        Returns:
            Instantiated adapter

        Raises:
            ImportError: If module or class not found
        """
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
        except Exception as e:
            logger.error(f"Failed to instantiate {class_name}: {e}")
            raise

    def _import_adapter_universal(self) -> MCPAdapter:
        """Import and instantiate UniversalAdapter"""
        # We need to pass the definition, but registry expects 0-arg constructor for lazy loading?
        # Wait, UniversalAdapter requires 'definition'.
        # This means UniversalAdapter cannot be a standard registered adapter in the same way
        # unless we have a 'UniversalAdapterFactory' or similar.
        # For now, let's register a 'GenericUniversalAdapter' that can load definitions dynamically?
        # Or better: The Runtime should handle 'adapter: universal' specially.
        
        # Actually, for the 'universal' adapter type in YAML:
        # - id: my_step
        #   adapter: universal
        #   definition: path/to/def.yaml
        
        # The runtime needs to instantiate it with the definition.
        # So 'universal' in registry should probably return a Factory or a special wrapper.
        
        # Let's import the class so it's available, but maybe we don't instantiate it fully here.
        from blackbox.core.universal import UniversalAdapter
        # We return the class itself? No, registry expects an instance.
        
        # HACK: Return a dummy instance or handle in runtime.
        # Let's make UniversalAdapter accept optional definition for registration.
        return UniversalAdapter({})

    def register(self, name: str, adapter: MCPAdapter):
        """
        Register an already-instantiated adapter.

        Args:
            name: Adapter name/alias
            adapter: Adapter instance
        """
        self._instances[name] = adapter
        logger.debug(f"Registered adapter: {name}")

    def register_lazy(self, names: list, factory: Callable[[], MCPAdapter]):
        """
        Register adapter with lazy loading.

        Args:
            names: List of names/aliases for this adapter
            factory: Callable that creates adapter instance
        """
        for name in names:
            self._factories[name] = factory
            logger.debug(f"Registered lazy adapter: {name}")

    def get_adapter(self, name: str) -> Optional[MCPAdapter]:
        """
        Get adapter by name (with lazy loading).

        Args:
            name: Adapter name/alias

        Returns:
            Adapter instance or None if not found
        """
        # Check if already instantiated
        if name in self._instances:
            return self._instances[name]

        # Check if factory registered
        if name in self._factories:
            try:
                # Instantiate adapter
                adapter = self._factories[name]()

                # Cache instance
                self._instances[name] = adapter

                logger.info(f"Loaded adapter: {name}")
                return adapter
            except Exception as e:
                logger.error(f"Failed to load adapter {name}: {e}")
                return None

        logger.warning(f"Adapter not found: {name}")
        return None

    def list_adapters(self) -> list:
        """
        List all registered adapter names.

        Returns:
            List of adapter names
        """
        all_names = set(self._instances.keys()) | set(self._factories.keys())
        return sorted(all_names)

    def is_loaded(self, name: str) -> bool:
        """
        Check if adapter is already instantiated.

        Args:
            name: Adapter name

        Returns:
            True if loaded, False otherwise
        """
        return name in self._instances

    # Legacy compatibility
    @property
    def servers(self) -> Dict[str, MCPAdapter]:
        """Legacy property for backward compatibility"""
        return self._instances


# Global registry instance
_registry = None


def get_registry() -> MCPRegistry:
    """Get global registry instance (singleton)"""
    global _registry
    if _registry is None:
        _registry = MCPRegistry()
    return _registry


# Legacy compatibility
registry = get_registry()


__all__ = ["MCPRegistry", "get_registry", "registry"]
