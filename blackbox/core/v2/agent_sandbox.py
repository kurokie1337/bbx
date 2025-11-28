# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Agent Sandbox - Flatpak-Inspired Isolation

Inspired by Flatpak sandboxing, provides:
- Agent isolation and containment
- Permission-based access control (portals)
- Filesystem namespacing
- Network policies
- Resource limits (via AgentQuotas)
- Secure IPC between agents

Key concepts:
- Sandbox: Isolated execution environment
- Portal: Controlled access to resources
- Permission: Granted capability
- Namespace: Isolated resource view
- Seccomp: System call filtering
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("bbx.sandbox")


# =============================================================================
# Permission System
# =============================================================================

class Permission(Enum):
    """
    Permissions that can be granted to sandboxed agents.

    Inspired by Flatpak permissions and Android permissions.
    """

    # Filesystem access
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    FILESYSTEM_HOME = "filesystem.home"
    FILESYSTEM_HOST = "filesystem.host"

    # Network access
    NETWORK_ACCESS = "network.access"
    NETWORK_HOST = "network.host"
    NETWORK_LOCALHOST = "network.localhost"

    # Process control
    PROCESS_SPAWN = "process.spawn"
    PROCESS_SIGNAL = "process.signal"

    # IPC
    IPC_DBUS = "ipc.dbus"
    IPC_SHARED_MEMORY = "ipc.shm"
    IPC_UNIX_SOCKET = "ipc.socket"

    # Hardware
    DEVICE_ALL = "device.all"
    DEVICE_GPU = "device.gpu"
    DEVICE_USB = "device.usb"

    # System
    SYSTEM_ENV = "system.env"
    SYSTEM_CLOCK = "system.clock"
    SYSTEM_LOCALE = "system.locale"

    # BBX-specific
    BBX_ADAPTERS = "bbx.adapters"
    BBX_HOOKS = "bbx.hooks"
    BBX_CONTEXT = "bbx.context"
    BBX_REGISTRY = "bbx.registry"
    BBX_STATE = "bbx.state"

    # Dangerous
    DANGEROUS_HOST_COMMANDS = "dangerous.host_commands"
    DANGEROUS_FULL_ACCESS = "dangerous.full_access"


@dataclass
class PermissionRequest:
    """A request for permission."""

    permission: Permission
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionGrant:
    """A granted permission."""

    permission: Permission
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if permission is still valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


# =============================================================================
# Filesystem Namespacing
# =============================================================================

class FilesystemMode(Enum):
    """Filesystem access modes."""
    NONE = "none"
    READ_ONLY = "ro"
    READ_WRITE = "rw"


@dataclass
class FilesystemMount:
    """A filesystem mount point."""

    source: Path           # Real path on host
    target: Path           # Path inside sandbox
    mode: FilesystemMode = FilesystemMode.READ_ONLY
    required: bool = False

    def validate(self) -> bool:
        """Validate mount configuration."""
        if not self.source.exists() and self.required:
            return False
        return True


@dataclass
class FilesystemNamespace:
    """
    Filesystem namespace for a sandbox.

    Provides an isolated filesystem view.
    """

    # Base paths
    root: Path             # Sandbox root
    home: Path             # Sandbox home directory
    tmp: Path              # Sandbox temp directory

    # Mount points
    mounts: Dict[str, FilesystemMount] = field(default_factory=dict)

    # Overlay
    use_overlay: bool = True
    overlay_upper: Optional[Path] = None
    overlay_work: Optional[Path] = None

    def __post_init__(self):
        # Ensure directories exist
        self.root.mkdir(parents=True, exist_ok=True)
        self.home.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)

        if self.use_overlay:
            self.overlay_upper = self.root / ".overlay" / "upper"
            self.overlay_work = self.root / ".overlay" / "work"
            self.overlay_upper.mkdir(parents=True, exist_ok=True)
            self.overlay_work.mkdir(parents=True, exist_ok=True)

    def add_mount(
        self,
        name: str,
        source: Union[str, Path],
        target: Union[str, Path],
        mode: FilesystemMode = FilesystemMode.READ_ONLY,
        required: bool = False,
    ):
        """Add a mount point."""
        self.mounts[name] = FilesystemMount(
            source=Path(source),
            target=self.root / Path(target).relative_to("/") if str(target).startswith("/") else self.root / target,
            mode=mode,
            required=required,
        )

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve a path within the sandbox."""
        path = Path(path)

        # Check if it's under a mount point
        for mount in self.mounts.values():
            try:
                rel = path.relative_to(mount.target)
                return mount.source / rel
            except ValueError:
                continue

        # Return path under sandbox root
        if path.is_absolute():
            return self.root / path.relative_to("/")
        return self.root / path


# =============================================================================
# Network Policy
# =============================================================================

class NetworkAction(Enum):
    """Network policy actions."""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"


@dataclass
class NetworkRule:
    """A network policy rule."""

    action: NetworkAction
    protocol: str = "any"    # tcp, udp, icmp, any
    direction: str = "both"  # inbound, outbound, both

    # Filters
    hosts: List[str] = field(default_factory=list)     # Hostnames/IPs
    ports: List[int] = field(default_factory=list)     # Port numbers
    port_range: Optional[Tuple[int, int]] = None       # Port range

    # Metadata
    description: str = ""

    def matches(
        self,
        host: str,
        port: int,
        protocol: str = "tcp",
        direction: str = "outbound",
    ) -> bool:
        """Check if this rule matches the connection."""
        # Check direction
        if self.direction != "both" and self.direction != direction:
            return False

        # Check protocol
        if self.protocol != "any" and self.protocol != protocol:
            return False

        # Check hosts
        if self.hosts and host not in self.hosts:
            # Check for wildcards
            matched = False
            for h in self.hosts:
                if h.startswith("*."):
                    if host.endswith(h[1:]):
                        matched = True
                        break
                elif h == "*":
                    matched = True
                    break
            if not matched:
                return False

        # Check ports
        if self.ports and port not in self.ports:
            return False

        if self.port_range:
            if not (self.port_range[0] <= port <= self.port_range[1]):
                return False

        return True


@dataclass
class NetworkPolicy:
    """Network policy for a sandbox."""

    # Default action
    default_action: NetworkAction = NetworkAction.DENY

    # Rules (evaluated in order)
    rules: List[NetworkRule] = field(default_factory=list)

    # Quick allows
    allow_localhost: bool = True
    allow_dns: bool = True

    def evaluate(
        self,
        host: str,
        port: int,
        protocol: str = "tcp",
        direction: str = "outbound",
    ) -> NetworkAction:
        """Evaluate network request against policy."""
        # Quick checks
        if self.allow_localhost and host in ("localhost", "127.0.0.1", "::1"):
            return NetworkAction.ALLOW

        if self.allow_dns and port == 53:
            return NetworkAction.ALLOW

        # Evaluate rules
        for rule in self.rules:
            if rule.matches(host, port, protocol, direction):
                return rule.action

        return self.default_action

    def add_allow_rule(
        self,
        hosts: Optional[List[str]] = None,
        ports: Optional[List[int]] = None,
        **kwargs
    ):
        """Add an allow rule."""
        self.rules.append(NetworkRule(
            action=NetworkAction.ALLOW,
            hosts=hosts or [],
            ports=ports or [],
            **kwargs
        ))

    def add_deny_rule(
        self,
        hosts: Optional[List[str]] = None,
        ports: Optional[List[int]] = None,
        **kwargs
    ):
        """Add a deny rule."""
        self.rules.append(NetworkRule(
            action=NetworkAction.DENY,
            hosts=hosts or [],
            ports=ports or [],
            **kwargs
        ))


# =============================================================================
# Portal System
# =============================================================================

class Portal(ABC):
    """
    Base class for portals - controlled access to host resources.

    Like Flatpak portals, these provide secure APIs for sandboxed
    agents to access host resources.
    """

    name: str = "base"

    @abstractmethod
    async def request(
        self,
        sandbox_id: str,
        method: str,
        params: Dict[str, Any],
    ) -> Any:
        """Handle a portal request."""
        pass

    @abstractmethod
    def required_permissions(self) -> List[Permission]:
        """Get required permissions for this portal."""
        pass


class FileChooserPortal(Portal):
    """Portal for file selection (like xdg-desktop-portal)."""

    name = "file-chooser"

    def __init__(self):
        self._allowed_paths: Dict[str, List[Path]] = {}  # sandbox_id -> paths

    async def request(
        self,
        sandbox_id: str,
        method: str,
        params: Dict[str, Any],
    ) -> Any:
        if method == "open":
            # Would show file picker dialog
            return {"path": "/selected/file.txt"}
        elif method == "save":
            return {"path": "/save/location/file.txt"}
        elif method == "check_access":
            path = Path(params.get("path", ""))
            allowed = self._allowed_paths.get(sandbox_id, [])
            return {"allowed": any(path.is_relative_to(p) for p in allowed)}
        return None

    def required_permissions(self) -> List[Permission]:
        return [Permission.FILESYSTEM_READ]


class NetworkPortal(Portal):
    """Portal for network access."""

    name = "network"

    async def request(
        self,
        sandbox_id: str,
        method: str,
        params: Dict[str, Any],
    ) -> Any:
        if method == "http_request":
            url = params.get("url")
            method = params.get("method", "GET")
            # Would proxy HTTP request
            return {"status": 200, "body": "..."}
        elif method == "check_host":
            host = params.get("host")
            port = params.get("port", 443)
            # Would check network policy
            return {"allowed": True}
        return None

    def required_permissions(self) -> List[Permission]:
        return [Permission.NETWORK_ACCESS]


class SecretPortal(Portal):
    """Portal for secret/credential access."""

    name = "secret"

    def __init__(self):
        self._secrets: Dict[str, Dict[str, str]] = {}

    async def request(
        self,
        sandbox_id: str,
        method: str,
        params: Dict[str, Any],
    ) -> Any:
        if method == "get":
            key = params.get("key")
            secrets = self._secrets.get(sandbox_id, {})
            return {"value": secrets.get(key)}
        elif method == "set":
            key = params.get("key")
            value = params.get("value")
            if sandbox_id not in self._secrets:
                self._secrets[sandbox_id] = {}
            self._secrets[sandbox_id][key] = value
            return {"success": True}
        return None

    def required_permissions(self) -> List[Permission]:
        return [Permission.BBX_STATE]


class AdapterPortal(Portal):
    """Portal for BBX adapter access."""

    name = "adapter"

    def __init__(self):
        self._allowed_adapters: Dict[str, Set[str]] = {}

    async def request(
        self,
        sandbox_id: str,
        method: str,
        params: Dict[str, Any],
    ) -> Any:
        if method == "execute":
            adapter = params.get("adapter")
            allowed = self._allowed_adapters.get(sandbox_id, set())
            if adapter not in allowed and "*" not in allowed:
                return {"error": f"Adapter not allowed: {adapter}"}
            # Would execute adapter
            return {"result": "..."}
        elif method == "list":
            return {"adapters": list(self._allowed_adapters.get(sandbox_id, []))}
        return None

    def required_permissions(self) -> List[Permission]:
        return [Permission.BBX_ADAPTERS]

    def allow_adapter(self, sandbox_id: str, adapter: str):
        """Allow an adapter for a sandbox."""
        if sandbox_id not in self._allowed_adapters:
            self._allowed_adapters[sandbox_id] = set()
        self._allowed_adapters[sandbox_id].add(adapter)


# =============================================================================
# Sandbox Configuration
# =============================================================================

@dataclass
class SandboxConfig:
    """Configuration for a sandbox."""

    # Identity
    name: str = ""
    agent_id: Optional[str] = None

    # Permissions
    permissions: List[Permission] = field(default_factory=list)
    permission_grants: Dict[str, PermissionGrant] = field(default_factory=dict)

    # Filesystem
    filesystem: Optional[FilesystemNamespace] = None
    shared_paths: List[str] = field(default_factory=list)  # Paths shared with host

    # Network
    network_policy: Optional[NetworkPolicy] = None
    allow_network: bool = False

    # Resources (integrated with AgentQuotas)
    max_memory_mb: int = 512
    max_cpu_percent: float = 100.0
    max_io_mbps: float = 100.0

    # Execution
    max_runtime_seconds: int = 3600
    allow_spawn: bool = False
    max_processes: int = 10

    # IPC
    allow_ipc: bool = False
    ipc_channels: List[str] = field(default_factory=list)

    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)
    env_passthrough: List[str] = field(default_factory=list)  # Pass through from host

    def has_permission(self, permission: Permission) -> bool:
        """Check if permission is granted."""
        if permission in self.permissions:
            return True

        # Check grants
        perm_str = permission.value
        if perm_str in self.permission_grants:
            grant = self.permission_grants[perm_str]
            return grant.is_valid

        return False


# =============================================================================
# Sandbox Instance
# =============================================================================

class SandboxState(Enum):
    """State of a sandbox."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class Sandbox:
    """
    An isolated agent execution environment.

    Provides containment similar to Flatpak sandboxing.
    """

    # Identity
    id: str
    config: SandboxConfig
    state: SandboxState = SandboxState.CREATED

    # Runtime
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    # Resources
    current_memory_mb: float = 0
    current_cpu_percent: float = 0

    # IPC
    input_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Logs
    logs: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.config.filesystem:
            # Create default filesystem namespace
            base = Path(tempfile.gettempdir()) / "bbx-sandbox" / self.id
            self.config.filesystem = FilesystemNamespace(
                root=base / "root",
                home=base / "home",
                tmp=base / "tmp",
            )

    def log(self, message: str, level: str = "INFO"):
        """Add a log entry."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] [{level}] {message}")

    async def send(self, message: Any):
        """Send a message to the sandbox."""
        await self.input_queue.put(message)

    async def receive(self, timeout: Optional[float] = None) -> Any:
        """Receive a message from the sandbox."""
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.output_queue.get(),
                    timeout=timeout
                )
            return await self.output_queue.get()
        except asyncio.TimeoutError:
            return None


# =============================================================================
# Sandbox Manager
# =============================================================================

class SandboxManager:
    """
    Central manager for agent sandboxes.

    Handles:
    - Sandbox creation and lifecycle
    - Permission management
    - Portal access
    - Resource monitoring
    """

    def __init__(
        self,
        sandboxes_path: Optional[Path] = None,
        quota_manager: Optional[Any] = None,  # AgentQuotas
    ):
        self.sandboxes_path = sandboxes_path or (Path.home() / ".bbx" / "sandboxes")
        self.sandboxes_path.mkdir(parents=True, exist_ok=True)

        self.quota_manager = quota_manager

        # Active sandboxes
        self._sandboxes: Dict[str, Sandbox] = {}

        # Portals
        self._portals: Dict[str, Portal] = {}
        self._register_default_portals()

        # Permission handlers
        self._permission_handlers: Dict[Permission, Callable] = {}

        # Statistics
        self._stats = {
            "sandboxes_created": 0,
            "sandboxes_active": 0,
            "portal_requests": 0,
            "permission_denials": 0,
        }

    def _register_default_portals(self):
        """Register default portals."""
        self._portals["file-chooser"] = FileChooserPortal()
        self._portals["network"] = NetworkPortal()
        self._portals["secret"] = SecretPortal()
        self._portals["adapter"] = AdapterPortal()

    def register_portal(self, portal: Portal):
        """Register a custom portal."""
        self._portals[portal.name] = portal

    async def create_sandbox(
        self,
        config: SandboxConfig,
        sandbox_id: Optional[str] = None,
    ) -> Sandbox:
        """Create a new sandbox."""
        sandbox_id = sandbox_id or str(uuid.uuid4())

        # Set up filesystem
        sandbox_path = self.sandboxes_path / sandbox_id
        if not config.filesystem:
            config.filesystem = FilesystemNamespace(
                root=sandbox_path / "root",
                home=sandbox_path / "home",
                tmp=sandbox_path / "tmp",
            )

        # Set up network policy
        if config.allow_network and not config.network_policy:
            config.network_policy = NetworkPolicy(
                default_action=NetworkAction.ALLOW,
                allow_localhost=True,
                allow_dns=True,
            )
        elif not config.allow_network and not config.network_policy:
            config.network_policy = NetworkPolicy(
                default_action=NetworkAction.DENY,
                allow_localhost=True,
                allow_dns=True,
            )

        sandbox = Sandbox(id=sandbox_id, config=config)

        self._sandboxes[sandbox_id] = sandbox
        self._stats["sandboxes_created"] += 1

        sandbox.log(f"Sandbox created: {sandbox_id}")

        return sandbox

    async def start_sandbox(self, sandbox_id: str) -> bool:
        """Start a sandbox."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        if sandbox.state != SandboxState.CREATED:
            return False

        sandbox.state = SandboxState.STARTING
        sandbox.log("Starting sandbox")

        try:
            # Set up resource limits
            if self.quota_manager:
                await self.quota_manager.create_group(
                    sandbox_id,
                    parent=None,
                    limits={
                        "memory": sandbox.config.max_memory_mb * 1024 * 1024,
                        "cpu": sandbox.config.max_cpu_percent,
                    }
                )

            # Start execution environment
            # In production, would create actual container/namespace
            sandbox.state = SandboxState.RUNNING
            sandbox.started_at = datetime.now()
            self._stats["sandboxes_active"] += 1

            sandbox.log("Sandbox started")
            return True

        except Exception as e:
            sandbox.state = SandboxState.FAILED
            sandbox.log(f"Failed to start: {e}", "ERROR")
            return False

    async def stop_sandbox(self, sandbox_id: str, force: bool = False) -> bool:
        """Stop a sandbox."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        if sandbox.state not in (SandboxState.RUNNING, SandboxState.PAUSED):
            return False

        sandbox.state = SandboxState.STOPPING
        sandbox.log("Stopping sandbox")

        try:
            # Clean up resources
            if self.quota_manager:
                await self.quota_manager.delete_group(sandbox_id)

            sandbox.state = SandboxState.STOPPED
            sandbox.stopped_at = datetime.now()
            self._stats["sandboxes_active"] -= 1

            sandbox.log("Sandbox stopped")
            return True

        except Exception as e:
            sandbox.log(f"Error stopping: {e}", "ERROR")
            if force:
                sandbox.state = SandboxState.STOPPED
                return True
            return False

    async def destroy_sandbox(self, sandbox_id: str) -> bool:
        """Destroy a sandbox and clean up resources."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        if sandbox.state == SandboxState.RUNNING:
            await self.stop_sandbox(sandbox_id, force=True)

        # Clean up filesystem
        import shutil
        sandbox_path = self.sandboxes_path / sandbox_id
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path, ignore_errors=True)

        del self._sandboxes[sandbox_id]
        return True

    def get_sandbox(self, sandbox_id: str) -> Optional[Sandbox]:
        """Get a sandbox by ID."""
        return self._sandboxes.get(sandbox_id)

    def list_sandboxes(
        self,
        state: Optional[SandboxState] = None
    ) -> List[Sandbox]:
        """List all sandboxes."""
        sandboxes = list(self._sandboxes.values())

        if state:
            sandboxes = [s for s in sandboxes if s.state == state]

        return sandboxes

    async def portal_request(
        self,
        sandbox_id: str,
        portal_name: str,
        method: str,
        params: Dict[str, Any],
    ) -> Any:
        """Handle a portal request from a sandbox."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            raise ValueError(f"Sandbox not found: {sandbox_id}")

        portal = self._portals.get(portal_name)
        if not portal:
            raise ValueError(f"Portal not found: {portal_name}")

        # Check permissions
        for perm in portal.required_permissions():
            if not sandbox.config.has_permission(perm):
                self._stats["permission_denials"] += 1
                sandbox.log(f"Permission denied: {perm.value}", "WARN")
                raise PermissionError(f"Permission denied: {perm.value}")

        self._stats["portal_requests"] += 1
        sandbox.log(f"Portal request: {portal_name}.{method}")

        return await portal.request(sandbox_id, method, params)

    async def grant_permission(
        self,
        sandbox_id: str,
        permission: Permission,
        expires_in_seconds: Optional[int] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Grant a permission to a sandbox."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        expires_at = None
        if expires_in_seconds:
            from datetime import timedelta
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)

        grant = PermissionGrant(
            permission=permission,
            granted_at=datetime.now(),
            expires_at=expires_at,
            conditions=conditions or {},
        )

        sandbox.config.permission_grants[permission.value] = grant
        sandbox.log(f"Permission granted: {permission.value}")

        return True

    async def revoke_permission(
        self,
        sandbox_id: str,
        permission: Permission,
    ) -> bool:
        """Revoke a permission from a sandbox."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox:
            return False

        if permission.value in sandbox.config.permission_grants:
            del sandbox.config.permission_grants[permission.value]
            sandbox.log(f"Permission revoked: {permission.value}")
            return True

        return False

    async def check_network(
        self,
        sandbox_id: str,
        host: str,
        port: int,
        protocol: str = "tcp",
    ) -> bool:
        """Check if network access is allowed."""
        sandbox = self._sandboxes.get(sandbox_id)
        if not sandbox or not sandbox.config.network_policy:
            return False

        action = sandbox.config.network_policy.evaluate(
            host, port, protocol, "outbound"
        )

        return action == NetworkAction.ALLOW

    @asynccontextmanager
    async def sandbox_context(
        self,
        config: SandboxConfig
    ):
        """Context manager for temporary sandbox."""
        sandbox = await self.create_sandbox(config)
        try:
            await self.start_sandbox(sandbox.id)
            yield sandbox
        finally:
            await self.destroy_sandbox(sandbox.id)

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self._stats,
            "portals_registered": len(self._portals),
        }


# =============================================================================
# Sandbox Templates
# =============================================================================

class SandboxTemplates:
    """Pre-defined sandbox configurations."""

    @staticmethod
    def minimal() -> SandboxConfig:
        """Minimal sandbox with no permissions."""
        return SandboxConfig(
            name="minimal",
            permissions=[],
            allow_network=False,
            allow_spawn=False,
            allow_ipc=False,
            max_memory_mb=128,
            max_runtime_seconds=60,
        )

    @staticmethod
    def standard() -> SandboxConfig:
        """Standard sandbox for most use cases."""
        return SandboxConfig(
            name="standard",
            permissions=[
                Permission.FILESYSTEM_READ,
                Permission.BBX_ADAPTERS,
                Permission.BBX_STATE,
            ],
            allow_network=False,
            allow_spawn=False,
            allow_ipc=True,
            max_memory_mb=512,
            max_runtime_seconds=3600,
        )

    @staticmethod
    def network() -> SandboxConfig:
        """Sandbox with network access."""
        config = SandboxTemplates.standard()
        config.name = "network"
        config.permissions.append(Permission.NETWORK_ACCESS)
        config.allow_network = True
        config.network_policy = NetworkPolicy(
            default_action=NetworkAction.ALLOW,
            allow_localhost=True,
            allow_dns=True,
        )
        return config

    @staticmethod
    def developer() -> SandboxConfig:
        """Developer sandbox with more permissions."""
        config = SandboxConfig(
            name="developer",
            permissions=[
                Permission.FILESYSTEM_READ,
                Permission.FILESYSTEM_WRITE,
                Permission.FILESYSTEM_HOME,
                Permission.NETWORK_ACCESS,
                Permission.PROCESS_SPAWN,
                Permission.BBX_ADAPTERS,
                Permission.BBX_HOOKS,
                Permission.BBX_STATE,
            ],
            allow_network=True,
            allow_spawn=True,
            allow_ipc=True,
            max_memory_mb=2048,
            max_processes=50,
            max_runtime_seconds=86400,
        )
        config.network_policy = NetworkPolicy(
            default_action=NetworkAction.ALLOW,
        )
        return config

    @staticmethod
    def untrusted() -> SandboxConfig:
        """Highly restricted sandbox for untrusted code."""
        return SandboxConfig(
            name="untrusted",
            permissions=[],
            allow_network=False,
            allow_spawn=False,
            allow_ipc=False,
            max_memory_mb=64,
            max_cpu_percent=10.0,
            max_runtime_seconds=30,
            max_processes=1,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_sandbox(
    config: Optional[SandboxConfig] = None,
    template: str = "standard",
) -> Sandbox:
    """Create a sandbox with default manager."""
    manager = SandboxManager()

    if not config:
        if template == "minimal":
            config = SandboxTemplates.minimal()
        elif template == "network":
            config = SandboxTemplates.network()
        elif template == "developer":
            config = SandboxTemplates.developer()
        elif template == "untrusted":
            config = SandboxTemplates.untrusted()
        else:
            config = SandboxTemplates.standard()

    sandbox = await manager.create_sandbox(config)
    await manager.start_sandbox(sandbox.id)

    return sandbox


async def run_in_sandbox(
    func: Callable,
    config: Optional[SandboxConfig] = None,
) -> Any:
    """Run a function in a sandbox."""
    manager = SandboxManager()

    config = config or SandboxTemplates.standard()

    async with manager.sandbox_context(config) as sandbox:
        # Would actually run in isolated environment
        if asyncio.iscoroutinefunction(func):
            return await func()
        return func()
