# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Config Registry - Windows Registry Inspired

Implements a hierarchical configuration store similar to Windows Registry.

Key concepts:
- Hives: Root-level configuration namespaces
- Keys: Hierarchical containers (like folders)
- Values: Named data entries with typed values
- Security: ACL-based access control per key

Standard Hives:
  HKBX_SYSTEM     - System-wide configuration
  HKBX_AGENTS     - Per-agent configuration
  HKBX_WORKFLOWS  - Workflow templates and defaults
  HKBX_SERVICES   - Service configuration
  HKBX_SECURITY   - Security policies and ACLs
  HKBX_CURRENT    - Current session configuration

Path format: HKBX_SYSTEM\\Network\\Mesh\\timeout_ms

Value types:
  REG_SZ         - String
  REG_DWORD      - Integer (32-bit)
  REG_QWORD      - Integer (64-bit)
  REG_BINARY     - Binary data
  REG_MULTI_SZ   - String array
  REG_JSON       - JSON object (BBX extension)
  REG_EXPAND_SZ  - String with variable expansion
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import asyncio
import copy
import fnmatch
import json
import logging
import os
import re
import threading
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ValueType(Enum):
    """Registry value types."""
    REG_NONE = auto()        # No type
    REG_SZ = auto()          # String
    REG_EXPAND_SZ = auto()   # Expandable string
    REG_BINARY = auto()      # Binary data
    REG_DWORD = auto()       # 32-bit integer
    REG_QWORD = auto()       # 64-bit integer
    REG_MULTI_SZ = auto()    # String array
    REG_JSON = auto()        # JSON object


class Hive(Enum):
    """Standard registry hives."""
    HKBX_SYSTEM = "HKBX_SYSTEM"
    HKBX_AGENTS = "HKBX_AGENTS"
    HKBX_WORKFLOWS = "HKBX_WORKFLOWS"
    HKBX_SERVICES = "HKBX_SERVICES"
    HKBX_SECURITY = "HKBX_SECURITY"
    HKBX_CURRENT = "HKBX_CURRENT"


class AccessRights(Enum):
    """Registry access rights."""
    KEY_READ = 0x20019
    KEY_WRITE = 0x20006
    KEY_ALL_ACCESS = 0xF003F
    KEY_QUERY_VALUE = 0x0001
    KEY_SET_VALUE = 0x0002
    KEY_CREATE_SUB_KEY = 0x0004
    KEY_ENUMERATE_SUB_KEYS = 0x0008
    KEY_NOTIFY = 0x0010
    KEY_CREATE_LINK = 0x0020


class NotifyFilter(Enum):
    """Filters for change notifications."""
    NAME = auto()           # Key name changes
    ATTRIBUTES = auto()     # Attribute changes
    LAST_SET = auto()       # Last write time changes
    SECURITY = auto()       # Security descriptor changes


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RegistryValue:
    """A registry value entry."""
    name: str
    type: ValueType
    data: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        """Validate data matches type."""
        if self.type == ValueType.REG_SZ:
            if not isinstance(self.data, str):
                self.data = str(self.data)
        elif self.type == ValueType.REG_DWORD:
            self.data = int(self.data) & 0xFFFFFFFF
        elif self.type == ValueType.REG_QWORD:
            self.data = int(self.data) & 0xFFFFFFFFFFFFFFFF
        elif self.type == ValueType.REG_MULTI_SZ:
            if not isinstance(self.data, list):
                self.data = [str(self.data)]
            else:
                self.data = [str(x) for x in self.data]
        elif self.type == ValueType.REG_BINARY:
            if isinstance(self.data, str):
                self.data = self.data.encode()
            elif not isinstance(self.data, bytes):
                self.data = bytes(self.data)


@dataclass
class RegistryKeySecurity:
    """Security descriptor for a registry key."""
    owner: str = "system"
    group: str = "system"
    acl: Dict[str, int] = field(default_factory=dict)  # sid -> access mask

    def check_access(self, caller: str, desired: int) -> bool:
        """Check if caller has desired access."""
        if caller == self.owner or caller == "system":
            return True

        if caller in self.acl:
            return (self.acl[caller] & desired) == desired

        # Check for wildcard entries
        if "*" in self.acl:
            return (self.acl["*"] & desired) == desired

        return False


@dataclass
class RegistryKey:
    """A registry key (container)."""
    name: str
    parent_path: str
    values: Dict[str, RegistryValue] = field(default_factory=dict)
    subkeys: Dict[str, "RegistryKey"] = field(default_factory=dict)
    security: RegistryKeySecurity = field(default_factory=RegistryKeySecurity)
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    volatile: bool = False  # If True, not persisted

    @property
    def path(self) -> str:
        """Get full path."""
        if self.parent_path:
            return f"{self.parent_path}\\{self.name}"
        return self.name

    def get_value(self, name: str) -> Optional[RegistryValue]:
        """Get a value by name."""
        return self.values.get(name)

    def set_value(
        self,
        name: str,
        data: Any,
        value_type: ValueType = ValueType.REG_SZ
    ) -> RegistryValue:
        """Set a value."""
        if name in self.values:
            value = self.values[name]
            value.data = data
            value.type = value_type
            value.modified_at = datetime.utcnow()
        else:
            value = RegistryValue(name=name, type=value_type, data=data)
            self.values[name] = value

        self.modified_at = datetime.utcnow()
        return value

    def delete_value(self, name: str) -> bool:
        """Delete a value."""
        if name in self.values:
            del self.values[name]
            self.modified_at = datetime.utcnow()
            return True
        return False

    def create_subkey(self, name: str, volatile: bool = False) -> "RegistryKey":
        """Create a subkey."""
        if name in self.subkeys:
            return self.subkeys[name]

        subkey = RegistryKey(
            name=name,
            parent_path=self.path,
            volatile=volatile,
            security=RegistryKeySecurity(owner=self.security.owner)
        )
        self.subkeys[name] = subkey
        self.modified_at = datetime.utcnow()
        return subkey

    def delete_subkey(self, name: str, recursive: bool = False) -> bool:
        """Delete a subkey."""
        if name not in self.subkeys:
            return False

        subkey = self.subkeys[name]
        if subkey.subkeys and not recursive:
            return False  # Has children, need recursive

        del self.subkeys[name]
        self.modified_at = datetime.utcnow()
        return True


@dataclass
class ChangeNotification:
    """Notification of registry changes."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key_path: str = ""
    filters: Set[NotifyFilter] = field(default_factory=set)
    watch_subtree: bool = False
    callback: Optional[Callable[[str, str, Any], None]] = None
    active: bool = True


# =============================================================================
# Registry Transaction
# =============================================================================

class RegistryTransaction:
    """
    Transaction for atomic registry operations.

    Similar to Windows registry transactions (KTM).
    """

    def __init__(self, registry: "ConfigRegistry"):
        self._registry = registry
        self._id = str(uuid.uuid4())
        self._operations: List[Tuple[str, Dict[str, Any]]] = []
        self._rollback_ops: List[Tuple[str, Dict[str, Any]]] = []
        self._committed = False
        self._rolled_back = False

    @property
    def id(self) -> str:
        return self._id

    async def set_value(
        self,
        path: str,
        name: str,
        data: Any,
        value_type: ValueType = ValueType.REG_SZ
    ) -> None:
        """Stage a set value operation."""
        # Get current value for rollback
        current = await self._registry.get_value(path, name)

        self._operations.append(("set_value", {
            "path": path,
            "name": name,
            "data": data,
            "type": value_type,
        }))

        if current:
            self._rollback_ops.append(("set_value", {
                "path": path,
                "name": current.name,
                "data": current.data,
                "type": current.type,
            }))
        else:
            self._rollback_ops.append(("delete_value", {
                "path": path,
                "name": name,
            }))

    async def delete_value(self, path: str, name: str) -> None:
        """Stage a delete value operation."""
        current = await self._registry.get_value(path, name)

        self._operations.append(("delete_value", {
            "path": path,
            "name": name,
        }))

        if current:
            self._rollback_ops.append(("set_value", {
                "path": path,
                "name": current.name,
                "data": current.data,
                "type": current.type,
            }))

    async def create_key(self, path: str, volatile: bool = False) -> None:
        """Stage a create key operation."""
        self._operations.append(("create_key", {
            "path": path,
            "volatile": volatile,
        }))
        self._rollback_ops.append(("delete_key", {
            "path": path,
        }))

    async def delete_key(self, path: str) -> None:
        """Stage a delete key operation."""
        self._operations.append(("delete_key", {
            "path": path,
        }))
        # Can't easily rollback key deletion with all values/subkeys
        # Would need full export

    async def commit(self) -> bool:
        """Commit all operations."""
        if self._committed or self._rolled_back:
            return False

        try:
            for op, params in self._operations:
                if op == "set_value":
                    await self._registry.set_value(
                        params["path"],
                        params["name"],
                        params["data"],
                        params["type"],
                    )
                elif op == "delete_value":
                    await self._registry.delete_value(
                        params["path"],
                        params["name"],
                    )
                elif op == "create_key":
                    await self._registry.create_key(
                        params["path"],
                        params["volatile"],
                    )
                elif op == "delete_key":
                    await self._registry.delete_key(params["path"])

            self._committed = True
            return True

        except Exception as e:
            logger.error(f"Transaction commit failed: {e}")
            await self.rollback()
            return False

    async def rollback(self) -> bool:
        """Rollback all operations."""
        if self._committed or self._rolled_back:
            return False

        try:
            # Apply rollback operations in reverse
            for op, params in reversed(self._rollback_ops):
                if op == "set_value":
                    await self._registry.set_value(
                        params["path"],
                        params["name"],
                        params["data"],
                        params["type"],
                    )
                elif op == "delete_value":
                    await self._registry.delete_value(
                        params["path"],
                        params["name"],
                    )
                elif op == "delete_key":
                    await self._registry.delete_key(params["path"])

            self._rolled_back = True
            return True

        except Exception as e:
            logger.error(f"Transaction rollback failed: {e}")
            return False


# =============================================================================
# Config Registry
# =============================================================================

class ConfigRegistry:
    """
    Centralized hierarchical configuration store.

    Similar to Windows Registry.
    """

    PATH_SEPARATOR = "\\"

    def __init__(self):
        self._hives: Dict[str, RegistryKey] = {}
        self._notifications: Dict[str, ChangeNotification] = {}
        self._lock = asyncio.Lock()
        self._variables: Dict[str, str] = {}  # For REG_EXPAND_SZ

        # Initialize standard hives
        self._init_hives()

    def _init_hives(self) -> None:
        """Initialize standard hives."""
        for hive in Hive:
            self._hives[hive.value] = RegistryKey(
                name=hive.value,
                parent_path="",
                security=RegistryKeySecurity(owner="system", group="system")
            )

        # Set up default structure
        self._setup_default_structure()

    def _setup_default_structure(self) -> None:
        """Set up default registry structure."""
        # HKBX_SYSTEM defaults
        system = self._hives[Hive.HKBX_SYSTEM.value]
        network = system.create_subkey("Network")
        network.create_subkey("Mesh")
        network.create_subkey("Services")

        runtime = system.create_subkey("Runtime")
        runtime.set_value("version", "2.0.0")
        runtime.set_value("max_agents", 100, ValueType.REG_DWORD)

        system.create_subkey("Policies")
        system.create_subkey("Quotas")

        # HKBX_SECURITY defaults
        security = self._hives[Hive.HKBX_SECURITY.value]
        security.create_subkey("Users")
        security.create_subkey("Groups")
        security.create_subkey("Policies")
        security.create_subkey("Audit")

        # HKBX_SERVICES defaults
        services = self._hives[Hive.HKBX_SERVICES.value]
        services.create_subkey("Adapters")
        services.create_subkey("Backends")

    def _parse_path(self, path: str) -> Tuple[str, List[str]]:
        """Parse path into hive and key parts."""
        parts = path.split(self.PATH_SEPARATOR)
        hive = parts[0]
        key_parts = parts[1:] if len(parts) > 1 else []
        return hive, key_parts

    def _navigate_to_key(
        self,
        path: str,
        create: bool = False
    ) -> Optional[RegistryKey]:
        """Navigate to a key by path."""
        hive_name, key_parts = self._parse_path(path)

        if hive_name not in self._hives:
            return None

        current = self._hives[hive_name]

        for part in key_parts:
            if not part:
                continue

            if part in current.subkeys:
                current = current.subkeys[part]
            elif create:
                current = current.create_subkey(part)
            else:
                return None

        return current

    # =========================================================================
    # Key Operations
    # =========================================================================

    async def create_key(
        self,
        path: str,
        volatile: bool = False,
        caller: str = "system"
    ) -> Optional[RegistryKey]:
        """Create a registry key."""
        async with self._lock:
            parent_path = self.PATH_SEPARATOR.join(path.split(self.PATH_SEPARATOR)[:-1])
            key_name = path.split(self.PATH_SEPARATOR)[-1]

            parent = self._navigate_to_key(parent_path, create=True)
            if not parent:
                return None

            # Check access
            if not parent.security.check_access(caller, AccessRights.KEY_CREATE_SUB_KEY.value):
                logger.warning(f"Access denied: {caller} cannot create key at {path}")
                return None

            key = parent.create_subkey(key_name, volatile)
            key.security.owner = caller

            self._notify_change(path, NotifyFilter.NAME, key)
            logger.debug(f"Created registry key: {path}")
            return key

    async def open_key(
        self,
        path: str,
        caller: str = "system",
        access: int = AccessRights.KEY_READ.value
    ) -> Optional[RegistryKey]:
        """Open a registry key."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return None

            if not key.security.check_access(caller, access):
                logger.warning(f"Access denied: {caller} cannot access {path}")
                return None

            return key

    async def delete_key(
        self,
        path: str,
        recursive: bool = False,
        caller: str = "system"
    ) -> bool:
        """Delete a registry key."""
        async with self._lock:
            parent_path = self.PATH_SEPARATOR.join(path.split(self.PATH_SEPARATOR)[:-1])
            key_name = path.split(self.PATH_SEPARATOR)[-1]

            parent = self._navigate_to_key(parent_path)
            if not parent:
                return False

            # Check access
            if not parent.security.check_access(caller, AccessRights.KEY_ALL_ACCESS.value):
                logger.warning(f"Access denied: {caller} cannot delete key {path}")
                return False

            result = parent.delete_subkey(key_name, recursive)
            if result:
                self._notify_change(path, NotifyFilter.NAME, None)
                logger.debug(f"Deleted registry key: {path}")

            return result

    async def key_exists(self, path: str) -> bool:
        """Check if key exists."""
        async with self._lock:
            return self._navigate_to_key(path) is not None

    async def enumerate_keys(
        self,
        path: str,
        caller: str = "system"
    ) -> List[str]:
        """Enumerate subkeys."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return []

            if not key.security.check_access(caller, AccessRights.KEY_ENUMERATE_SUB_KEYS.value):
                return []

            return list(key.subkeys.keys())

    # =========================================================================
    # Value Operations
    # =========================================================================

    async def set_value(
        self,
        path: str,
        name: str,
        data: Any,
        value_type: ValueType = ValueType.REG_SZ,
        caller: str = "system"
    ) -> bool:
        """Set a registry value."""
        async with self._lock:
            key = self._navigate_to_key(path, create=True)
            if not key:
                return False

            if not key.security.check_access(caller, AccessRights.KEY_SET_VALUE.value):
                logger.warning(f"Access denied: {caller} cannot set value at {path}")
                return False

            key.set_value(name, data, value_type)
            self._notify_change(path, NotifyFilter.LAST_SET, (name, data))
            logger.debug(f"Set registry value: {path}\\{name}")
            return True

    async def get_value(
        self,
        path: str,
        name: str,
        caller: str = "system"
    ) -> Optional[RegistryValue]:
        """Get a registry value."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return None

            if not key.security.check_access(caller, AccessRights.KEY_QUERY_VALUE.value):
                return None

            value = key.get_value(name)
            if value and value.type == ValueType.REG_EXPAND_SZ:
                # Expand variables
                value = copy.copy(value)
                value.data = self._expand_string(value.data)

            return value

    async def get_value_data(
        self,
        path: str,
        name: str,
        default: Any = None,
        caller: str = "system"
    ) -> Any:
        """Get registry value data with default."""
        value = await self.get_value(path, name, caller)
        if value:
            return value.data
        return default

    async def delete_value(
        self,
        path: str,
        name: str,
        caller: str = "system"
    ) -> bool:
        """Delete a registry value."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return False

            if not key.security.check_access(caller, AccessRights.KEY_SET_VALUE.value):
                logger.warning(f"Access denied: {caller} cannot delete value at {path}")
                return False

            result = key.delete_value(name)
            if result:
                self._notify_change(path, NotifyFilter.LAST_SET, (name, None))
                logger.debug(f"Deleted registry value: {path}\\{name}")

            return result

    async def enumerate_values(
        self,
        path: str,
        caller: str = "system"
    ) -> List[str]:
        """Enumerate values in a key."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return []

            if not key.security.check_access(caller, AccessRights.KEY_QUERY_VALUE.value):
                return []

            return list(key.values.keys())

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def get_string(
        self,
        path: str,
        name: str,
        default: str = ""
    ) -> str:
        """Get string value."""
        data = await self.get_value_data(path, name, default)
        return str(data) if data is not None else default

    async def get_int(
        self,
        path: str,
        name: str,
        default: int = 0
    ) -> int:
        """Get integer value."""
        data = await self.get_value_data(path, name, default)
        try:
            return int(data)
        except (TypeError, ValueError):
            return default

    async def get_json(
        self,
        path: str,
        name: str,
        default: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Get JSON value."""
        data = await self.get_value_data(path, name, default)
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return default
        return default

    async def set_string(self, path: str, name: str, value: str) -> bool:
        """Set string value."""
        return await self.set_value(path, name, value, ValueType.REG_SZ)

    async def set_int(self, path: str, name: str, value: int) -> bool:
        """Set integer value."""
        return await self.set_value(path, name, value, ValueType.REG_DWORD)

    async def set_json(self, path: str, name: str, value: Dict) -> bool:
        """Set JSON value."""
        return await self.set_value(path, name, value, ValueType.REG_JSON)

    # =========================================================================
    # Variable Expansion
    # =========================================================================

    def set_variable(self, name: str, value: str) -> None:
        """Set a variable for expansion."""
        self._variables[name] = value

    def _expand_string(self, s: str) -> str:
        """Expand variables in string."""
        pattern = r'%([^%]+)%'

        def replace(match):
            var_name = match.group(1)
            return self._variables.get(var_name, match.group(0))

        return re.sub(pattern, replace, s)

    # =========================================================================
    # Notifications
    # =========================================================================

    async def register_notify(
        self,
        path: str,
        callback: Callable[[str, str, Any], None],
        filters: Optional[Set[NotifyFilter]] = None,
        watch_subtree: bool = False
    ) -> str:
        """Register for change notifications."""
        notification = ChangeNotification(
            key_path=path,
            filters=filters or {NotifyFilter.LAST_SET},
            watch_subtree=watch_subtree,
            callback=callback,
        )

        async with self._lock:
            self._notifications[notification.id] = notification

        return notification.id

    async def unregister_notify(self, notification_id: str) -> bool:
        """Unregister change notification."""
        async with self._lock:
            if notification_id in self._notifications:
                self._notifications[notification_id].active = False
                del self._notifications[notification_id]
                return True
            return False

    def _notify_change(
        self,
        path: str,
        filter_type: NotifyFilter,
        data: Any
    ) -> None:
        """Send change notifications."""
        for notif in self._notifications.values():
            if not notif.active:
                continue

            if filter_type not in notif.filters:
                continue

            # Check path match
            if notif.watch_subtree:
                if not path.startswith(notif.key_path):
                    continue
            else:
                if path != notif.key_path:
                    continue

            # Call callback (non-blocking)
            if notif.callback:
                try:
                    notif.callback(path, filter_type.name, data)
                except Exception as e:
                    logger.error(f"Notification callback error: {e}")

    # =========================================================================
    # Transactions
    # =========================================================================

    def begin_transaction(self) -> RegistryTransaction:
        """Begin a registry transaction."""
        return RegistryTransaction(self)

    # =========================================================================
    # Export/Import
    # =========================================================================

    async def export_key(
        self,
        path: str,
        caller: str = "system"
    ) -> Optional[Dict[str, Any]]:
        """Export a key and all subkeys/values to dict."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return None

            if not key.security.check_access(caller, AccessRights.KEY_READ.value):
                return None

            return self._export_key_recursive(key)

    def _export_key_recursive(self, key: RegistryKey) -> Dict[str, Any]:
        """Recursively export key."""
        result = {
            "name": key.name,
            "path": key.path,
            "values": {},
            "subkeys": {},
        }

        for name, value in key.values.items():
            result["values"][name] = {
                "type": value.type.name,
                "data": value.data if not isinstance(value.data, bytes) else value.data.hex(),
            }

        for name, subkey in key.subkeys.items():
            result["subkeys"][name] = self._export_key_recursive(subkey)

        return result

    async def import_key(
        self,
        path: str,
        data: Dict[str, Any],
        caller: str = "system"
    ) -> bool:
        """Import key from dict."""
        async with self._lock:
            key = self._navigate_to_key(path, create=True)
            if not key:
                return False

            if not key.security.check_access(caller, AccessRights.KEY_ALL_ACCESS.value):
                return False

            return self._import_key_recursive(key, data)

    def _import_key_recursive(
        self,
        key: RegistryKey,
        data: Dict[str, Any]
    ) -> bool:
        """Recursively import key."""
        try:
            # Import values
            for name, value_data in data.get("values", {}).items():
                value_type = ValueType[value_data["type"]]
                data_val = value_data["data"]

                # Convert hex back to bytes for binary
                if value_type == ValueType.REG_BINARY and isinstance(data_val, str):
                    data_val = bytes.fromhex(data_val)

                key.set_value(name, data_val, value_type)

            # Import subkeys
            for name, subkey_data in data.get("subkeys", {}).items():
                subkey = key.create_subkey(name)
                self._import_key_recursive(subkey, subkey_data)

            return True

        except Exception as e:
            logger.error(f"Import error: {e}")
            return False

    # =========================================================================
    # Query
    # =========================================================================

    async def query(
        self,
        path_pattern: str,
        value_pattern: Optional[str] = None,
        caller: str = "system"
    ) -> List[Tuple[str, str, Any]]:
        """
        Query registry with patterns.

        Returns list of (path, value_name, data) tuples.
        """
        results = []

        async with self._lock:
            for hive_name, hive in self._hives.items():
                self._query_recursive(
                    hive,
                    path_pattern,
                    value_pattern,
                    results,
                    caller
                )

        return results

    def _query_recursive(
        self,
        key: RegistryKey,
        path_pattern: str,
        value_pattern: Optional[str],
        results: List[Tuple[str, str, Any]],
        caller: str
    ) -> None:
        """Recursive query helper."""
        # Check if path matches
        if fnmatch.fnmatch(key.path, path_pattern):
            if key.security.check_access(caller, AccessRights.KEY_QUERY_VALUE.value):
                # Add matching values
                for name, value in key.values.items():
                    if value_pattern is None or fnmatch.fnmatch(name, value_pattern):
                        results.append((key.path, name, value.data))

        # Recurse into subkeys
        for subkey in key.subkeys.values():
            self._query_recursive(subkey, path_pattern, value_pattern, results, caller)

    # =========================================================================
    # Security
    # =========================================================================

    async def set_key_security(
        self,
        path: str,
        owner: Optional[str] = None,
        group: Optional[str] = None,
        acl: Optional[Dict[str, int]] = None,
        caller: str = "system"
    ) -> bool:
        """Set security on a key."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return False

            # Only owner or system can change security
            if caller != "system" and caller != key.security.owner:
                return False

            if owner:
                key.security.owner = owner
            if group:
                key.security.group = group
            if acl:
                key.security.acl = acl

            self._notify_change(path, NotifyFilter.SECURITY, None)
            return True

    async def get_key_security(
        self,
        path: str,
        caller: str = "system"
    ) -> Optional[RegistryKeySecurity]:
        """Get security descriptor for a key."""
        async with self._lock:
            key = self._navigate_to_key(path)
            if not key:
                return None

            # Need at least read access
            if not key.security.check_access(caller, AccessRights.KEY_READ.value):
                return None

            return key.security


# =============================================================================
# Registry Helper
# =============================================================================

class RegistryHelper:
    """Helper for common registry operations."""

    def __init__(self, registry: ConfigRegistry, base_path: str):
        self._registry = registry
        self._base_path = base_path

    def _full_path(self, subpath: str = "") -> str:
        """Get full path."""
        if subpath:
            return f"{self._base_path}\\{subpath}"
        return self._base_path

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value."""
        parts = key.rsplit("\\", 1)
        if len(parts) == 2:
            path, name = self._full_path(parts[0]), parts[1]
        else:
            path, name = self._base_path, key

        return await self._registry.get_value_data(path, name, default)

    async def set(self, key: str, value: Any) -> bool:
        """Set a value."""
        parts = key.rsplit("\\", 1)
        if len(parts) == 2:
            path, name = self._full_path(parts[0]), parts[1]
        else:
            path, name = self._base_path, key

        # Determine type
        if isinstance(value, bool):
            return await self._registry.set_value(path, name, int(value), ValueType.REG_DWORD)
        elif isinstance(value, int):
            return await self._registry.set_value(path, name, value, ValueType.REG_DWORD)
        elif isinstance(value, str):
            return await self._registry.set_value(path, name, value, ValueType.REG_SZ)
        elif isinstance(value, list):
            if all(isinstance(x, str) for x in value):
                return await self._registry.set_value(path, name, value, ValueType.REG_MULTI_SZ)
            else:
                return await self._registry.set_value(path, name, value, ValueType.REG_JSON)
        elif isinstance(value, dict):
            return await self._registry.set_value(path, name, value, ValueType.REG_JSON)
        else:
            return await self._registry.set_value(path, name, str(value), ValueType.REG_SZ)

    async def delete(self, key: str) -> bool:
        """Delete a value."""
        parts = key.rsplit("\\", 1)
        if len(parts) == 2:
            path, name = self._full_path(parts[0]), parts[1]
        else:
            path, name = self._base_path, key

        return await self._registry.delete_value(path, name)

    def subkey(self, path: str) -> "RegistryHelper":
        """Get helper for subkey."""
        return RegistryHelper(self._registry, self._full_path(path))


# =============================================================================
# Singleton Access
# =============================================================================

_config_registry: Optional[ConfigRegistry] = None


def get_config_registry() -> ConfigRegistry:
    """Get or create the global config registry."""
    global _config_registry
    if _config_registry is None:
        _config_registry = ConfigRegistry()
    return _config_registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ValueType",
    "Hive",
    "AccessRights",
    "NotifyFilter",

    # Data classes
    "RegistryValue",
    "RegistryKeySecurity",
    "RegistryKey",
    "ChangeNotification",

    # Transaction
    "RegistryTransaction",

    # Main class
    "ConfigRegistry",
    "RegistryHelper",

    # Singleton
    "get_config_registry",
]
