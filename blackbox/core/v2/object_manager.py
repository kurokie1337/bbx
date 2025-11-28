# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Object Manager - Windows Object Manager-Inspired

Inspired by Windows NT Object Manager, provides:
- Unified object namespace (/agents/alice/contexts/session_1)
- All resources are objects with ACLs
- Reference counting and lifecycle management
- Handle-based access
- Object inheritance and composition

Key concepts:
- Object: Any resource (agent, context, tool, message, snapshot)
- Handle: Reference to an object with specific access rights
- Namespace: Hierarchical object organization
- Security Descriptor: ACL attached to each object
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar, Union

logger = logging.getLogger("bbx.object_manager")


# =============================================================================
# Object Types
# =============================================================================

class ObjectType(Enum):
    """Types of objects in the BBX namespace."""

    # Core objects
    DIRECTORY = "directory"      # Container for other objects
    SYMBOLIC_LINK = "symlink"    # Reference to another object

    # Agent objects
    AGENT = "agent"
    CONTEXT = "context"
    MESSAGE = "message"
    TOOL = "tool"

    # Runtime objects
    WORKFLOW = "workflow"
    STEP = "step"
    RING = "ring"
    HOOK = "hook"

    # State objects
    SNAPSHOT = "snapshot"
    STATE = "state"
    CHECKPOINT = "checkpoint"

    # Security objects
    TOKEN = "token"
    POLICY = "policy"

    # Resource objects
    QUOTA = "quota"
    SANDBOX = "sandbox"

    # Communication objects
    CHANNEL = "channel"
    PIPE = "pipe"
    EVENT = "event"
    SEMAPHORE = "semaphore"
    MUTEX = "mutex"


# =============================================================================
# Access Rights
# =============================================================================

class AccessMask:
    """
    Access rights bitmask (Windows-style).

    Bits 0-15: Object-specific rights
    Bits 16-23: Standard rights
    Bits 24-27: Reserved
    Bits 28-31: Generic rights
    """

    # Generic rights (high bits)
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    GENERIC_EXECUTE = 0x20000000
    GENERIC_ALL = 0x10000000

    # Standard rights
    DELETE = 0x00010000
    READ_CONTROL = 0x00020000
    WRITE_DAC = 0x00040000
    WRITE_OWNER = 0x00080000
    SYNCHRONIZE = 0x00100000

    # Standard combinations
    STANDARD_RIGHTS_READ = READ_CONTROL
    STANDARD_RIGHTS_WRITE = READ_CONTROL
    STANDARD_RIGHTS_EXECUTE = READ_CONTROL
    STANDARD_RIGHTS_ALL = DELETE | READ_CONTROL | WRITE_DAC | WRITE_OWNER | SYNCHRONIZE

    # Object-specific rights (low bits) - for agents
    AGENT_QUERY = 0x0001
    AGENT_MODIFY = 0x0002
    AGENT_EXECUTE = 0x0004
    AGENT_PAUSE = 0x0008
    AGENT_TERMINATE = 0x0010
    AGENT_FORK = 0x0020
    AGENT_READ_CONTEXT = 0x0040
    AGENT_WRITE_CONTEXT = 0x0080
    AGENT_USE_TOOLS = 0x0100
    AGENT_CREATE_SNAPSHOT = 0x0200

    # Full control
    AGENT_ALL_ACCESS = (
        STANDARD_RIGHTS_ALL |
        AGENT_QUERY | AGENT_MODIFY | AGENT_EXECUTE |
        AGENT_PAUSE | AGENT_TERMINATE | AGENT_FORK |
        AGENT_READ_CONTEXT | AGENT_WRITE_CONTEXT |
        AGENT_USE_TOOLS | AGENT_CREATE_SNAPSHOT
    )

    def __init__(self, mask: int = 0):
        self.mask = mask

    def has(self, right: int) -> bool:
        return (self.mask & right) == right

    def grant(self, right: int):
        self.mask |= right

    def revoke(self, right: int):
        self.mask &= ~right

    def __or__(self, other: "AccessMask") -> "AccessMask":
        return AccessMask(self.mask | other.mask)

    def __and__(self, other: "AccessMask") -> "AccessMask":
        return AccessMask(self.mask & other.mask)


# =============================================================================
# Security Descriptor
# =============================================================================

class ACEType(Enum):
    """Access Control Entry types."""
    ACCESS_ALLOWED = "allow"
    ACCESS_DENIED = "deny"
    SYSTEM_AUDIT = "audit"


@dataclass
class ACE:
    """Access Control Entry."""

    sid: str                      # Security Identifier (user:alice, group:admins)
    ace_type: ACEType
    access_mask: int
    flags: int = 0                # Inheritance flags

    # Inheritance flags
    OBJECT_INHERIT = 0x01
    CONTAINER_INHERIT = 0x02
    NO_PROPAGATE_INHERIT = 0x04
    INHERIT_ONLY = 0x08

    def matches_sid(self, subject_sid: str, group_resolver: Optional[Callable] = None) -> bool:
        """Check if this ACE applies to the subject."""
        if self.sid == subject_sid:
            return True
        if self.sid == "everyone" or self.sid == "*":
            return True

        # Check group membership
        if group_resolver:
            groups = group_resolver(subject_sid)
            if self.sid in groups:
                return True

        return False


@dataclass
class ACL:
    """Access Control List."""

    aces: List[ACE] = field(default_factory=list)
    is_protected: bool = False    # Protected from inheritance

    def add_ace(self, ace: ACE, position: int = -1):
        """Add an ACE to the list."""
        if position < 0:
            self.aces.append(ace)
        else:
            self.aces.insert(position, ace)

    def remove_ace(self, sid: str):
        """Remove ACEs for a SID."""
        self.aces = [ace for ace in self.aces if ace.sid != sid]


@dataclass
class SecurityDescriptor:
    """Security descriptor for an object."""

    owner: str                    # Owner SID
    group: str                    # Primary group SID
    dacl: Optional[ACL] = None    # Discretionary ACL (permissions)
    sacl: Optional[ACL] = None    # System ACL (audit)
    flags: int = 0

    # Flags
    SE_DACL_PRESENT = 0x0004
    SE_SACL_PRESENT = 0x0010
    SE_DACL_PROTECTED = 0x1000
    SE_SACL_PROTECTED = 0x2000

    def check_access(
        self,
        subject_sid: str,
        desired_access: int,
        group_resolver: Optional[Callable] = None
    ) -> bool:
        """Check if subject has the desired access."""
        # Owner always has full control
        if subject_sid == self.owner:
            return True

        if not self.dacl:
            return True  # No DACL = full access

        # Walk ACEs in order: DENY first, then ALLOW
        denied = False
        allowed = False

        for ace in self.dacl.aces:
            if not ace.matches_sid(subject_sid, group_resolver):
                continue

            if ace.ace_type == ACEType.ACCESS_DENIED:
                if (ace.access_mask & desired_access) != 0:
                    denied = True
                    break  # DENY wins immediately

            elif ace.ace_type == ACEType.ACCESS_ALLOWED:
                if (ace.access_mask & desired_access) == desired_access:
                    allowed = True

        return not denied and allowed

    def audit_access(
        self,
        subject_sid: str,
        access_mask: int,
        success: bool
    ) -> bool:
        """Check if access should be audited."""
        if not self.sacl:
            return False

        for ace in self.sacl.aces:
            if ace.ace_type != ACEType.SYSTEM_AUDIT:
                continue

            if not ace.matches_sid(subject_sid):
                continue

            if (ace.access_mask & access_mask) == 0:
                continue

            # Check audit flags
            # 0x01 = audit success, 0x02 = audit failure
            if success and (ace.flags & 0x01):
                return True
            if not success and (ace.flags & 0x02):
                return True

        return False


# =============================================================================
# Base Object
# =============================================================================

@dataclass
class ObjectHeader:
    """Header information for all objects."""

    name: str
    object_type: ObjectType
    security: SecurityDescriptor

    # Object metadata
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)

    # Reference counting
    ref_count: int = 0
    handle_count: int = 0

    # Parent directory
    parent: Optional["DirectoryObject"] = None

    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_path(self) -> str:
        """Get full path from root."""
        if self.parent is None:
            return "/" + self.name if self.name else "/"

        parent_path = self.parent.header.full_path
        if parent_path == "/":
            return "/" + self.name
        return parent_path + "/" + self.name


class BBXObject(ABC):
    """
    Base class for all BBX objects.

    Every resource in BBX is an object with:
    - A header containing metadata and security
    - Type-specific body/data
    - Reference counting
    - Handle tracking
    """

    def __init__(
        self,
        name: str,
        object_type: ObjectType,
        owner: str = "system",
        group: str = "system",
    ):
        self.header = ObjectHeader(
            name=name,
            object_type=object_type,
            security=SecurityDescriptor(
                owner=owner,
                group=group,
                dacl=ACL(),
            ),
        )

        self._lock = threading.RLock()
        self._handles: weakref.WeakSet = weakref.WeakSet()
        self._deleted = False

    def add_ref(self):
        """Increment reference count."""
        with self._lock:
            self.header.ref_count += 1

    def release(self) -> int:
        """Decrement reference count, return new count."""
        with self._lock:
            self.header.ref_count -= 1
            if self.header.ref_count <= 0 and self.header.handle_count <= 0:
                self._cleanup()
            return self.header.ref_count

    def _cleanup(self):
        """Clean up resources when object is deleted."""
        self._deleted = True

    @abstractmethod
    def get_body(self) -> Any:
        """Get the object body/data."""
        pass


# =============================================================================
# Directory Object
# =============================================================================

class DirectoryObject(BBXObject):
    """
    Directory object - container for other objects.

    Like a file system directory but for all BBX objects.
    """

    def __init__(self, name: str, owner: str = "system"):
        super().__init__(name, ObjectType.DIRECTORY, owner)
        self._children: Dict[str, BBXObject] = {}

    def get_body(self) -> Dict[str, BBXObject]:
        return self._children

    def add_object(self, obj: BBXObject) -> bool:
        """Add an object to this directory."""
        with self._lock:
            if obj.header.name in self._children:
                return False

            self._children[obj.header.name] = obj
            obj.header.parent = self
            obj.add_ref()
            return True

    def remove_object(self, name: str) -> Optional[BBXObject]:
        """Remove an object from this directory."""
        with self._lock:
            if name not in self._children:
                return None

            obj = self._children.pop(name)
            obj.header.parent = None
            obj.release()
            return obj

    def lookup(self, name: str) -> Optional[BBXObject]:
        """Look up an object by name."""
        with self._lock:
            return self._children.get(name)

    def list_objects(self) -> List[str]:
        """List all object names in this directory."""
        with self._lock:
            return list(self._children.keys())


# =============================================================================
# Symbolic Link Object
# =============================================================================

class SymbolicLinkObject(BBXObject):
    """Symbolic link to another object."""

    def __init__(self, name: str, target_path: str, owner: str = "system"):
        super().__init__(name, ObjectType.SYMBOLIC_LINK, owner)
        self.target_path = target_path

    def get_body(self) -> str:
        return self.target_path


# =============================================================================
# Handle
# =============================================================================

@dataclass
class Handle:
    """
    Handle to an object.

    A handle represents an open reference to an object
    with specific access rights.
    """

    id: str
    obj: BBXObject
    access_mask: int
    owner_sid: str

    # Handle state
    is_valid: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.obj.header.handle_count += 1
        self.obj._handles.add(self)

    def close(self):
        """Close this handle."""
        if self.is_valid:
            self.is_valid = False
            self.obj.header.handle_count -= 1
            self.obj.release()

    def check_access(self, required: int) -> bool:
        """Check if handle has required access."""
        return (self.access_mask & required) == required


# =============================================================================
# Object Manager
# =============================================================================

class ObjectManager:
    """
    Central Object Manager.

    Manages the object namespace, handles, and security checks.
    Like Windows NT Object Manager but for AI agents.
    """

    def __init__(self):
        # Root of the namespace
        self._root = DirectoryObject("", "system")
        self._root.header.parent = None

        # Handle table
        self._handles: Dict[str, Handle] = {}

        # Group resolver for ACL checks
        self._group_resolver: Optional[Callable[[str], List[str]]] = None

        # Object creation callbacks
        self._create_callbacks: Dict[ObjectType, List[Callable]] = {}
        self._delete_callbacks: Dict[ObjectType, List[Callable]] = {}

        # Statistics
        self._stats = {
            "objects_created": 0,
            "objects_deleted": 0,
            "handles_opened": 0,
            "handles_closed": 0,
            "access_checks": 0,
            "access_denied": 0,
        }

        # Initialize standard directories
        self._init_namespace()

    def _init_namespace(self):
        """Initialize standard namespace directories."""
        standard_dirs = [
            "/agents",       # All agents
            "/contexts",     # All contexts
            "/tools",        # Available tools
            "/workflows",    # Workflow definitions
            "/snapshots",    # State snapshots
            "/security",     # Security objects
            "/devices",      # Device objects (backends via AAL)
        ]

        for path in standard_dirs:
            self.create_directory(path, "system")

    def set_group_resolver(self, resolver: Callable[[str], List[str]]):
        """Set the function that resolves group memberships."""
        self._group_resolver = resolver

    # -------------------------------------------------------------------------
    # Path Operations
    # -------------------------------------------------------------------------

    def _parse_path(self, path: str) -> List[str]:
        """Parse a path into components."""
        path = path.strip()
        if not path.startswith("/"):
            path = "/" + path

        parts = [p for p in path.split("/") if p]
        return parts

    def _lookup_path(
        self,
        path: str,
        follow_symlinks: bool = True
    ) -> Optional[BBXObject]:
        """Look up an object by path."""
        parts = self._parse_path(path)

        current: BBXObject = self._root
        for part in parts:
            if not isinstance(current, DirectoryObject):
                return None

            obj = current.lookup(part)
            if obj is None:
                return None

            # Follow symbolic links
            if follow_symlinks and isinstance(obj, SymbolicLinkObject):
                obj = self._lookup_path(obj.target_path, follow_symlinks)
                if obj is None:
                    return None

            current = obj

        return current

    def _lookup_parent(self, path: str) -> Optional[DirectoryObject]:
        """Look up parent directory of a path."""
        parts = self._parse_path(path)
        if not parts:
            return None

        parent_parts = parts[:-1]
        if not parent_parts:
            return self._root

        current: BBXObject = self._root
        for part in parent_parts:
            if not isinstance(current, DirectoryObject):
                return None

            obj = current.lookup(part)
            if obj is None:
                return None

            current = obj

        if isinstance(current, DirectoryObject):
            return current
        return None

    # -------------------------------------------------------------------------
    # Object Creation
    # -------------------------------------------------------------------------

    def create_directory(
        self,
        path: str,
        owner: str,
        inherit_acl: bool = True
    ) -> Optional[DirectoryObject]:
        """Create a directory object."""
        parent = self._lookup_parent(path)
        if parent is None:
            # Create parent directories
            parts = self._parse_path(path)
            current = self._root
            for part in parts[:-1]:
                if not isinstance(current, DirectoryObject):
                    return None

                child = current.lookup(part)
                if child is None:
                    child = DirectoryObject(part, owner)
                    current.add_object(child)

                current = child

            parent = current

        name = self._parse_path(path)[-1] if self._parse_path(path) else ""
        if not name:
            return self._root if isinstance(self._root, DirectoryObject) else None

        # Check if already exists
        existing = parent.lookup(name)
        if existing:
            return existing if isinstance(existing, DirectoryObject) else None

        directory = DirectoryObject(name, owner)

        # Inherit ACL from parent
        if inherit_acl and parent.header.security.dacl:
            directory.header.security.dacl = ACL(
                aces=[
                    ACE(ace.sid, ace.ace_type, ace.access_mask, ace.flags)
                    for ace in parent.header.security.dacl.aces
                    if ace.flags & ACE.CONTAINER_INHERIT
                ]
            )

        parent.add_object(directory)
        self._stats["objects_created"] += 1

        return directory

    def create_object(
        self,
        path: str,
        obj: BBXObject,
        owner: str,
        desired_access: int = AccessMask.GENERIC_ALL,
    ) -> Optional[Handle]:
        """Create an object and return a handle to it."""
        parent = self._lookup_parent(path)
        if parent is None:
            logger.error(f"Parent directory not found: {path}")
            return None

        name = self._parse_path(path)[-1]
        obj.header.name = name
        obj.header.security.owner = owner

        # Check parent access
        self._stats["access_checks"] += 1
        if not parent.header.security.check_access(
            owner,
            AccessMask.GENERIC_WRITE,
            self._group_resolver
        ):
            self._stats["access_denied"] += 1
            logger.warning(f"Access denied creating object: {path}")
            return None

        # Add to parent
        if not parent.add_object(obj):
            logger.error(f"Object already exists: {path}")
            return None

        self._stats["objects_created"] += 1

        # Create handle
        handle = self._create_handle(obj, desired_access, owner)

        # Call creation callbacks
        for callback in self._create_callbacks.get(obj.header.object_type, []):
            try:
                callback(obj, handle)
            except Exception as e:
                logger.error(f"Create callback error: {e}")

        return handle

    def delete_object(self, path: str, caller_sid: str) -> bool:
        """Delete an object."""
        obj = self._lookup_path(path)
        if obj is None:
            return False

        # Check access
        self._stats["access_checks"] += 1
        if not obj.header.security.check_access(
            caller_sid,
            AccessMask.DELETE,
            self._group_resolver
        ):
            self._stats["access_denied"] += 1
            return False

        # Can't delete if handles are open
        if obj.header.handle_count > 0:
            logger.warning(f"Cannot delete object with open handles: {path}")
            return False

        # Remove from parent
        parent = obj.header.parent
        if parent:
            parent.remove_object(obj.header.name)

        # Call delete callbacks
        for callback in self._delete_callbacks.get(obj.header.object_type, []):
            try:
                callback(obj)
            except Exception as e:
                logger.error(f"Delete callback error: {e}")

        self._stats["objects_deleted"] += 1
        return True

    # -------------------------------------------------------------------------
    # Handle Operations
    # -------------------------------------------------------------------------

    def _create_handle(
        self,
        obj: BBXObject,
        access_mask: int,
        owner_sid: str
    ) -> Handle:
        """Create a handle to an object."""
        handle_id = str(uuid.uuid4())
        handle = Handle(
            id=handle_id,
            obj=obj,
            access_mask=access_mask,
            owner_sid=owner_sid,
        )
        self._handles[handle_id] = handle
        self._stats["handles_opened"] += 1
        return handle

    def open_object(
        self,
        path: str,
        caller_sid: str,
        desired_access: int
    ) -> Optional[Handle]:
        """Open a handle to an existing object."""
        obj = self._lookup_path(path)
        if obj is None:
            return None

        # Check access
        self._stats["access_checks"] += 1
        if not obj.header.security.check_access(
            caller_sid,
            desired_access,
            self._group_resolver
        ):
            self._stats["access_denied"] += 1

            # Audit failed access
            if obj.header.security.audit_access(caller_sid, desired_access, False):
                logger.warning(f"Audit: Access denied for {caller_sid} to {path}")

            return None

        # Audit successful access
        if obj.header.security.audit_access(caller_sid, desired_access, True):
            logger.info(f"Audit: Access granted for {caller_sid} to {path}")

        return self._create_handle(obj, desired_access, caller_sid)

    def close_handle(self, handle_id: str) -> bool:
        """Close a handle."""
        handle = self._handles.get(handle_id)
        if handle is None:
            return False

        handle.close()
        del self._handles[handle_id]
        self._stats["handles_closed"] += 1
        return True

    def get_handle(self, handle_id: str) -> Optional[Handle]:
        """Get a handle by ID."""
        return self._handles.get(handle_id)

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def query_object(self, path: str) -> Optional[ObjectHeader]:
        """Query object information."""
        obj = self._lookup_path(path)
        if obj:
            obj.header.accessed_at = datetime.now()
            return obj.header
        return None

    def query_security(self, path: str) -> Optional[SecurityDescriptor]:
        """Query object security descriptor."""
        obj = self._lookup_path(path)
        if obj:
            return obj.header.security
        return None

    def set_security(
        self,
        path: str,
        caller_sid: str,
        security: SecurityDescriptor
    ) -> bool:
        """Set object security descriptor."""
        obj = self._lookup_path(path)
        if obj is None:
            return False

        # Check WRITE_DAC access
        self._stats["access_checks"] += 1
        if not obj.header.security.check_access(
            caller_sid,
            AccessMask.WRITE_DAC,
            self._group_resolver
        ):
            self._stats["access_denied"] += 1
            return False

        obj.header.security = security
        obj.header.modified_at = datetime.now()
        return True

    def list_directory(
        self,
        path: str,
        caller_sid: str
    ) -> Optional[List[str]]:
        """List objects in a directory."""
        obj = self._lookup_path(path)
        if obj is None or not isinstance(obj, DirectoryObject):
            return None

        # Check read access
        self._stats["access_checks"] += 1
        if not obj.header.security.check_access(
            caller_sid,
            AccessMask.GENERIC_READ,
            self._group_resolver
        ):
            self._stats["access_denied"] += 1
            return None

        return obj.list_objects()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def register_create_callback(
        self,
        object_type: ObjectType,
        callback: Callable[[BBXObject, Handle], None]
    ):
        """Register a callback for object creation."""
        if object_type not in self._create_callbacks:
            self._create_callbacks[object_type] = []
        self._create_callbacks[object_type].append(callback)

    def register_delete_callback(
        self,
        object_type: ObjectType,
        callback: Callable[[BBXObject], None]
    ):
        """Register a callback for object deletion."""
        if object_type not in self._delete_callbacks:
            self._delete_callbacks[object_type] = []
        self._delete_callbacks[object_type].append(callback)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get Object Manager statistics."""
        return {
            **self._stats,
            "active_handles": len(self._handles),
        }


# =============================================================================
# Concrete Object Types
# =============================================================================

class AgentObject(BBXObject):
    """Agent object."""

    def __init__(
        self,
        name: str,
        owner: str,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, ObjectType.AGENT, owner)
        self.config = config or {}
        self.state: Dict[str, Any] = {}
        self.contexts: List[str] = []

    def get_body(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "state": self.state,
            "contexts": self.contexts,
        }


class ContextObject(BBXObject):
    """Context object for agent memory."""

    def __init__(
        self,
        name: str,
        owner: str,
        max_size: int = 100000
    ):
        super().__init__(name, ObjectType.CONTEXT, owner)
        self.max_size = max_size
        self.messages: List[Dict[str, Any]] = []
        self.variables: Dict[str, Any] = {}

    def get_body(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "messages": self.messages,
            "variables": self.variables,
        }


class ToolObject(BBXObject):
    """Tool object."""

    def __init__(
        self,
        name: str,
        owner: str,
        handler: Optional[Callable] = None
    ):
        super().__init__(name, ObjectType.TOOL, owner)
        self.handler = handler
        self.schema: Optional[Dict] = None
        self.permissions: List[str] = []

    def get_body(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "permissions": self.permissions,
        }


# =============================================================================
# Global Object Manager
# =============================================================================

_global_object_manager: Optional[ObjectManager] = None


def get_object_manager() -> ObjectManager:
    """Get the global Object Manager instance."""
    global _global_object_manager
    if _global_object_manager is None:
        _global_object_manager = ObjectManager()
    return _global_object_manager


# =============================================================================
# Convenience Functions
# =============================================================================

def create_agent(
    name: str,
    owner: str,
    config: Optional[Dict[str, Any]] = None
) -> Optional[Handle]:
    """Create an agent object."""
    om = get_object_manager()
    agent = AgentObject(name, owner, config)
    return om.create_object(f"/agents/{name}", agent, owner)


def open_agent(
    name: str,
    caller_sid: str,
    access: int = AccessMask.GENERIC_READ
) -> Optional[Handle]:
    """Open a handle to an agent."""
    om = get_object_manager()
    return om.open_object(f"/agents/{name}", caller_sid, access)


def create_context(
    agent_name: str,
    context_name: str,
    owner: str,
    max_size: int = 100000
) -> Optional[Handle]:
    """Create a context for an agent."""
    om = get_object_manager()

    # Ensure contexts directory exists for agent
    om.create_directory(f"/agents/{agent_name}/contexts", owner)

    context = ContextObject(context_name, owner, max_size)
    return om.create_object(
        f"/agents/{agent_name}/contexts/{context_name}",
        context,
        owner
    )
