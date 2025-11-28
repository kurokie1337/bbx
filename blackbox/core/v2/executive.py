# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Executive - Hybrid Kernel Architecture

The BBX Executive is the core kernel layer that integrates all NT-inspired
components into a unified system. Similar to Windows NT Executive (ntoskrnl.exe).

Architecture:
                    ┌──────────────────────────────────────┐
                    │          User Mode (Agents)          │
                    ├──────────────────────────────────────┤
                    │         System Call Interface        │
                    ├──────────────────────────────────────┤
    ┌───────────────┼──────────────────────────────────────┼───────────────┐
    │               │         BBX Executive                │               │
    │  ┌────────────┴────────────────────────────────────┴────────────┐   │
    │  │                                                               │   │
    │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
    │  │  │   Object    │  │   Filter    │  │    Config Registry  │   │   │
    │  │  │   Manager   │  │   Manager   │  │                     │   │   │
    │  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
    │  │                                                               │   │
    │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
    │  │  │  Working    │  │   Security  │  │     I/O Manager     │   │   │
    │  │  │  Set Mgr    │  │   Ref Mon   │  │    (AgentRing)      │   │   │
    │  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
    │  │                                                               │   │
    │  └───────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    │  ┌───────────────────────────────────────────────────────────────┐   │
    │  │           Agent Abstraction Layer (AAL / HAL)                 │   │
    │  └───────────────────────────────────────────────────────────────┘   │
    │                                                                       │
    └───────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
            [LLM Backends, Vector Stores, External Services]

Key Components:
- Object Manager: Unified namespace for all BBX objects
- Filter Manager: Extensible request processing pipeline
- Config Registry: Hierarchical configuration store
- Working Set Manager: Context memory management
- Security Reference Monitor: Unified access control
- I/O Manager: AgentRing-based operation processing
- AAL: Hardware abstraction for LLM/vector backends
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
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
import logging
import time
import uuid

# Import all NT components
from .object_manager import (
    ObjectManager,
    BBXObject,
    DirectoryObject,
    Handle,
    SecurityDescriptor,
    ACL,
    ACE,
    ACEType,
    AccessMask,
    ObjectType,
    get_object_manager,
)
from .filter_stack import (
    FilterManager,
    FilterContext,
    FilterResult,
    OperationClass,
    Filter,
    AuditFilter,
    SecurityFilter,
    QuotaFilter,
    RateLimitFilter,
    MetricsFilter,
    CacheFilter,
    FilterStackBuilder,
    get_filter_manager,
)
from .working_set import (
    WorkingSetManager,
    AgentWorkingSet,
    AllocationRequest,
    PageState,
    MemoryPressure,
    WorkingSetLimits,
    get_working_set_manager,
)
from .config_registry import (
    ConfigRegistry,
    RegistryHelper,
    RegistryKey,
    RegistryValue,
    ValueType,
    Hive,
    AccessRights,
    get_config_registry,
)
from .aal import (
    AAL,
    LLMProvider,
    VectorProvider,
    LLMRequest,
    LLMResponse,
    VectorQuery,
    VectorResult,
    BackendRouter,
    Capability,
    RoutingStrategy,
    get_aal,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ExecutiveState(Enum):
    """State of the BBX Executive."""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


class SystemCallType(Enum):
    """Types of system calls."""
    # Object operations
    OBJ_CREATE = auto()
    OBJ_OPEN = auto()
    OBJ_CLOSE = auto()
    OBJ_DELETE = auto()
    OBJ_QUERY = auto()

    # Agent operations
    AGENT_CREATE = auto()
    AGENT_INVOKE = auto()
    AGENT_TERMINATE = auto()

    # Context operations
    CTX_ALLOCATE = auto()
    CTX_READ = auto()
    CTX_WRITE = auto()
    CTX_FREE = auto()

    # Registry operations
    REG_CREATE_KEY = auto()
    REG_OPEN_KEY = auto()
    REG_DELETE_KEY = auto()
    REG_SET_VALUE = auto()
    REG_GET_VALUE = auto()
    REG_DELETE_VALUE = auto()

    # Security operations
    SEC_CHECK_ACCESS = auto()
    SEC_SET_SECURITY = auto()
    SEC_GET_TOKEN = auto()

    # I/O operations
    IO_SUBMIT = auto()
    IO_WAIT = auto()
    IO_CANCEL = auto()


class ServiceType(Enum):
    """Types of executive services."""
    OBJECT_MANAGER = auto()
    FILTER_MANAGER = auto()
    WORKING_SET = auto()
    REGISTRY = auto()
    AAL = auto()
    SECURITY = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SystemCall:
    """Represents a system call to the executive."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_type: SystemCallType = SystemCallType.OBJ_QUERY
    caller_id: str = ""
    caller_token: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SystemCallResult:
    """Result of a system call."""
    call_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class SecurityToken:
    """Security token for caller identification."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    groups: List[str] = field(default_factory=list)
    privileges: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def is_valid(self) -> bool:
        """Check if token is valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def has_privilege(self, privilege: str) -> bool:
        """Check if token has privilege."""
        return privilege in self.privileges or "SE_ALL_PRIVILEGES" in self.privileges


@dataclass
class ExecutiveStats:
    """Statistics for the executive."""
    uptime_seconds: float = 0.0
    total_syscalls: int = 0
    successful_syscalls: int = 0
    failed_syscalls: int = 0
    avg_latency_ms: float = 0.0
    active_handles: int = 0
    active_tokens: int = 0
    memory_pressure: MemoryPressure = MemoryPressure.LOW


@dataclass
class ExecutiveConfig:
    """Configuration for the executive."""
    enable_audit: bool = True
    enable_security: bool = True
    enable_quotas: bool = True
    enable_rate_limit: bool = True
    enable_cache: bool = True

    rate_limit_rps: float = 100.0
    rate_limit_burst: int = 200

    cache_size: int = 10000
    cache_ttl: float = 300.0

    working_set_minimum: int = 100
    working_set_maximum: int = 10000
    working_set_soft_limit: int = 5000


# =============================================================================
# Security Reference Monitor
# =============================================================================

class SecurityReferenceMonitor:
    """
    Security Reference Monitor (SRM).

    Unified access control decision point.
    """

    def __init__(self, registry: ConfigRegistry, object_manager: ObjectManager):
        self._registry = registry
        self._object_manager = object_manager
        self._tokens: Dict[str, SecurityToken] = {}
        self._system_token = SecurityToken(
            subject_id="system",
            groups=["SYSTEM", "ADMINISTRATORS"],
            privileges={"SE_ALL_PRIVILEGES"},
        )

    async def create_token(
        self,
        subject_id: str,
        groups: Optional[List[str]] = None,
        privileges: Optional[Set[str]] = None,
        ttl_seconds: Optional[float] = None
    ) -> SecurityToken:
        """Create a security token."""
        token = SecurityToken(
            subject_id=subject_id,
            groups=groups or [],
            privileges=privileges or set(),
        )

        if ttl_seconds:
            from datetime import timedelta
            token.expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        self._tokens[token.id] = token
        return token

    async def validate_token(self, token_id: str) -> Optional[SecurityToken]:
        """Validate and retrieve token."""
        token = self._tokens.get(token_id)
        if token and token.is_valid():
            return token
        return None

    async def revoke_token(self, token_id: str) -> bool:
        """Revoke a token."""
        if token_id in self._tokens:
            del self._tokens[token_id]
            return True
        return False

    async def check_access(
        self,
        token: SecurityToken,
        resource_path: str,
        desired_access: int
    ) -> bool:
        """Check access to a resource."""
        # System has full access
        if token.subject_id == "system":
            return True

        # Check object manager ACL
        handle = self._object_manager.open_object(
            resource_path,
            token.subject_id,
            desired_access
        )

        if handle:
            self._object_manager.close_handle(handle.id)
            return True

        return False

    async def check_privilege(
        self,
        token: SecurityToken,
        privilege: str
    ) -> bool:
        """Check if token has privilege."""
        return token.has_privilege(privilege)

    def get_system_token(self) -> SecurityToken:
        """Get the system token."""
        return self._system_token


# =============================================================================
# I/O Manager
# =============================================================================

class IOManager:
    """
    I/O Manager for operation processing.

    Integrates with AgentRing for async I/O.
    """

    def __init__(
        self,
        filter_manager: FilterManager,
        aal: AAL
    ):
        self._filter_manager = filter_manager
        self._aal = aal
        self._pending_ops: Dict[str, asyncio.Future] = {}

    async def submit(
        self,
        ctx: FilterContext,
        processor: Callable[[FilterContext], Awaitable[Any]]
    ) -> FilterContext:
        """Submit I/O operation through filter stack."""
        return await self._filter_manager.process(ctx, processor)

    async def invoke_llm(
        self,
        request: LLMRequest,
        caller_id: str
    ) -> LLMResponse:
        """Invoke LLM through AAL."""
        if not self._aal.llm:
            raise RuntimeError("No LLM provider configured")

        ctx = FilterContext(
            operation=OperationClass.AGENT_INVOKE,
            operation_name="llm_complete",
            caller_id=caller_id,
            data={"request": request},
        )

        async def process(c: FilterContext) -> LLMResponse:
            return await self._aal.llm.complete(request)

        result = await self.submit(ctx, process)
        return result.result

    async def query_vectors(
        self,
        query: VectorQuery,
        caller_id: str
    ) -> VectorResult:
        """Query vectors through AAL."""
        if not self._aal.vector:
            raise RuntimeError("No vector provider configured")

        ctx = FilterContext(
            operation=OperationClass.CONTEXT_READ,
            operation_name="vector_query",
            caller_id=caller_id,
            data={"query": query},
        )

        async def process(c: FilterContext) -> VectorResult:
            return await self._aal.vector.query(query)

        result = await self.submit(ctx, process)
        return result.result


# =============================================================================
# BBX Executive
# =============================================================================

class BBXExecutive:
    """
    BBX Executive - The Hybrid Kernel.

    Central integration point for all NT-inspired components.
    """

    def __init__(self, config: Optional[ExecutiveConfig] = None):
        self._config = config or ExecutiveConfig()
        self._state = ExecutiveState.STOPPED
        self._start_time: Optional[datetime] = None

        # Core services
        self._object_manager: Optional[ObjectManager] = None
        self._filter_manager: Optional[FilterManager] = None
        self._working_set: Optional[WorkingSetManager] = None
        self._registry: Optional[ConfigRegistry] = None
        self._aal: Optional[AAL] = None

        # Higher-level services
        self._srm: Optional[SecurityReferenceMonitor] = None
        self._io_manager: Optional[IOManager] = None

        # Statistics
        self._stats = ExecutiveStats()
        self._syscall_history: List[Tuple[SystemCall, SystemCallResult]] = []
        self._max_history = 1000

        # Lock for thread safety
        self._lock = asyncio.Lock()

    @property
    def state(self) -> ExecutiveState:
        """Get executive state."""
        return self._state

    @property
    def object_manager(self) -> ObjectManager:
        """Get object manager."""
        if not self._object_manager:
            raise RuntimeError("Executive not started")
        return self._object_manager

    @property
    def filter_manager(self) -> FilterManager:
        """Get filter manager."""
        if not self._filter_manager:
            raise RuntimeError("Executive not started")
        return self._filter_manager

    @property
    def working_set(self) -> WorkingSetManager:
        """Get working set manager."""
        if not self._working_set:
            raise RuntimeError("Executive not started")
        return self._working_set

    @property
    def registry(self) -> ConfigRegistry:
        """Get config registry."""
        if not self._registry:
            raise RuntimeError("Executive not started")
        return self._registry

    @property
    def aal(self) -> AAL:
        """Get AAL."""
        if not self._aal:
            raise RuntimeError("Executive not started")
        return self._aal

    @property
    def security(self) -> SecurityReferenceMonitor:
        """Get security reference monitor."""
        if not self._srm:
            raise RuntimeError("Executive not started")
        return self._srm

    @property
    def io(self) -> IOManager:
        """Get I/O manager."""
        if not self._io_manager:
            raise RuntimeError("Executive not started")
        return self._io_manager

    async def start(self) -> None:
        """Start the BBX Executive."""
        async with self._lock:
            if self._state == ExecutiveState.RUNNING:
                return

            self._state = ExecutiveState.STARTING
            logger.info("Starting BBX Executive...")

            try:
                # Initialize core services
                await self._init_services()

                # Set up filter stack
                await self._setup_filters()

                # Initialize registry defaults
                await self._init_registry()

                # Start background tasks
                await self._start_background_tasks()

                self._state = ExecutiveState.RUNNING
                self._start_time = datetime.utcnow()
                logger.info("BBX Executive started successfully")

            except Exception as e:
                self._state = ExecutiveState.ERROR
                logger.error(f"Failed to start BBX Executive: {e}")
                raise

    async def stop(self) -> None:
        """Stop the BBX Executive."""
        async with self._lock:
            if self._state != ExecutiveState.RUNNING:
                return

            self._state = ExecutiveState.STOPPING
            logger.info("Stopping BBX Executive...")

            try:
                # Stop background tasks
                if self._working_set:
                    await self._working_set.stop()

                if self._filter_manager:
                    await self._filter_manager.stop()

                self._state = ExecutiveState.STOPPED
                logger.info("BBX Executive stopped")

            except Exception as e:
                self._state = ExecutiveState.ERROR
                logger.error(f"Error stopping BBX Executive: {e}")

    async def _init_services(self) -> None:
        """Initialize core services."""
        # Object Manager
        self._object_manager = ObjectManager()

        # Filter Manager
        self._filter_manager = FilterManager()
        await self._filter_manager.start()

        # Working Set Manager
        self._working_set = WorkingSetManager(
            limits=WorkingSetLimits(
                minimum=self._config.working_set_minimum,
                maximum=self._config.working_set_maximum,
                soft_limit=self._config.working_set_soft_limit,
            )
        )

        # Config Registry
        self._registry = ConfigRegistry()

        # AAL
        self._aal = AAL()
        await self._aal.start()

        # Security Reference Monitor
        self._srm = SecurityReferenceMonitor(self._registry, self._object_manager)

        # I/O Manager
        self._io_manager = IOManager(self._filter_manager, self._aal)

    async def _setup_filters(self) -> None:
        """Set up the filter stack."""
        builder = FilterStackBuilder(self._filter_manager)

        if self._config.enable_audit:
            builder.add_audit()
            builder.add_metrics()

        if self._config.enable_security:
            builder.add_security(self._security_check)

        if self._config.enable_quotas:
            builder.add_quota(self._quota_check)

        if self._config.enable_rate_limit:
            builder.add_rate_limit(
                rate_per_second=self._config.rate_limit_rps,
                burst_size=self._config.rate_limit_burst
            )

        if self._config.enable_cache:
            builder.add_cache(
                max_size=self._config.cache_size,
                ttl_seconds=self._config.cache_ttl
            )

        await builder.build()

    async def _security_check(
        self,
        caller_id: str,
        target_id: str,
        operation: OperationClass
    ) -> bool:
        """Security check callback for filter."""
        # For now, allow all - integrate with SRM
        return True

    async def _quota_check(
        self,
        caller_id: str,
        operation: OperationClass
    ) -> Tuple[bool, str]:
        """Quota check callback for filter."""
        # For now, allow all
        return True, ""

    async def _init_registry(self) -> None:
        """Initialize registry with executive defaults."""
        # Executive configuration
        await self._registry.set_value(
            f"{Hive.HKBX_SYSTEM.value}\\Executive",
            "version",
            "2.0.0",
            ValueType.REG_SZ
        )
        await self._registry.set_value(
            f"{Hive.HKBX_SYSTEM.value}\\Executive",
            "enable_audit",
            int(self._config.enable_audit),
            ValueType.REG_DWORD
        )
        await self._registry.set_value(
            f"{Hive.HKBX_SYSTEM.value}\\Executive",
            "enable_security",
            int(self._config.enable_security),
            ValueType.REG_DWORD
        )

    async def _start_background_tasks(self) -> None:
        """Start background tasks."""
        await self._working_set.start()

    # =========================================================================
    # System Call Interface
    # =========================================================================

    async def syscall(self, call: SystemCall) -> SystemCallResult:
        """
        Process a system call.

        Main entry point for all executive operations.
        """
        start_time = time.time()
        result = SystemCallResult(call_id=call.id, success=False)

        try:
            # Validate token if provided
            token = None
            if call.caller_token:
                token = await self._srm.validate_token(call.caller_token)
                if not token:
                    result.error = "Invalid or expired token"
                    return result
            else:
                # Use caller_id directly
                token = SecurityToken(subject_id=call.caller_id)

            # Route to appropriate handler
            handler = self._get_syscall_handler(call.call_type)
            if not handler:
                result.error = f"Unknown syscall type: {call.call_type}"
                return result

            result.result = await handler(call, token)
            result.success = True

        except Exception as e:
            result.error = str(e)
            logger.error(f"Syscall {call.call_type.name} failed: {e}")

        finally:
            result.latency_ms = (time.time() - start_time) * 1000
            self._record_syscall(call, result)

        return result

    def _get_syscall_handler(
        self,
        call_type: SystemCallType
    ) -> Optional[Callable[[SystemCall, SecurityToken], Awaitable[Any]]]:
        """Get handler for syscall type."""
        handlers = {
            # Object operations
            SystemCallType.OBJ_CREATE: self._handle_obj_create,
            SystemCallType.OBJ_OPEN: self._handle_obj_open,
            SystemCallType.OBJ_CLOSE: self._handle_obj_close,
            SystemCallType.OBJ_DELETE: self._handle_obj_delete,
            SystemCallType.OBJ_QUERY: self._handle_obj_query,

            # Context operations
            SystemCallType.CTX_ALLOCATE: self._handle_ctx_allocate,
            SystemCallType.CTX_READ: self._handle_ctx_read,
            SystemCallType.CTX_WRITE: self._handle_ctx_write,
            SystemCallType.CTX_FREE: self._handle_ctx_free,

            # Registry operations
            SystemCallType.REG_CREATE_KEY: self._handle_reg_create_key,
            SystemCallType.REG_OPEN_KEY: self._handle_reg_open_key,
            SystemCallType.REG_DELETE_KEY: self._handle_reg_delete_key,
            SystemCallType.REG_SET_VALUE: self._handle_reg_set_value,
            SystemCallType.REG_GET_VALUE: self._handle_reg_get_value,
            SystemCallType.REG_DELETE_VALUE: self._handle_reg_delete_value,

            # Security operations
            SystemCallType.SEC_CHECK_ACCESS: self._handle_sec_check_access,
            SystemCallType.SEC_SET_SECURITY: self._handle_sec_set_security,
            SystemCallType.SEC_GET_TOKEN: self._handle_sec_get_token,
        }
        return handlers.get(call_type)

    def _record_syscall(self, call: SystemCall, result: SystemCallResult) -> None:
        """Record syscall for statistics."""
        self._stats.total_syscalls += 1
        if result.success:
            self._stats.successful_syscalls += 1
        else:
            self._stats.failed_syscalls += 1

        # Update average latency
        total = self._stats.total_syscalls
        self._stats.avg_latency_ms = (
            (self._stats.avg_latency_ms * (total - 1) + result.latency_ms) / total
        )

        # Store history
        self._syscall_history.append((call, result))
        if len(self._syscall_history) > self._max_history:
            self._syscall_history = self._syscall_history[-self._max_history // 2:]

    # =========================================================================
    # Object Operation Handlers
    # =========================================================================

    async def _handle_obj_create(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Optional[str]:
        """Handle object creation."""
        path = call.params.get("path")
        obj_type = call.params.get("type", ObjectType.DIRECTORY)
        obj_data = call.params.get("data")

        if obj_type == ObjectType.DIRECTORY:
            obj = DirectoryObject(path.split("/")[-1], token.subject_id)
        else:
            obj = BBXObject(
                name=path.split("/")[-1],
                object_type=obj_type,
                owner=token.subject_id,
                data=obj_data
            )

        handle = self._object_manager.create_object(
            path,
            obj,
            token.subject_id,
            AccessMask.GENERIC_ALL
        )

        return handle.id if handle else None

    async def _handle_obj_open(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Optional[str]:
        """Handle object open."""
        path = call.params.get("path")
        access = call.params.get("access", AccessMask.GENERIC_READ)

        handle = self._object_manager.open_object(
            path,
            token.subject_id,
            access
        )

        return handle.id if handle else None

    async def _handle_obj_close(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle object close."""
        handle_id = call.params.get("handle_id")
        return self._object_manager.close_handle(handle_id)

    async def _handle_obj_delete(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle object deletion."""
        path = call.params.get("path")
        return self._object_manager.delete_object(path, token.subject_id)

    async def _handle_obj_query(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Optional[Dict[str, Any]]:
        """Handle object query."""
        path = call.params.get("path")

        handle = self._object_manager.open_object(
            path,
            token.subject_id,
            AccessMask.GENERIC_READ
        )

        if not handle:
            return None

        obj = handle.object
        result = {
            "name": obj.name,
            "type": obj.object_type.name,
            "owner": obj.owner,
            "created_at": obj.created_at.isoformat(),
        }

        self._object_manager.close_handle(handle.id)
        return result

    # =========================================================================
    # Context Operation Handlers
    # =========================================================================

    async def _handle_ctx_allocate(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Optional[str]:
        """Handle context allocation."""
        key = call.params.get("key")
        priority = call.params.get("priority", 5)

        result = await self._working_set.allocate(AllocationRequest(
            owner_id=token.subject_id,
            key=key,
            priority=priority,
        ))

        return result.page_id if result.success else None

    async def _handle_ctx_read(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Any:
        """Handle context read."""
        page_id = call.params.get("page_id")
        return await self._working_set.read(page_id)

    async def _handle_ctx_write(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle context write."""
        page_id = call.params.get("page_id")
        data = call.params.get("data")
        return await self._working_set.write(page_id, data)

    async def _handle_ctx_free(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle context free."""
        page_id = call.params.get("page_id")
        return await self._working_set.free(page_id)

    # =========================================================================
    # Registry Operation Handlers
    # =========================================================================

    async def _handle_reg_create_key(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Optional[str]:
        """Handle registry key creation."""
        path = call.params.get("path")
        volatile = call.params.get("volatile", False)

        key = await self._registry.create_key(path, volatile, token.subject_id)
        return key.path if key else None

    async def _handle_reg_open_key(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Optional[str]:
        """Handle registry key open."""
        path = call.params.get("path")
        access = call.params.get("access", AccessRights.KEY_READ.value)

        key = await self._registry.open_key(path, token.subject_id, access)
        return key.path if key else None

    async def _handle_reg_delete_key(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle registry key deletion."""
        path = call.params.get("path")
        recursive = call.params.get("recursive", False)
        return await self._registry.delete_key(path, recursive, token.subject_id)

    async def _handle_reg_set_value(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle registry value set."""
        path = call.params.get("path")
        name = call.params.get("name")
        data = call.params.get("data")
        value_type = call.params.get("type", ValueType.REG_SZ)

        return await self._registry.set_value(
            path, name, data, value_type, token.subject_id
        )

    async def _handle_reg_get_value(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Any:
        """Handle registry value get."""
        path = call.params.get("path")
        name = call.params.get("name")
        default = call.params.get("default")

        return await self._registry.get_value_data(
            path, name, default, token.subject_id
        )

    async def _handle_reg_delete_value(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle registry value deletion."""
        path = call.params.get("path")
        name = call.params.get("name")
        return await self._registry.delete_value(path, name, token.subject_id)

    # =========================================================================
    # Security Operation Handlers
    # =========================================================================

    async def _handle_sec_check_access(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle access check."""
        resource_path = call.params.get("resource")
        desired_access = call.params.get("access", AccessMask.GENERIC_READ)

        return await self._srm.check_access(token, resource_path, desired_access)

    async def _handle_sec_set_security(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> bool:
        """Handle set security."""
        path = call.params.get("path")
        owner = call.params.get("owner")
        acl = call.params.get("acl")

        return await self._registry.set_key_security(
            path, owner=owner, acl=acl, caller=token.subject_id
        )

    async def _handle_sec_get_token(
        self,
        call: SystemCall,
        token: SecurityToken
    ) -> Optional[str]:
        """Handle token creation."""
        subject_id = call.params.get("subject_id", token.subject_id)
        groups = call.params.get("groups", [])
        privileges = call.params.get("privileges", set())
        ttl = call.params.get("ttl_seconds")

        new_token = await self._srm.create_token(
            subject_id, groups, privileges, ttl
        )
        return new_token.id

    # =========================================================================
    # High-Level APIs
    # =========================================================================

    def get_agent_working_set(self, agent_id: str) -> AgentWorkingSet:
        """Get working set for an agent."""
        return AgentWorkingSet(agent_id, self._working_set)

    def get_registry_helper(self, base_path: str) -> RegistryHelper:
        """Get registry helper for a base path."""
        return RegistryHelper(self._registry, base_path)

    async def invoke_with_filter(
        self,
        operation: OperationClass,
        caller_id: str,
        target_id: str,
        data: Dict[str, Any],
        processor: Callable[[FilterContext], Awaitable[Any]]
    ) -> FilterContext:
        """Invoke operation through filter stack."""
        ctx = FilterContext(
            operation=operation,
            caller_id=caller_id,
            target_id=target_id,
            data=data,
        )
        return await self._io_manager.submit(ctx, processor)

    def get_stats(self) -> ExecutiveStats:
        """Get executive statistics."""
        if self._start_time:
            self._stats.uptime_seconds = (
                datetime.utcnow() - self._start_time
            ).total_seconds()

        if self._working_set:
            ws_stats = self._working_set.get_stats()
            self._stats.memory_pressure = ws_stats.current_pressure

        if self._object_manager:
            self._stats.active_handles = len(self._object_manager._handles)

        if self._srm:
            self._stats.active_tokens = len(self._srm._tokens)

        return self._stats


# =============================================================================
# Singleton Access
# =============================================================================

_executive: Optional[BBXExecutive] = None


def get_executive() -> BBXExecutive:
    """Get or create the global BBX Executive."""
    global _executive
    if _executive is None:
        _executive = BBXExecutive()
    return _executive


async def start_executive(config: Optional[ExecutiveConfig] = None) -> BBXExecutive:
    """Start the global BBX Executive."""
    global _executive
    if _executive is None:
        _executive = BBXExecutive(config)
    await _executive.start()
    return _executive


async def stop_executive() -> None:
    """Stop the global BBX Executive."""
    global _executive
    if _executive:
        await _executive.stop()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ExecutiveState",
    "SystemCallType",
    "ServiceType",

    # Data classes
    "SystemCall",
    "SystemCallResult",
    "SecurityToken",
    "ExecutiveStats",
    "ExecutiveConfig",

    # Components
    "SecurityReferenceMonitor",
    "IOManager",

    # Main class
    "BBXExecutive",

    # Singleton
    "get_executive",
    "start_executive",
    "stop_executive",
]
