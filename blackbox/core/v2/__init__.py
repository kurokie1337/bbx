# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 Core Module - Operating System for AI Agents

This module contains the complete BBX 2.0 architecture, synthesizing
innovations from Linux and Windows NT:

KERNEL LEVEL - Linux Inspired:
- AgentRing: io_uring-inspired batch operation system
- Hooks: eBPF-inspired dynamic workflow programming
- ContextTiering: MGLRU-inspired multi-generation context memory
- FlowIntegrity: CET-inspired control flow protection
- AgentQuotas: Cgroups v2-inspired resource limits
- StateSnapshots: XFS Reflink-inspired CoW state management

NT KERNEL LEVEL - Windows Inspired:
- BBXExecutive: Hybrid kernel architecture (ntoskrnl)
- ObjectManager: Unified namespace with ACLs (ObMgr)
- FilterStack: Extensible processing pipeline (Filter Drivers)
- WorkingSet: Memory compression and paging (Mm)
- ConfigRegistry: Hierarchical configuration store (Registry)
- AAL: Agent Abstraction Layer for backends (HAL)

DISTRIBUTION LEVEL (Ecosystem):
- Declarative: NixOS-inspired infrastructure as code
- Flakes: Nix Flakes-inspired reproducible environments
- AgentRegistry: AUR-inspired community marketplace
- AgentBundles: Kali-inspired specialized toolkits
- AgentSandbox: Flatpak-inspired isolation
- NetworkFabric: Istio-inspired service mesh for agents
- PolicyEngine: OPA/SELinux-inspired policy system

These components are designed to work together to provide
a powerful, efficient, and secure OS for AI agents.
"""

# =============================================================================
# Core Kernel Components
# =============================================================================

from .ring import (
    AgentRing,
    Operation,
    Completion,
    OperationType,
    OperationPriority,
    OperationStatus,
    RingConfig,
    RingStats,
    get_ring,
    submit_batch,
)

from .hooks import (
    HookManager,
    HookDefinition,
    HookContext,
    HookResult,
    HookFilter,
    HookType,
    AttachPoint,
    HookAction,
    HookVerifier,
    get_hook_manager,
    hook,
)

from .context_tiering import (
    ContextTiering,
    ContextItem,
    TieringConfig,
    TieringStats,
    GenerationTier,
    Generation,
    RefaultTracker,
    get_context_tiering,
)

from .flow_integrity import (
    FlowIntegrityEngine,
    FlowState,
    TransitionGraph,
    ShadowTrace,
    Violation,
    ViolationType,
    FlowConfig,
    get_flow_integrity,
)

from .agent_quotas import (
    QuotaManager,
    QuotaGroup,
    ResourceLimit,
    ResourceUsage,
    ResourceType,
    ThrottlePolicy,
    QuotaConfig as ResourceQuotaConfig,
    get_quota_manager,
)

from .state_snapshots import (
    StateSnapshotEngine,
    Snapshot,
    Branch,
    Transaction,
    SnapshotType,
    CoWStore,
    BranchManager,
    TransactionManager,
    get_snapshot_engine,
)

# =============================================================================
# Distribution Level Components
# =============================================================================

from .declarative import (
    BBXConfig,
    AgentConfig,
    QuotaConfig,
    StateConfig,
    SecretConfig,
    AdapterConfig,
    HookConfig,
    Generation as ConfigGeneration,
    GenerationManager,
    DeclarativeManager,
    BBXFlake,
    FlakeInput as DeclFlakeInput,
    FlakeLock as DeclFlakeLock,
    create_example_config,
)

from .flakes import (
    FlakeManager,
    Flake,
    FlakeRef,
    FlakeRefType,
    FlakeLock,
    FlakeOutput,
    FlakeOutputType,
    FlakeInput,
    LockedInput,
    ContentStore,
    FlakeTemplates,
    init_flake,
    build_flake,
    update_flake_lock,
)

from .agent_registry import (
    AgentRegistry,
    Package,
    PackageVersion,
    PackageType,
    PackageStatus,
    PackageSource,
    PKGBUILD,
    DependencyResolver,
    PackageBuilder,
    SearchQuery,
    SearchResult,
    SearchField,
    SortBy,
    RegistryHelper,
    search_registry,
    install_package,
    get_package_info,
)

from .agent_bundles import (
    BundleManager,
    Bundle,
    BundleType,
    BundleCategory,
    Tool,
    ToolStatus,
    ToolDependency,
    Profile,
    get_bundle_manager,
    list_available_bundles,
    list_available_tools,
    execute_tool,
)

from .agent_sandbox import (
    SandboxManager,
    Sandbox,
    SandboxConfig,
    SandboxState,
    Permission,
    PermissionGrant,
    PermissionRequest,
    FilesystemNamespace,
    FilesystemMount,
    FilesystemMode,
    NetworkPolicy,
    NetworkRule,
    NetworkAction,
    Portal,
    SandboxTemplates,
    create_sandbox,
    run_in_sandbox,
)

from .network_fabric import (
    NetworkMesh,
    Service,
    ServiceEndpoint,
    ServiceStatus,
    ServiceRegistry,
    VirtualService,
    DestinationRule,
    TrafficRoute,
    TrafficMatch,
    TrafficRetry,
    TrafficTimeout,
    LoadBalancer,
    LoadBalanceAlgorithm,
    CircuitBreaker,
    CircuitState,
    Sidecar,
    get_mesh,
    send_to_agent,
    register_agent_service,
)

from .policy_engine import (
    PolicyEngine,
    Policy,
    PolicySet,
    Rule,
    Condition,
    AttributeCondition,
    CompositeCondition,
    TimeCondition,
    ExpressionCondition,
    Subject,
    Resource,
    Action,
    Context,
    PolicyRequest,
    PolicyResponse,
    Decision,
    Effect,
    CombiningAlgorithm,
    RBACEngine,
    Role,
    Permission as RBACPermission,
    AuditLogger,
    AuditEntry,
    get_policy_engine,
    check_permission,
    create_policy,
)

# =============================================================================
# NT Kernel Components (Windows-inspired)
# =============================================================================

from .executive import (
    BBXExecutive,
    ExecutiveConfig,
    ExecutiveState,
    ExecutiveStats,
    SystemCall,
    SystemCallType,
    SystemCallResult,
    SecurityToken,
    ServiceType,
    SecurityReferenceMonitor,
    IOManager,
    get_executive,
    start_executive,
    stop_executive,
)

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
    FilterInstance,
    FilterStats,
    FilterRegistration,
    FilterResult,
    FilterStatus,
    OperationClass,
    FilterAltitude,
    Filter,
    AuditFilter,
    SecurityFilter,
    QuotaFilter,
    RateLimitFilter,
    TransformFilter,
    CacheFilter,
    ValidationFilter,
    MetricsFilter,
    FilterStackBuilder,
    get_filter_manager,
)

from .working_set import (
    WorkingSetManager,
    AgentWorkingSet,
    PageMetadata,
    WorkingSetLimits,
    CompressionStats,
    WorkingSetStats,
    AllocationRequest,
    AllocationResult,
    PageState,
    MemoryPressure,
    FaultType,
    TrimPriority,
    CompressedStore,
    PageFile,
    InMemoryPageFile,
    FileSystemPageFile,
    get_working_set_manager,
)

from .config_registry import (
    ConfigRegistry,
    RegistryKey,
    RegistryValue,
    RegistryKeySecurity,
    RegistryTransaction,
    RegistryHelper as ConfigRegistryHelper,
    ValueType,
    Hive,
    AccessRights as RegistryAccessRights,
    NotifyFilter,
    ChangeNotification,
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
    VectorDocument,
    Backend,
    BackendConfig,
    BackendHealth,
    BackendMetrics,
    BackendRouter,
    RoutingRule,
    Capability,
    RoutingStrategy,
    get_aal,
)

# =============================================================================
# Enhanced Components (Production-ready)
# =============================================================================

from .ring_enhanced import (
    EnhancedAgentRing,
    EnhancedOperation,
    EnhancedCompletion,
    EnhancedRingConfig,
    EnhancedRingStats,
    WALManager,
    WALEntry,
    IdempotencyManager,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState as RingCircuitState,
    SharedMemoryRingBuffer,
    get_enhanced_ring,
    create_enhanced_ring,
)

from .context_tiering_enhanced import (
    EnhancedContextTiering,
    EnhancedTieringConfig,
    EnhancedTieringStats,
    ImportanceScorer,
    PrefetchManager,
    AsyncMigrationEngine,
    CompressionManager,
    MemoryBackend,
    DiskBackend,
    ContentType,
    get_enhanced_tiering,
    create_enhanced_tiering,
)

from .quotas_enforced import (
    EnforcedQuotaManager,
    EnforcedQuotaConfig,
    TokenBucket,
    CgroupsManager,
    GPUQuotaManager,
    EnforcementAction,
    ThrottleLevel,
    QuotaExceededError,
    enforce_quota,
    get_enforced_quota_manager,
    create_enforced_quota_manager,
)

from .snapshots_distributed import (
    DistributedSnapshotManager,
    DistributedSnapshotConfig,
    DistributedSnapshotStats,
    LocalDiskBackend,
    S3Backend,
    RedisBackend,
    ReplicationManager,
    AsyncSnapshotWriter,
    PointInTimeRecovery,
    StorageBackendType,
    ReplicationStatus,
    get_distributed_snapshot_manager,
    create_distributed_snapshot_manager,
)

from .flow_integrity_enhanced import (
    EnhancedFlowIntegrity,
    EnhancedFlowConfig,
    FlowVerificationResult,
    AnomalyDetector,
    AnomalyEvent,
    AnomalyType,
    PolicyEngine as FlowPolicyEngine,
    PolicyRule,
    PolicyEvaluation,
    PolicyDecision,
    MemoryAccessControl,
    MemoryAccessRule,
    ToolCallValidator,
    ToolValidationRule,
    VerificationResult,
    get_enhanced_flow_integrity,
    create_enhanced_flow_integrity,
)

from .semantic_memory import (
    SemanticMemory,
    SemanticMemoryConfig,
    MemoryEntry,
    MemoryType,
    ContentModality,
    SearchResult,
    EmbeddingService,
    OpenAIEmbedding,
    LocalEmbedding,
    VectorStore,
    QdrantStore,
    InMemoryStore,
    ForgettingManager,
    get_semantic_memory,
    create_semantic_memory,
)

from .message_bus import (
    MessageBus,
    MessageBusConfig,
    Message,
    ConsumerGroup,
    DeliveryGuarantee,
    InMemoryBackend as MessageBusInMemoryBackend,
    RedisStreamsBackend,
    get_message_bus,
    create_message_bus,
)

from .goal_engine import (
    GoalEngine,
    GoalEngineConfig,
    Goal,
    Milestone,
    Task,
    GoalStatus,
    TaskPriority,
    LLMPlanner,
    SimplePlanner,
    DAGExecutor,
    get_goal_engine,
    create_goal_engine,
)

from .auth import (
    AuthManager,
    AuthConfig,
    Identity,
    JWTManager,
    JWTClaims,
    APIKeyManager,
    OIDCProvider,
    GenericOIDCProvider,
    AuthorizationEngine,
    AuthorizationRule,
    AuthMethod,
    Permission as AuthPermission,
    get_auth_manager,
    create_auth_manager,
)

from .monitoring import (
    MonitoringManager,
    MonitoringConfig,
    MetricsRegistry,
    MetricValue,
    MetricType,
    Tracer,
    Span,
    StructuredLogger,
    AlertManager,
    AlertRule,
    Alert,
    DashboardManager,
    Dashboard,
    DashboardPanel,
    get_monitoring,
    create_monitoring,
)

from .deployment import (
    DeploymentManager,
    DeploymentConfig,
    DeploymentTarget,
    AgentDeploymentSpec,
    ResourceRequirements,
    HealthCheck,
    AutoscalingConfig,
    DockerfileGenerator,
    HelmChartGenerator,
    KubernetesOperator,
    get_deployment_manager,
    create_deployment_manager,
)

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Ring
    "AgentRing",
    "Operation",
    "Completion",
    "OperationType",
    "OperationPriority",
    "OperationStatus",
    "RingConfig",
    "RingStats",
    "get_ring",
    "submit_batch",

    # Hooks
    "HookManager",
    "HookDefinition",
    "HookContext",
    "HookResult",
    "HookFilter",
    "HookType",
    "AttachPoint",
    "HookAction",
    "HookVerifier",
    "get_hook_manager",
    "hook",

    # Context Tiering
    "ContextTiering",
    "ContextItem",
    "TieringConfig",
    "TieringStats",
    "GenerationTier",
    "Generation",
    "RefaultTracker",
    "get_context_tiering",

    # Flow Integrity
    "FlowIntegrityEngine",
    "FlowState",
    "TransitionGraph",
    "ShadowTrace",
    "Violation",
    "ViolationType",
    "FlowConfig",
    "get_flow_integrity",

    # Agent Quotas
    "QuotaManager",
    "QuotaGroup",
    "ResourceLimit",
    "ResourceUsage",
    "ResourceType",
    "ThrottlePolicy",
    "ResourceQuotaConfig",
    "get_quota_manager",

    # State Snapshots
    "StateSnapshotEngine",
    "Snapshot",
    "Branch",
    "Transaction",
    "SnapshotType",
    "CoWStore",
    "BranchManager",
    "TransactionManager",
    "get_snapshot_engine",

    # Declarative
    "BBXConfig",
    "AgentConfig",
    "QuotaConfig",
    "StateConfig",
    "SecretConfig",
    "AdapterConfig",
    "HookConfig",
    "ConfigGeneration",
    "GenerationManager",
    "DeclarativeManager",
    "BBXFlake",
    "DeclFlakeInput",
    "DeclFlakeLock",
    "create_example_config",

    # Flakes
    "FlakeManager",
    "Flake",
    "FlakeRef",
    "FlakeRefType",
    "FlakeLock",
    "FlakeOutput",
    "FlakeOutputType",
    "FlakeInput",
    "LockedInput",
    "ContentStore",
    "FlakeTemplates",
    "init_flake",
    "build_flake",
    "update_flake_lock",

    # Agent Registry
    "AgentRegistry",
    "Package",
    "PackageVersion",
    "PackageType",
    "PackageStatus",
    "PackageSource",
    "PKGBUILD",
    "DependencyResolver",
    "PackageBuilder",
    "SearchQuery",
    "SearchResult",
    "SearchField",
    "SortBy",
    "RegistryHelper",
    "search_registry",
    "install_package",
    "get_package_info",

    # Agent Bundles
    "BundleManager",
    "Bundle",
    "BundleType",
    "BundleCategory",
    "Tool",
    "ToolStatus",
    "ToolDependency",
    "Profile",
    "get_bundle_manager",
    "list_available_bundles",
    "list_available_tools",
    "execute_tool",

    # Agent Sandbox
    "SandboxManager",
    "Sandbox",
    "SandboxConfig",
    "SandboxState",
    "Permission",
    "PermissionGrant",
    "PermissionRequest",
    "FilesystemNamespace",
    "FilesystemMount",
    "FilesystemMode",
    "NetworkPolicy",
    "NetworkRule",
    "NetworkAction",
    "Portal",
    "SandboxTemplates",
    "create_sandbox",
    "run_in_sandbox",

    # Network Fabric
    "NetworkMesh",
    "Service",
    "ServiceEndpoint",
    "ServiceStatus",
    "ServiceRegistry",
    "VirtualService",
    "DestinationRule",
    "TrafficRoute",
    "TrafficMatch",
    "TrafficRetry",
    "TrafficTimeout",
    "LoadBalancer",
    "LoadBalanceAlgorithm",
    "CircuitBreaker",
    "CircuitState",
    "Sidecar",
    "get_mesh",
    "send_to_agent",
    "register_agent_service",

    # Policy Engine
    "PolicyEngine",
    "Policy",
    "PolicySet",
    "Rule",
    "Condition",
    "AttributeCondition",
    "CompositeCondition",
    "TimeCondition",
    "ExpressionCondition",
    "Subject",
    "Resource",
    "Action",
    "Context",
    "PolicyRequest",
    "PolicyResponse",
    "Decision",
    "Effect",
    "CombiningAlgorithm",
    "RBACEngine",
    "Role",
    "RBACPermission",
    "AuditLogger",
    "AuditEntry",
    "get_policy_engine",
    "check_permission",
    "create_policy",

    # NT Kernel - Executive
    "BBXExecutive",
    "ExecutiveConfig",
    "ExecutiveState",
    "ExecutiveStats",
    "SystemCall",
    "SystemCallType",
    "SystemCallResult",
    "SecurityToken",
    "ServiceType",
    "SecurityReferenceMonitor",
    "IOManager",
    "get_executive",
    "start_executive",
    "stop_executive",

    # NT Kernel - Object Manager
    "ObjectManager",
    "BBXObject",
    "DirectoryObject",
    "Handle",
    "SecurityDescriptor",
    "ACL",
    "ACE",
    "ACEType",
    "AccessMask",
    "ObjectType",
    "get_object_manager",

    # NT Kernel - Filter Stack
    "FilterManager",
    "FilterContext",
    "FilterInstance",
    "FilterStats",
    "FilterRegistration",
    "FilterResult",
    "FilterStatus",
    "OperationClass",
    "FilterAltitude",
    "Filter",
    "AuditFilter",
    "SecurityFilter",
    "QuotaFilter",
    "RateLimitFilter",
    "TransformFilter",
    "CacheFilter",
    "ValidationFilter",
    "MetricsFilter",
    "FilterStackBuilder",
    "get_filter_manager",

    # NT Kernel - Working Set
    "WorkingSetManager",
    "AgentWorkingSet",
    "PageMetadata",
    "WorkingSetLimits",
    "CompressionStats",
    "WorkingSetStats",
    "AllocationRequest",
    "AllocationResult",
    "PageState",
    "MemoryPressure",
    "FaultType",
    "TrimPriority",
    "CompressedStore",
    "PageFile",
    "InMemoryPageFile",
    "FileSystemPageFile",
    "get_working_set_manager",

    # NT Kernel - Config Registry
    "ConfigRegistry",
    "RegistryKey",
    "RegistryValue",
    "RegistryKeySecurity",
    "RegistryTransaction",
    "ConfigRegistryHelper",
    "ValueType",
    "Hive",
    "RegistryAccessRights",
    "NotifyFilter",
    "ChangeNotification",
    "get_config_registry",

    # NT Kernel - AAL
    "AAL",
    "LLMProvider",
    "VectorProvider",
    "LLMRequest",
    "LLMResponse",
    "VectorQuery",
    "VectorResult",
    "VectorDocument",
    "Backend",
    "BackendConfig",
    "BackendHealth",
    "BackendMetrics",
    "BackendRouter",
    "RoutingRule",
    "Capability",
    "RoutingStrategy",
    "get_aal",

    # =========================================================================
    # Enhanced Components (Production-ready)
    # =========================================================================

    # Enhanced Ring
    "EnhancedAgentRing",
    "EnhancedOperation",
    "EnhancedCompletion",
    "EnhancedRingConfig",
    "EnhancedRingStats",
    "WALManager",
    "WALEntry",
    "IdempotencyManager",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "RingCircuitState",
    "SharedMemoryRingBuffer",
    "get_enhanced_ring",
    "create_enhanced_ring",

    # Enhanced Context Tiering
    "EnhancedContextTiering",
    "EnhancedTieringConfig",
    "EnhancedTieringStats",
    "ImportanceScorer",
    "PrefetchManager",
    "AsyncMigrationEngine",
    "CompressionManager",
    "MemoryBackend",
    "DiskBackend",
    "ContentType",
    "get_enhanced_tiering",
    "create_enhanced_tiering",

    # Enforced Quotas
    "EnforcedQuotaManager",
    "EnforcedQuotaConfig",
    "TokenBucket",
    "CgroupsManager",
    "GPUQuotaManager",
    "EnforcementAction",
    "ThrottleLevel",
    "QuotaExceededError",
    "enforce_quota",
    "get_enforced_quota_manager",
    "create_enforced_quota_manager",

    # Distributed Snapshots
    "DistributedSnapshotManager",
    "DistributedSnapshotConfig",
    "DistributedSnapshotStats",
    "LocalDiskBackend",
    "S3Backend",
    "RedisBackend",
    "ReplicationManager",
    "AsyncSnapshotWriter",
    "PointInTimeRecovery",
    "StorageBackendType",
    "ReplicationStatus",
    "get_distributed_snapshot_manager",
    "create_distributed_snapshot_manager",

    # Enhanced Flow Integrity
    "EnhancedFlowIntegrity",
    "EnhancedFlowConfig",
    "FlowVerificationResult",
    "AnomalyDetector",
    "AnomalyEvent",
    "AnomalyType",
    "FlowPolicyEngine",
    "PolicyRule",
    "PolicyEvaluation",
    "PolicyDecision",
    "MemoryAccessControl",
    "MemoryAccessRule",
    "ToolCallValidator",
    "ToolValidationRule",
    "VerificationResult",
    "get_enhanced_flow_integrity",
    "create_enhanced_flow_integrity",

    # Semantic Memory
    "SemanticMemory",
    "SemanticMemoryConfig",
    "MemoryEntry",
    "MemoryType",
    "ContentModality",
    "SearchResult",
    "EmbeddingService",
    "OpenAIEmbedding",
    "LocalEmbedding",
    "VectorStore",
    "QdrantStore",
    "InMemoryStore",
    "ForgettingManager",
    "get_semantic_memory",
    "create_semantic_memory",

    # Message Bus
    "MessageBus",
    "MessageBusConfig",
    "Message",
    "ConsumerGroup",
    "DeliveryGuarantee",
    "MessageBusInMemoryBackend",
    "RedisStreamsBackend",
    "get_message_bus",
    "create_message_bus",

    # Goal Engine
    "GoalEngine",
    "GoalEngineConfig",
    "Goal",
    "Milestone",
    "Task",
    "GoalStatus",
    "TaskPriority",
    "LLMPlanner",
    "SimplePlanner",
    "DAGExecutor",
    "get_goal_engine",
    "create_goal_engine",

    # Authentication
    "AuthManager",
    "AuthConfig",
    "Identity",
    "JWTManager",
    "JWTClaims",
    "APIKeyManager",
    "OIDCProvider",
    "GenericOIDCProvider",
    "AuthorizationEngine",
    "AuthorizationRule",
    "AuthMethod",
    "AuthPermission",
    "get_auth_manager",
    "create_auth_manager",

    # Monitoring
    "MonitoringManager",
    "MonitoringConfig",
    "MetricsRegistry",
    "MetricValue",
    "MetricType",
    "Tracer",
    "Span",
    "StructuredLogger",
    "AlertManager",
    "AlertRule",
    "Alert",
    "DashboardManager",
    "Dashboard",
    "DashboardPanel",
    "get_monitoring",
    "create_monitoring",

    # Deployment
    "DeploymentManager",
    "DeploymentConfig",
    "DeploymentTarget",
    "AgentDeploymentSpec",
    "ResourceRequirements",
    "HealthCheck",
    "AutoscalingConfig",
    "DockerfileGenerator",
    "HelmChartGenerator",
    "KubernetesOperator",
    "get_deployment_manager",
    "create_deployment_manager",
]

__version__ = "2.0.0"
