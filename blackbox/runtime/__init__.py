# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Runtime - Background Execution and API Layer

This module provides the runtime components for BBX:
- BBXDaemon: Background service for managing agents (like Docker daemon)
- WorkflowEngine: Multi-step workflow execution with recovery
- BBXAPIServer: REST/WebSocket API for GUI clients
"""

from .daemon import (
    BBXDaemon,
    AgentConfig,
    AgentStatus,
    AgentInstance,
    EventBus,
    FileWatcher,
    Scheduler,
    MemoryManager as DaemonMemoryManager,
    SnapshotManager,
    AgentManager,
    MemoryTier,
    get_daemon,
    start_daemon,
    stop_daemon,
)

from .workflows import (
    WorkflowEngine,
    WorkflowConfig,
    WorkflowStatus,
    WorkflowInstance,
    StepConfig,
    StepStatus,
    StepResult,
    RecoveryStrategy,
    RecoveryConfig,
    WorkflowLoader,
    create_workflow_engine,
    load_example_workflows,
)

from .api import (
    BBXAPIServer,
    APIResponse,
    WSEvent,
    WSEventType,
    create_api_server,
)

from .llm_provider import (
    LLMManager,
    AnthropicProvider,
    OpenAIProvider,
    OllamaProvider,
    Message,
    LLMRequest,
    LLMResponse,
    get_llm_manager,
)

from .vectordb_provider import (
    ChromaDBProvider,
    MemoryStore,
    Document,
    SearchResult,
    get_vectordb,
    get_memory_store,
)

__all__ = [
    # Daemon
    "BBXDaemon",
    "AgentConfig",
    "AgentStatus",
    "AgentInstance",
    "EventBus",
    "FileWatcher",
    "Scheduler",
    "DaemonMemoryManager",
    "SnapshotManager",
    "AgentManager",
    "MemoryTier",
    "get_daemon",
    "start_daemon",
    "stop_daemon",
    # Workflows
    "WorkflowEngine",
    "WorkflowConfig",
    "WorkflowStatus",
    "WorkflowInstance",
    "StepConfig",
    "StepStatus",
    "StepResult",
    "RecoveryStrategy",
    "RecoveryConfig",
    "WorkflowLoader",
    "create_workflow_engine",
    "load_example_workflows",
    # API
    "BBXAPIServer",
    "APIResponse",
    "WSEvent",
    "WSEventType",
    "create_api_server",
    # LLM Providers
    "LLMManager",
    "AnthropicProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "Message",
    "LLMRequest",
    "LLMResponse",
    "get_llm_manager",
    # VectorDB
    "ChromaDBProvider",
    "MemoryStore",
    "Document",
    "SearchResult",
    "get_vectordb",
    "get_memory_store",
]

__version__ = "1.0.0"
