# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Blackbox Core - Init file

BBX: Operating System for AI Agents

Exports main runtime functions with BBX version support.
"""

from .cache import WorkflowCache, get_cache
from .context import WorkflowContext
from .dag import DAGError, WorkflowDAG
from .events import Event, EventBus, EventType
from .expressions import ExpressionError, SafeExpr
from .parsers.v6 import BBXv6Parser
from .registry import MCPRegistry
from .runtime import run_file

# Workspace system (OS-like isolated environments)
from .workspace import Workspace, WorkspacePaths, ExecutionRecord
from .workspace_manager import (
    WorkspaceManager,
    get_workspace_manager,
    get_current_workspace,
    require_workspace,
)

# Background execution (process-like management)
from .execution_store import (
    Execution,
    ExecutionStatus,
    ExecutionStore,
    get_execution_store,
)
from .execution_manager import (
    ExecutionManager,
    get_execution_manager,
    run_background,
    ps,
    kill,
    wait,
)

__all__ = [
    # Runtime
    "run_file",
    "WorkflowContext",
    "MCPRegistry",
    "EventBus",
    "Event",
    "EventType",
    "SafeExpr",
    "ExpressionError",
    "WorkflowDAG",
    "DAGError",
    "get_cache",
    "WorkflowCache",
    "BBXv6Parser",
    # Workspace system
    "Workspace",
    "WorkspacePaths",
    "ExecutionRecord",
    "WorkspaceManager",
    "get_workspace_manager",
    "get_current_workspace",
    "require_workspace",
    # Background execution
    "Execution",
    "ExecutionStatus",
    "ExecutionStore",
    "get_execution_store",
    "ExecutionManager",
    "get_execution_manager",
    "run_background",
    "ps",
    "kill",
    "wait",
]
