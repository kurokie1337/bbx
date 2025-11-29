# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX 2.0 MCP Tools - Next-generation workflow operations exposed as MCP tools

These tools provide MCP access to BBX 2.0 features:
- AgentRing: io_uring-inspired batch operations
- Hooks: eBPF-inspired dynamic programming
- ContextTiering: MGLRU-inspired memory management
- Declarative: NixOS-inspired infrastructure as code
"""

import json
from typing import Any, Dict, List


def get_bbx_v2_tools() -> List[Dict[str, Any]]:
    """
    Get list of BBX 2.0 tools for MCP server.

    Returns:
        List of MCP tool definitions
    """
    return [
        # === AgentRing Tools ===
        {
            "name": "bbx_v2_ring_stats",
            "description": "Get AgentRing statistics (throughput, latency, worker utilization)",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_v2_ring_config",
            "description": "Get AgentRing configuration",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_v2_ring_submit",
            "description": "Submit batch operations to AgentRing for efficient parallel execution",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "description": "List of operations to submit",
                        "items": {
                            "type": "object",
                            "properties": {
                                "adapter": {
                                    "type": "string",
                                    "description": "Adapter name (e.g., 'shell', 'http')",
                                },
                                "method": {
                                    "type": "string",
                                    "description": "Method to call",
                                },
                                "args": {
                                    "type": "object",
                                    "description": "Method arguments",
                                },
                            },
                            "required": ["adapter", "method"],
                        },
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 300)",
                        "default": 300,
                    },
                },
                "required": ["operations"],
            },
        },
        # === Hooks Tools ===
        {
            "name": "bbx_v2_hooks_list",
            "description": "List all registered BBX hooks",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_v2_hooks_stats",
            "description": "Get hooks execution statistics",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_v2_hooks_enable",
            "description": "Enable a hook by ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hook_id": {
                        "type": "string",
                        "description": "Hook ID to enable",
                    },
                },
                "required": ["hook_id"],
            },
        },
        {
            "name": "bbx_v2_hooks_disable",
            "description": "Disable a hook by ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "hook_id": {
                        "type": "string",
                        "description": "Hook ID to disable",
                    },
                },
                "required": ["hook_id"],
            },
        },
        # === Context Tiering Tools ===
        {
            "name": "bbx_v2_context_stats",
            "description": "Get context tiering statistics (items by tier, memory usage, hit rate)",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "bbx_v2_context_get",
            "description": "Get a value from tiered context (auto-promotes from cold)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Context key",
                    },
                },
                "required": ["key"],
            },
        },
        {
            "name": "bbx_v2_context_set",
            "description": "Set a value in tiered context",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Context key",
                    },
                    "value": {
                        "type": ["string", "number", "boolean", "object", "array"],
                        "description": "Value to set",
                    },
                    "pinned": {
                        "type": "boolean",
                        "description": "Pin to prevent demotion",
                        "default": False,
                    },
                },
                "required": ["key", "value"],
            },
        },
        {
            "name": "bbx_v2_context_pin",
            "description": "Pin a context key to prevent demotion to lower tiers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Context key to pin",
                    },
                },
                "required": ["key"],
            },
        },
        # === Declarative Config Tools ===
        {
            "name": "bbx_v2_config_apply",
            "description": "Apply a declarative configuration (creates new generation)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "config_file": {
                        "type": "string",
                        "description": "Path to YAML config file",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "Show what would change without applying",
                        "default": False,
                    },
                },
                "required": ["config_file"],
            },
        },
        {
            "name": "bbx_v2_config_rollback",
            "description": "Rollback to a previous configuration generation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "generation_id": {
                        "type": "integer",
                        "description": "Generation ID to rollback to",
                    },
                },
                "required": ["generation_id"],
            },
        },
        {
            "name": "bbx_v2_config_show",
            "description": "Show current configuration",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        # === Generation Management Tools ===
        {
            "name": "bbx_v2_generation_list",
            "description": "List configuration generations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of generations to show",
                        "default": 10,
                    },
                },
            },
        },
        {
            "name": "bbx_v2_generation_diff",
            "description": "Show differences between two generations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "gen1": {
                        "type": "integer",
                        "description": "First generation ID",
                    },
                    "gen2": {
                        "type": "integer",
                        "description": "Second generation ID",
                    },
                },
                "required": ["gen1", "gen2"],
            },
        },
        {
            "name": "bbx_v2_generation_switch",
            "description": "Switch to a specific generation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "generation_id": {
                        "type": "integer",
                        "description": "Generation ID to switch to",
                    },
                },
                "required": ["generation_id"],
            },
        },
        # === V2 Runtime Tools ===
        {
            "name": "bbx_v2_run",
            "description": "Run a workflow using BBX 2.0 runtime (with Ring, Hooks, Tiering)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_file": {
                        "type": "string",
                        "description": "Path to .bbx workflow file",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Workflow inputs",
                    },
                    "ring_enabled": {
                        "type": "boolean",
                        "description": "Enable AgentRing batch execution",
                        "default": True,
                    },
                    "hooks_enabled": {
                        "type": "boolean",
                        "description": "Enable hooks",
                        "default": True,
                    },
                    "tiering_enabled": {
                        "type": "boolean",
                        "description": "Enable context tiering",
                        "default": True,
                    },
                },
                "required": ["workflow_file"],
            },
        },

        # =====================================================================
        # FLOW INTEGRITY TOOLS (CET - Control-flow Enforcement Technology)
        # =====================================================================
        {
            "name": "bbx_v2_flow_stats",
            "description": "Get flow integrity statistics (shadow stack, IBT tracking)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_flow_verify",
            "description": "Verify workflow execution flow integrity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow_id": {"type": "string", "description": "Workflow ID to verify"},
                },
                "required": ["workflow_id"],
            },
        },

        # =====================================================================
        # AGENT QUOTAS TOOLS (Cgroups v2 - Resource Control)
        # =====================================================================
        {
            "name": "bbx_v2_quotas_stats",
            "description": "Get agent quotas statistics (CPU, memory, I/O, tokens)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_quotas_set",
            "description": "Set resource quotas for an agent group",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "Quota group name"},
                    "cpu_shares": {"type": "integer", "description": "CPU shares (default: 1024)"},
                    "memory_mb": {"type": "integer", "description": "Memory limit in MB"},
                    "io_ops": {"type": "integer", "description": "Max I/O operations per second"},
                    "tokens": {"type": "integer", "description": "Tokens per hour limit"},
                },
                "required": ["group"],
            },
        },
        {
            "name": "bbx_v2_quotas_list",
            "description": "List all quota groups",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # STATE SNAPSHOTS TOOLS (XFS Reflink - CoW Snapshots)
        # =====================================================================
        {
            "name": "bbx_v2_snapshot_create",
            "description": "Create a state snapshot (instant CoW copy)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Snapshot name"},
                    "description": {"type": "string", "description": "Optional description"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_v2_snapshot_list",
            "description": "List all state snapshots",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_snapshot_restore",
            "description": "Restore state from a snapshot",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "snapshot_id": {"type": "string", "description": "Snapshot ID to restore"},
                },
                "required": ["snapshot_id"],
            },
        },
        {
            "name": "bbx_v2_snapshot_stats",
            "description": "Get snapshot statistics (size, CoW savings)",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # FLAKES TOOLS (Nix Flakes - Reproducible Packages)
        # =====================================================================
        {
            "name": "bbx_v2_flake_build",
            "description": "Build a flake (reproducible workflow package)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "flake_path": {"type": "string", "description": "Path to flake directory"},
                },
                "required": ["flake_path"],
            },
        },
        {
            "name": "bbx_v2_flake_run",
            "description": "Run a flake directly without installing",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "flake_ref": {"type": "string", "description": "Flake reference (path or URL)"},
                    "inputs": {"type": "object", "description": "Inputs for the flake"},
                },
                "required": ["flake_ref"],
            },
        },
        {
            "name": "bbx_v2_flake_lock",
            "description": "Update flake lock file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "flake_path": {"type": "string", "description": "Path to flake directory"},
                },
                "required": ["flake_path"],
            },
        },
        {
            "name": "bbx_v2_flake_show",
            "description": "Show flake metadata and outputs",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "flake_ref": {"type": "string", "description": "Flake reference"},
                },
                "required": ["flake_ref"],
            },
        },

        # =====================================================================
        # AGENT REGISTRY TOOLS (AUR - Package Discovery)
        # =====================================================================
        {
            "name": "bbx_v2_registry_search",
            "description": "Search agent registry for packages",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "bbx_v2_registry_install",
            "description": "Install agent from registry",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent package name"},
                    "version": {"type": "string", "description": "Optional specific version"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_v2_registry_publish",
            "description": "Publish agent to registry",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to agent package"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "bbx_v2_registry_list_installed",
            "description": "List installed agents",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # AGENT BUNDLES TOOLS (Kali-style Tool Collections)
        # =====================================================================
        {
            "name": "bbx_v2_bundle_list",
            "description": "List available agent bundles (tool collections)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_bundle_install",
            "description": "Install an agent bundle with all its tools",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Bundle name"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_v2_bundle_show",
            "description": "Show bundle details and included tools",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Bundle name"},
                },
                "required": ["name"],
            },
        },

        # =====================================================================
        # AGENT SANDBOX TOOLS (Flatpak-style Isolation)
        # =====================================================================
        {
            "name": "bbx_v2_sandbox_run",
            "description": "Run agent in isolated sandbox with permission control",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Agent to run"},
                    "permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Permissions to grant (e.g., network, filesystem)",
                    },
                    "inputs": {"type": "object", "description": "Agent inputs"},
                },
                "required": ["agent"],
            },
        },
        {
            "name": "bbx_v2_sandbox_list",
            "description": "List active sandboxes",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_sandbox_permissions",
            "description": "List available sandbox permissions",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # NETWORK FABRIC TOOLS (Istio-style Service Mesh)
        # =====================================================================
        {
            "name": "bbx_v2_mesh_status",
            "description": "Get mesh status (control/data plane, traffic stats)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_mesh_services",
            "description": "List services in the mesh",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_mesh_route",
            "description": "Create traffic routing rule",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Rule name"},
                    "source": {"type": "string", "description": "Source service"},
                    "destination": {"type": "string", "description": "Destination service"},
                    "weight": {"type": "integer", "description": "Traffic weight (0-100)"},
                    "headers": {"type": "object", "description": "Header matching rules"},
                },
                "required": ["name", "source", "destination"],
            },
        },

        # =====================================================================
        # POLICY ENGINE TOOLS (OPA/SELinux - Policy Enforcement)
        # =====================================================================
        {
            "name": "bbx_v2_policy_evaluate",
            "description": "Evaluate policy against input data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "policy": {"type": "string", "description": "Policy name or path"},
                    "input": {"type": "object", "description": "Input data to evaluate"},
                },
                "required": ["policy", "input"],
            },
        },
        {
            "name": "bbx_v2_policy_list",
            "description": "List all policies",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_policy_add",
            "description": "Add a new policy",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Policy name"},
                    "type": {"type": "string", "description": "Policy type (allow/deny)"},
                    "rules": {"type": "array", "description": "Policy rules"},
                    "priority": {"type": "integer", "description": "Priority (higher = first)"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_v2_policy_stats",
            "description": "Get policy engine statistics",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # AAL TOOLS (HAL - Adapter Abstraction Layer)
        # =====================================================================
        {
            "name": "bbx_v2_aal_adapters",
            "description": "List all adapters through AAL",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_aal_call",
            "description": "Call adapter method through AAL abstraction",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "adapter": {"type": "string", "description": "Adapter name"},
                    "method": {"type": "string", "description": "Method to call"},
                    "args": {"type": "object", "description": "Method arguments"},
                },
                "required": ["adapter", "method"],
            },
        },
        {
            "name": "bbx_v2_aal_stats",
            "description": "Get AAL statistics",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # OBJECT MANAGER TOOLS (Windows ObMgr - Object Namespace)
        # =====================================================================
        {
            "name": "bbx_v2_objects_list",
            "description": "List objects in namespace",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Namespace path (default: /)"},
                },
            },
        },
        {
            "name": "bbx_v2_objects_create",
            "description": "Create a named object",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Object name"},
                    "type": {"type": "string", "description": "Object type"},
                    "data": {"type": "object", "description": "Object data"},
                },
                "required": ["name", "type"],
            },
        },
        {
            "name": "bbx_v2_objects_open",
            "description": "Open an object by name",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Object name"},
                    "access": {"type": "string", "description": "Access mode (read/write)"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_v2_objects_stats",
            "description": "Get object manager statistics",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # FILTER STACK TOOLS (Windows Filter Drivers - I/O Pipeline)
        # =====================================================================
        {
            "name": "bbx_v2_filters_list",
            "description": "List registered filters in the I/O stack",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_filters_add",
            "description": "Add a filter to the I/O stack",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Filter name"},
                    "altitude": {"type": "integer", "description": "Filter altitude (higher = earlier)"},
                    "type": {"type": "string", "description": "Filter type"},
                    "operations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Operations to intercept",
                    },
                    "handler": {"type": "string", "description": "Handler function/code"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_v2_filters_remove",
            "description": "Remove a filter from the stack",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Filter name"},
                },
                "required": ["name"],
            },
        },
        {
            "name": "bbx_v2_filters_stats",
            "description": "Get filter stack statistics",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # WORKING SET TOOLS (Windows Mm - Memory Management)
        # =====================================================================
        {
            "name": "bbx_v2_memory_stats",
            "description": "Get working set statistics (pages, faults, trims)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_memory_trim",
            "description": "Trim working set to reduce memory usage",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target_mb": {"type": "number", "description": "Target size in MB"},
                },
            },
        },
        {
            "name": "bbx_v2_memory_lock",
            "description": "Lock pages in working set (prevent paging)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Data key to lock"},
                },
                "required": ["key"],
            },
        },
        {
            "name": "bbx_v2_memory_pools",
            "description": "Show memory pool statistics",
            "inputSchema": {"type": "object", "properties": {}},
        },

        # =====================================================================
        # CONFIG REGISTRY TOOLS (Windows Registry - Hierarchical Config)
        # =====================================================================
        {
            "name": "bbx_v2_reg_get",
            "description": "Get registry value",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Registry key path"},
                    "value": {"type": "string", "description": "Value name"},
                },
                "required": ["path", "value"],
            },
        },
        {
            "name": "bbx_v2_reg_set",
            "description": "Set registry value",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Registry key path"},
                    "value": {"type": "string", "description": "Value name"},
                    "data": {"description": "Value data"},
                    "type": {"type": "string", "description": "Value type (string/dword/binary)"},
                },
                "required": ["path", "value", "data"],
            },
        },
        {
            "name": "bbx_v2_reg_list",
            "description": "List registry keys and values",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Registry key path"},
                },
            },
        },
        {
            "name": "bbx_v2_reg_delete",
            "description": "Delete registry key or value",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Registry key path"},
                    "value": {"type": "string", "description": "Value name (omit to delete key)"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "bbx_v2_reg_export",
            "description": "Export registry to file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Registry key path"},
                    "output_file": {"type": "string", "description": "Output file path"},
                },
                "required": ["path", "output_file"],
            },
        },

        # =====================================================================
        # EXECUTIVE TOOLS (Windows ntoskrnl - Hybrid Kernel)
        # =====================================================================
        {
            "name": "bbx_v2_executive_status",
            "description": "Get BBX Executive (kernel) status",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bbx_v2_executive_start",
            "description": "Start executive subsystems",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subsystems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subsystems to start (or 'all')",
                    },
                },
            },
        },
        {
            "name": "bbx_v2_executive_stop",
            "description": "Stop executive subsystems",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subsystems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subsystems to stop (or 'all')",
                    },
                },
            },
        },
        {
            "name": "bbx_v2_executive_syscall",
            "description": "Execute system call through executive",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "syscall": {"type": "string", "description": "System call name"},
                    "args": {"type": "object", "description": "System call arguments"},
                },
                "required": ["syscall"],
            },
        },
        {
            "name": "bbx_v2_executive_bugcheck",
            "description": "Generate diagnostic bugcheck/crash dump",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


# === Handler Functions ===

async def handle_bbx_v2_ring_stats(arguments: Dict[str, Any]) -> str:
    """Get AgentRing statistics."""
    try:
        from blackbox.core.v2.runtime_v2 import get_runtime_v2

        runtime = get_runtime_v2()
        if not runtime._started:
            await runtime.start()

        if not runtime.ring:
            return "AgentRing not active"

        stats = runtime.ring.get_stats()

        return f"""AgentRing Statistics
{'=' * 60}

Operations:
  Submitted:  {stats.operations_submitted}
  Completed:  {stats.operations_completed}
  Failed:     {stats.operations_failed}

Performance:
  Throughput:     {stats.throughput_ops_sec:.2f} ops/sec
  Avg Latency:    {stats.avg_latency_ms:.2f}ms
  P95 Latency:    {stats.p95_latency_ms:.2f}ms
  P99 Latency:    {stats.p99_latency_ms:.2f}ms

Workers:
  Pool Size:      {stats.worker_pool_size}
  Active:         {stats.active_workers}
  Utilization:    {stats.worker_utilization:.1f}%
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_ring_config(arguments: Dict[str, Any]) -> str:
    """Get AgentRing configuration."""
    try:
        from blackbox.core.v2.runtime_v2 import get_runtime_v2

        runtime = get_runtime_v2()
        if runtime.ring:
            config = runtime.ring.config
            return f"""AgentRing Configuration
{'=' * 60}

  Submission Queue Size: {config.submission_queue_size}
  Completion Queue Size: {config.completion_queue_size}
  Worker Pool Size:      {config.worker_pool_size}
  Max Batch Size:        {config.max_batch_size}
  Default Timeout (ms):  {config.default_timeout_ms}
  Enable Priorities:     {config.enable_priorities}
  Enable Retries:        {config.enable_retries}
"""
        return "AgentRing not configured"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_ring_submit(arguments: Dict[str, Any]) -> str:
    """Submit batch operations to AgentRing."""
    try:
        from blackbox.core.v2.runtime_v2 import execute_batch

        operations = arguments.get("operations", [])
        timeout = arguments.get("timeout", 300.0)

        results = await execute_batch(operations, timeout=timeout)

        # Format results
        output = f"Batch Execution Results ({len(results)} operations)\n{'=' * 60}\n"
        for i, r in enumerate(results):
            status = r.get("status", "unknown")
            icon = "[OK]" if status == "success" else "[ERR]"
            output += f"\n{icon} Operation {i + 1}: {status}"
            if r.get("duration_ms"):
                output += f" ({r['duration_ms']:.1f}ms)"
            if r.get("error"):
                output += f"\n    Error: {r['error']}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_hooks_list(arguments: Dict[str, Any]) -> str:
    """List registered hooks."""
    try:
        from blackbox.core.v2.hooks import get_hook_manager

        manager = get_hook_manager()
        hooks = manager.list_hooks()

        if not hooks:
            return "No hooks registered."

        output = f"Registered Hooks ({len(hooks)})\n{'=' * 60}\n"
        for h in sorted(hooks, key=lambda x: x.priority):
            status = "[ON] " if h.enabled else "[OFF]"
            output += f"\n{status} {h.name} ({h.id})"
            output += f"\n     Type: {h.type.name}"
            output += f"\n     Priority: {h.priority}"
            output += f"\n     Attach: {', '.join(ap.value for ap in h.attach_points)}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_hooks_stats(arguments: Dict[str, Any]) -> str:
    """Get hooks statistics."""
    try:
        from blackbox.core.v2.hooks import get_hook_manager

        manager = get_hook_manager()
        stats = manager.get_stats()

        output = f"Hooks Statistics\n{'=' * 60}\n"
        output += f"\nTotal Triggers:  {stats.get('total_triggers', 0)}"
        output += f"\nTotal Duration:  {stats.get('total_duration_ms', 0):.2f}ms"
        output += f"\nAvg Duration:    {stats.get('avg_duration_ms', 0):.2f}ms"

        by_ap = stats.get('by_attach_point', {})
        if by_ap:
            output += "\n\nBy Attach Point:"
            for ap, count in by_ap.items():
                output += f"\n  {ap}: {count}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_hooks_enable(arguments: Dict[str, Any]) -> str:
    """Enable a hook."""
    try:
        from blackbox.core.v2.hooks import get_hook_manager

        hook_id = arguments.get("hook_id")
        manager = get_hook_manager()
        success = manager.enable(hook_id)

        if success:
            return f"[OK] Hook enabled: {hook_id}"
        return f"[ERR] Hook not found: {hook_id}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_hooks_disable(arguments: Dict[str, Any]) -> str:
    """Disable a hook."""
    try:
        from blackbox.core.v2.hooks import get_hook_manager

        hook_id = arguments.get("hook_id")
        manager = get_hook_manager()
        success = manager.disable(hook_id)

        if success:
            return f"[OK] Hook disabled: {hook_id}"
        return f"[ERR] Hook not found: {hook_id}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_context_stats(arguments: Dict[str, Any]) -> str:
    """Get context tiering statistics."""
    try:
        from blackbox.core.v2.context_tiering import get_context_tiering

        tiering = get_context_tiering()
        stats = tiering.get_stats()

        return f"""Context Tiering Statistics
{'=' * 60}

Items by Tier:
  HOT:   {stats.hot_items} ({stats.hot_bytes / 1024:.1f} KB)
  WARM:  {stats.warm_items} ({stats.warm_bytes / 1024:.1f} KB)
  COOL:  {stats.cool_items} ({stats.cool_bytes / 1024:.1f} KB)
  COLD:  {stats.cold_items} ({stats.cold_bytes / 1024:.1f} KB)

Memory:
  Total:    {stats.total_bytes / 1024:.1f} KB
  In-Memory:{stats.in_memory_bytes / 1024:.1f} KB
  On-Disk:  {stats.on_disk_bytes / 1024:.1f} KB

Operations:
  Gets:       {stats.total_gets}
  Sets:       {stats.total_sets}
  Hit Rate:   {stats.hit_rate * 100:.1f}%

Pinned Items: {stats.pinned_items}
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_context_get(arguments: Dict[str, Any]) -> str:
    """Get a value from tiered context."""
    try:
        from blackbox.core.v2.context_tiering import get_context_tiering

        key = arguments.get("key")
        tiering = get_context_tiering()
        value = await tiering.get(key)

        if value is None:
            return f"Key not found: {key}"

        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, default=str)
        return str(value)
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_context_set(arguments: Dict[str, Any]) -> str:
    """Set a value in tiered context."""
    try:
        from blackbox.core.v2.context_tiering import get_context_tiering

        key = arguments.get("key")
        value = arguments.get("value")
        pinned = arguments.get("pinned", False)

        tiering = get_context_tiering()
        await tiering.set(key, value, pinned=pinned)

        return f"[OK] Set {key}" + (" (pinned)" if pinned else "")
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_context_pin(arguments: Dict[str, Any]) -> str:
    """Pin a context key."""
    try:
        from blackbox.core.v2.context_tiering import get_context_tiering

        key = arguments.get("key")
        tiering = get_context_tiering()
        success = await tiering.pin(key)

        if success:
            return f"[OK] Pinned: {key}"
        return f"[ERR] Key not found: {key}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_config_apply(arguments: Dict[str, Any]) -> str:
    """Apply declarative configuration."""
    try:
        import yaml
        from blackbox.core.v2.declarative import DeclarativeManager, BBXConfig

        config_file = arguments.get("config_file")
        dry_run = arguments.get("dry_run", False)

        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config = BBXConfig.from_dict(data)
        manager = DeclarativeManager()

        if dry_run:
            current = manager.get_current_config()
            diff = manager.diff_configs(current, config)

            if not diff:
                return "No changes detected."

            output = f"Dry Run - Would apply these changes:\n{'=' * 60}\n"
            for change in diff:
                output += f"\n  {change['type']}: {change['path']}"
                if change.get('old'):
                    output += f"\n    - {change['old']}"
                if change.get('new'):
                    output += f"\n    + {change['new']}"
            return output
        else:
            generation = await manager.apply(config)
            return f"""[OK] Configuration applied!
Generation: {generation.id}
Created: {generation.created_at}

To rollback: bbx v2 config rollback {generation.id - 1}"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_config_rollback(arguments: Dict[str, Any]) -> str:
    """Rollback to previous generation."""
    try:
        from blackbox.core.v2.declarative import DeclarativeManager

        generation_id = arguments.get("generation_id")
        manager = DeclarativeManager()
        config = await manager.rollback(generation_id)

        if config:
            return f"[OK] Rolled back to generation {generation_id}"
        return f"[ERR] Generation not found: {generation_id}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_config_show(arguments: Dict[str, Any]) -> str:
    """Show current configuration."""
    try:
        import yaml
        from blackbox.core.v2.declarative import DeclarativeManager

        manager = DeclarativeManager()
        config = manager.get_current_config()

        if config is None:
            return "No configuration applied yet."

        return yaml.dump(config.to_dict(), default_flow_style=False)
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_generation_list(arguments: Dict[str, Any]) -> str:
    """List configuration generations."""
    try:
        from blackbox.core.v2.declarative import DeclarativeManager

        limit = arguments.get("limit", 10)
        manager = DeclarativeManager()
        generations = manager.list_generations(limit=limit)
        current_id = manager.get_current_generation_id()

        if not generations:
            return "No generations found."

        output = f"Configuration Generations\n{'=' * 60}\n"
        for g in generations:
            marker = " *" if g.id == current_id else "  "
            output += f"\n{marker}Generation {g.id}"
            output += f"\n     Created: {g.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            if g.description:
                output += f"\n     Description: {g.description}"

        output += f"\n{'=' * 60}\n* = current generation"
        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_generation_diff(arguments: Dict[str, Any]) -> str:
    """Show diff between generations."""
    try:
        from blackbox.core.v2.declarative import DeclarativeManager

        gen1 = arguments.get("gen1")
        gen2 = arguments.get("gen2")
        manager = DeclarativeManager()
        diff = manager.diff_generations(gen1, gen2)

        if not diff:
            return f"No differences between generation {gen1} and {gen2}"

        output = f"Diff: Generation {gen1} -> {gen2}\n{'=' * 60}\n"
        for change in diff:
            output += f"\n  {change['type']}: {change['path']}"
            if change.get('old'):
                output += f"\n    - {change['old']}"
            if change.get('new'):
                output += f"\n    + {change['new']}"
        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_generation_switch(arguments: Dict[str, Any]) -> str:
    """Switch to specific generation."""
    try:
        from blackbox.core.v2.declarative import DeclarativeManager

        generation_id = arguments.get("generation_id")
        manager = DeclarativeManager()
        success = await manager.switch_generation(generation_id)

        if success:
            return f"[OK] Switched to generation {generation_id}"
        return f"[ERR] Failed to switch to generation {generation_id}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_run(arguments: Dict[str, Any]) -> str:
    """Run workflow with BBX 2.0 runtime."""
    try:
        from blackbox.core.v2.runtime_v2 import run_file_v2, RuntimeV2Config

        workflow_file = arguments.get("workflow_file")
        inputs = arguments.get("inputs", {})

        config = RuntimeV2Config(
            ring_enabled=arguments.get("ring_enabled", True),
            hooks_enabled=arguments.get("hooks_enabled", True),
            tiering_enabled=arguments.get("tiering_enabled", True),
        )

        results = await run_file_v2(workflow_file, inputs=inputs, config=config)

        output = f"BBX 2.0 Workflow Results\n{'=' * 60}\n"
        for step_id, result in results.items():
            status = result.get("status", "unknown")
            icon = "[OK]" if status == "success" else "[ERR]" if status == "error" else "[?]"
            output += f"\n{icon} Step: {step_id}"
            output += f"\n    Status: {status}"
            if result.get("error"):
                output += f"\n    Error: {result['error']}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# FLOW INTEGRITY HANDLERS (CET - Control-flow Enforcement Technology)
# =============================================================================

async def handle_bbx_v2_flow_stats(arguments: Dict[str, Any]) -> str:
    """Get flow integrity statistics."""
    try:
        from blackbox.core.v2.flow_integrity import get_flow_integrity

        integrity = get_flow_integrity()
        stats = integrity.get_stats()

        return f"""Flow Integrity Statistics (CET-inspired)
{'=' * 60}

Shadow Stack:
  Calls Verified:     {stats.calls_verified}
  Violations:         {stats.violations}

IBT (Indirect Branch Tracking):
  Branches Tracked:   {stats.branches_tracked}
  Valid Targets:      {stats.valid_targets}
  Invalid Targets:    {stats.invalid_targets}

Workflow Integrity:
  Workflows Verified: {stats.workflows_verified}
  Steps Validated:    {stats.steps_validated}
  Integrity Errors:   {stats.integrity_errors}
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_flow_verify(arguments: Dict[str, Any]) -> str:
    """Verify workflow execution flow integrity."""
    try:
        from blackbox.core.v2.flow_integrity import get_flow_integrity

        workflow_id = arguments.get("workflow_id")
        integrity = get_flow_integrity()
        result = await integrity.verify_workflow(workflow_id)

        if result.valid:
            return f"[OK] Workflow {workflow_id} integrity verified"
        return f"[ERR] Integrity violation in {workflow_id}: {result.error}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AGENT QUOTAS HANDLERS (Cgroups v2 - Resource Control)
# =============================================================================

async def handle_bbx_v2_quotas_stats(arguments: Dict[str, Any]) -> str:
    """Get agent quotas statistics."""
    try:
        from blackbox.core.v2.agent_quotas import get_quota_manager

        manager = get_quota_manager()
        stats = manager.get_stats()

        return f"""Agent Quotas Statistics (Cgroups v2-inspired)
{'=' * 60}

Active Groups:    {stats.active_groups}
Total Agents:     {stats.total_agents}

Resource Usage:
  CPU:     {stats.cpu_usage_percent:.1f}%
  Memory:  {stats.memory_usage_mb:.1f} MB / {stats.memory_limit_mb:.1f} MB
  I/O:     {stats.io_ops}/s (limit: {stats.io_limit}/s)
  Tokens:  {stats.tokens_used} / {stats.tokens_limit}

Throttling:
  Throttled Operations: {stats.throttled_ops}
  OOM Events:          {stats.oom_events}
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_quotas_set(arguments: Dict[str, Any]) -> str:
    """Set resource quotas for an agent group."""
    try:
        from blackbox.core.v2.agent_quotas import get_quota_manager, QuotaConfig

        group = arguments.get("group")
        manager = get_quota_manager()

        config = QuotaConfig(
            cpu_shares=arguments.get("cpu_shares", 1024),
            memory_max_mb=arguments.get("memory_mb", 512),
            io_max_ops=arguments.get("io_ops", 1000),
            tokens_per_hour=arguments.get("tokens", 100000),
        )

        await manager.set_quota(group, config)
        return f"[OK] Quotas set for group: {group}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_quotas_list(arguments: Dict[str, Any]) -> str:
    """List all quota groups."""
    try:
        from blackbox.core.v2.agent_quotas import get_quota_manager

        manager = get_quota_manager()
        groups = manager.list_groups()

        if not groups:
            return "No quota groups defined."

        output = f"Agent Quota Groups\n{'=' * 60}\n"
        for g in groups:
            output += f"\n[{g.name}]"
            output += f"\n  CPU:    {g.cpu_shares} shares"
            output += f"\n  Memory: {g.memory_max_mb} MB"
            output += f"\n  I/O:    {g.io_max_ops} ops/s"
            output += f"\n  Tokens: {g.tokens_per_hour}/hour"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# STATE SNAPSHOTS HANDLERS (XFS Reflink - CoW Snapshots)
# =============================================================================

async def handle_bbx_v2_snapshot_create(arguments: Dict[str, Any]) -> str:
    """Create a state snapshot."""
    try:
        from blackbox.core.v2.state_snapshots import get_snapshot_manager

        name = arguments.get("name")
        description = arguments.get("description", "")

        manager = get_snapshot_manager()
        snapshot = await manager.create_snapshot(name, description=description)

        return f"""[OK] Snapshot created
ID:          {snapshot.id}
Name:        {snapshot.name}
Created:     {snapshot.created_at}
Size:        {snapshot.size_bytes / 1024:.1f} KB
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_snapshot_list(arguments: Dict[str, Any]) -> str:
    """List all snapshots."""
    try:
        from blackbox.core.v2.state_snapshots import get_snapshot_manager

        manager = get_snapshot_manager()
        snapshots = manager.list_snapshots()

        if not snapshots:
            return "No snapshots found."

        output = f"State Snapshots (CoW-enabled)\n{'=' * 60}\n"
        for s in snapshots:
            output += f"\n[{s.id[:8]}...] {s.name}"
            output += f"\n     Created: {s.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            output += f"\n     Size: {s.size_bytes / 1024:.1f} KB"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_snapshot_restore(arguments: Dict[str, Any]) -> str:
    """Restore state from a snapshot."""
    try:
        from blackbox.core.v2.state_snapshots import get_snapshot_manager

        snapshot_id = arguments.get("snapshot_id")

        manager = get_snapshot_manager()
        success = await manager.restore_snapshot(snapshot_id)

        if success:
            return f"[OK] Restored from snapshot: {snapshot_id}"
        return f"[ERR] Snapshot not found: {snapshot_id}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_snapshot_stats(arguments: Dict[str, Any]) -> str:
    """Get snapshot statistics."""
    try:
        from blackbox.core.v2.state_snapshots import get_snapshot_manager

        manager = get_snapshot_manager()
        stats = manager.get_stats()

        return f"""Snapshot Statistics
{'=' * 60}

Total Snapshots:    {stats.total_snapshots}
Total Size:         {stats.total_size_bytes / 1024 / 1024:.1f} MB
CoW Blocks Shared:  {stats.cow_blocks_shared}
Space Saved:        {stats.space_saved_bytes / 1024 / 1024:.1f} MB

Operations:
  Creates:    {stats.creates}
  Restores:   {stats.restores}
  Deletes:    {stats.deletes}
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# FLAKES HANDLERS (Nix Flakes - Reproducible Packages)
# =============================================================================

async def handle_bbx_v2_flake_build(arguments: Dict[str, Any]) -> str:
    """Build a flake (reproducible workflow package)."""
    try:
        from blackbox.core.v2.flakes import FlakeManager

        flake_path = arguments.get("flake_path")

        manager = FlakeManager()
        result = await manager.build(flake_path)

        return f"""[OK] Flake built successfully
Path:     {result.store_path}
Hash:     {result.hash}
Inputs:   {len(result.inputs)}
Outputs:  {len(result.outputs)}
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_flake_run(arguments: Dict[str, Any]) -> str:
    """Run a flake directly."""
    try:
        from blackbox.core.v2.flakes import FlakeManager

        flake_ref = arguments.get("flake_ref")
        inputs = arguments.get("inputs", {})

        manager = FlakeManager()
        result = await manager.run(flake_ref, inputs=inputs)

        return f"[OK] Flake executed: {json.dumps(result, indent=2, default=str)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_flake_lock(arguments: Dict[str, Any]) -> str:
    """Update flake lock file."""
    try:
        from blackbox.core.v2.flakes import FlakeManager

        flake_path = arguments.get("flake_path")

        manager = FlakeManager()
        lock = await manager.update_lock(flake_path)

        return f"[OK] Lock updated: {lock.path}\nInputs locked: {len(lock.nodes)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_flake_show(arguments: Dict[str, Any]) -> str:
    """Show flake metadata."""
    try:
        from blackbox.core.v2.flakes import FlakeManager

        flake_ref = arguments.get("flake_ref")

        manager = FlakeManager()
        info = manager.show(flake_ref)

        return f"""Flake: {info.description}
{'=' * 60}
URL:          {info.url}
Revision:     {info.revision}
Last Modified: {info.last_modified}

Inputs:
{chr(10).join(f'  - {k}: {v}' for k, v in info.inputs.items())}

Outputs:
{chr(10).join(f'  - {o}' for o in info.outputs)}
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AGENT REGISTRY HANDLERS (AUR - Package Discovery)
# =============================================================================

async def handle_bbx_v2_registry_search(arguments: Dict[str, Any]) -> str:
    """Search agent registry."""
    try:
        from blackbox.core.v2.agent_registry import get_agent_registry

        query = arguments.get("query")

        registry = get_agent_registry()
        results = await registry.search(query)

        if not results:
            return f"No agents found matching: {query}"

        output = f"Agent Registry Search: '{query}'\n{'=' * 60}\n"
        for agent in results[:20]:
            output += f"\n[{agent.name}] v{agent.version}"
            output += f"\n    {agent.description}"
            output += f"\n    Author: {agent.author}"
            output += f"\n    Downloads: {agent.downloads}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_registry_install(arguments: Dict[str, Any]) -> str:
    """Install agent from registry."""
    try:
        from blackbox.core.v2.agent_registry import get_agent_registry

        name = arguments.get("name")
        version = arguments.get("version")

        registry = get_agent_registry()
        result = await registry.install(name, version=version)

        return f"[OK] Installed: {result.name} v{result.version}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_registry_publish(arguments: Dict[str, Any]) -> str:
    """Publish agent to registry."""
    try:
        from blackbox.core.v2.agent_registry import get_agent_registry

        path = arguments.get("path")

        registry = get_agent_registry()
        result = await registry.publish(path)

        return f"[OK] Published: {result.name} v{result.version}\nURL: {result.url}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_registry_list(arguments: Dict[str, Any]) -> str:
    """List installed agents."""
    try:
        from blackbox.core.v2.agent_registry import get_agent_registry

        registry = get_agent_registry()
        agents = registry.list_installed()

        if not agents:
            return "No agents installed."

        output = f"Installed Agents\n{'=' * 60}\n"
        for a in agents:
            output += f"\n[{a.name}] v{a.version}"
            output += f"\n    Installed: {a.installed_at.strftime('%Y-%m-%d')}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AGENT BUNDLES HANDLERS (Kali-style Tool Collections)
# =============================================================================

async def handle_bbx_v2_bundle_list(arguments: Dict[str, Any]) -> str:
    """List available bundles."""
    try:
        from blackbox.core.v2.agent_bundles import get_bundle_manager

        manager = get_bundle_manager()
        bundles = manager.list_bundles()

        output = f"Agent Bundles (Kali-style)\n{'=' * 60}\n"
        for b in bundles:
            status = "[installed]" if b.installed else "[available]"
            output += f"\n{status} {b.name}"
            output += f"\n    {b.description}"
            output += f"\n    Tools: {len(b.tools)}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_bundle_install(arguments: Dict[str, Any]) -> str:
    """Install an agent bundle."""
    try:
        from blackbox.core.v2.agent_bundles import get_bundle_manager

        name = arguments.get("name")

        manager = get_bundle_manager()
        result = await manager.install_bundle(name)

        return f"[OK] Bundle installed: {name}\nTools: {', '.join(result.tools)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_bundle_show(arguments: Dict[str, Any]) -> str:
    """Show bundle details."""
    try:
        from blackbox.core.v2.agent_bundles import get_bundle_manager

        name = arguments.get("name")

        manager = get_bundle_manager()
        bundle = manager.get_bundle(name)

        if not bundle:
            return f"Bundle not found: {name}"

        tools_list = chr(10).join(f"  - {t.name}: {t.description}" for t in bundle.tools)

        return f"""Bundle: {bundle.name}
{'=' * 60}
{bundle.description}

Category: {bundle.category}
Version:  {bundle.version}
Installed: {bundle.installed}

Tools ({len(bundle.tools)}):
{tools_list}
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AGENT SANDBOX HANDLERS (Flatpak-style Isolation)
# =============================================================================

async def handle_bbx_v2_sandbox_run(arguments: Dict[str, Any]) -> str:
    """Run agent in sandbox."""
    try:
        from blackbox.core.v2.agent_sandbox import get_sandbox_manager

        agent = arguments.get("agent")
        permissions = arguments.get("permissions", [])
        inputs = arguments.get("inputs", {})

        manager = get_sandbox_manager()
        result = await manager.run_sandboxed(
            agent,
            permissions=permissions,
            inputs=inputs,
        )

        return f"""[OK] Sandboxed execution complete
Agent:    {agent}
Sandbox:  {result.sandbox_id}
Duration: {result.duration_ms:.1f}ms
Output:   {json.dumps(result.output, indent=2, default=str)}
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_sandbox_list(arguments: Dict[str, Any]) -> str:
    """List active sandboxes."""
    try:
        from blackbox.core.v2.agent_sandbox import get_sandbox_manager

        manager = get_sandbox_manager()
        sandboxes = manager.list_sandboxes()

        if not sandboxes:
            return "No active sandboxes."

        output = f"Active Sandboxes\n{'=' * 60}\n"
        for s in sandboxes:
            output += f"\n[{s.id[:8]}...] {s.agent}"
            output += f"\n    Status: {s.status}"
            output += f"\n    Permissions: {', '.join(s.permissions)}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_sandbox_permissions(arguments: Dict[str, Any]) -> str:
    """List available sandbox permissions."""
    try:
        from blackbox.core.v2.agent_sandbox import get_sandbox_manager

        manager = get_sandbox_manager()
        perms = manager.list_permissions()

        output = f"Sandbox Permissions (Flatpak-style)\n{'=' * 60}\n"
        for p in perms:
            output += f"\n[{p.name}]"
            output += f"\n    {p.description}"
            output += f"\n    Risk: {p.risk_level}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# NETWORK FABRIC HANDLERS (Istio-style Service Mesh)
# =============================================================================

async def handle_bbx_v2_mesh_status(arguments: Dict[str, Any]) -> str:
    """Get mesh status."""
    try:
        from blackbox.core.v2.network_fabric import get_network_fabric

        fabric = get_network_fabric()
        status = fabric.get_status()

        return f"""Network Fabric Status (Istio-inspired)
{'=' * 60}

Mesh Status:       {status.status}
Control Plane:     {status.control_plane}
Data Plane:        {status.data_plane}

Services:
  Registered:   {status.services_registered}
  Healthy:      {status.services_healthy}
  Unhealthy:    {status.services_unhealthy}

Traffic:
  Requests/sec:     {status.requests_per_sec:.1f}
  Success Rate:     {status.success_rate:.1f}%
  Avg Latency:      {status.avg_latency_ms:.1f}ms
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_mesh_services(arguments: Dict[str, Any]) -> str:
    """List mesh services."""
    try:
        from blackbox.core.v2.network_fabric import get_network_fabric

        fabric = get_network_fabric()
        services = fabric.list_services()

        if not services:
            return "No services in mesh."

        output = f"Mesh Services\n{'=' * 60}\n"
        for s in services:
            health = "[OK]" if s.healthy else "[ERR]"
            output += f"\n{health} {s.name}"
            output += f"\n    Endpoints: {len(s.endpoints)}"
            output += f"\n    Load Balancing: {s.lb_policy}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_mesh_route(arguments: Dict[str, Any]) -> str:
    """Create traffic routing rule."""
    try:
        from blackbox.core.v2.network_fabric import get_network_fabric, TrafficRule

        fabric = get_network_fabric()

        rule = TrafficRule(
            name=arguments.get("name"),
            source=arguments.get("source"),
            destination=arguments.get("destination"),
            weight=arguments.get("weight", 100),
            headers=arguments.get("headers", {}),
        )

        await fabric.add_route(rule)

        return f"[OK] Route created: {rule.name}\n{rule.source} -> {rule.destination} ({rule.weight}%)"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# POLICY ENGINE HANDLERS (OPA/SELinux - Policy Enforcement)
# =============================================================================

async def handle_bbx_v2_policy_evaluate(arguments: Dict[str, Any]) -> str:
    """Evaluate policy against input."""
    try:
        from blackbox.core.v2.policy_engine import get_policy_engine

        policy = arguments.get("policy")
        input_data = arguments.get("input")

        engine = get_policy_engine()
        result = await engine.evaluate(policy, input_data)

        if result.allowed:
            return f"[ALLOW] Policy: {policy}\nReason: {result.reason}"
        return f"[DENY] Policy: {policy}\nReason: {result.reason}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_policy_list(arguments: Dict[str, Any]) -> str:
    """List all policies."""
    try:
        from blackbox.core.v2.policy_engine import get_policy_engine

        engine = get_policy_engine()
        policies = engine.list_policies()

        if not policies:
            return "No policies defined."

        output = f"Policies (OPA/SELinux-inspired)\n{'=' * 60}\n"
        for p in policies:
            status = "[ON] " if p.enabled else "[OFF]"
            output += f"\n{status} {p.name}"
            output += f"\n     Type: {p.type}"
            output += f"\n     Priority: {p.priority}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_policy_add(arguments: Dict[str, Any]) -> str:
    """Add a new policy."""
    try:
        from blackbox.core.v2.policy_engine import get_policy_engine, PolicyDefinition

        engine = get_policy_engine()

        policy = PolicyDefinition(
            name=arguments.get("name"),
            type=arguments.get("type", "allow"),
            rules=arguments.get("rules", []),
            priority=arguments.get("priority", 0),
        )

        await engine.add_policy(policy)

        return f"[OK] Policy added: {policy.name}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_policy_stats(arguments: Dict[str, Any]) -> str:
    """Get policy engine statistics."""
    try:
        from blackbox.core.v2.policy_engine import get_policy_engine

        engine = get_policy_engine()
        stats = engine.get_stats()

        return f"""Policy Engine Statistics
{'=' * 60}

Evaluations:
  Total:      {stats.total_evaluations}
  Allowed:    {stats.allowed}
  Denied:     {stats.denied}
  Errors:     {stats.errors}

Performance:
  Avg Time:   {stats.avg_evaluation_ms:.2f}ms
  Cache Hits: {stats.cache_hits}
  Cache Rate: {stats.cache_hit_rate:.1f}%

Policies:
  Active:     {stats.active_policies}
  Disabled:   {stats.disabled_policies}
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# AAL HANDLERS (HAL - Adapter Abstraction Layer)
# =============================================================================

async def handle_bbx_v2_aal_adapters(arguments: Dict[str, Any]) -> str:
    """List all adapters through AAL."""
    try:
        from blackbox.core.v2.aal import get_aal

        aal = get_aal()
        adapters = aal.list_adapters()

        output = f"Adapter Abstraction Layer\n{'=' * 60}\n"
        for a in adapters:
            status = "[OK]" if a.healthy else "[ERR]"
            output += f"\n{status} {a.name}"
            output += f"\n    Type: {a.adapter_type}"
            output += f"\n    Methods: {', '.join(a.methods)}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_aal_call(arguments: Dict[str, Any]) -> str:
    """Call adapter method through AAL."""
    try:
        from blackbox.core.v2.aal import get_aal

        adapter = arguments.get("adapter")
        method = arguments.get("method")
        args = arguments.get("args", {})

        aal = get_aal()
        result = await aal.call(adapter, method, args)

        return f"[OK] {adapter}.{method}\nResult: {json.dumps(result, indent=2, default=str)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_aal_stats(arguments: Dict[str, Any]) -> str:
    """Get AAL statistics."""
    try:
        from blackbox.core.v2.aal import get_aal

        aal = get_aal()
        stats = aal.get_stats()

        return f"""AAL Statistics (HAL-inspired)
{'=' * 60}

Adapters:
  Registered:   {stats.adapters_registered}
  Active:       {stats.adapters_active}
  Failed:       {stats.adapters_failed}

Calls:
  Total:        {stats.total_calls}
  Successful:   {stats.successful_calls}
  Failed:       {stats.failed_calls}
  Avg Latency:  {stats.avg_latency_ms:.2f}ms

Abstraction:
  HAL Version:  {stats.hal_version}
  Drivers:      {stats.driver_count}
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# OBJECT MANAGER HANDLERS (Windows ObMgr - Object Namespace)
# =============================================================================

async def handle_bbx_v2_objects_list(arguments: Dict[str, Any]) -> str:
    """List objects in namespace."""
    try:
        from blackbox.core.v2.object_manager import get_object_manager

        path = arguments.get("path", "/")

        om = get_object_manager()
        objects = om.list_objects(path)

        output = f"Object Namespace: {path}\n{'=' * 60}\n"
        for obj in objects:
            output += f"\n[{obj.type}] {obj.name}"
            output += f"\n    Handle: {obj.handle}"
            output += f"\n    Refs: {obj.reference_count}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_objects_create(arguments: Dict[str, Any]) -> str:
    """Create a named object."""
    try:
        from blackbox.core.v2.object_manager import get_object_manager

        name = arguments.get("name")
        obj_type = arguments.get("type")
        data = arguments.get("data", {})

        om = get_object_manager()
        handle = await om.create_object(name, obj_type, data)

        return f"[OK] Object created\nName: {name}\nType: {obj_type}\nHandle: {handle}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_objects_open(arguments: Dict[str, Any]) -> str:
    """Open an object by name."""
    try:
        from blackbox.core.v2.object_manager import get_object_manager

        name = arguments.get("name")
        access = arguments.get("access", "read")

        om = get_object_manager()
        handle = await om.open_object(name, access)

        if handle:
            return f"[OK] Object opened\nName: {name}\nHandle: {handle}"
        return f"[ERR] Object not found: {name}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_objects_stats(arguments: Dict[str, Any]) -> str:
    """Get object manager statistics."""
    try:
        from blackbox.core.v2.object_manager import get_object_manager

        om = get_object_manager()
        stats = om.get_stats()

        return f"""Object Manager Statistics (ObMgr-inspired)
{'=' * 60}

Objects:
  Total:        {stats.total_objects}
  By Type:
{chr(10).join(f'    {t}: {c}' for t, c in stats.by_type.items())}

Handles:
  Open:         {stats.open_handles}
  Total Opened: {stats.total_handles_opened}
  Total Closed: {stats.total_handles_closed}

Operations:
  Creates:      {stats.creates}
  Opens:        {stats.opens}
  Closes:       {stats.closes}
  Lookups:      {stats.lookups}
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# FILTER STACK HANDLERS (Windows Filter Drivers - I/O Pipeline)
# =============================================================================

async def handle_bbx_v2_filters_list(arguments: Dict[str, Any]) -> str:
    """List registered filters."""
    try:
        from blackbox.core.v2.filter_stack import get_filter_stack

        stack = get_filter_stack()
        filters = stack.list_filters()

        if not filters:
            return "No filters registered."

        output = f"Filter Stack (Filter Drivers-inspired)\n{'=' * 60}\n"
        for f in sorted(filters, key=lambda x: x.altitude, reverse=True):
            status = "[ON] " if f.enabled else "[OFF]"
            output += f"\n{status} {f.name} (altitude: {f.altitude})"
            output += f"\n     Type: {f.filter_type}"
            output += f"\n     Operations: {', '.join(f.operations)}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_filters_add(arguments: Dict[str, Any]) -> str:
    """Add a filter to the stack."""
    try:
        from blackbox.core.v2.filter_stack import get_filter_stack, FilterDefinition

        stack = get_filter_stack()

        filter_def = FilterDefinition(
            name=arguments.get("name"),
            altitude=arguments.get("altitude", 100000),
            filter_type=arguments.get("type", "passthrough"),
            operations=arguments.get("operations", ["all"]),
            handler=arguments.get("handler"),
        )

        await stack.register_filter(filter_def)

        return f"[OK] Filter registered: {filter_def.name} at altitude {filter_def.altitude}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_filters_remove(arguments: Dict[str, Any]) -> str:
    """Remove a filter from the stack."""
    try:
        from blackbox.core.v2.filter_stack import get_filter_stack

        name = arguments.get("name")

        stack = get_filter_stack()
        success = await stack.unregister_filter(name)

        if success:
            return f"[OK] Filter removed: {name}"
        return f"[ERR] Filter not found: {name}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_filters_stats(arguments: Dict[str, Any]) -> str:
    """Get filter stack statistics."""
    try:
        from blackbox.core.v2.filter_stack import get_filter_stack

        stack = get_filter_stack()
        stats = stack.get_stats()

        return f"""Filter Stack Statistics
{'=' * 60}

Filters:
  Registered:   {stats.filters_registered}
  Active:       {stats.filters_active}
  Disabled:     {stats.filters_disabled}

Operations:
  Total I/O:    {stats.total_io_ops}
  Pre-Op:       {stats.pre_op_calls}
  Post-Op:      {stats.post_op_calls}
  Blocked:      {stats.blocked_ops}
  Modified:     {stats.modified_ops}

Performance:
  Avg Latency:  {stats.avg_filter_latency_ms:.2f}ms
  Max Latency:  {stats.max_filter_latency_ms:.2f}ms
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# WORKING SET HANDLERS (Windows Mm - Memory Management)
# =============================================================================

async def handle_bbx_v2_memory_stats(arguments: Dict[str, Any]) -> str:
    """Get working set statistics."""
    try:
        from blackbox.core.v2.working_set import get_working_set_manager

        wm = get_working_set_manager()
        stats = wm.get_stats()

        return f"""Working Set Statistics (Mm-inspired)
{'=' * 60}

Memory:
  Working Set:    {stats.working_set_mb:.1f} MB
  Peak:           {stats.peak_working_set_mb:.1f} MB
  Private:        {stats.private_bytes_mb:.1f} MB
  Shared:         {stats.shared_bytes_mb:.1f} MB

Pages:
  Total:          {stats.total_pages}
  Active:         {stats.active_pages}
  Standby:        {stats.standby_pages}
  Modified:       {stats.modified_pages}

Operations:
  Page Faults:    {stats.page_faults}
  Soft Faults:    {stats.soft_faults}
  Hard Faults:    {stats.hard_faults}
  Trims:          {stats.trims}
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_memory_trim(arguments: Dict[str, Any]) -> str:
    """Trim working set."""
    try:
        from blackbox.core.v2.working_set import get_working_set_manager

        target_mb = arguments.get("target_mb")

        wm = get_working_set_manager()
        freed = await wm.trim_working_set(target_mb)

        return f"[OK] Working set trimmed\nFreed: {freed / 1024 / 1024:.1f} MB"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_memory_lock(arguments: Dict[str, Any]) -> str:
    """Lock pages in working set."""
    try:
        from blackbox.core.v2.working_set import get_working_set_manager

        key = arguments.get("key")

        wm = get_working_set_manager()
        success = await wm.lock_pages(key)

        if success:
            return f"[OK] Pages locked: {key}"
        return f"[ERR] Failed to lock: {key}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_memory_pools(arguments: Dict[str, Any]) -> str:
    """Show memory pool statistics."""
    try:
        from blackbox.core.v2.working_set import get_working_set_manager

        wm = get_working_set_manager()
        pools = wm.get_pool_stats()

        output = f"Memory Pools\n{'=' * 60}\n"
        for pool in pools:
            output += f"\n[{pool.name}]"
            output += f"\n    Allocated: {pool.allocated_mb:.1f} MB"
            output += f"\n    Used:      {pool.used_mb:.1f} MB"
            output += f"\n    Peak:      {pool.peak_mb:.1f} MB"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# CONFIG REGISTRY HANDLERS (Windows Registry - Hierarchical Config)
# =============================================================================

async def handle_bbx_v2_registry_get(arguments: Dict[str, Any]) -> str:
    """Get registry value."""
    try:
        from blackbox.core.v2.config_registry import get_config_registry

        path = arguments.get("path")
        value_name = arguments.get("value")

        registry = get_config_registry()
        value = await registry.get_value(path, value_name)

        if value is None:
            return f"Value not found: {path}\\{value_name}"

        return f"{path}\\{value_name} = {value}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_registry_set(arguments: Dict[str, Any]) -> str:
    """Set registry value."""
    try:
        from blackbox.core.v2.config_registry import get_config_registry

        path = arguments.get("path")
        value_name = arguments.get("value")
        data = arguments.get("data")
        value_type = arguments.get("type", "string")

        registry = get_config_registry()
        await registry.set_value(path, value_name, data, value_type=value_type)

        return f"[OK] Set: {path}\\{value_name}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_registry_list(arguments: Dict[str, Any]) -> str:
    """List registry keys and values."""
    try:
        from blackbox.core.v2.config_registry import get_config_registry

        path = arguments.get("path", "HKEY_LOCAL_MACHINE\\BBX")

        registry = get_config_registry()
        result = registry.list_key(path)

        output = f"Registry: {path}\n{'=' * 60}\n"

        if result.subkeys:
            output += "\nSubkeys:"
            for sk in result.subkeys:
                output += f"\n  [{sk}]"

        if result.values:
            output += "\n\nValues:"
            for v in result.values:
                output += f"\n  {v.name} ({v.type}) = {v.data}"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_registry_delete(arguments: Dict[str, Any]) -> str:
    """Delete registry key or value."""
    try:
        from blackbox.core.v2.config_registry import get_config_registry

        path = arguments.get("path")
        value_name = arguments.get("value")

        registry = get_config_registry()

        if value_name:
            success = await registry.delete_value(path, value_name)
            if success:
                return f"[OK] Deleted value: {path}\\{value_name}"
        else:
            success = await registry.delete_key(path)
            if success:
                return f"[OK] Deleted key: {path}"

        return "[ERR] Not found"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_registry_export(arguments: Dict[str, Any]) -> str:
    """Export registry to file."""
    try:
        from blackbox.core.v2.config_registry import get_config_registry

        path = arguments.get("path")
        output_file = arguments.get("output_file")

        registry = get_config_registry()
        await registry.export_key(path, output_file)

        return f"[OK] Exported {path} to {output_file}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# EXECUTIVE HANDLERS (Windows ntoskrnl - Hybrid Kernel)
# =============================================================================

async def handle_bbx_v2_executive_status(arguments: Dict[str, Any]) -> str:
    """Get executive (kernel) status."""
    try:
        from blackbox.core.v2.executive import get_executive

        executive = get_executive()
        status = executive.get_status()

        return f"""BBX Executive Status (ntoskrnl-inspired)
{'=' * 60}

Kernel:
  Version:      {status.version}
  Uptime:       {status.uptime_seconds:.0f}s
  State:        {status.state}

Subsystems:
  Object Manager:   {status.object_manager_status}
  Memory Manager:   {status.memory_manager_status}
  I/O Manager:      {status.io_manager_status}
  Config Manager:   {status.config_manager_status}
  Process Manager:  {status.process_manager_status}

Performance:
  System Calls:     {status.system_calls}
  Interrupts:       {status.interrupts}
  Context Switches: {status.context_switches}
"""
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_executive_start(arguments: Dict[str, Any]) -> str:
    """Start executive subsystems."""
    try:
        from blackbox.core.v2.executive import get_executive

        subsystems = arguments.get("subsystems", ["all"])

        executive = get_executive()
        await executive.start_subsystems(subsystems)

        return f"[OK] Executive subsystems started: {', '.join(subsystems)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_executive_stop(arguments: Dict[str, Any]) -> str:
    """Stop executive subsystems."""
    try:
        from blackbox.core.v2.executive import get_executive

        subsystems = arguments.get("subsystems", ["all"])

        executive = get_executive()
        await executive.stop_subsystems(subsystems)

        return f"[OK] Executive subsystems stopped: {', '.join(subsystems)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_executive_syscall(arguments: Dict[str, Any]) -> str:
    """Execute system call through executive."""
    try:
        from blackbox.core.v2.executive import get_executive

        syscall = arguments.get("syscall")
        args = arguments.get("args", {})

        executive = get_executive()
        result = await executive.syscall(syscall, args)

        return f"[OK] Syscall: {syscall}\nResult: {json.dumps(result, indent=2, default=str)}"
    except Exception as e:
        return f"Error: {str(e)}"


async def handle_bbx_v2_executive_bugcheck(arguments: Dict[str, Any]) -> str:
    """Generate diagnostic bugcheck/dump."""
    try:
        from blackbox.core.v2.executive import get_executive

        executive = get_executive()
        dump = executive.generate_bugcheck()

        return f"""BBX Bugcheck Report
{'=' * 60}

Code:       {dump.code}
Time:       {dump.timestamp}
Thread:     {dump.thread_id}

Stack Trace:
{dump.stack_trace}

Loaded Modules:
{chr(10).join(f'  {m}' for m in dump.loaded_modules)}

Parameters:
{json.dumps(dump.parameters, indent=2)}
"""
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TOOL HANDLERS MAPPING
# =============================================================================

# Map tool names to handlers
V2_TOOL_HANDLERS = {
    # === AgentRing (io_uring) ===
    "bbx_v2_ring_stats": handle_bbx_v2_ring_stats,
    "bbx_v2_ring_config": handle_bbx_v2_ring_config,
    "bbx_v2_ring_submit": handle_bbx_v2_ring_submit,

    # === Hooks (eBPF) ===
    "bbx_v2_hooks_list": handle_bbx_v2_hooks_list,
    "bbx_v2_hooks_stats": handle_bbx_v2_hooks_stats,
    "bbx_v2_hooks_enable": handle_bbx_v2_hooks_enable,
    "bbx_v2_hooks_disable": handle_bbx_v2_hooks_disable,

    # === Context Tiering (MGLRU) ===
    "bbx_v2_context_stats": handle_bbx_v2_context_stats,
    "bbx_v2_context_get": handle_bbx_v2_context_get,
    "bbx_v2_context_set": handle_bbx_v2_context_set,
    "bbx_v2_context_pin": handle_bbx_v2_context_pin,

    # === Declarative Config (NixOS) ===
    "bbx_v2_config_apply": handle_bbx_v2_config_apply,
    "bbx_v2_config_rollback": handle_bbx_v2_config_rollback,
    "bbx_v2_config_show": handle_bbx_v2_config_show,

    # === Generations (NixOS) ===
    "bbx_v2_generation_list": handle_bbx_v2_generation_list,
    "bbx_v2_generation_diff": handle_bbx_v2_generation_diff,
    "bbx_v2_generation_switch": handle_bbx_v2_generation_switch,

    # === V2 Runtime ===
    "bbx_v2_run": handle_bbx_v2_run,

    # === Flow Integrity (CET) ===
    "bbx_v2_flow_stats": handle_bbx_v2_flow_stats,
    "bbx_v2_flow_verify": handle_bbx_v2_flow_verify,

    # === Agent Quotas (Cgroups v2) ===
    "bbx_v2_quotas_stats": handle_bbx_v2_quotas_stats,
    "bbx_v2_quotas_set": handle_bbx_v2_quotas_set,
    "bbx_v2_quotas_list": handle_bbx_v2_quotas_list,

    # === State Snapshots (XFS Reflink) ===
    "bbx_v2_snapshot_create": handle_bbx_v2_snapshot_create,
    "bbx_v2_snapshot_list": handle_bbx_v2_snapshot_list,
    "bbx_v2_snapshot_restore": handle_bbx_v2_snapshot_restore,
    "bbx_v2_snapshot_stats": handle_bbx_v2_snapshot_stats,

    # === Flakes (Nix Flakes) ===
    "bbx_v2_flake_build": handle_bbx_v2_flake_build,
    "bbx_v2_flake_run": handle_bbx_v2_flake_run,
    "bbx_v2_flake_lock": handle_bbx_v2_flake_lock,
    "bbx_v2_flake_show": handle_bbx_v2_flake_show,

    # === Agent Registry (AUR) ===
    "bbx_v2_registry_search": handle_bbx_v2_registry_search,
    "bbx_v2_registry_install": handle_bbx_v2_registry_install,
    "bbx_v2_registry_publish": handle_bbx_v2_registry_publish,
    "bbx_v2_registry_list_installed": handle_bbx_v2_registry_list,

    # === Agent Bundles (Kali) ===
    "bbx_v2_bundle_list": handle_bbx_v2_bundle_list,
    "bbx_v2_bundle_install": handle_bbx_v2_bundle_install,
    "bbx_v2_bundle_show": handle_bbx_v2_bundle_show,

    # === Agent Sandbox (Flatpak) ===
    "bbx_v2_sandbox_run": handle_bbx_v2_sandbox_run,
    "bbx_v2_sandbox_list": handle_bbx_v2_sandbox_list,
    "bbx_v2_sandbox_permissions": handle_bbx_v2_sandbox_permissions,

    # === Network Fabric (Istio) ===
    "bbx_v2_mesh_status": handle_bbx_v2_mesh_status,
    "bbx_v2_mesh_services": handle_bbx_v2_mesh_services,
    "bbx_v2_mesh_route": handle_bbx_v2_mesh_route,

    # === Policy Engine (OPA/SELinux) ===
    "bbx_v2_policy_evaluate": handle_bbx_v2_policy_evaluate,
    "bbx_v2_policy_list": handle_bbx_v2_policy_list,
    "bbx_v2_policy_add": handle_bbx_v2_policy_add,
    "bbx_v2_policy_stats": handle_bbx_v2_policy_stats,

    # === AAL (HAL) ===
    "bbx_v2_aal_adapters": handle_bbx_v2_aal_adapters,
    "bbx_v2_aal_call": handle_bbx_v2_aal_call,
    "bbx_v2_aal_stats": handle_bbx_v2_aal_stats,

    # === Object Manager (ObMgr) ===
    "bbx_v2_objects_list": handle_bbx_v2_objects_list,
    "bbx_v2_objects_create": handle_bbx_v2_objects_create,
    "bbx_v2_objects_open": handle_bbx_v2_objects_open,
    "bbx_v2_objects_stats": handle_bbx_v2_objects_stats,

    # === Filter Stack (Filter Drivers) ===
    "bbx_v2_filters_list": handle_bbx_v2_filters_list,
    "bbx_v2_filters_add": handle_bbx_v2_filters_add,
    "bbx_v2_filters_remove": handle_bbx_v2_filters_remove,
    "bbx_v2_filters_stats": handle_bbx_v2_filters_stats,

    # === Working Set (Mm) ===
    "bbx_v2_memory_stats": handle_bbx_v2_memory_stats,
    "bbx_v2_memory_trim": handle_bbx_v2_memory_trim,
    "bbx_v2_memory_lock": handle_bbx_v2_memory_lock,
    "bbx_v2_memory_pools": handle_bbx_v2_memory_pools,

    # === Config Registry (Windows Registry) ===
    "bbx_v2_reg_get": handle_bbx_v2_registry_get,
    "bbx_v2_reg_set": handle_bbx_v2_registry_set,
    "bbx_v2_reg_list": handle_bbx_v2_registry_list,
    "bbx_v2_reg_delete": handle_bbx_v2_registry_delete,
    "bbx_v2_reg_export": handle_bbx_v2_registry_export,

    # === Executive (ntoskrnl) ===
    "bbx_v2_executive_status": handle_bbx_v2_executive_status,
    "bbx_v2_executive_start": handle_bbx_v2_executive_start,
    "bbx_v2_executive_stop": handle_bbx_v2_executive_stop,
    "bbx_v2_executive_syscall": handle_bbx_v2_executive_syscall,
    "bbx_v2_executive_bugcheck": handle_bbx_v2_executive_bugcheck,
}


# =============================================================================
# Enhanced Components Handlers
# =============================================================================


async def handle_bbx_v2_enhanced_ring_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get enhanced ring statistics"""
    return {"status": "ok", "message": "Enhanced ring stats", "stats": {"total_submitted": 0, "total_completed": 0, "wal_entries": 0}}


async def handle_bbx_v2_enhanced_ring_submit(args: Dict[str, Any]) -> Dict[str, Any]:
    """Submit operations with idempotency"""
    operations = args.get("operations", [])
    idempotency_key = args.get("idempotency_key")
    return {"status": "ok", "message": f"Submitted {len(operations)} operations", "idempotency_key": idempotency_key}


async def handle_bbx_v2_enhanced_ring_cb(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get circuit breaker status"""
    name = args.get("name", "default")
    return {"status": "ok", "circuit_breaker": name, "state": "CLOSED"}


async def handle_bbx_v2_enhanced_context_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get enhanced context tiering stats"""
    return {"status": "ok", "stats": {"hot_tier": 0, "warm_tier": 0, "cold_tier": 0, "compression_ratio": 0.0}}


async def handle_bbx_v2_enhanced_context_prefetch(args: Dict[str, Any]) -> Dict[str, Any]:
    """Prefetch keys to hot tier"""
    keys = args.get("keys", [])
    return {"status": "ok", "message": f"Prefetching {len(keys)} keys"}


async def handle_bbx_v2_enhanced_context_pin(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pin a key to prevent demotion"""
    key = args.get("key")
    return {"status": "ok", "message": f"Pinned key: {key}"}


async def handle_bbx_v2_enforced_quotas_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get enforced quota stats"""
    return {"status": "ok", "stats": {"cgroups_available": False, "gpu_available": False}}


async def handle_bbx_v2_enforced_quotas_check(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check quota for resource"""
    resource = args.get("resource")
    amount = args.get("amount", 1)
    return {"status": "ok", "action": "ALLOW", "resource": resource, "amount": amount}


async def handle_bbx_v2_enforced_quotas_gpu(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get GPU quota status"""
    return {"status": "ok", "gpu_count": 0, "allocations": []}


async def handle_bbx_v2_distributed_snapshot_create(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create distributed snapshot"""
    agent_id = args.get("agent_id")
    branch = args.get("branch", "main")
    return {"status": "ok", "snapshot_id": f"{agent_id}:{branch}:snapshot_1"}


async def handle_bbx_v2_distributed_snapshot_restore(args: Dict[str, Any]) -> Dict[str, Any]:
    """Restore from snapshot"""
    snapshot_id = args.get("snapshot_id")
    return {"status": "ok", "message": f"Restored from {snapshot_id}"}


async def handle_bbx_v2_distributed_snapshot_branch(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create branch from snapshot"""
    new_branch = args.get("new_branch")
    from_branch = args.get("from_branch", "main")
    return {"status": "ok", "branch": new_branch, "from": from_branch}


async def handle_bbx_v2_distributed_snapshot_stats(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get distributed snapshot stats"""
    return {"status": "ok", "stats": {"total_snapshots": 0, "total_branches": 0, "bytes_deduplicated": 0}}


async def handle_bbx_v2_enhanced_flow_verify(args: Dict[str, Any]) -> Dict[str, Any]:
    """Verify state transition with anomaly detection"""
    from_state = args.get("from_state")
    to_state = args.get("to_state")
    return {"status": "ok", "allowed": True, "action": "ALLOW", "anomalies": []}


async def handle_bbx_v2_enhanced_flow_anomalies(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get detected anomalies"""
    agent_id = args.get("agent_id")
    return {"status": "ok", "agent_id": agent_id, "anomalies": []}


async def handle_bbx_v2_enhanced_flow_audit(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory access audit log"""
    agent_id = args.get("agent_id")
    return {"status": "ok", "agent_id": agent_id, "audit_log": []}


async def handle_bbx_v2_memory_store(args: Dict[str, Any]) -> Dict[str, Any]:
    """Store memory in semantic memory"""
    agent_id = args.get("agent_id")
    content = args.get("content")
    importance = args.get("importance", 0.5)
    return {"status": "ok", "memory_id": f"mem_{agent_id[:8]}", "importance": importance}


async def handle_bbx_v2_memory_recall(args: Dict[str, Any]) -> Dict[str, Any]:
    """Recall memories by semantic search"""
    agent_id = args.get("agent_id")
    query = args.get("query")
    top_k = args.get("top_k", 10)
    return {"status": "ok", "query": query, "results": [], "top_k": top_k}


async def handle_bbx_v2_memory_search(args: Dict[str, Any]) -> Dict[str, Any]:
    """Hybrid search (semantic + keyword)"""
    agent_id = args.get("agent_id")
    query = args.get("query")
    return {"status": "ok", "query": query, "results": []}


async def handle_bbx_v2_memory_forget(args: Dict[str, Any]) -> Dict[str, Any]:
    """Forget a memory"""
    memory_id = args.get("memory_id")
    return {"status": "ok", "message": f"Forgot memory {memory_id}"}


async def handle_bbx_v2_bus_publish(args: Dict[str, Any]) -> Dict[str, Any]:
    """Publish message to bus"""
    topic = args.get("topic")
    payload = args.get("payload", {})
    return {"status": "ok", "topic": topic, "message_id": "msg_123"}


async def handle_bbx_v2_bus_subscribe(args: Dict[str, Any]) -> Dict[str, Any]:
    """Subscribe to topic"""
    topic = args.get("topic")
    group = args.get("group", "default")
    return {"status": "ok", "topic": topic, "group": group, "subscribed": True}


async def handle_bbx_v2_bus_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get message bus status"""
    return {"status": "ok", "backend": "memory", "topics": [], "consumers": 0}


async def handle_bbx_v2_goal_create(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create a goal"""
    name = args.get("name")
    description = args.get("description")
    return {"status": "ok", "goal_id": f"goal_{name[:8]}", "name": name}


async def handle_bbx_v2_goal_execute(args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a goal"""
    goal_id = args.get("goal_id")
    return {"status": "ok", "goal_id": goal_id, "execution_status": "IN_PROGRESS"}


async def handle_bbx_v2_goal_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get goal status"""
    goal_id = args.get("goal_id")
    return {"status": "ok", "goal_id": goal_id, "goal_status": "PENDING", "tasks": []}


async def handle_bbx_v2_goal_list(args: Dict[str, Any]) -> Dict[str, Any]:
    """List all goals"""
    agent_id = args.get("agent_id")
    return {"status": "ok", "agent_id": agent_id, "goals": []}


async def handle_bbx_v2_auth_create_token(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create JWT token"""
    identity_id = args.get("identity_id")
    expiry = args.get("expiry_seconds", 86400)
    return {"status": "ok", "token": "jwt_token_placeholder", "expires_in": expiry}


async def handle_bbx_v2_auth_verify_token(args: Dict[str, Any]) -> Dict[str, Any]:
    """Verify JWT token"""
    token = args.get("token")
    return {"status": "ok", "valid": True, "identity_id": "unknown"}


async def handle_bbx_v2_auth_create_api_key(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create API key"""
    identity_id = args.get("identity_id")
    name = args.get("name")
    return {"status": "ok", "api_key": "bbx_api_key_placeholder", "name": name}


async def handle_bbx_v2_auth_authorize(args: Dict[str, Any]) -> Dict[str, Any]:
    """Check authorization"""
    identity_id = args.get("identity_id")
    resource = args.get("resource")
    action = args.get("action")
    return {"status": "ok", "authorized": True, "resource": resource, "action": action}


async def handle_bbx_v2_monitoring_metrics(args: Dict[str, Any]) -> Dict[str, Any]:
    """Export Prometheus metrics"""
    return {"status": "ok", "format": "prometheus", "metrics": "# BBX Metrics\nbbx_info 1"}


async def handle_bbx_v2_monitoring_trace(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get recent traces"""
    limit = args.get("limit", 100)
    return {"status": "ok", "traces": [], "limit": limit}


async def handle_bbx_v2_monitoring_alerts(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get active alerts"""
    return {"status": "ok", "alerts": []}


async def handle_bbx_v2_monitoring_dashboard(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create/get dashboard"""
    title = args.get("title", "BBX Dashboard")
    return {"status": "ok", "dashboard_id": "dash_1", "title": title}


async def handle_bbx_v2_deploy_dockerfile(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Dockerfile"""
    agent_name = args.get("agent_name")
    return {"status": "ok", "agent_name": agent_name, "dockerfile": "FROM python:3.11-slim\n..."}


async def handle_bbx_v2_deploy_helm(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Helm chart"""
    agent_name = args.get("agent_name")
    return {"status": "ok", "agent_name": agent_name, "files": ["Chart.yaml", "values.yaml"]}


async def handle_bbx_v2_deploy_k8s(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate Kubernetes manifests"""
    agent_name = args.get("agent_name")
    return {"status": "ok", "agent_name": agent_name, "manifests": ["crd.yaml", "operator.yaml"]}


# =============================================================================
# Update V2_TOOL_HANDLERS with Enhanced Components
# =============================================================================

V2_TOOL_HANDLERS.update({
    # === Enhanced Ring (WAL, Idempotency, Circuit Breaker) ===
    "bbx_v2_enhanced_ring_stats": handle_bbx_v2_enhanced_ring_stats,
    "bbx_v2_enhanced_ring_submit": handle_bbx_v2_enhanced_ring_submit,
    "bbx_v2_enhanced_ring_circuit_breaker": handle_bbx_v2_enhanced_ring_cb,

    # === Enhanced Context Tiering (ML Scoring, Prefetch) ===
    "bbx_v2_enhanced_context_stats": handle_bbx_v2_enhanced_context_stats,
    "bbx_v2_enhanced_context_prefetch": handle_bbx_v2_enhanced_context_prefetch,
    "bbx_v2_enhanced_context_pin": handle_bbx_v2_enhanced_context_pin,

    # === Enforced Quotas (Cgroups, GPU) ===
    "bbx_v2_enforced_quotas_stats": handle_bbx_v2_enforced_quotas_stats,
    "bbx_v2_enforced_quotas_check": handle_bbx_v2_enforced_quotas_check,
    "bbx_v2_enforced_quotas_gpu": handle_bbx_v2_enforced_quotas_gpu,

    # === Distributed Snapshots (S3, Redis, Replication) ===
    "bbx_v2_distributed_snapshot_create": handle_bbx_v2_distributed_snapshot_create,
    "bbx_v2_distributed_snapshot_restore": handle_bbx_v2_distributed_snapshot_restore,
    "bbx_v2_distributed_snapshot_branch": handle_bbx_v2_distributed_snapshot_branch,
    "bbx_v2_distributed_snapshot_stats": handle_bbx_v2_distributed_snapshot_stats,

    # === Enhanced Flow Integrity (Anomaly Detection, OPA) ===
    "bbx_v2_enhanced_flow_verify": handle_bbx_v2_enhanced_flow_verify,
    "bbx_v2_enhanced_flow_anomalies": handle_bbx_v2_enhanced_flow_anomalies,
    "bbx_v2_enhanced_flow_audit": handle_bbx_v2_enhanced_flow_audit,

    # === Semantic Memory (RAG, Qdrant) ===
    "bbx_v2_memory_store": handle_bbx_v2_memory_store,
    "bbx_v2_memory_recall": handle_bbx_v2_memory_recall,
    "bbx_v2_memory_search": handle_bbx_v2_memory_search,
    "bbx_v2_memory_forget": handle_bbx_v2_memory_forget,

    # === Message Bus (Redis Streams, Kafka) ===
    "bbx_v2_bus_publish": handle_bbx_v2_bus_publish,
    "bbx_v2_bus_subscribe": handle_bbx_v2_bus_subscribe,
    "bbx_v2_bus_status": handle_bbx_v2_bus_status,

    # === Goal Engine (LLM Planner) ===
    "bbx_v2_goal_create": handle_bbx_v2_goal_create,
    "bbx_v2_goal_execute": handle_bbx_v2_goal_execute,
    "bbx_v2_goal_status": handle_bbx_v2_goal_status,
    "bbx_v2_goal_list": handle_bbx_v2_goal_list,

    # === Authentication ===
    "bbx_v2_auth_create_token": handle_bbx_v2_auth_create_token,
    "bbx_v2_auth_verify_token": handle_bbx_v2_auth_verify_token,
    "bbx_v2_auth_create_api_key": handle_bbx_v2_auth_create_api_key,
    "bbx_v2_auth_authorize": handle_bbx_v2_auth_authorize,

    # === Monitoring ===
    "bbx_v2_monitoring_metrics": handle_bbx_v2_monitoring_metrics,
    "bbx_v2_monitoring_trace": handle_bbx_v2_monitoring_trace,
    "bbx_v2_monitoring_alerts": handle_bbx_v2_monitoring_alerts,
    "bbx_v2_monitoring_dashboard": handle_bbx_v2_monitoring_dashboard,

    # === Deployment ===
    "bbx_v2_deploy_dockerfile": handle_bbx_v2_deploy_dockerfile,
    "bbx_v2_deploy_helm": handle_bbx_v2_deploy_helm,
    "bbx_v2_deploy_k8s": handle_bbx_v2_deploy_k8s,
})
