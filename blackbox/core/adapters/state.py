# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX State Adapter - Persistent State Management

Provides persistent key-value state storage for AI agents:
- Global workspace state
- Workflow-scoped state (namespace per workflow)
- Atomic operations (increment, append, merge)
- Memory/context for AI agents

This is like environment variables + configuration in Linux,
but designed for AI agent memory and context persistence.

Examples:
    # Set a value
    - id: save_config
      use: state.set
      inputs:
        key: user_preference
        value: dark_mode

    # Get a value
    - id: load_config
      use: state.get
      inputs:
        key: user_preference
        default: light_mode

    # Increment a counter
    - id: increment_runs
      use: state.increment
      inputs:
        key: run_count
        by: 1

    # Append to a list (agent memory)
    - id: remember
      use: state.append
      inputs:
        key: conversation_history
        value:
          role: assistant
          content: "Completed deploy"

    # Get workflow-scoped state
    - id: get_local
      use: state.get
      inputs:
        key: last_result
        namespace: deploy_workflow

    # List all keys
    - id: list_keys
      use: state.keys
      inputs:
        pattern: "user_*"

    # Delete a key
    - id: cleanup
      use: state.delete
      inputs:
        key: temp_data
"""

import fnmatch
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bbx.adapters.state")


class StateAdapter:
    """
    BBX Adapter for persistent state management.

    Provides key-value storage with:
    - Workspace-level persistence
    - Optional namespacing (per-workflow state)
    - Atomic operations
    - List/dict merge operations
    - Pattern-based key listing
    """

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute state method"""

        if method == "get":
            return await self._get(
                key=inputs["key"],
                default=inputs.get("default"),
                namespace=inputs.get("namespace"),
            )

        elif method == "set":
            return await self._set(
                key=inputs["key"],
                value=inputs["value"],
                namespace=inputs.get("namespace"),
            )

        elif method == "delete":
            return await self._delete(
                key=inputs["key"],
                namespace=inputs.get("namespace"),
            )

        elif method == "exists":
            return await self._exists(
                key=inputs["key"],
                namespace=inputs.get("namespace"),
            )

        elif method == "keys":
            return await self._keys(
                pattern=inputs.get("pattern", "*"),
                namespace=inputs.get("namespace"),
            )

        elif method == "all":
            return await self._all(
                namespace=inputs.get("namespace"),
            )

        elif method == "clear":
            return await self._clear(
                namespace=inputs.get("namespace"),
            )

        elif method == "increment":
            return await self._increment(
                key=inputs["key"],
                by=inputs.get("by", 1),
                namespace=inputs.get("namespace"),
            )

        elif method == "append":
            return await self._append(
                key=inputs["key"],
                value=inputs["value"],
                max_items=inputs.get("max_items"),
                namespace=inputs.get("namespace"),
            )

        elif method == "merge":
            return await self._merge(
                key=inputs["key"],
                value=inputs["value"],
                namespace=inputs.get("namespace"),
            )

        elif method == "pop":
            return await self._pop(
                key=inputs["key"],
                default=inputs.get("default"),
                namespace=inputs.get("namespace"),
            )

        else:
            raise ValueError(f"Unknown state method: {method}")

    def _get_state_file(self, namespace: Optional[str] = None) -> Path:
        """Get path to state file"""
        from blackbox.core.workspace_manager import get_current_workspace
        from blackbox.core.config import get_config

        workspace = get_current_workspace()

        if workspace:
            state_dir = workspace.paths.state_dir
        else:
            # Fall back to global state
            config = get_config()
            state_dir = config.paths.bbx_home / "state"
            state_dir.mkdir(parents=True, exist_ok=True)

        if namespace:
            # Sanitize namespace for filename
            safe_namespace = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)
            return state_dir / f"{safe_namespace}.json"
        else:
            return state_dir / "vars.json"

    def _load_state(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Load state from file"""
        state_file = self._get_state_file(namespace)

        if not state_file.exists():
            return {}

        try:
            return json.loads(state_file.read_text())
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            return {}

    def _save_state(self, state: Dict[str, Any], namespace: Optional[str] = None) -> None:
        """Save state to file"""
        state_file = self._get_state_file(namespace)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2, default=str))

    async def _get(
        self,
        key: str,
        default: Any = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a state value"""
        state = self._load_state(namespace)
        value = state.get(key, default)

        return {
            "key": key,
            "value": value,
            "found": key in state,
            "namespace": namespace,
        }

    async def _set(
        self,
        key: str,
        value: Any,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set a state value"""
        state = self._load_state(namespace)
        old_value = state.get(key)
        state[key] = value
        self._save_state(state, namespace)

        return {
            "key": key,
            "value": value,
            "old_value": old_value,
            "namespace": namespace,
        }

    async def _delete(
        self,
        key: str,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a state key"""
        state = self._load_state(namespace)

        if key in state:
            old_value = state.pop(key)
            self._save_state(state, namespace)
            return {
                "key": key,
                "deleted": True,
                "old_value": old_value,
                "namespace": namespace,
            }

        return {
            "key": key,
            "deleted": False,
            "namespace": namespace,
        }

    async def _exists(
        self,
        key: str,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Check if a key exists"""
        state = self._load_state(namespace)

        return {
            "key": key,
            "exists": key in state,
            "namespace": namespace,
        }

    async def _keys(
        self,
        pattern: str = "*",
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List keys matching pattern"""
        state = self._load_state(namespace)

        matching_keys = [
            key for key in state.keys()
            if fnmatch.fnmatch(key, pattern)
        ]

        return {
            "pattern": pattern,
            "keys": sorted(matching_keys),
            "count": len(matching_keys),
            "namespace": namespace,
        }

    async def _all(
        self,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get all state"""
        state = self._load_state(namespace)

        return {
            "state": state,
            "keys_count": len(state),
            "namespace": namespace,
        }

    async def _clear(
        self,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clear all state"""
        state = self._load_state(namespace)
        keys_count = len(state)

        self._save_state({}, namespace)

        return {
            "cleared": True,
            "keys_removed": keys_count,
            "namespace": namespace,
        }

    async def _increment(
        self,
        key: str,
        by: int = 1,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Increment a numeric value (atomic)"""
        state = self._load_state(namespace)

        old_value = state.get(key, 0)
        if not isinstance(old_value, (int, float)):
            return {
                "key": key,
                "error": f"Cannot increment non-numeric value: {type(old_value).__name__}",
                "namespace": namespace,
            }

        new_value = old_value + by
        state[key] = new_value
        self._save_state(state, namespace)

        return {
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "by": by,
            "namespace": namespace,
        }

    async def _append(
        self,
        key: str,
        value: Any,
        max_items: Optional[int] = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append to a list (great for agent memory/history)"""
        state = self._load_state(namespace)

        current = state.get(key, [])
        if not isinstance(current, list):
            return {
                "key": key,
                "error": f"Cannot append to non-list value: {type(current).__name__}",
                "namespace": namespace,
            }

        current.append(value)

        # Trim if max_items specified (keep most recent)
        if max_items and len(current) > max_items:
            current = current[-max_items:]

        state[key] = current
        self._save_state(state, namespace)

        return {
            "key": key,
            "appended": value,
            "list_length": len(current),
            "namespace": namespace,
        }

    async def _merge(
        self,
        key: str,
        value: Dict[str, Any],
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge a dict into existing dict value"""
        state = self._load_state(namespace)

        current = state.get(key, {})
        if not isinstance(current, dict):
            return {
                "key": key,
                "error": f"Cannot merge into non-dict value: {type(current).__name__}",
                "namespace": namespace,
            }

        # Deep merge
        current.update(value)
        state[key] = current
        self._save_state(state, namespace)

        return {
            "key": key,
            "merged": value,
            "result": current,
            "namespace": namespace,
        }

    async def _pop(
        self,
        key: str,
        default: Any = None,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get and delete a value (useful for queues)"""
        state = self._load_state(namespace)

        if key in state:
            value = state.pop(key)
            self._save_state(state, namespace)
            return {
                "key": key,
                "value": value,
                "found": True,
                "namespace": namespace,
            }

        return {
            "key": key,
            "value": default,
            "found": False,
            "namespace": namespace,
        }
