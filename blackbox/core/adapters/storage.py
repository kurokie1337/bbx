# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

import json
import sqlite3
from typing import Any, Dict

from blackbox.core.base_adapter import MCPAdapter

DB_PATH = "blackbox.db"


class StorageAdapter(MCPAdapter):
    """
    Adapter for persistent storage using SQLite.
    Provides Key-Value store capabilities.
    """

    def __init__(self):
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        if method == "kv.set":
            return self._kv_set(inputs)
        elif method == "kv.get":
            return self._kv_get(inputs)
        elif method == "kv.delete":
            return self._kv_delete(inputs)
        else:
            raise ValueError(f"Unknown storage method: {method}")

    def _kv_set(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        key = inputs.get("key")
        value = inputs.get("value")

        if not key:
            raise ValueError("Key is required")

        # Serialize value if it's not a string
        if not isinstance(value, str):
            value = json.dumps(value)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                (key, value),
            )

        return {"status": "success", "key": key}

    def _kv_get(self, inputs: Dict[str, Any]) -> Any:
        key = inputs.get("key")
        default = inputs.get("default")

        if not key:
            raise ValueError("Key is required")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()

        if row:
            val = row[0]
            # Try to deserialize JSON
            try:
                return json.loads(val)
            except Exception:
                return val
        return default

    def _kv_delete(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        key = inputs.get("key")

        if not key:
            raise ValueError("Key is required")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))

        return {"status": "deleted", "key": key}
