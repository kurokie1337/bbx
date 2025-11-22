# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sqlite3
import json
from typing import Dict, Any
from blackbox.core.base_adapter import MCPAdapter

DB_PATH = "blackbox.db"

class QueueAdapter(MCPAdapter):
    """
    Adapter for message queuing using SQLite.
    Enables asynchronous task processing and decoupling.
    """

    def __init__(self):
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
            """)

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        if method == "push":
            return self._push(inputs)
        elif method == "pop":
            return self._pop(inputs)
        elif method == "peek":
            return self._peek(inputs)
        else:
            raise ValueError(f"Unknown queue method: {method}")

    def _push(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        queue_name = inputs.get("queue")
        payload = inputs.get("payload")

        if not queue_name or payload is None:
            raise ValueError("Queue name and payload are required")

        if not isinstance(payload, str):
            payload = json.dumps(payload)

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO queues (queue_name, payload) VALUES (?, ?)",
                (queue_name, payload)
            )
            job_id = cursor.lastrowid

        return {"status": "queued", "job_id": job_id, "queue": queue_name}

    def _pop(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        queue_name = inputs.get("queue")

        if not queue_name:
            raise ValueError("Queue name is required")

        with sqlite3.connect(DB_PATH) as conn:
            # Simple FIFO pop: find oldest pending
            cursor = conn.execute(
                """
                SELECT id, payload FROM queues
                WHERE queue_name = ? AND status = 'pending'
                ORDER BY id ASC LIMIT 1
                """,
                (queue_name,)
            )
            row = cursor.fetchone()

            if row:
                job_id, payload_str = row
                # Mark as processing/done (for simplicity we just delete or mark done)
                # In a real system we'd have a 'processing' state and a timeout.
                # Here we'll just mark it 'processed' to simulate consumption.
                conn.execute(
                    "UPDATE queues SET status = 'processed', processed_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (job_id,)
                )

                try:
                    payload = json.loads(payload_str)
                except Exception:
                    payload = payload_str

                return {"status": "found", "job_id": job_id, "payload": payload}
            else:
                return {"status": "empty"}

    def _peek(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        queue_name = inputs.get("queue")

        if not queue_name:
            raise ValueError("Queue name is required")

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                """
                SELECT count(*) FROM queues
                WHERE queue_name = ? AND status = 'pending'
                """,
                (queue_name,)
            )
            count = cursor.fetchone()[0]

        return {"queue": queue_name, "pending_count": count}
