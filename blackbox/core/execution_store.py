# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Execution Store

SQLite-based persistence for workflow executions.
Tracks running, completed, and failed workflows.

This is like /proc in Linux - provides information about running processes.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config

logger = logging.getLogger("bbx.execution_store")


class ExecutionStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Execution:
    """Represents a workflow execution (like a process in Linux)"""
    execution_id: str
    workflow_path: str
    workflow_id: str
    status: ExecutionStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    inputs: Optional[Dict[str, Any]] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    workspace_path: Optional[str] = None
    pid: Optional[int] = None  # Process ID if running in subprocess
    background: bool = False

    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate duration in milliseconds"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None

    @property
    def is_running(self) -> bool:
        return self.status == ExecutionStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        return self.status in (
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_path": self.workflow_path,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "workspace_path": self.workspace_path,
            "pid": self.pid,
            "background": self.background,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "Execution":
        """Create Execution from database row"""
        return cls(
            execution_id=row[0],
            workflow_path=row[1],
            workflow_id=row[2],
            status=ExecutionStatus(row[3]),
            created_at=datetime.fromisoformat(row[4]),
            started_at=datetime.fromisoformat(row[5]) if row[5] else None,
            completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
            inputs=json.loads(row[7]) if row[7] else None,
            outputs=json.loads(row[8]) if row[8] else None,
            error=row[9],
            workspace_path=row[10],
            pid=row[11],
            background=bool(row[12]),
        )


class ExecutionStore:
    """
    SQLite-based storage for workflow executions.

    Like /proc filesystem in Linux, but persistent.

    Usage:
        store = ExecutionStore()

        # Create new execution
        exec = store.create(
            execution_id="abc123",
            workflow_path="deploy.bbx",
            workflow_id="deploy"
        )

        # Update status
        store.update_status("abc123", ExecutionStatus.RUNNING)

        # Get all running
        running = store.list(status=ExecutionStatus.RUNNING)

        # Get by ID
        exec = store.get("abc123")
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize ExecutionStore.

        Args:
            db_path: Path to SQLite database.
                    Defaults to ~/.bbx/executions.db
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            config = get_config()
            self.db_path = config.paths.bbx_home / "executions.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    workflow_path TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    inputs TEXT,
                    outputs TEXT,
                    error TEXT,
                    workspace_path TEXT,
                    pid INTEGER,
                    background INTEGER DEFAULT 0
                )
            """)

            # Index for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_status
                ON executions(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_created
                ON executions(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_workspace
                ON executions(workspace_path)
            """)

            # Logs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    execution_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    step_id TEXT,
                    data TEXT,
                    FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_execution
                ON execution_logs(execution_id)
            """)

            conn.commit()

    @contextmanager
    def _connect(self):
        """Context manager for database connection"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    # ==========================================================================
    # CRUD Operations
    # ==========================================================================

    def create(
        self,
        execution_id: str,
        workflow_path: str,
        workflow_id: str,
        inputs: Optional[Dict[str, Any]] = None,
        workspace_path: Optional[str] = None,
        background: bool = False,
    ) -> Execution:
        """Create a new execution record"""
        execution = Execution(
            execution_id=execution_id,
            workflow_path=workflow_path,
            workflow_id=workflow_id,
            status=ExecutionStatus.PENDING,
            created_at=datetime.now(),
            inputs=inputs,
            workspace_path=workspace_path,
            background=background,
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO executions
                (execution_id, workflow_path, workflow_id, status, created_at,
                 inputs, workspace_path, background)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution.execution_id,
                    execution.workflow_path,
                    execution.workflow_id,
                    execution.status.value,
                    execution.created_at.isoformat(),
                    json.dumps(execution.inputs) if execution.inputs else None,
                    execution.workspace_path,
                    int(execution.background),
                ),
            )
            conn.commit()

        return execution

    def get(self, execution_id: str) -> Optional[Execution]:
        """Get execution by ID"""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM executions WHERE execution_id = ?",
                (execution_id,),
            )
            row = cursor.fetchone()
            if row:
                return Execution.from_row(row)
        return None

    def update_status(
        self,
        execution_id: str,
        status: ExecutionStatus,
        error: Optional[str] = None,
    ) -> bool:
        """Update execution status"""
        now = datetime.now().isoformat()

        with self._connect() as conn:
            if status == ExecutionStatus.RUNNING:
                conn.execute(
                    """
                    UPDATE executions
                    SET status = ?, started_at = ?
                    WHERE execution_id = ?
                    """,
                    (status.value, now, execution_id),
                )
            elif status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
                conn.execute(
                    """
                    UPDATE executions
                    SET status = ?, completed_at = ?, error = ?
                    WHERE execution_id = ?
                    """,
                    (status.value, now, error, execution_id),
                )
            else:
                conn.execute(
                    "UPDATE executions SET status = ? WHERE execution_id = ?",
                    (status.value, execution_id),
                )
            conn.commit()
            return conn.total_changes > 0

    def update_outputs(
        self,
        execution_id: str,
        outputs: Dict[str, Any],
    ) -> bool:
        """Update execution outputs"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE executions SET outputs = ? WHERE execution_id = ?",
                (json.dumps(outputs, default=str), execution_id),
            )
            conn.commit()
            return conn.total_changes > 0

    def update_pid(self, execution_id: str, pid: int) -> bool:
        """Update process ID for background execution"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE executions SET pid = ? WHERE execution_id = ?",
                (pid, execution_id),
            )
            conn.commit()
            return conn.total_changes > 0

    def delete(self, execution_id: str) -> bool:
        """Delete execution record"""
        with self._connect() as conn:
            # Delete logs first
            conn.execute(
                "DELETE FROM execution_logs WHERE execution_id = ?",
                (execution_id,),
            )
            conn.execute(
                "DELETE FROM executions WHERE execution_id = ?",
                (execution_id,),
            )
            conn.commit()
            return conn.total_changes > 0

    # ==========================================================================
    # Listing and Querying
    # ==========================================================================

    def list(
        self,
        status: Optional[ExecutionStatus] = None,
        workspace_path: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        background_only: bool = False,
    ) -> List[Execution]:
        """
        List executions with optional filters.

        Args:
            status: Filter by status
            workspace_path: Filter by workspace
            limit: Maximum results
            offset: Pagination offset
            background_only: Only return background executions
        """
        query = "SELECT * FROM executions WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if workspace_path:
            query += " AND workspace_path = ?"
            params.append(workspace_path)

        if background_only:
            query += " AND background = 1"

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return [Execution.from_row(row) for row in cursor.fetchall()]

    def list_running(self) -> List[Execution]:
        """List all running executions"""
        return self.list(status=ExecutionStatus.RUNNING)

    def list_recent(self, limit: int = 10) -> List[Execution]:
        """List most recent executions"""
        return self.list(limit=limit)

    def count(self, status: Optional[ExecutionStatus] = None) -> int:
        """Count executions"""
        if status:
            query = "SELECT COUNT(*) FROM executions WHERE status = ?"
            params = (status.value,)
        else:
            query = "SELECT COUNT(*) FROM executions"
            params = ()

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchone()[0]

    # ==========================================================================
    # Logging
    # ==========================================================================

    def add_log(
        self,
        execution_id: str,
        level: str,
        message: str,
        step_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a log entry for an execution"""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO execution_logs
                (execution_id, timestamp, level, message, step_id, data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    datetime.now().isoformat(),
                    level,
                    message,
                    step_id,
                    json.dumps(data) if data else None,
                ),
            )
            conn.commit()

    def get_logs(
        self,
        execution_id: str,
        limit: int = 100,
        offset: int = 0,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get logs for an execution"""
        query = """
            SELECT timestamp, level, message, step_id, data
            FROM execution_logs
            WHERE execution_id = ?
        """
        params: List[Any] = [execution_id]

        if level:
            query += " AND level = ?"
            params.append(level)

        query += " ORDER BY id ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            cursor = conn.execute(query, params)
            logs = []
            for row in cursor.fetchall():
                logs.append({
                    "timestamp": row[0],
                    "level": row[1],
                    "message": row[2],
                    "step_id": row[3],
                    "data": json.loads(row[4]) if row[4] else None,
                })
            return logs

    def get_logs_since(
        self,
        execution_id: str,
        since_id: int = 0,
        limit: int = 100,
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Get logs since a specific ID (for streaming/follow).

        Returns:
            Tuple of (logs, last_id)
        """
        query = """
            SELECT id, timestamp, level, message, step_id, data
            FROM execution_logs
            WHERE execution_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
        """

        with self._connect() as conn:
            cursor = conn.execute(query, (execution_id, since_id, limit))
            logs = []
            last_id = since_id
            for row in cursor.fetchall():
                last_id = row[0]
                logs.append({
                    "timestamp": row[1],
                    "level": row[2],
                    "message": row[3],
                    "step_id": row[4],
                    "data": json.loads(row[5]) if row[5] else None,
                })
            return logs, last_id

    # ==========================================================================
    # Cleanup
    # ==========================================================================

    def cleanup_old(self, keep_days: int = 7) -> int:
        """
        Clean up old completed executions.

        Args:
            keep_days: Keep executions from the last N days

        Returns:
            Number of executions deleted
        """
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = cutoff.replace(day=cutoff.day - keep_days)

        with self._connect() as conn:
            # Get IDs to delete
            cursor = conn.execute(
                """
                SELECT execution_id FROM executions
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND created_at < ?
                """,
                (cutoff.isoformat(),),
            )
            ids_to_delete = [row[0] for row in cursor.fetchall()]

            if not ids_to_delete:
                return 0

            # Delete logs
            placeholders = ",".join("?" * len(ids_to_delete))
            conn.execute(
                f"DELETE FROM execution_logs WHERE execution_id IN ({placeholders})",
                ids_to_delete,
            )

            # Delete executions
            conn.execute(
                f"DELETE FROM executions WHERE execution_id IN ({placeholders})",
                ids_to_delete,
            )

            conn.commit()
            return len(ids_to_delete)

    def mark_stale_as_failed(self) -> int:
        """
        Mark executions that are stuck in 'running' state as failed.

        This handles cases where the process crashed without updating status.

        Returns:
            Number of executions marked as failed
        """
        # Consider running executions older than 1 hour as stale
        cutoff = datetime.now()
        cutoff = cutoff.replace(hour=cutoff.hour - 1)

        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE executions
                SET status = ?, error = 'Process terminated unexpectedly', completed_at = ?
                WHERE status = 'running' AND started_at < ?
                """,
                (ExecutionStatus.FAILED.value, datetime.now().isoformat(), cutoff.isoformat()),
            )
            conn.commit()
            return cursor.rowcount


# ==========================================================================
# Global Store Instance
# ==========================================================================

_store: Optional[ExecutionStore] = None


def get_execution_store() -> ExecutionStore:
    """Get global ExecutionStore instance"""
    global _store
    if _store is None:
        _store = ExecutionStore()
    return _store
