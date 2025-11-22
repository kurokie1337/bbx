# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
BBX Audit Logging System for Universal Adapter
Tracks all adapter executions for security and compliance
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditLogEntry:
    """Single audit log entry."""

    timestamp: str
    adapter_id: str
    docker_image: str
    command: list
    user: str
    success: bool
    exit_code: int
    duration_ms: float
    env_hash: str
    volumes: dict
    metadata: dict


class AuditLogger:
    """
    Audit logger for Universal Adapter executions.

    Features:
    - JSON-based log format
    - Rotating log files
    - Security-conscious (hashes sensitive data)
    - Queryable logs
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize audit logger."""
        self.log_dir = log_dir or Path.home() / ".bbx" / "audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_log = (
            self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        self.logger = logging.getLogger("bbx.audit")

    def log_execution(
        self,
        adapter_id: str,
        docker_image: str,
        command: list,
        env: dict,
        volumes: dict,
        success: bool,
        exit_code: int,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log an adapter execution."""
        import os

        # Hash environment variables (security)
        env_str = json.dumps(env, sort_keys=True)
        env_hash = hashlib.sha256(env_str.encode()).hexdigest()[:16]

        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            adapter_id=adapter_id,
            docker_image=docker_image,
            command=command,
            user=os.getenv("USER", os.getenv("USERNAME", "unknown")),
            success=success,
            exit_code=exit_code,
            duration_ms=duration_ms,
            env_hash=env_hash,
            volumes=volumes,
            metadata=metadata or {},
        )

        # Write to JSONL file
        with open(self.current_log, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

        self.logger.info(
            f"Audit: {adapter_id} - {'✅' if success else '❌'} - {duration_ms:.0f}ms"
        )

    def query(
        self,
        adapter_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        success_only: bool = False,
    ) -> List[AuditLogEntry]:
        """Query audit logs."""
        results = []

        # Find all log files in range
        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))

        for log_file in log_files:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        entry_dict = json.loads(line)

                        # Apply filters
                        if adapter_id and entry_dict["adapter_id"] != adapter_id:
                            continue

                        if start_date and entry_dict["timestamp"] < start_date:
                            continue

                        if end_date and entry_dict["timestamp"] > end_date:
                            continue

                        if success_only and not entry_dict["success"]:
                            continue

                        results.append(AuditLogEntry(**entry_dict))
                    except Exception as e:
                        self.logger.warning(f"Failed to parse audit entry: {e}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        all_entries = self.query()

        if not all_entries:
            return {"total": 0}

        total = len(all_entries)
        successful = sum(1 for e in all_entries if e.success)
        failed = total - successful

        avg_duration = (
            sum(e.duration_ms for e in all_entries) / total if total > 0 else 0
        )

        # Most used adapters
        adapter_counts: Dict[str, int] = {}
        for entry in all_entries:
            adapter_counts[entry.adapter_id] = (
                adapter_counts.get(entry.adapter_id, 0) + 1
            )

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "avg_duration_ms": avg_duration,
            "most_used": sorted(
                adapter_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


# Global audit logger
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger (singleton)."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
