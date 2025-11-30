"""
Execution API schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepState(BaseModel):
    """Step execution state"""
    step_id: str
    status: str  # pending, waiting, running, success, failed, skipped, timeout, cancelled
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0


class ExecutionState(BaseModel):
    """Workflow execution state"""
    id: str
    workflow_id: str
    workflow_name: Optional[str] = None
    status: str  # pending, running, completed, failed, cancelled
    inputs: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    steps: Dict[str, StepState] = Field(default_factory=dict)
    current_level: int = 0
    progress: float = 0.0  # 0-100


class ExecutionListItem(BaseModel):
    """Execution list item"""
    id: str
    workflow_id: str
    workflow_name: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    step_count: int = 0
    steps_completed: int = 0


class ExecutionListRequest(BaseModel):
    """Execution list filter"""
    workflow_id: Optional[str] = None
    status: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class LogEntry(BaseModel):
    """Execution log entry"""
    timestamp: datetime
    level: str  # info, warning, error, debug
    step_id: Optional[str] = None
    message: str
    data: Optional[Dict[str, Any]] = None


class ExecutionLogsResponse(BaseModel):
    """Execution logs response"""
    execution_id: str
    logs: List[LogEntry]
