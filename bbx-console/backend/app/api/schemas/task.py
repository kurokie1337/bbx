"""
Task management API schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    """Create task request"""
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    priority: str = Field(default="medium")  # low, medium, high, critical
    parent_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskUpdate(BaseModel):
    """Update task request"""
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None  # pending, in_progress, completed, failed, cancelled
    priority: Optional[str] = None
    assigned_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskResponse(BaseModel):
    """Task response"""
    id: str
    title: str
    description: Optional[str] = None
    status: str
    priority: str
    parent_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    execution_id: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    subtasks: List["TaskResponse"] = Field(default_factory=list)


class TaskListRequest(BaseModel):
    """Task list filter"""
    status: Optional[str] = None
    priority: Optional[str] = None
    parent_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)


class DecomposeRequest(BaseModel):
    """AI decomposition request"""
    description: str = Field(..., min_length=10)
    context: Optional[str] = None


class DecomposedSubtask(BaseModel):
    """Decomposed subtask"""
    title: str
    description: str
    assigned_agent: str
    depends_on: List[str] = Field(default_factory=list)
    priority: str = "medium"


class DecomposeResponse(BaseModel):
    """AI decomposition response"""
    original_task: str
    subtasks: List[DecomposedSubtask]
    suggested_workflow: str  # Generated BBX workflow YAML
    confidence: float = 0.0


class TaskBoardColumn(BaseModel):
    """Task board column"""
    status: str
    title: str
    tasks: List[TaskResponse]
    count: int


class TaskBoardResponse(BaseModel):
    """Task board response (Kanban view)"""
    columns: List[TaskBoardColumn]
    total_tasks: int


# Update forward refs
TaskResponse.model_rebuild()
