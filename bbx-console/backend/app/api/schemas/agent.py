"""
Agent API schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentListItem(BaseModel):
    """Agent list item"""
    id: str
    name: str
    description: str
    status: str  # idle, working, queued, error
    current_task: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    model: str = "sonnet"
    metrics: "AgentMetricsSummary"


class AgentMetricsSummary(BaseModel):
    """Agent metrics summary"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_duration_ms: float = 0.0
    success_rate: float = 0.0


class AgentDetail(BaseModel):
    """Detailed agent information"""
    id: str
    name: str
    description: str
    status: str
    current_task: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    model: str
    system_prompt: str
    file_path: str
    metrics: AgentMetricsSummary
    recent_tasks: List["TaskHistoryItem"] = Field(default_factory=list)


class TaskHistoryItem(BaseModel):
    """Task history item"""
    id: str
    prompt: str
    status: str
    started_at: datetime
    duration_ms: Optional[int] = None


class AgentMetricsResponse(BaseModel):
    """Agent metrics response"""
    agent_id: str
    period: str
    metrics: List["MetricPoint"]


class MetricPoint(BaseModel):
    """Single metric point"""
    timestamp: datetime
    tasks_completed: int
    tasks_failed: int
    avg_duration_ms: float


class AgentStatsResponse(BaseModel):
    """Overall agent statistics"""
    total_agents: int
    busy_agents: int
    idle_agents: int
    total_tasks_today: int
    average_success_rate: float
    queued_tasks: int


# Update forward refs
AgentListItem.model_rebuild()
AgentDetail.model_rebuild()
AgentMetricsResponse.model_rebuild()
