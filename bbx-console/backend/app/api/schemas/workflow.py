"""
Workflow API schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WorkflowInput(BaseModel):
    """Workflow input definition"""
    name: str
    type: str
    required: bool = True
    default: Optional[Any] = None
    description: Optional[str] = None


class WorkflowStep(BaseModel):
    """Workflow step definition"""
    id: str
    mcp: str
    method: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    timeout: Optional[int] = None
    retry: Optional[int] = None
    when: Optional[str] = None


class WorkflowListItem(BaseModel):
    """Workflow list item"""
    id: str
    name: str
    description: Optional[str] = None
    file_path: str
    step_count: int
    last_run: Optional["LastRunInfo"] = None


class LastRunInfo(BaseModel):
    """Last run information"""
    execution_id: str
    status: str
    started_at: datetime
    duration_ms: Optional[int] = None


class WorkflowDetail(BaseModel):
    """Detailed workflow information"""
    id: str
    name: str
    description: Optional[str] = None
    file_path: str
    bbx_version: str
    inputs: List[WorkflowInput]
    steps: List[WorkflowStep]
    dag: "DAGVisualization"


class DAGVisualization(BaseModel):
    """DAG visualization data"""
    nodes: List["DAGNode"]
    edges: List["DAGEdge"]
    levels: List[List[str]]


class DAGNode(BaseModel):
    """DAG node"""
    id: str
    label: str
    level: int
    mcp: str
    method: str
    position: Dict[str, float] = Field(default_factory=dict)


class DAGEdge(BaseModel):
    """DAG edge"""
    source: str
    target: str


class WorkflowValidationRequest(BaseModel):
    """Workflow validation request"""
    content: str


class WorkflowValidationResponse(BaseModel):
    """Workflow validation response"""
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class WorkflowRunRequest(BaseModel):
    """Workflow run request"""
    inputs: Dict[str, Any] = Field(default_factory=dict)


class WorkflowRunResponse(BaseModel):
    """Workflow run response"""
    execution_id: str
    workflow_id: str
    status: str


# Update forward refs
WorkflowListItem.model_rebuild()
WorkflowDetail.model_rebuild()
DAGVisualization.model_rebuild()
