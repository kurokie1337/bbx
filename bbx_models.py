from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"

class LicenseType(str, Enum):
    FREE = "free"
    PAID = "paid"
    ENTERPRISE = "enterprise"

class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    bbx_yaml: str
    license_type: LicenseType = LicenseType.FREE
    max_holders: int = 1000
    signature: Optional[str] = None

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    bbx_yaml: Optional[str] = None
    status: Optional[WorkflowStatus] = None

class WorkflowResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    bbx_yaml: str
    status: WorkflowStatus
    version: int
    created_at: datetime
    updated_at: datetime

class WorkflowListResponse(BaseModel):
    items: List[WorkflowResponse]
    total: int
    page: int
    size: int

class ExecutionCreate(BaseModel):
    workflow_id: UUID
    trigger_data: Dict[str, Any] = Field(default_factory=dict)

class ExecutionResponse(BaseModel):
    id: str
    workflow_id: str
    status: str
    started_at: datetime
    current_step: Optional[str] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)

class AuthLogin(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: Optional[int] = None
