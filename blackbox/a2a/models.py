# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
A2A Protocol Models

Pydantic models implementing Google Agent2Agent Protocol v0.3 specification.
Based on: https://a2a-protocol.org/latest/

Key concepts:
- AgentCard: JSON metadata describing agent capabilities (/.well-known/agent-card.json)
- Task: Unit of work with lifecycle (pending -> in_progress -> completed/failed)
- Message: Communication unit within a task
- Artifact: Output data from task execution
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, HttpUrl
import uuid


# =============================================================================
# Enums
# =============================================================================

class A2ATaskStatus(str, Enum):
    """Task lifecycle states per A2A spec."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class A2AMessageRole(str, Enum):
    """Message sender role."""
    USER = "user"
    AGENT = "agent"


class A2AArtifactType(str, Enum):
    """Types of artifacts agents can produce."""
    TEXT = "text"
    FILE = "file"
    IMAGE = "image"
    JSON = "json"
    CODE = "code"


# =============================================================================
# Agent Card Models (Capability Advertisement)
# =============================================================================

class AgentSkillParameter(BaseModel):
    """Parameter definition for a skill."""
    name: str
    description: Optional[str] = None
    type: str = "string"
    required: bool = False
    default: Optional[Any] = None


class AgentSkill(BaseModel):
    """
    A capability that an agent can perform.

    Maps to BBX workflows or adapters.
    """
    id: str = Field(..., description="Unique skill identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="What this skill does")

    # Input/output schemas
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="inputSchema",
        description="JSON Schema for skill inputs"
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="outputSchema",
        description="JSON Schema for skill outputs"
    )

    # Metadata
    tags: List[str] = Field(default_factory=list)
    examples: List[Dict[str, Any]] = Field(default_factory=list)

    # Execution hints
    estimated_duration: Optional[str] = Field(
        default=None,
        alias="estimatedDuration",
        description="Expected execution time (e.g., '30s', '5m')"
    )
    supports_streaming: bool = Field(
        default=False,
        alias="supportsStreaming"
    )

    class Config:
        populate_by_name = True


class AgentAuthentication(BaseModel):
    """Authentication requirements for the agent."""
    schemes: List[str] = Field(
        default_factory=lambda: ["none"],
        description="Supported auth schemes: none, bearer, apiKey, oauth2"
    )
    oauth2_config: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="oauth2Config"
    )

    class Config:
        populate_by_name = True


class AgentEndpoints(BaseModel):
    """A2A protocol endpoints."""
    task: str = Field(default="/a2a/tasks", description="Create task endpoint")
    task_status: str = Field(
        default="/a2a/tasks/{task_id}",
        alias="taskStatus",
        description="Get task status"
    )
    task_cancel: str = Field(
        default="/a2a/tasks/{task_id}/cancel",
        alias="taskCancel",
        description="Cancel task"
    )
    stream: str = Field(
        default="/a2a/tasks/{task_id}/stream",
        description="SSE stream endpoint"
    )

    class Config:
        populate_by_name = True


class AgentCard(BaseModel):
    """
    Agent Card - the core A2A discovery mechanism.

    Hosted at /.well-known/agent-card.json
    Describes what this agent can do and how to interact with it.
    """
    # Identity
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="What this agent does")
    url: str = Field(..., description="Base URL of the agent")
    version: str = Field(default="1.0.0", description="Agent version")

    # Protocol info
    protocol_version: str = Field(
        default="0.3",
        alias="protocolVersion",
        description="A2A protocol version"
    )

    # Capabilities
    skills: List[AgentSkill] = Field(
        default_factory=list,
        description="List of capabilities"
    )

    # Communication
    endpoints: AgentEndpoints = Field(
        default_factory=AgentEndpoints
    )

    # Security
    authentication: AgentAuthentication = Field(
        default_factory=AgentAuthentication
    )

    # Metadata
    provider: Optional[str] = Field(
        default=None,
        description="Organization providing this agent"
    )
    documentation_url: Optional[str] = Field(
        default=None,
        alias="documentationUrl"
    )
    contact_email: Optional[str] = Field(
        default=None,
        alias="contactEmail"
    )
    tags: List[str] = Field(default_factory=list)

    # BBX-specific extensions
    bbx_version: Optional[str] = Field(
        default=None,
        alias="bbxVersion",
        description="BBX engine version"
    )
    mcp_tools_count: Optional[int] = Field(
        default=None,
        alias="mcpToolsCount",
        description="Number of MCP tools available"
    )

    class Config:
        populate_by_name = True


# =============================================================================
# Task Models (Work Units)
# =============================================================================

class A2AMessage(BaseModel):
    """
    A message within a task conversation.

    Agents exchange messages during task execution.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: A2AMessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional structured data
    data: Optional[Dict[str, Any]] = None

    # For multi-part messages
    parts: Optional[List[Dict[str, Any]]] = None


class A2AArtifact(BaseModel):
    """
    An output artifact from task execution.

    Can be text, files, images, structured data, etc.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: A2AArtifactType
    name: str
    description: Optional[str] = None

    # Content (one of these should be set)
    content: Optional[str] = None  # For text/code/json
    url: Optional[str] = None  # For files/images
    data: Optional[Dict[str, Any]] = None  # For structured data

    # Metadata
    mime_type: Optional[str] = Field(default=None, alias="mimeType")
    size_bytes: Optional[int] = Field(default=None, alias="sizeBytes")
    created_at: datetime = Field(default_factory=datetime.utcnow, alias="createdAt")

    class Config:
        populate_by_name = True


class A2ATaskInput(BaseModel):
    """Input for creating a new task."""
    skill_id: str = Field(..., alias="skillId", description="Which skill to invoke")
    input: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for the skill"
    )

    # Optional context
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for execution"
    )

    # Callback URL for async notifications
    callback_url: Optional[str] = Field(
        default=None,
        alias="callbackUrl",
        description="URL for push notifications"
    )

    # Metadata
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True


class A2ATask(BaseModel):
    """
    A task - the primary unit of work in A2A.

    Maps to BBX Execution.
    """
    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # What to do
    skill_id: str = Field(..., alias="skillId")
    input: Dict[str, Any] = Field(default_factory=dict)

    # Status
    status: A2ATaskStatus = Field(default=A2ATaskStatus.PENDING)
    progress: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Completion percentage"
    )
    status_message: Optional[str] = Field(
        default=None,
        alias="statusMessage"
    )

    # Output
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    artifacts: List[A2AArtifact] = Field(default_factory=list)

    # Conversation history
    messages: List[A2AMessage] = Field(default_factory=list)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, alias="createdAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    completed_at: Optional[datetime] = Field(default=None, alias="completedAt")

    # Metadata
    metadata: Optional[Dict[str, Any]] = None

    # BBX-specific
    bbx_execution_id: Optional[str] = Field(
        default=None,
        alias="bbxExecutionId",
        description="Linked BBX execution ID"
    )

    class Config:
        populate_by_name = True

    def to_sse_event(self, event_type: str = "update") -> str:
        """Format task as SSE event."""
        import json
        data = self.model_dump(mode="json", by_alias=True)
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# =============================================================================
# JSON-RPC Models (A2A uses JSON-RPC 2.0)
# =============================================================================

class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request."""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object."""
    code: int
    message: str
    data: Optional[Any] = None


# =============================================================================
# Discovery Models
# =============================================================================

class AgentDiscoveryEntry(BaseModel):
    """Entry in an agent registry."""
    url: str
    name: str
    description: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    last_seen: datetime = Field(default_factory=datetime.utcnow, alias="lastSeen")
    healthy: bool = True

    class Config:
        populate_by_name = True


class AgentRegistry(BaseModel):
    """Local registry of known agents."""
    agents: Dict[str, AgentDiscoveryEntry] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.utcnow, alias="updatedAt")

    class Config:
        populate_by_name = True
