"""
A2A (Agent-to-Agent) API routes
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class AgentCardResponse(BaseModel):
    """Agent Card response"""
    name: str
    description: str
    url: str
    version: str
    skills: List[Dict[str, Any]]
    capabilities: Dict[str, Any]


class DiscoverRequest(BaseModel):
    """Agent discovery request"""
    url: str


class TaskRequest(BaseModel):
    """A2A task request"""
    agent_url: str
    skill_id: str
    input: Dict[str, Any] = {}


class TaskResponse(BaseModel):
    """A2A task response"""
    id: str
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None


@router.post("/discover", response_model=AgentCardResponse)
async def discover_agent(request: DiscoverRequest):
    """Discover an agent's capabilities"""
    try:
        from blackbox.a2a.client import A2AClient

        async with A2AClient() as client:
            card = await client.discover(request.url)
            return AgentCardResponse(
                name=card.name,
                description=card.description,
                url=card.url,
                version=card.version,
                skills=[s.model_dump() for s in card.skills],
                capabilities=card.capabilities.model_dump() if card.capabilities else {},
            )
    except ImportError:
        raise HTTPException(status_code=501, detail="A2A client not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks", response_model=TaskResponse)
async def create_a2a_task(request: TaskRequest):
    """Create a task on a remote agent"""
    try:
        from blackbox.a2a.client import A2AClient

        async with A2AClient() as client:
            task = await client.create_task(
                agent_url=request.agent_url,
                skill_id=request.skill_id,
                input=request.input,
            )
            return TaskResponse(
                id=task["id"],
                status=task.get("status", "pending"),
            )
    except ImportError:
        raise HTTPException(status_code=501, detail="A2A client not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_a2a_task(task_id: str, agent_url: str):
    """Get task status from remote agent"""
    try:
        from blackbox.a2a.client import A2AClient

        async with A2AClient() as client:
            task = await client.get_task(agent_url, task_id)
            return TaskResponse(
                id=task["id"],
                status=task.get("status", "unknown"),
                output=task.get("output"),
                error=task.get("error"),
            )
    except ImportError:
        raise HTTPException(status_code=501, detail="A2A client not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/registry")
async def list_registered_agents():
    """List locally registered agents"""
    # Would maintain a registry of known agents
    return {"agents": []}
