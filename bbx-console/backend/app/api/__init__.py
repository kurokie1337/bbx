# API module
from fastapi import APIRouter

from .routes import workflows, executions, agents, memory, ring, tasks, a2a, mcp

api_router = APIRouter()

# Include all route modules
api_router.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
api_router.include_router(executions.router, prefix="/executions", tags=["executions"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(memory.router, prefix="/memory", tags=["memory"])
api_router.include_router(ring.router, prefix="/ring", tags=["ring"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
api_router.include_router(a2a.router, prefix="/a2a", tags=["a2a"])
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])

__all__ = ["api_router"]
