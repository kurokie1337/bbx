"""
Workspaces API routes

Manage BBX workspaces via MCP tools.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    path: Optional[str] = None


class WorkspaceSet(BaseModel):
    path: str


async def call_bbx_tool(tool_name: str, arguments: dict):
    """Call a BBX MCP tool"""
    try:
        from blackbox.mcp.tools import TOOL_HANDLERS

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            raise ValueError(f"Tool not found: {tool_name}")

        return await handler(arguments)
    except ImportError:
        raise HTTPException(status_code=501, detail="BBX MCP tools not available")


@router.get("/")
async def list_workspaces():
    """List all available workspaces"""
    try:
        result = await call_bbx_tool("bbx_workspace_list", {})

        # Parse the result
        if isinstance(result, dict) and 'workspaces' in result:
            return result['workspaces']
        elif isinstance(result, list):
            return result
        else:
            # Return default workspace info
            return [{
                "name": "default",
                "path": "~/.bbx/workspaces/default",
                "active": True
            }]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_workspace():
    """Get information about current workspace"""
    try:
        result = await call_bbx_tool("bbx_workspace_info", {})
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_workspace(workspace: WorkspaceCreate):
    """Create a new workspace"""
    try:
        args = {"name": workspace.name}
        if workspace.description:
            args["description"] = workspace.description
        if workspace.path:
            args["path"] = workspace.path

        result = await call_bbx_tool("bbx_workspace_create", args)
        return {"success": True, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set")
async def set_workspace(workspace: WorkspaceSet):
    """Set current workspace"""
    try:
        result = await call_bbx_tool("bbx_workspace_set", {"path": workspace.path})
        return {"success": True, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_workspace():
    """Clear current workspace context"""
    try:
        result = await call_bbx_tool("bbx_workspace_clear", {})
        return {"success": True, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
