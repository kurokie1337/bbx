"""
MCP (Model Context Protocol) API routes
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class MCPToolInfo(BaseModel):
    """MCP tool information"""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPServerInfo(BaseModel):
    """MCP server information"""
    name: str
    status: str
    tools_count: int


class MCPCallRequest(BaseModel):
    """MCP tool call request"""
    server: str
    tool: str
    arguments: Dict[str, Any] = {}


class MCPCallResponse(BaseModel):
    """MCP tool call response"""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


@router.get("/servers", response_model=List[MCPServerInfo])
async def list_mcp_servers():
    """List configured MCP servers"""
    try:
        # Would read from MCP client configuration
        return [
            MCPServerInfo(name="bbx", status="connected", tools_count=50),
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/servers/{server}/tools", response_model=List[MCPToolInfo])
async def list_server_tools(server: str):
    """List tools from a specific MCP server"""
    try:
        from blackbox.mcp.tools import get_bbx_tools

        if server == "bbx":
            tools = get_bbx_tools()
            return [
                MCPToolInfo(
                    name=t["name"],
                    description=t["description"],
                    input_schema=t["inputSchema"],
                )
                for t in tools
            ]

        raise HTTPException(status_code=404, detail=f"Server not found: {server}")

    except ImportError:
        raise HTTPException(status_code=501, detail="MCP tools not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/call", response_model=MCPCallResponse)
async def call_mcp_tool(request: MCPCallRequest):
    """Call an MCP tool"""
    try:
        from blackbox.mcp.tools import TOOL_HANDLERS

        if request.server != "bbx":
            raise HTTPException(status_code=404, detail=f"Server not found: {request.server}")

        handler = TOOL_HANDLERS.get(request.tool)
        if not handler:
            raise HTTPException(status_code=404, detail=f"Tool not found: {request.tool}")

        result = await handler(request.arguments)

        return MCPCallResponse(success=True, result=result)

    except ImportError:
        raise HTTPException(status_code=501, detail="MCP tools not available")
    except HTTPException:
        raise
    except Exception as e:
        return MCPCallResponse(success=False, error=str(e))


@router.get("/tools/{tool}/schema")
async def get_tool_schema(tool: str):
    """Get input schema for a specific tool"""
    try:
        from blackbox.mcp.tools import get_bbx_tools

        tools = get_bbx_tools()
        for t in tools:
            if t["name"] == tool:
                return {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["inputSchema"],
                }

        raise HTTPException(status_code=404, detail=f"Tool not found: {tool}")

    except ImportError:
        raise HTTPException(status_code=501, detail="MCP tools not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
