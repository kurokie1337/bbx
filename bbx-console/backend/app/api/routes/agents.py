"""
Agent API routes
"""

from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.bbx import bbx_bridge
from app.api.schemas.agent import (
    AgentListItem,
    AgentDetail,
    AgentMetricsResponse,
    AgentStatsResponse,
    AgentMetricsSummary,
)

router = APIRouter()


@router.get("/", response_model=List[AgentListItem])
async def list_agents():
    """List all available agents"""
    agents = await bbx_bridge.list_agents()

    return [
        AgentListItem(
            id=a["id"],
            name=a["name"],
            description=a["description"],
            status=a.get("status", "idle"),
            current_task=a.get("current_task"),
            tools=a.get("tools", []),
            model=a.get("model", "sonnet"),
            metrics=AgentMetricsSummary(**a.get("metrics", {})),
        )
        for a in agents
    ]


@router.get("/stats", response_model=AgentStatsResponse)
async def get_agent_stats():
    """Get overall agent statistics"""
    agents = await bbx_bridge.list_agents()

    total = len(agents)
    busy = sum(1 for a in agents if a.get("status") == "working")
    idle = sum(1 for a in agents if a.get("status") == "idle")

    return AgentStatsResponse(
        total_agents=total,
        busy_agents=busy,
        idle_agents=idle,
        total_tasks_today=0,  # Would fetch from DB
        average_success_rate=0.95,  # Would calculate
        queued_tasks=0,
    )


@router.get("/{agent_id}", response_model=AgentDetail)
async def get_agent(agent_id: str):
    """Get agent details"""
    agent = await bbx_bridge.get_agent(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return AgentDetail(
        id=agent["id"],
        name=agent["name"],
        description=agent["description"],
        status=agent.get("status", "idle"),
        current_task=agent.get("current_task"),
        tools=agent.get("tools", []),
        model=agent.get("model", "sonnet"),
        system_prompt=agent.get("system_prompt", ""),
        file_path=agent.get("file_path", ""),
        metrics=AgentMetricsSummary(**agent.get("metrics", {})),
        recent_tasks=[],
    )


@router.get("/{agent_id}/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    agent_id: str,
    period: str = Query(default="24h", regex="^(1h|24h|7d)$"),
):
    """Get agent metrics history"""
    agent = await bbx_bridge.get_agent(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Would fetch from database
    return AgentMetricsResponse(
        agent_id=agent_id,
        period=period,
        metrics=[],
    )
