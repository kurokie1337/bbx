"""
Ring (AgentRing) API routes
"""

from fastapi import APIRouter

from app.bbx import bbx_bridge

router = APIRouter()


@router.get("/stats")
async def get_ring_stats():
    """Get AgentRing statistics"""
    return await bbx_bridge.get_ring_stats()


@router.get("/queues/submission")
async def get_submission_queue():
    """Get items in submission queue"""
    # Would implement in bridge
    return {"queue": "submission", "items": []}


@router.get("/queues/completion")
async def get_completion_queue():
    """Get items in completion queue"""
    # Would implement in bridge
    return {"queue": "completion", "items": []}


@router.get("/workers")
async def get_workers():
    """Get worker status"""
    stats = await bbx_bridge.get_ring_stats()
    return {
        "active_workers": stats.get("active_workers", 0),
        "max_workers": stats.get("worker_pool_size", 32),
        "utilization": stats.get("worker_utilization", 0),
    }
