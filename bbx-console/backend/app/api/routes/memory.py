"""
Memory (Context Tiering) API routes
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException

from app.bbx import bbx_bridge
from app.api.schemas.common import SuccessResponse

router = APIRouter()


@router.get("/stats")
async def get_memory_stats():
    """Get memory tiering statistics"""
    return await bbx_bridge.get_memory_stats()


@router.get("/tiers/{tier}")
async def get_tier_items(tier: str):
    """Get items in a specific memory tier"""
    valid_tiers = ["hot", "warm", "cool", "cold"]
    if tier.lower() not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier. Must be one of: {valid_tiers}"
        )

    items = await bbx_bridge.get_memory_items(tier.upper())
    return {"tier": tier.upper(), "items": items}


@router.post("/items/{key}/pin", response_model=SuccessResponse)
async def pin_item(key: str):
    """Pin a memory item to prevent demotion"""
    success = await bbx_bridge.pin_memory_item(key)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return SuccessResponse(success=True, message=f"Item '{key}' pinned")


@router.post("/items/{key}/unpin", response_model=SuccessResponse)
async def unpin_item(key: str):
    """Unpin a memory item to allow demotion"""
    success = await bbx_bridge.unpin_memory_item(key)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return SuccessResponse(success=True, message=f"Item '{key}' unpinned")


@router.delete("/items/{key}", response_model=SuccessResponse)
async def delete_item(key: str):
    """Delete a memory item"""
    # Would implement in bridge
    return SuccessResponse(success=True, message=f"Item '{key}' deleted")
