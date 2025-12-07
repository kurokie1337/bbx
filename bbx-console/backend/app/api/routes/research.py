
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from blackbox.core.registry import registry

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    categories: str = "general"
    limit: int = 10

class BrowseRequest(BaseModel):
    url: str
    wait: int = 2

@router.post("/search")
async def search(request: SearchRequest):
    """
    Perform a sovereign search using local SearXNG.
    """
    searx = registry.get_adapter("searx")
    if not searx:
        raise HTTPException(500, "SearXNG adapter not found")
        
    result = await searx.execute("search", request.dict())
    
    if not result.get("success"):
        raise HTTPException(500, result.get("error"))
        
    return result.get("data")

@router.post("/browse")
async def browse(request: BrowseRequest):
    """
    Explore a URL using headless browser and extract text.
    """
    browser = registry.get_adapter("browser")
    if not browser:
        raise HTTPException(500, "Browser adapter not found")
        
    result = await browser.execute("extract_text", request.dict())
    
    if not result.get("success"):
        raise HTTPException(500, result.get("error"))
        
    return result.get("data")

@router.post("/ensure-engine")
async def ensure_engine():
    """Start the search engine container if not running."""
    searx = registry.get_adapter("searx")
    if not searx:
        raise HTTPException(500, "SearXNG adapter not found")
        
    result = await searx.execute("ensure_server", {})
    return result
