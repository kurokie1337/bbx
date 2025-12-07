
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
    
class SynthesizeRequest(BaseModel):
    text: str
    query: str
    model: Optional[str] = "qwen2.5:0.5b"

@router.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize knowledge from text using local LLM.
    """
    from blackbox.runtime.llm_provider import complete
    
    prompt = f"""
    You are a helpful research assistant. 
    User Query: "{request.query}"
    
    Based ONLY on the provided text, answer the query concisely.
    If the text doesn't contain the answer, say "I couldn't find the answer in the provided text."
    
    Text content:
    {request.text[:12000]}  # Limit context to avoid overflow
    
    Answer:
    """
    
    try:
        response = await complete(prompt, model=request.model)
        return {"answer": response.content, "model": response.model}
    except Exception as e:
        raise HTTPException(500, f"Synthesis failed: {str(e)}")
