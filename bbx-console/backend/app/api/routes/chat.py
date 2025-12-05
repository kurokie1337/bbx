"""
LLM Chat API routes - Direct interaction with Ollama LLM
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json

from app.bbx import bbx_bridge

router = APIRouter()


class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen2.5:0.5b"
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    model: str
    tokens: Optional[int] = 0


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a prompt to the LLM and get a response.

    This is the main chat endpoint for direct LLM interaction.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    result = await bbx_bridge.chat(
        prompt=request.prompt,
        model=request.model or "qwen2.5:0.5b"
    )

    return ChatResponse(
        success=result.get("success", False),
        response=result.get("response"),
        error=result.get("error"),
        model=result.get("model", request.model),
        tokens=result.get("tokens", 0)
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream LLM response token by token.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    async def generate():
        async for chunk in bbx_bridge.chat_stream(
            prompt=request.prompt,
            model=request.model or "qwen2.5:0.5b"
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
