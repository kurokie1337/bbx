"""
BBX Console Backend - Main Application

FastAPI application with WebSocket support for real-time updates.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api import api_router
from app.api.schemas.common import HealthResponse
from app.bbx import bbx_bridge
from app.db import init_db, close_db
from app.ws import ws_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
)
logger = logging.getLogger(__name__)

# Startup time for uptime calculation
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting BBX Console...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize BBX Bridge
    await bbx_bridge.initialize()
    logger.info("BBX Bridge initialized")

    yield

    # Shutdown
    logger.info("Shutting down BBX Console...")

    await bbx_bridge.shutdown()
    await close_db()

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Web console for BBX workflow engine",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API router
app.include_router(api_router, prefix="/api")


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["health"])
@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        uptime_seconds=time.time() - START_TIME,
        bbx_connected=bbx_bridge.is_initialized,
        database_connected=True,  # Would check actual connection
        websocket_connections=ws_manager.connection_count,
    )


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await ws_manager.handle_connection(websocket)


# WebSocket stats endpoint
@app.get("/api/ws/stats", tags=["websocket"])
async def websocket_stats():
    """Get WebSocket manager statistics"""
    return ws_manager.get_stats()


# WebSocket channels endpoint
@app.get("/api/ws/channels", tags=["websocket"])
async def websocket_channels():
    """Get active WebSocket channels"""
    return ws_manager.get_channels()


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else None,
        },
    )


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
