# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Blackbox API Server - Enhanced Production Version
Provides REST API for executing workflows with metrics, health checks, and validation.
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import sys
import time
import uuid
import logging

sys.path.insert(0, 'c:\\Users\\User\\Desktop\\Новая папка\\workflow_test')

from blackbox.core import run_file
from blackbox.core.config import get_settings
from blackbox.core.validation import validate_workflow
from blackbox.core.exceptions import WorkflowValidationError
from blackbox.core.metrics import init_metrics
import tempfile
import os
import yaml

# Initialize settings
settings = get_settings()

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Blackbox API",
    version="1.0.0",
    description="Workflow automation engine API"
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for editor
# Ensure the editor directory exists
EDITOR_DIR = os.path.join(os.path.dirname(__file__), "editor")
if os.path.exists(EDITOR_DIR):
    app.mount("/ui", StaticFiles(directory=EDITOR_DIR, html=True), name="ui")
else:
    logger.warning(f"Editor directory not found at {EDITOR_DIR}")

@app.get("/")
async def root():
    """Redirect to the visual editor."""
    return RedirectResponse(url="/ui")

# Initialize metrics
init_metrics(version="1.0.0")

# Track server start time
SERVER_START_TIME = time.time()


class WorkflowExecute(BaseModel):
    """Request model for workflow execution."""
    workflow: Dict[str, Any] = Field(..., description="Workflow definition")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context variables")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    uptime_seconds: float
    dependencies: Dict[str, str]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with UUID."""
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    logger.info(f"Request {request_id} completed in {duration:.3f}s with status {response.status_code}")
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Blackbox API Server",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns server status and dependency health.
    """
    uptime = time.time() - SERVER_START_TIME
    
    # Check dependencies
    dependencies = {
        "cache": "ok",
        "registry": "ok",
        "filesystem": "ok"
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime,
        dependencies=dependencies
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Returns metrics in Prometheus format.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/api/execute")
async def execute_workflow(request: WorkflowExecute):
    """
    Execute a workflow from the visual editor or API client.
    
    Args:
        request: Workflow execution request
        
    Returns:
        Execution results
        
    Raises:
        HTTPException: If workflow is invalid or execution fails
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Executing workflow {request_id}")
    
    try:
        # Validate workflow structure
        try:
            validate_workflow(request.workflow)
        except WorkflowValidationError as e:
            logger.error(f"Workflow validation failed: {e.errors}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Workflow validation failed",
                    "errors": e.errors
                }
            )
        
        # Create temporary BBX file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.bbx',
            delete=False,
            encoding='utf-8'
        ) as f:
            yaml.dump(request.workflow, f)
            temp_file = f.name
        
        try:
            # Execute workflow
            logger.info(f"Running workflow from {temp_file}")
            start_time = time.time()
            
            results = await run_file(temp_file, use_cache=False)
            
            duration = time.time() - start_time
            logger.info(f"Workflow completed in {duration:.3f}s")
            
            # Format results for browser
            formatted_results = []
            for step_id, result in results.items():
                formatted_results.append({
                    "step_id": step_id,
                    "status": result.get("status"),
                    "output": result.get("output"),
                    "error": result.get("error")
                })
            
            return {
                "success": True,
                "request_id": request_id,
                "results": formatted_results,
                "duration_ms": duration * 1000,
                "message": "Workflow executed successfully!"
            }
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temp file: {temp_file}")
    
    except WorkflowValidationError:
        # Already handled above
        raise
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Workflow execution failed",
                "message": str(e),
                "request_id": request_id
            }
        )


@app.post("/api/validate")
async def validate_workflow_endpoint(request: WorkflowExecute):
    """
    Validate a workflow without executing it.
    
    Args:
        request: Workflow to validate
        
    Returns:
        Validation result
    """
    try:
        validate_workflow(request.workflow)
        return {
            "valid": True,
            "message": "Workflow is valid"
        }
    except WorkflowValidationError as e:
        return {
            "valid": False,
            "errors": e.errors
        }


@app.get("/api/workflows")
async def list_workflows():
    """
    List available workflows in the project.
    """
    workflows = []
    root_dir = os.getcwd()
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".bbx"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                
                # Try to read metadata
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        workflows.append({
                            "path": rel_path,
                            "id": data.get("id", "unknown"),
                            "name": data.get("name", file),
                            "description": data.get("description", ""),
                            "content": data  # Include full content for execution
                        })
                except Exception:
                    workflows.append({
                        "path": rel_path,
                        "id": "error",
                        "name": file,
                        "error": "Failed to parse"
                    })
                    
    return {"workflows": workflows}


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 BLACKBOX API SERVER v1.0.0")
    print("="*80)
    print(f"\n📡 Starting server at http://{settings.api_host}:{settings.api_port}")
    print(f"📊 Metrics available at http://{settings.api_host}:{settings.api_port}/metrics")
    print(f"❤️  Health check at http://{settings.api_host}:{settings.api_port}/health")
    print(f"📖 API docs at http://{settings.api_host}:{settings.api_port}/docs")
    print("🌐 Visual Editor can now execute workflows!\n")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )
