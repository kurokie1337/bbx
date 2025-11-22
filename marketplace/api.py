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
Blackbox Marketplace API

FastAPI-based marketplace for sharing and discovering workflows.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

app = FastAPI(
    title="Blackbox Marketplace API",
    description="Discover, publish, and share BBX workflows",
    version="1.0.0"
)

# Models
class WorkflowPublish(BaseModel):
    name: str
    description: str
    bbx_content: str
    category: str
    tags: List[str] = []
    license: str = "MIT"


class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    author: str
    downloads: int
    rating: float
    created_at: datetime
    updated_at: datetime


class WorkflowDetail(WorkflowResponse):
    bbx_content: str
    license: str


# In-memory storage (replace with database in production)
workflows_db = {}


@app.get("/")
def root():
    return {
        "message": "Blackbox Marketplace API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/marketplace/publish", response_model=WorkflowResponse)
def publish_workflow(workflow: WorkflowPublish):
    """Publish a new workflow to marketplace"""
    workflow_id = str(uuid.uuid4())
    
    workflow_data = {
        "id": workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "category": workflow.category,
        "tags": workflow.tags,
        "author": "anonymous",  # TODO: Get from auth
        "downloads": 0,
        "rating": 0.0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "bbx_content": workflow.bbx_content,
        "license": workflow.license
    }
    
    workflows_db[workflow_id] = workflow_data
    
    return WorkflowResponse(**workflow_data)


@app.get("/marketplace/workflows", response_model=List[WorkflowResponse])
def list_workflows(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """List all workflows with optional filters"""
    results = list(workflows_db.values())
    
    # Filter by category
    if category:
        results = [w for w in results if w["category"] == category]
    
    # Filter by tag
    if tag:
        results = [w for w in results if tag in w["tags"]]
    
    # Search
    if search:
        search_lower = search.lower()
        results = [
            w for w in results 
            if search_lower in w["name"].lower() or search_lower in w["description"].lower()
        ]
    
    # Pagination
    results = results[offset:offset + limit]
    
    return [WorkflowResponse(**w) for w in results]


@app.get("/marketplace/workflows/{workflow_id}", response_model=WorkflowDetail)
def get_workflow(workflow_id: str):
    """Get workflow details including BBX content"""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = workflows_db[workflow_id]
    
    # Increment downloads
    workflow["downloads"] += 1
    
    return WorkflowDetail(**workflow)


@app.post("/marketplace/workflows/{workflow_id}/rate")
def rate_workflow(workflow_id: str, rating: int):
    """Rate a workflow (1-5 stars)"""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    if rating < 1 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    workflow = workflows_db[workflow_id]
    
    # Simple average (in production, store individual ratings)
    current_rating = workflow["rating"]
    downloads = workflow["downloads"]
    
    new_rating = ((current_rating * downloads) + rating) / (downloads + 1)
    workflow["rating"] = round(new_rating, 2)
    
    return {"message": "Rating submitted", "new_rating": workflow["rating"]}


@app.get("/marketplace/categories")
def get_categories():
    """Get all workflow categories"""
    categories = set()
    for workflow in workflows_db.values():
        categories.add(workflow["category"])
    
    return {"categories": sorted(list(categories))}


@app.get("/marketplace/trending")
def get_trending(limit: int = 10):
    """Get trending workflows"""
    workflows = sorted(
        workflows_db.values(),
        key=lambda w: w["downloads"] * w["rating"],
        reverse=True
    )
    
    return [WorkflowResponse(**w) for w in workflows[:limit]]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
