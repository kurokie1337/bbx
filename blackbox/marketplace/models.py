"""Marketplace models."""

from datetime import datetime
from typing import List

from pydantic import BaseModel


class WorkflowTemplate(BaseModel):
    id: str
    name: str
    description: str
    author: str
    version: str
    downloads: int = 0
    rating: float = 0.0
    tags: List[str] = []
    category: str
    workflow_content: str
    created_at: datetime
    updated_at: datetime


class Review(BaseModel):
    user: str
    rating: int  # 1-5
    comment: str
    created_at: datetime
