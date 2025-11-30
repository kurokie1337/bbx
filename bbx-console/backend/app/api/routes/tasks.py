"""
Task management API routes
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db
from app.db.models import Task
from app.api.schemas.task import (
    TaskCreate,
    TaskUpdate,
    TaskResponse,
    TaskBoardResponse,
    TaskBoardColumn,
    DecomposeRequest,
    DecomposeResponse,
    DecomposedSubtask,
)
from app.services.task_decomposer import TaskDecomposer

router = APIRouter()


@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    parent_id: Optional[str] = None,
    assigned_agent: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List tasks with filters"""
    query = select(Task)

    if status:
        query = query.where(Task.status == status)
    if priority:
        query = query.where(Task.priority == priority)
    if parent_id:
        query = query.where(Task.parent_id == parent_id)
    if assigned_agent:
        query = query.where(Task.assigned_agent == assigned_agent)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    tasks = result.scalars().all()

    return [TaskResponse(**t.to_dict()) for t in tasks]


@router.get("/board", response_model=TaskBoardResponse)
async def get_task_board(db: AsyncSession = Depends(get_db)):
    """Get tasks organized as Kanban board"""
    statuses = ["pending", "in_progress", "completed", "failed"]
    columns = []

    for status in statuses:
        query = select(Task).where(Task.status == status).where(Task.parent_id == None)
        result = await db.execute(query)
        tasks = result.scalars().all()

        columns.append(TaskBoardColumn(
            status=status,
            title=status.replace("_", " ").title(),
            tasks=[TaskResponse(**t.to_dict()) for t in tasks],
            count=len(tasks),
        ))

    total = sum(c.count for c in columns)

    return TaskBoardResponse(columns=columns, total_tasks=total)


@router.post("/", response_model=TaskResponse)
async def create_task(
    request: TaskCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new task"""
    task = Task(
        id=str(uuid.uuid4()),
        title=request.title,
        description=request.description,
        status="pending",
        priority=request.priority,
        parent_id=request.parent_id,
        assigned_agent=request.assigned_agent,
        metadata=request.metadata,
        created_at=datetime.utcnow(),
    )

    db.add(task)
    await db.commit()
    await db.refresh(task)

    return TaskResponse(**task.to_dict())


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: AsyncSession = Depends(get_db)):
    """Get task by ID"""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskResponse(**task.to_dict(include_subtasks=True))


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    request: TaskUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a task"""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Update fields
    if request.title is not None:
        task.title = request.title
    if request.description is not None:
        task.description = request.description
    if request.status is not None:
        task.status = request.status
        if request.status == "completed":
            task.completed_at = datetime.utcnow()
    if request.priority is not None:
        task.priority = request.priority
    if request.assigned_agent is not None:
        task.assigned_agent = request.assigned_agent
    if request.metadata is not None:
        task.metadata = request.metadata

    await db.commit()
    await db.refresh(task)

    return TaskResponse(**task.to_dict())


@router.delete("/{task_id}")
async def delete_task(task_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a task"""
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    await db.delete(task)
    await db.commit()

    return {"success": True, "message": "Task deleted"}


@router.post("/decompose", response_model=DecomposeResponse)
async def decompose_task(request: DecomposeRequest):
    """Use AI to decompose a task into subtasks"""
    decomposer = TaskDecomposer()

    try:
        result = await decomposer.decompose(
            request.description,
            context=request.context,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
