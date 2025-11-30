"""
Task model

For task management and AI decomposition feature.
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.db.session import Base


class Task(Base):
    """Task record for task management"""

    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    status = Column(String(50), nullable=False, default="pending", index=True)
    priority = Column(String(20), nullable=False, default="medium")

    # Hierarchy
    parent_id = Column(String(36), ForeignKey("tasks.id"), index=True)

    # Assignment
    assigned_agent = Column(String(100))

    # Execution link
    execution_id = Column(String(36), ForeignKey("executions.id"))

    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)

    # Extra data
    metadata = Column(JSON)

    # Relationships
    subtasks = relationship("Task", backref="parent", remote_side=[id])

    def to_dict(self, include_subtasks: bool = False) -> dict:
        result = {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "parent_id": self.parent_id,
            "assigned_agent": self.assigned_agent,
            "execution_id": self.execution_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

        if include_subtasks:
            result["subtasks"] = [s.to_dict() for s in self.subtasks]

        return result
