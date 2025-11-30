"""
Execution and StepLog models

Track workflow executions and individual step results.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.db.session import Base


class Execution(Base):
    """Workflow execution record"""

    __tablename__ = "executions"

    id = Column(String(36), primary_key=True)
    workflow_id = Column(String(255), nullable=False, index=True)
    workflow_name = Column(String(255))
    status = Column(String(50), nullable=False, default="pending", index=True)
    inputs = Column(JSON)
    results = Column(JSON)
    error = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    steps = relationship("StepLog", back_populates="execution", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "inputs": self.inputs,
            "results": self.results,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class StepLog(Base):
    """Step execution log"""

    __tablename__ = "step_logs"

    id = Column(String(36), primary_key=True)
    execution_id = Column(String(36), ForeignKey("executions.id"), nullable=False, index=True)
    step_id = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    output = Column(JSON)
    error = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    retry_count = Column(Integer, default=0)

    # Relationships
    execution = relationship("Execution", back_populates="steps")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
        }
