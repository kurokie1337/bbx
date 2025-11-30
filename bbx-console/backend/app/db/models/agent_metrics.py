"""
Agent metrics model

Store historical metrics for agents.
"""

from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, DateTime

from app.db.session import Base


class AgentMetric(Base):
    """Agent metrics snapshot"""

    __tablename__ = "agent_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Counters
    tasks_completed = Column(Integer, default=0)
    tasks_failed = Column(Integer, default=0)

    # Performance
    avg_duration_ms = Column(Float, default=0)
    min_duration_ms = Column(Float)
    max_duration_ms = Column(Float)

    # Tokens (if tracked)
    total_tokens = Column(Integer, default=0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "total_tokens": self.total_tokens,
        }
