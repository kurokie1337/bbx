"""
Logs API routes

Real-time logs from BBX runtime via Python logging.
"""

import logging
from datetime import datetime
from typing import List, Optional
from collections import deque
from fastapi import APIRouter

router = APIRouter(prefix="/logs", tags=["logs"])

# In-memory log buffer (last 500 entries)
log_buffer: deque = deque(maxlen=500)

class LogCapture(logging.Handler):
    """Custom handler to capture logs into buffer"""

    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
                "level": record.levelname,
                "source": record.name,
                "message": record.getMessage()
            }
            log_buffer.append(log_entry)
        except Exception:
            pass

# Install log capture on BBX loggers
def setup_log_capture():
    handler = LogCapture()
    handler.setLevel(logging.DEBUG)

    # Capture logs from BBX modules
    for logger_name in ['bbx', 'app', 'uvicorn']:
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)

# Setup on module load
setup_log_capture()


@router.get("/")
async def get_logs(
    limit: int = 50,
    level: Optional[str] = None,
    source: Optional[str] = None
):
    """Get recent logs from buffer"""
    logs = list(log_buffer)

    # Filter by level
    if level and level != 'all':
        logs = [l for l in logs if l['level'] == level.upper()]

    # Filter by source
    if source:
        logs = [l for l in logs if source.lower() in l['source'].lower()]

    # Return latest entries
    return {
        "entries": logs[-limit:],
        "total": len(logs)
    }


@router.get("/stream")
async def get_log_stream(since: Optional[str] = None):
    """Get logs since a timestamp (for polling)"""
    logs = list(log_buffer)

    if since:
        logs = [l for l in logs if l['timestamp'] > since]

    return {
        "entries": logs,
        "latest": logs[-1]['timestamp'] if logs else None
    }


@router.delete("/")
async def clear_logs():
    """Clear log buffer"""
    log_buffer.clear()
    return {"success": True, "message": "Logs cleared"}
