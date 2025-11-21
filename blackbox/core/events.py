from typing import Callable, List, Any, Dict
from enum import Enum
import asyncio

class EventType(Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    STEP_ERROR = "step_error"

class Event:
    def __init__(self, type: EventType, data: Dict[str, Any]):
        self.type = type
        self.data = data

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], Any]]] = {}

    def subscribe(self, event_type: EventType, callback: Callable[[Event], Any]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def emit(self, event: Event):
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
