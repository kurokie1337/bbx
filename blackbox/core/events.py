# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

import asyncio
from enum import Enum
from typing import Any, Callable, Dict, List


class EventType(Enum):
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    STEP_COMPLETE = "step_complete"  # Alias for STEP_END (used by ExecutionManager)
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
