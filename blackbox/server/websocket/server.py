# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""WebSocket server for real-time workflow updates."""

import json
import logging
from typing import Dict, Set

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, workflow_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()

        if workflow_id not in self.active_connections:
            self.active_connections[workflow_id] = set()

        self.active_connections[workflow_id].add(websocket)
        logger.info(f"Client connected to workflow {workflow_id}")

    def disconnect(self, websocket: WebSocket, workflow_id: str):
        """Remove WebSocket connection."""
        if workflow_id in self.active_connections:
            self.active_connections[workflow_id].discard(websocket)
            logger.info(f"Client disconnected from workflow {workflow_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client."""
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(message)

    async def broadcast_to_workflow(self, workflow_id: str, message: dict):
        """Broadcast message to all clients watching a workflow."""
        if workflow_id not in self.active_connections:
            return

        disconnected = set()
        message_json = json.dumps(message)

        for connection in self.active_connections[workflow_id]:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_text(message_json)
                else:
                    disconnected.add(connection)
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections[workflow_id].discard(conn)

    async def broadcast_global(self, message: dict):
        """Broadcast message to all connected clients."""
        message_json = json.dumps(message)

        for connections in self.active_connections.values():
            for connection in connections:
                try:
                    if connection.client_state == WebSocketState.CONNECTED:
                        await connection.send_text(message_json)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")


manager = ConnectionManager()
