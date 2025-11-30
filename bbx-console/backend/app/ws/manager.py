"""
WebSocket Connection Manager

Handles WebSocket connections, subscriptions, and broadcasting.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Set, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import settings

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and message routing.

    Features:
    - Connection tracking with unique IDs
    - Channel-based subscriptions
    - Broadcast to specific channels
    - Heartbeat/ping-pong for connection health
    """

    def __init__(self):
        # Connection ID -> WebSocket
        self._connections: Dict[str, WebSocket] = {}

        # Channel -> Set of connection IDs
        self._subscriptions: Dict[str, Set[str]] = {}

        # Connection ID -> Set of subscribed channels
        self._connection_channels: Dict[str, Set[str]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Stats
        self._stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
        }

    @property
    def connection_count(self) -> int:
        """Number of active connections"""
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept and register a new WebSocket connection.

        Returns:
            Connection ID
        """
        await websocket.accept()

        conn_id = str(uuid.uuid4())

        async with self._lock:
            if len(self._connections) >= settings.ws_max_connections:
                await websocket.close(code=1013, reason="Too many connections")
                raise ConnectionRefusedError("Maximum connections reached")

            self._connections[conn_id] = websocket
            self._connection_channels[conn_id] = set()
            self._stats["total_connections"] += 1

        logger.info(f"WebSocket connected: {conn_id}")

        # Send welcome message
        await self.send_to_connection(conn_id, {
            "type": "connected",
            "connectionId": conn_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        return conn_id

    async def disconnect(self, conn_id: str):
        """Disconnect and cleanup a WebSocket connection"""
        async with self._lock:
            # Remove from all subscriptions
            if conn_id in self._connection_channels:
                for channel in self._connection_channels[conn_id]:
                    if channel in self._subscriptions:
                        self._subscriptions[channel].discard(conn_id)
                        if not self._subscriptions[channel]:
                            del self._subscriptions[channel]

                del self._connection_channels[conn_id]

            # Remove connection
            if conn_id in self._connections:
                del self._connections[conn_id]

        logger.info(f"WebSocket disconnected: {conn_id}")

    async def subscribe(self, conn_id: str, channel: str):
        """Subscribe a connection to a channel"""
        async with self._lock:
            if conn_id not in self._connections:
                return

            if channel not in self._subscriptions:
                self._subscriptions[channel] = set()

            self._subscriptions[channel].add(conn_id)
            self._connection_channels[conn_id].add(channel)

        logger.debug(f"Connection {conn_id} subscribed to {channel}")

        await self.send_to_connection(conn_id, {
            "type": "subscribed",
            "channel": channel,
        })

    async def unsubscribe(self, conn_id: str, channel: str):
        """Unsubscribe a connection from a channel"""
        async with self._lock:
            if channel in self._subscriptions:
                self._subscriptions[channel].discard(conn_id)
                if not self._subscriptions[channel]:
                    del self._subscriptions[channel]

            if conn_id in self._connection_channels:
                self._connection_channels[conn_id].discard(channel)

        logger.debug(f"Connection {conn_id} unsubscribed from {channel}")

        await self.send_to_connection(conn_id, {
            "type": "unsubscribed",
            "channel": channel,
        })

    async def send_to_connection(self, conn_id: str, message: dict):
        """Send message to a specific connection"""
        if conn_id not in self._connections:
            return

        try:
            ws = self._connections[conn_id]
            await ws.send_json(message)
            self._stats["total_messages_sent"] += 1
        except Exception as e:
            logger.error(f"Failed to send to {conn_id}: {e}")
            await self.disconnect(conn_id)

    async def broadcast(self, channel: str, event: str, data: Any):
        """Broadcast message to all subscribers of a channel"""
        message = {
            "type": "event",
            "channel": channel,
            "event": event,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        conn_ids = self._subscriptions.get(channel, set()).copy()

        for conn_id in conn_ids:
            await self.send_to_connection(conn_id, message)

        logger.debug(f"Broadcast to {channel}: {event} ({len(conn_ids)} recipients)")

    async def broadcast_all(self, event: str, data: Any):
        """Broadcast message to all connected clients"""
        message = {
            "type": "event",
            "channel": "*",
            "event": event,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        for conn_id in list(self._connections.keys()):
            await self.send_to_connection(conn_id, message)

    async def handle_message(self, conn_id: str, message: dict):
        """Handle incoming message from client"""
        self._stats["total_messages_received"] += 1

        msg_type = message.get("type")

        if msg_type == "subscribe":
            channel = message.get("channel")
            if channel:
                await self.subscribe(conn_id, channel)

        elif msg_type == "unsubscribe":
            channel = message.get("channel")
            if channel:
                await self.unsubscribe(conn_id, channel)

        elif msg_type == "ping":
            await self.send_to_connection(conn_id, {"type": "pong"})

        elif msg_type == "action":
            # Forward to action handler
            action = message.get("action")
            data = message.get("data", {})
            logger.info(f"Action from {conn_id}: {action}")
            # Actions handled by API routes

        else:
            logger.warning(f"Unknown message type from {conn_id}: {msg_type}")

    async def handle_connection(self, websocket: WebSocket):
        """Main handler for WebSocket connection lifecycle"""
        conn_id = await self.connect(websocket)

        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    await self.handle_message(conn_id, data)
                except json.JSONDecodeError:
                    await self.send_to_connection(conn_id, {
                        "type": "error",
                        "message": "Invalid JSON",
                    })
        except WebSocketDisconnect:
            pass
        finally:
            await self.disconnect(conn_id)

    def get_stats(self) -> dict:
        """Get WebSocket manager statistics"""
        return {
            "active_connections": len(self._connections),
            "active_channels": len(self._subscriptions),
            "total_connections": self._stats["total_connections"],
            "total_messages_sent": self._stats["total_messages_sent"],
            "total_messages_received": self._stats["total_messages_received"],
        }

    def get_channels(self) -> Dict[str, int]:
        """Get all channels with subscriber counts"""
        return {channel: len(subs) for channel, subs in self._subscriptions.items()}


# Global WebSocket manager instance
ws_manager = WebSocketManager()
