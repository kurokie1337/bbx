"""
BBX Console Backend - WebSocket Tests
"""
import pytest
import json
from httpx import AsyncClient
from starlette.testclient import TestClient

from app.main import app


def test_websocket_connection():
    """Test WebSocket connection."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            # Connection should be established
            assert websocket is not None


def test_websocket_subscribe():
    """Test WebSocket channel subscription."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            # Send subscribe message
            websocket.send_json({
                "action": "subscribe",
                "channels": ["executions", "agents"]
            })

            # Should receive confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscribed"
            assert "channels" in data


def test_websocket_unsubscribe():
    """Test WebSocket channel unsubscription."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            # Subscribe first
            websocket.send_json({
                "action": "subscribe",
                "channels": ["executions"]
            })
            websocket.receive_json()

            # Unsubscribe
            websocket.send_json({
                "action": "unsubscribe",
                "channels": ["executions"]
            })

            data = websocket.receive_json()
            assert data["type"] == "unsubscribed"


def test_websocket_ping_pong():
    """Test WebSocket ping/pong."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"action": "ping"})
            data = websocket.receive_json()
            assert data["type"] == "pong"


def test_websocket_invalid_action():
    """Test WebSocket with invalid action."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_json({"action": "invalid_action"})
            data = websocket.receive_json()
            assert data["type"] == "error"


def test_websocket_malformed_message():
    """Test WebSocket with malformed message."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text("not valid json")
            data = websocket.receive_json()
            assert data["type"] == "error"
