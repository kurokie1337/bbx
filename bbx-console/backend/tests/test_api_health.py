"""
BBX Console Backend - Health API Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_components(client: AsyncClient):
    """Test that health check includes all components."""
    response = await client.get("/api/health")
    data = response.json()

    components = data["components"]
    assert "database" in components
    assert "bbx_core" in components
    assert "websocket" in components


@pytest.mark.asyncio
async def test_root_redirect(client: AsyncClient):
    """Test root endpoint provides API info."""
    response = await client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["name"] == "BBX Console API"
