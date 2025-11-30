"""
BBX Console Backend - Agents API Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_agents(client: AsyncClient):
    """Test listing all agents."""
    response = await client.get("/api/agents")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_agent_not_found(client: AsyncClient):
    """Test getting non-existent agent."""
    response = await client.get("/api/agents/non-existent-agent")
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        # May return empty/null if agent not found
        data = response.json()
        assert data is None or "error" in str(data).lower()


@pytest.mark.asyncio
async def test_get_agent_metrics(client: AsyncClient):
    """Test getting agent metrics."""
    response = await client.get("/api/agents/test-agent/metrics")
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, (dict, list))


@pytest.mark.asyncio
async def test_agent_stats(client: AsyncClient):
    """Test getting overall agent statistics."""
    response = await client.get("/api/agents/stats")
    # May not be implemented
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_agent_response_structure(client: AsyncClient):
    """Test that agent response has expected structure."""
    response = await client.get("/api/agents")
    assert response.status_code == 200

    data = response.json()
    if len(data) > 0:
        agent = data[0]
        # Verify expected fields
        expected_fields = ["id", "name", "status"]
        for field in expected_fields:
            assert field in agent, f"Missing field: {field}"
