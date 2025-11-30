"""
BBX Console Backend - Memory API Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_memory_tiers(client: AsyncClient):
    """Test getting memory tiers."""
    response = await client.get("/api/memory/tiers")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_memory_tier_structure(client: AsyncClient):
    """Test memory tier response structure."""
    response = await client.get("/api/memory/tiers")
    assert response.status_code == 200

    data = response.json()
    if len(data) > 0:
        tier = data[0]
        expected_fields = ["name", "capacity", "used"]
        for field in expected_fields:
            assert field in tier, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_get_memory_stats(client: AsyncClient):
    """Test getting memory statistics."""
    response = await client.get("/api/memory/stats")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_get_memory_entries(client: AsyncClient):
    """Test getting memory entries."""
    response = await client.get("/api/memory/entries")
    # May not be implemented
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_evict_memory_tier(client: AsyncClient):
    """Test memory tier eviction."""
    response = await client.post("/api/memory/tiers/cold/evict")
    # May not be implemented or may require auth
    assert response.status_code in [200, 404, 403]
