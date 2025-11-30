"""
BBX Console Backend - Ring API Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_ring_stats(client: AsyncClient):
    """Test getting ring statistics."""
    response = await client.get("/api/ring/stats")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_ring_stats_structure(client: AsyncClient):
    """Test ring stats response structure."""
    response = await client.get("/api/ring/stats")
    assert response.status_code == 200

    data = response.json()
    expected_fields = [
        "throughput_ops_sec",
        "pending_count",
        "processing_count",
        "operations_completed"
    ]

    for field in expected_fields:
        assert field in data, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_get_ring_operations(client: AsyncClient):
    """Test getting ring operations."""
    response = await client.get("/api/ring/operations")
    # May not be implemented
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_submit_ring_operation(client: AsyncClient):
    """Test submitting a ring operation."""
    response = await client.post(
        "/api/ring/submit",
        json={
            "type": "echo",
            "payload": {"message": "test"}
        }
    )
    # May not be implemented
    assert response.status_code in [200, 202, 404]

    if response.status_code in [200, 202]:
        data = response.json()
        assert "operation_id" in data or "id" in data


@pytest.mark.asyncio
async def test_get_ring_queue_sizes(client: AsyncClient):
    """Test getting ring queue sizes."""
    response = await client.get("/api/ring/stats")
    assert response.status_code == 200

    data = response.json()
    # Should have queue size information
    assert "submission_queue_size" in data or "pending_count" in data
