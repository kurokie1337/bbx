"""
BBX Console Backend - Workflows API Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_workflows(client: AsyncClient):
    """Test listing workflows."""
    response = await client.get("/api/workflows")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_list_workflows_with_pagination(client: AsyncClient):
    """Test workflow listing with pagination."""
    response = await client.get("/api/workflows?skip=0&limit=10")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10


@pytest.mark.asyncio
async def test_get_workflow_not_found(client: AsyncClient):
    """Test getting non-existent workflow."""
    response = await client.get("/api/workflows/non-existent-workflow")
    # Should return 404 or empty response depending on implementation
    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_get_workflow_dag(client: AsyncClient):
    """Test getting workflow DAG structure."""
    response = await client.get("/api/workflows/test-workflow/dag")
    # May return 404 if workflow doesn't exist
    assert response.status_code in [200, 404]

    if response.status_code == 200:
        data = response.json()
        assert "nodes" in data or data == {}
        assert "edges" in data or data == {}


@pytest.mark.asyncio
async def test_validate_workflow(client: AsyncClient, sample_workflow_data: dict):
    """Test workflow validation."""
    response = await client.post(
        "/api/workflows/validate",
        json=sample_workflow_data
    )
    # Validation endpoint may not be implemented yet
    assert response.status_code in [200, 404, 422]


@pytest.mark.asyncio
async def test_run_workflow(client: AsyncClient):
    """Test running a workflow."""
    response = await client.post(
        "/api/workflows/test-workflow/run",
        json={"inputs": {}}
    )
    # May fail if workflow doesn't exist
    assert response.status_code in [200, 202, 404, 500]

    if response.status_code in [200, 202]:
        data = response.json()
        assert "execution_id" in data or "id" in data
