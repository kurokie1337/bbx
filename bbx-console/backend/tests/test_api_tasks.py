"""
BBX Console Backend - Tasks API Tests
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_task_board(client: AsyncClient):
    """Test getting task board with columns."""
    response = await client.get("/api/tasks/board")
    assert response.status_code == 200

    data = response.json()
    assert "columns" in data
    assert isinstance(data["columns"], list)

    # Verify column structure
    for column in data["columns"]:
        assert "status" in column
        assert "title" in column
        assert "tasks" in column
        assert "count" in column


@pytest.mark.asyncio
async def test_list_tasks(client: AsyncClient):
    """Test listing all tasks."""
    response = await client.get("/api/tasks")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_list_tasks_with_status_filter(client: AsyncClient):
    """Test listing tasks with status filter."""
    response = await client.get("/api/tasks?status=todo")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_create_task(client: AsyncClient, sample_task_data: dict):
    """Test creating a new task."""
    response = await client.post("/api/tasks", json=sample_task_data)
    assert response.status_code in [200, 201]

    data = response.json()
    assert "id" in data
    assert data["title"] == sample_task_data["title"]


@pytest.mark.asyncio
async def test_create_task_minimal(client: AsyncClient):
    """Test creating task with minimal data."""
    response = await client.post("/api/tasks", json={"title": "Minimal Task"})
    assert response.status_code in [200, 201]

    data = response.json()
    assert data["title"] == "Minimal Task"
    assert "id" in data


@pytest.mark.asyncio
async def test_update_task_status(client: AsyncClient):
    """Test updating task status."""
    # First create a task
    create_response = await client.post(
        "/api/tasks",
        json={"title": "Task to Update"}
    )
    assert create_response.status_code in [200, 201]
    task_id = create_response.json()["id"]

    # Update the task
    update_response = await client.patch(
        f"/api/tasks/{task_id}",
        json={"status": "in_progress"}
    )
    assert update_response.status_code == 200

    data = update_response.json()
    assert data["status"] == "in_progress"


@pytest.mark.asyncio
async def test_delete_task(client: AsyncClient):
    """Test deleting a task."""
    # First create a task
    create_response = await client.post(
        "/api/tasks",
        json={"title": "Task to Delete"}
    )
    assert create_response.status_code in [200, 201]
    task_id = create_response.json()["id"]

    # Delete the task
    delete_response = await client.delete(f"/api/tasks/{task_id}")
    assert delete_response.status_code in [200, 204]


@pytest.mark.asyncio
async def test_decompose_task(client: AsyncClient):
    """Test AI task decomposition."""
    response = await client.post(
        "/api/tasks/decompose",
        json={"description": "Build a web application with user authentication"}
    )
    # May fail if AI service not available
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "subtasks" in data
        assert isinstance(data["subtasks"], list)


@pytest.mark.asyncio
async def test_get_task_not_found(client: AsyncClient):
    """Test getting non-existent task."""
    response = await client.get("/api/tasks/non-existent-uuid")
    assert response.status_code in [404, 422]  # 422 for invalid UUID format
