"""
BBX Console Backend - Test Configuration
"""
import asyncio
from typing import AsyncGenerator, Generator
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db
from app.core.config import settings


# Test database URL (in-memory SQLite for speed)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for each test."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_workflow_data() -> dict:
    """Sample workflow data for testing."""
    return {
        "name": "test-workflow",
        "version": "1.0.0",
        "description": "Test workflow for unit tests",
        "steps": [
            {
                "id": "step1",
                "name": "First Step",
                "adapter": "shell",
                "config": {
                    "command": "echo 'Hello'"
                }
            },
            {
                "id": "step2",
                "name": "Second Step",
                "adapter": "shell",
                "config": {
                    "command": "echo 'World'"
                },
                "depends_on": ["step1"]
            }
        ]
    }


@pytest.fixture
def sample_task_data() -> dict:
    """Sample task data for testing."""
    return {
        "title": "Test Task",
        "description": "This is a test task",
        "priority": "medium",
        "status": "todo"
    }


@pytest.fixture
def sample_agent_data() -> dict:
    """Sample agent data for testing."""
    return {
        "id": "agent-001",
        "name": "TestAgent",
        "type": "worker",
        "status": "idle",
        "capabilities": ["shell", "python"],
        "current_task": None
    }
