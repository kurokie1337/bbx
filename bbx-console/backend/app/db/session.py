"""
Database session management

Uses SQLAlchemy async with aiosqlite for SQLite support.
"""

import logging
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create async engine with error handling
try:
    # Try to use configured database URL
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        future=True,
    )
    logger.info(f"Database engine created for: {settings.database_url.split('://')[0]}://...")
except Exception as e:
    logger.warning(f"Failed to create engine with configured URL: {e}")
    # Fallback to SQLite
    logger.info("Falling back to SQLite database")
    engine = create_async_engine(
        "sqlite+aiosqlite:///./data/bbx_console.db",
        echo=settings.debug,
        future=True,
    )

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    try:
        # Ensure data directory exists for SQLite
        db_url = str(engine.url)
        if "sqlite" in db_url:
            db_path = db_url.replace("sqlite+aiosqlite:///", "")
            if db_path.startswith("./"):
                db_path = db_path[2:]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Import models to register them
        try:
            from app.db.models import execution, task, agent_metrics  # noqa
        except ImportError as e:
            logger.warning(f"Could not import database models: {e}")
            logger.info("Running without database models")
            return

        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("Backend will run without database")


async def close_db():
    """Close database connections"""
    try:
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning(f"Error closing database: {e}")
