"""
BBX Console Configuration

All settings loaded from environment variables or .env file.
"""

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # App
    app_name: str = "BBX Console"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins as list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return self.cors_origins

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/bbx_console.db"

    # BBX Core path
    bbx_path: str = str(Path(__file__).parent.parent.parent.parent.parent)

    # WebSocket
    ws_heartbeat_interval: int = 30
    ws_max_connections: int = 100

    # Execution
    max_concurrent_executions: int = 10
    execution_timeout: int = 3600  # 1 hour

    # Authentication (optional)
    auth_enabled: bool = False
    auth_secret: str = "change-me-in-production"
    auth_algorithm: str = "HS256"
    auth_token_expire_minutes: int = 60 * 24  # 24 hours

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Data retention
    execution_retention_days: int = 30
    metrics_retention_days: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
