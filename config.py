from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    BLACKBOX_API_URL: str = "http://localhost:8000"
    BLACKBOX_USERNAME: str = ""
    BLACKBOX_PASSWORD: str = ""
    BLACKBOX_API_TIMEOUT: int = 30
    LOG_LEVEL: str = "INFO"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
