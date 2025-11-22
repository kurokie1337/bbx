"""Integrate Redis cache into runtime."""

from blackbox.cache.redis_backend import RedisCache
from blackbox.core.config import config

# Initialize Redis cache if enabled
redis_cache = None
if config.get("cache.backend") == "redis":
    redis_cache = RedisCache(
        host=config.get("redis.host", "localhost"),
        port=config.get("redis.port", 6379),
        db=config.get("redis.db", 0),
        password=config.get("redis.password"),
        prefix=config.get("redis.prefix", "bbx:"),
        ttl=config.get("cache.ttl", 3600),
    )


def get_cache():
    """Get cache instance."""
    return redis_cache
