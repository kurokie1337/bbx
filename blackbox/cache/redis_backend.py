"""Redis cache backend for BBX."""

import hashlib
import json
from typing import Any, Optional

import redis


class RedisCache:
    """Redis-backed cache for workflow results."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "bbx:",
        ttl: int = 3600,
    ):
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password, decode_responses=True
        )
        self.prefix = prefix
        self.ttl = ttl

    def _make_key(self, workflow_id: str, inputs: dict) -> str:
        """Generate cache key from workflow and inputs."""
        input_hash = hashlib.md5(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
        return f"{self.prefix}{workflow_id}:{input_hash}"

    def get(self, workflow_id: str, inputs: dict) -> Optional[Any]:
        """Get cached workflow result."""
        key = self._make_key(workflow_id, inputs)
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None

    def set(self, workflow_id: str, inputs: dict, result: Any):
        """Cache workflow result."""
        key = self._make_key(workflow_id, inputs)
        self.client.setex(key, self.ttl, json.dumps(result, default=str))

    def invalidate(self, workflow_id: str):
        """Invalidate all cache entries for a workflow."""
        pattern = f"{self.prefix}{workflow_id}:*"
        for key in self.client.scan_iter(match=pattern):
            self.client.delete(key)

    def clear(self):
        """Clear all BBX cache entries."""
        pattern = f"{self.prefix}*"
        for key in self.client.scan_iter(match=pattern):
            self.client.delete(key)

    def stats(self) -> dict:
        """Get cache statistics."""
        pattern = f"{self.prefix}*"
        keys = list(self.client.scan_iter(match=pattern))
        return {
            "total_keys": len(keys),
            "memory_usage": sum(self.client.memory_usage(key) or 0 for key in keys),
            "redis_info": self.client.info("stats"),
        }
