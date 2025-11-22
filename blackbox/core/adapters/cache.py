"""Cache management adapter."""

from blackbox.core.base_adapter import BaseAdapter
from blackbox.core.cache_integration import get_cache


class CacheAdapter(BaseAdapter):
    """Adapter for cache operations."""

    def __init__(self):
        super().__init__("cache")

    def clear(self):
        """Clear all cache."""
        cache = get_cache()
        if cache:
            cache.clear()
            return {"status": "cleared"}
        return {"status": "no_cache"}

    def invalidate(self, workflow_id: str):
        """Invalidate workflow cache."""
        cache = get_cache()
        if cache:
            cache.invalidate(workflow_id)
            return {"status": "invalidated", "workflow_id": workflow_id}
        return {"status": "no_cache"}

    def stats(self):
        """Get cache statistics."""
        cache = get_cache()
        if cache:
            return cache.stats()
        return {"status": "no_cache"}
