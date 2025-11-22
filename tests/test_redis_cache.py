"""Tests for Redis cache."""
import pytest
from blackbox.cache.redis_backend import RedisCache

@pytest.fixture
def cache():
    return RedisCache(host="localhost", port=6379, db=15)

def test_cache_set_get(cache):
    """Test basic cache operations."""
    cache.set("test_wf", {"input": "test"}, {"output": "result"})
    result = cache.get("test_wf", {"input": "test"})
    assert result == {"output": "result"}

def test_cache_invalidate(cache):
    """Test cache invalidation."""
    cache.set("test_wf", {"input": "test"}, {"output": "result"})
    cache.invalidate("test_wf")
    result = cache.get("test_wf", {"input": "test"})
    assert result is None
