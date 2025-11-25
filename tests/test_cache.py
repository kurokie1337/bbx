# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

import os
import tempfile
from blackbox.core.cache import WorkflowCache, get_cache

def test_cache_basic():
    """Test basic cache operations"""
    cache = WorkflowCache(max_size=10)
    
    data = {"workflow": {"id": "test", "steps": []}}
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        f.write("workflow:\n  id: test\n  steps: []")
        temp_path = f.name
    
    try:
        # First access - should parse
        result = cache.load_or_parse(temp_path)
        assert result == data
        assert cache.size() == 1
        
        # Second access - should use cache
        result2 = cache.load_or_parse(temp_path)
        assert result2 == data
        assert cache.size() == 1
    finally:
        os.unlink(temp_path)

def test_cache_invalidation():
    """Test cache invalidation on file modification"""
    cache = WorkflowCache()
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        f.write("workflow:\n  id: v1")
        temp_path = f.name
    
    try:
        # Load first version
        result1 = cache.load_or_parse(temp_path)
        assert result1['workflow']['id'] == 'v1'
        
        # Modify file
        import time
        time.sleep(0.1)  # Ensure mtime changes
        with open(temp_path, 'w') as f:
            f.write("workflow:\n  id: v2")
        
        # Should reload new version
        result2 = cache.load_or_parse(temp_path)
        assert result2['workflow']['id'] == 'v2'
    finally:
        os.unlink(temp_path)

def test_cache_lru_eviction():
    """Test LRU eviction when cache is full"""
    cache = WorkflowCache(max_size=2)
    
    files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            f.write(f"workflow:\n  id: file{i}")
            files.append(f.name)
    
    try:
        # Load 3 files (cache size = 2, so oldest should be evicted)
        cache.load_or_parse(files[0])
        cache.load_or_parse(files[1])
        cache.load_or_parse(files[2])
        
        assert cache.size() == 2
        
        # First file should be evicted
        assert cache.get(files[0]) is None
        assert cache.get(files[1]) is not None
        assert cache.get(files[2]) is not None
    finally:
        for f in files:
            os.unlink(f)

def test_global_cache():
    """Test global cache instance"""
    cache1 = get_cache()
    cache2 = get_cache()
    
    assert cache1 is cache2  # Should be same instance
