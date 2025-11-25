# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Workflow parsing cache for Blackbox.
Caches parsed YAML files to avoid repeated parsing.
"""

import os
from collections import OrderedDict
from typing import Any, Dict, Optional

import yaml


class WorkflowCache:
    """
    LRU cache for parsed workflow files.

    Caches YAML parsing results and invalidates on file modification.
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached workflows
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._mtimes: Dict[str, float] = {}

    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get parsed workflow from cache.

        Args:
            file_path: Path to workflow file

        Returns:
            Parsed workflow dictionary or None if not cached/invalid
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return None

        # Get current modification time
        current_mtime = os.path.getmtime(file_path)
        cached_mtime = self._mtimes.get(file_path)

        # Check if cached and not modified
        if (
            file_path in self._cache
            and cached_mtime is not None
            and current_mtime == cached_mtime
        ):
            # Move to end (mark as recently used)
            self._cache.move_to_end(file_path)
            return self._cache[file_path]

        return None

    def put(self, file_path: str, data: Dict[str, Any]):
        """
        Cache parsed workflow.

        Args:
            file_path: Path to workflow file
            data: Parsed workflow data
        """
        # Get modification time
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
        else:
            mtime = 0

        # Add to cache
        self._cache[file_path] = data
        self._mtimes[file_path] = mtime

        # Move to end (mark as recently used)
        self._cache.move_to_end(file_path)

        # Evict oldest if needed
        while len(self._cache) > self.max_size:
            oldest_path = next(iter(self._cache))
            del self._cache[oldest_path]
            del self._mtimes[oldest_path]

    def load_or_parse(self, file_path: str) -> Dict[str, Any]:
        """
        Load from cache or parse YAML file.

        Args:
            file_path: Path to workflow file

        Returns:
            Parsed workflow data
        """
        # Try cache first
        cached = self.get(file_path)
        if cached is not None:
            return cached

        # Parse file
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Cache result
        self.put(file_path, data)

        return data

    def invalidate(self, file_path: str):
        """
        Invalidate cache entry for a file.

        Args:
            file_path: Path to workflow file
        """
        if file_path in self._cache:
            del self._cache[file_path]
        if file_path in self._mtimes:
            del self._mtimes[file_path]

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._mtimes.clear()

    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)


# Global cache instance
_global_cache = WorkflowCache()


def get_cache() -> WorkflowCache:
    """Get global workflow cache instance"""
    return _global_cache
