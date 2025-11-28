# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Core Adapters Package

BBX 2.0 compatible adapters with support for:
- Hook integration
- Context tiering
- Batch operations via AgentRing
"""

# Note: Core adapters are imported by registry.py
# Don't import them here to avoid circular imports

# V2 Base classes (safe to import)
from .v2_base import (
    V2AdapterMixin,
    V2BaseAdapter,
    V2AdapterFactory,
    AdapterMetadata,
    MethodMetadata,
    ParamMetadata,
    v2_method,
    ensure_v2_compatible,
)

__all__ = [
    # V2 Base
    "V2AdapterMixin",
    "V2BaseAdapter",
    "V2AdapterFactory",
    "AdapterMetadata",
    "MethodMetadata",
    "ParamMetadata",
    "v2_method",
    "ensure_v2_compatible",
]
