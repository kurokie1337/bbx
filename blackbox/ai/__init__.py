# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX AI Module - Local LLM workflow generation
"""

from .generator import WorkflowGenerator
from .model_manager import ModelManager

__all__ = ["ModelManager", "WorkflowGenerator"]
