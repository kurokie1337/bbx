"""
BBX AI Module - Local LLM workflow generation
"""

from .generator import WorkflowGenerator
from .model_manager import ModelManager

__all__ = ["ModelManager", "WorkflowGenerator"]
