"""Plugin API for BBX."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BBXPlugin(ABC):
    """Base class for BBX plugins."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plugin logic."""

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate plugin inputs."""
        return True

    def on_load(self):
        """Called when plugin is loaded."""

    def on_unload(self):
        """Called when plugin is unloaded."""
