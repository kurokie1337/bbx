"""Plugin adapter for executing loaded plugins."""

from blackbox.core.base_adapter import BaseAdapter
from blackbox.plugins.loader import PluginLoader

loader = PluginLoader()
loader.load_all_plugins()


class PluginAdapter(BaseAdapter):
    """Adapter for plugin execution."""

    def __init__(self):
        super().__init__("plugin")

    def execute(self, plugin_name: str, inputs: dict):
        """Execute plugin."""
        return loader.execute_plugin(plugin_name, inputs)

    def list_plugins(self):
        """List all loaded plugins."""
        return [{"name": p.name, "version": p.version} for p in loader.plugins.values()]
