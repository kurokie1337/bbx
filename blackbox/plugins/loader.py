"""Plugin loader and manager."""

import importlib.util
import sys
from pathlib import Path
from typing import Dict

from blackbox.plugins.api import BBXPlugin


class PluginLoader:
    """Load and manage plugins."""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, BBXPlugin] = {}

    def load_plugin(self, plugin_path: Path) -> BBXPlugin:
        """Load a plugin from file."""
        spec = importlib.util.spec_from_file_location(plugin_path.stem, plugin_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[plugin_path.stem] = module
        spec.loader.exec_module(module)

        # Find BBXPlugin subclass
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BBXPlugin)
                and attr is not BBXPlugin
            ):
                plugin = attr()
                plugin.on_load()
                return plugin

        raise ValueError(f"No BBXPlugin found in {plugin_path}")

    def load_all_plugins(self):
        """Load all plugins from plugin directory."""
        if not self.plugin_dir.exists():
            self.plugin_dir.mkdir(parents=True)
            return

        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            try:
                plugin = self.load_plugin(plugin_file)
                self.plugins[plugin.name] = plugin
                print(f"Loaded plugin: {plugin.name} v{plugin.version}")
            except Exception as e:
                print(f"Failed to load {plugin_file}: {e}")

    def get_plugin(self, name: str) -> BBXPlugin:
        """Get plugin by name."""
        return self.plugins.get(name)

    def execute_plugin(self, name: str, inputs: Dict) -> Dict:
        """Execute plugin."""
        plugin = self.get_plugin(name)
        if not plugin:
            raise ValueError(f"Plugin not found: {name}")

        if not plugin.validate_inputs(inputs):
            raise ValueError(f"Invalid inputs for plugin: {name}")

        return plugin.execute(inputs)
