# BBX Plugin System

Create custom adapters as plugins.

## Creating a Plugin

\`\`\`python
from blackbox.plugins.api import BBXPlugin

class MyPlugin(BBXPlugin):
    def __init__(self):
        super().__init__("my_plugin", "1.0.0")

    def execute(self, inputs):
        # Your logic
        return {"result": "success"}
\`\`\`

## Installing Plugin

\`\`\`bash
cp my_plugin.py plugins/
python cli.py plugin list
\`\`\`

## Using in Workflow

\`\`\`yaml
- id: use_plugin
  mcp: bbx.plugin
  method: exec
  inputs:
    plugin_name: my_plugin
    inputs:
      param1: value1
\`\`\`
