"""Example BBX plugin."""
from blackbox.plugins.api import BBXPlugin

class ExamplePlugin(BBXPlugin):
    """Example plugin that processes text."""

    def __init__(self):
        super().__init__("example", "1.0.0")

    def execute(self, inputs):
        """Process text input."""
        text = inputs.get("text", "")
        return {
            "processed_text": text.upper(),
            "length": len(text)
        }

    def validate_inputs(self, inputs):
        """Validate inputs."""
        return "text" in inputs
