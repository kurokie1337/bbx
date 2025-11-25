# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Workflow Generator - Generate BBX workflows using local LLM
"""

import re
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

from .model_manager import ModelManager


class WorkflowGenerator:
    """Generate BBX workflows from natural language using local LLM"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize workflow generator

        Args:
            model_name: Name of model to use (defaults to qwen-0.5b)

        Raises:
            ImportError: If llama-cpp-python is not installed
            FileNotFoundError: If model is not downloaded
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "Install with: pip install llama-cpp-python"
            )

        manager = ModelManager()

        # Use default model if not specified
        if model_name is None:
            model_name = manager.load_default_model_from_config()

        # Get model path (raises FileNotFoundError if not exists)
        model_path = manager.get_model_path(model_name)

        print(f"🤖 Loading AI model: {model_name}...")

        # Initialize llama.cpp
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,  # Context window
            n_threads=4,  # CPU threads
            n_gpu_layers=0,  # CPU-only for now (portable)
            verbose=False,
        )

        self.model_name = model_name
        print("✅ Model loaded successfully")

    def generate(self, description: str, output_file: Optional[str] = None) -> str:
        """
        Generate BBX workflow from natural language description

        Args:
            description: Natural language task description
            output_file: Optional file path to save generated workflow

        Returns:
            Generated YAML workflow as string
        """
        print(f"🤖 Generating workflow with {self.model_name}...")

        # Create prompt
        prompt = self._create_prompt(description)

        # Generate
        response = self.llm(
            prompt,
            max_tokens=1024,
            temperature=0.1,  # Low temperature = more deterministic
            top_p=0.9,
            stop=["```", "---", "<|end|>"],  # Stop sequences
            echo=False,
        )

        # Extract YAML
        raw_output = response["choices"][0]["text"]
        yaml_content = self._extract_yaml(raw_output)

        # Validate basic YAML structure
        if not self._is_valid_yaml_structure(yaml_content):
            raise ValueError(
                "Generated workflow has invalid YAML structure.\n"
                f"Raw output:\n{raw_output}"
            )

        # Save to file if requested
        if output_file:
            Path(output_file).write_text(yaml_content, encoding="utf-8")
            print(f"✅ Saved to: {output_file}")

        return yaml_content

    def _create_prompt(self, description: str) -> str:
        """
        Create BBX-specific prompt for workflow generation

        Args:
            description: User's task description

        Returns:
            Formatted prompt for LLM
        """
        # BBX-specific prompt template
        return f"""<|system|>
You are BBX-GPT, an AI specialized in generating BBX workflow YAML files.

BBX workflows use this structure:
- Workflows have an 'id' and 'steps'
- Each step has: id, mcp, method, inputs
- Use "universal" adapter with docker:// images
- Commands are YAML arrays: [cmd, arg1, arg2]
- Use depends_on for step dependencies

Example:
```yaml
workflow:
  id: deploy_app
  steps:
    - id: build
      mcp: universal
      method: run
      inputs:
        uses: docker://node:20-alpine
        cmd: [npm, run, build]

    - id: deploy
      mcp: universal
      method: run
      inputs:
        uses: docker://amazon/aws-cli:latest
        cmd: [aws, s3, sync, dist/, s3://bucket/]
      depends_on: [build]
```
<|end|>
<|user|>
Generate a BBX workflow for this task: {description}
<|end|>
<|assistant|>
```yaml
workflow:
  id: generated_workflow
  steps:"""

    def _extract_yaml(self, text: str) -> str:
        """
        Extract and clean YAML from LLM output

        Args:
            text: Raw LLM output

        Returns:
            Cleaned YAML content
        """
        # Remove markdown code fences
        text = re.sub(r"```ya?ml\n?", "", text)
        text = re.sub(r"```\n?", "", text)

        # Remove everything after first "---" (common separator)
        if "---" in text:
            text = text.split("---")[0]

        # Prepend workflow header if missing
        if not text.strip().startswith("workflow:"):
            text = "workflow:\n  id: generated_workflow\n  steps:\n" + text

        # Trim whitespace
        return text.strip()

    def _is_valid_yaml_structure(self, yaml_content: str) -> bool:
        """
        Basic validation of YAML structure

        Args:
            yaml_content: YAML content to validate

        Returns:
            True if structure looks valid
        """
        # Check for required keywords
        required = ["workflow:", "steps:", "- id:", "mcp:", "method:"]
        return all(keyword in yaml_content for keyword in required)

    def get_model_info(self) -> dict:
        """
        Get information about current model

        Returns:
            Dict with model name and metadata
        """
        manager = ModelManager()
        model_info = manager.MODELS.get(self.model_name, {})

        return {
            "name": self.model_name,
            "size": model_info.get("size", "Unknown"),
            "description": model_info.get("description", ""),
        }


def check_dependencies() -> bool:
    """
    Check if all dependencies are installed

    Returns:
        True if all dependencies available
    """
    if not LLAMA_CPP_AVAILABLE:
        print("❌ llama-cpp-python is not installed")
        print("   Install with: pip install llama-cpp-python")
        return False

    return True
