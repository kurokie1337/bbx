# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Model Manager - Download and manage local LLM models for BBX
"""

from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


class ModelManager:
    """Manage downloadable AI models for BBX workflow generation"""

    MODELS: Dict[str, Dict[str, str]] = {
        "qwen-0.5b": {
            "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            "size": "250MB",
            "description": "Fast and compact AI for BBX workflow generation",
            "filename": "qwen-0.5b.gguf",
        },
    }

    DEFAULT_MODEL = "qwen-0.5b"  # Only one model for v1.0

    def __init__(self):
        self.models_dir = Path.home() / ".bbx" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download(self, model_name: str, force: bool = False) -> Path:
        """
        Download a model from HuggingFace

        Args:
            model_name: Name of the model (e.g., 'qwen-0.5b')
            force: Force re-download even if exists

        Returns:
            Path to downloaded model file

        Raises:
            ValueError: If model name is unknown
        """
        if model_name not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            raise ValueError(
                f"Unknown model: {model_name}\n" f"Available models: {available}"
            )

        model_info = self.MODELS[model_name]
        model_path = self.models_dir / model_info["filename"]

        # Check if already exists
        if model_path.exists() and not force:
            print(f"✅ Model '{model_name}' already downloaded")
            print(f"   Location: {model_path}")
            return model_path

        # Download
        print(f"📥 Downloading {model_name} ({model_info['size']})...")
        print(f"   From: {model_info['url']}")

        try:
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(model_path, "wb") as f:
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=model_name,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print("✅ Model downloaded successfully")
            print(f"   Saved to: {model_path}")
            return model_path

        except requests.RequestException as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")

    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """
        Get path to a model file

        Args:
            model_name: Name of the model (defaults to DEFAULT_MODEL)

        Returns:
            Path to model file

        Raises:
            FileNotFoundError: If model is not installed
        """
        if model_name is None:
            model_name = self.DEFAULT_MODEL

        if model_name not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            raise ValueError(f"Unknown model: {model_name}\nAvailable: {available}")

        model_path = self.models_dir / self.MODELS[model_name]["filename"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model '{model_name}' not found.\n"
                f"Download with: bbx model download {model_name}"
            )

        return model_path

    def list_installed(self) -> List[str]:
        """
        List installed models

        Returns:
            List of installed model names
        """
        installed = []
        for model_name, model_info in self.MODELS.items():
            model_path = self.models_dir / model_info["filename"]
            if model_path.exists():
                installed.append(model_name)
        return installed

    def list_available(self) -> Dict[str, Dict[str, str]]:
        """
        List all available models with installation status

        Returns:
            Dict mapping model names to their info + 'installed' status
        """
        result = {}
        for model_name, model_info in self.MODELS.items():
            model_path = self.models_dir / model_info["filename"]
            result[model_name] = {
                **model_info,
                "installed": model_path.exists(),
                "path": str(model_path) if model_path.exists() else None,
            }
        return result

    def remove(self, model_name: str) -> bool:
        """
        Remove an installed model

        Args:
            model_name: Name of the model to remove

        Returns:
            True if removed, False if not installed
        """
        if model_name not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            raise ValueError(f"Unknown model: {model_name}\nAvailable: {available}")

        model_path = self.models_dir / self.MODELS[model_name]["filename"]

        if model_path.exists():
            model_path.unlink()
            print(f"🗑️  Removed {model_name}")
            return True
        else:
            print(f"ℹ️  Model '{model_name}' is not installed")
            return False

    def get_default_model(self) -> str:
        """
        Get the default model name

        Returns:
            Name of default model
        """
        return self.DEFAULT_MODEL

    def set_default_model(self, model_name: str):
        """
        Set the default model

        Args:
            model_name: Name of the model to set as default

        Raises:
            ValueError: If model is unknown
        """
        if model_name not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            raise ValueError(f"Unknown model: {model_name}\nAvailable: {available}")

        # Save to config file
        config_file = Path.home() / ".bbx" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        import json

        config = {}
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)

        config["default_model"] = model_name

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Default model set to: {model_name}")

    def load_default_model_from_config(self) -> str:
        """
        Load default model from config file

        Returns:
            Model name from config, or DEFAULT_MODEL if not set
        """
        config_file = Path.home() / ".bbx" / "config.json"

        if not config_file.exists():
            return self.DEFAULT_MODEL

        import json

        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            return config.get("default_model", self.DEFAULT_MODEL)
        except (json.JSONDecodeError, IOError):
            return self.DEFAULT_MODEL
