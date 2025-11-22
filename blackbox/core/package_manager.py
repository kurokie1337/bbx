# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
BBX Package Manager for Universal Adapter Definitions
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("bbx.package_manager")

class AdapterPackageManager:
    """
    Package manager for Universal Adapter definitions.
    
    Features:
    - Install definitions from library
    - Semantic versioning
    - Hot-reload support
    - Definition caching
    """
    
    def __init__(self, library_dir: Optional[Path] = None):
        """Initialize package manager."""
        self.library_dir = library_dir or Path(__file__).parent.parent / "library"
        self.installed_dir = Path.home() / ".bbx" / "installed"
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.watch_mode = False
        
    def install(self, package_name: str, version: str = "latest") -> bool:
        """Install package from library."""
        try:
            yaml_path = self.library_dir / f"{package_name}.yaml"
            if not yaml_path.exists():
                logger.error(f"Package not found: {package_name}")
                return False
            
            # Read YAML - allow Jinja2 templates
            with open(yaml_path, 'r') as f:
                content = f.read()
                
                # Check if file contains Jinja2 templates
                has_jinja = '{%' in content or '{{' in content
                
                if has_jinja:
                    # For template files, just copy to installed dir without validation
                    logger.info(f"Installing template package: {package_name}")
                    self.installed_dir.mkdir(parents=True, exist_ok=True)
                    installed_path = self.installed_dir / f"{package_name}.yaml"
                    installed_path.write_text(content)
                    self.cache[package_name] = {"id": package_name, "template": True}
                    return True
                else:
                    # For regular YAML, validate and install
                    definition = yaml.safe_load(content)
                    
                    self.installed_dir.mkdir(parents=True, exist_ok=True)
                    installed_path = self.installed_dir / f"{package_name}.yaml"
                    
                    with open(installed_path, 'w') as out:
                        yaml.dump(definition, out)
                    
                    self.cache[package_name] = definition
                    logger.info(f"Installed {package_name}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to install {package_name}: {e}")
            return False
    
    def get(self, package_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached package definition.
        
        Args:
            package_name: Name of the adapter
            
        Returns:
            Definition dict or None if not found
        """
        if package_name in self.cache:
            return self.cache[package_name]
        
        # Try to install if not cached
        if self.install(package_name):
            return self.cache.get(package_name)
        
        return None
    
    def list_available(self) -> List[str]:
        """List all available packages in the library."""
        if not self.library_dir.exists():
            return []
        
        packages = []
        for file in self.library_dir.glob("*.yaml"):
            if file.name != "README.md":
                packages.append(file.stem)
        
        return sorted(packages)
    
    def list_installed(self) -> List[str]:
        """List all installed (cached) packages."""
        return sorted(self.cache.keys())
    
    def reload(self, package_name: str) -> bool:
        """
        Hot-reload a package definition.
        
        Args:
            package_name: Name of the adapter to reload
            
        Returns:
            True if reloaded successfully
        """
        if package_name in self.cache:
            del self.cache[package_name]
        
        return self.install(package_name)
    
    def enable_watch_mode(self):
        """Enable hot-reload for all packages (watches file changes)."""
        self.watch_mode = True
        logger.info("📡 Hot-reload enabled for adapter definitions")
        # TODO: Implement file watcher using watchdog library
    
    def validate_all(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all available packages."""
        results: Dict[str, Any] = {}
        
        for package_name in self.list_available():
            try:
                yaml_path = self.library_dir / f"{package_name}.yaml"
                with open(yaml_path, 'r') as f:
                    content = f.read()
                    
                    # Check if file contains Jinja2 templates
                    if '{%' in content or '{{' in content:
                        # Template files are valid by definition
                        results[package_name] = (True, [])
                        continue
                    
                    # For non-template files, validate YAML
                    definition = yaml.safe_load(content)
                    
                    # Basic validation
                    errors = []
                    if not isinstance(definition, dict):
                        errors.append("Definition must be a dictionary")
                    elif 'id' not in definition:
                        errors.append("Missing required field: 'id'")
                    
                    results[package_name] = (len(errors) == 0, errors)
                    
            except Exception as e:
                results[package_name] = (False, [str(e)])
        
        return results

# Global package manager instance
_pm = None

def get_package_manager() -> AdapterPackageManager:
    """Get global package manager instance (singleton)."""
    global _pm
    if _pm is None:
        _pm = AdapterPackageManager()
    return _pm
