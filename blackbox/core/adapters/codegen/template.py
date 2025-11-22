# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Template Rendering Adapter

Provides Jinja2-based template rendering for code generation.
Supports string templates, file templates, and directory trees.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template, TemplateError, TemplateSyntaxError
from blackbox.core.base_adapter import MCPAdapter


class TemplateAdapter(MCPAdapter):
    """
    Template rendering adapter using Jinja2.
    
    Enables code generation through template rendering with variable substitution.
    Supports inline string templates and file-based templates.
    """
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template adapter.
        
        Args:
            template_dir: Default directory for template files. If None, uses ./templates
        """
        self.template_dir = template_dir or os.path.join(os.getcwd(), "templates")
        self._env: Optional[Environment] = None

    @property
    def env(self) -> Environment:
        """Lazy-load Jinja2 environment"""
        if self._env is None:
            if os.path.exists(self.template_dir):
                self._env = Environment(
                    loader=FileSystemLoader(self.template_dir),
                    autoescape=False,  # Code generation needs raw output
                    trim_blocks=True,
                    lstrip_blocks=True,
                    keep_trailing_newline=True
                )
            else:
                # No template directory, use basic environment
                self._env = Environment(autoescape=False)

        # This is safe because _env is always assigned above
        assert self._env is not None
        return self._env
    
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute template rendering method"""
        
        if method == "render":
            return await self._render(inputs)
        elif method == "render_file":
            return await self._render_file(inputs)
        elif method == "render_to_file":
            return await self._render_to_file(inputs)
        elif method == "render_directory":
            return await self._render_directory(inputs)
        else:
            raise ValueError(f"Unknown codegen.template method: {method}")
    
    # ========== Template Rendering ==========
    
    async def _render(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render inline string template.
        
        Args:
            template: String template with Jinja2 syntax
            variables: Dictionary of variables for substitution
            
        Returns:
            Rendered content
        """
        template_str = inputs.get("template")
        variables = inputs.get("variables", {})
        
        if not template_str:
            raise ValueError("template is required")
        
        try:
            template = Template(template_str)
            rendered = template.render(**variables)
            
            return {
                "content": rendered,
                "size": len(rendered),
                "variables_used": list(variables.keys())
            }
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error at line {e.lineno}: {e.message}")
        except TemplateError as e:
            raise ValueError(f"Template rendering error: {str(e)}")
    
    async def _render_file(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render template from file.
        
        Args:
            template_path: Path to template file (relative to template_dir)
            variables: Dictionary of variables for substitution
            
        Returns:
            Rendered content
        """
        template_path = inputs.get("template_path") or inputs.get("template")
        variables = inputs.get("variables", {})
        
        if not template_path:
            raise ValueError("template_path is required")
        
        # Handle absolute vs relative paths
        if os.path.isabs(template_path):
            # Absolute path - load directly
            if not os.path.exists(template_path):
                raise FileNotFoundError(f"Template not found: {template_path}")
            
            template_str = Path(template_path).read_text(encoding='utf-8')
            template = Template(template_str)
        else:
            # Relative path - use template directory
            try:
                template = self.env.get_template(template_path)
            except Exception:
                raise FileNotFoundError(f"Template not found: {template_path} (in {self.template_dir})")
        
        try:
            rendered = template.render(**variables)
            
            return {
                "content": rendered,
                "size": len(rendered),
                "template": template_path,
                "variables_used": list(variables.keys())
            }
        except TemplateError as e:
            raise ValueError(f"Template rendering error: {str(e)}")
    
    async def _render_to_file(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render template and write to output file.
        
        Args:
            template_path: Path to template file OR inline template string
            output_path: Where to write rendered output
            variables: Dictionary of variables for substitution
            create_dirs: Create parent directories if they don't exist (default: True)
            
        Returns:
            File write info
        """
        template_path = inputs.get("template_path") or inputs.get("template")
        output_path = inputs.get("output_path") or inputs.get("output")
        variables = inputs.get("variables", {})
        create_dirs = inputs.get("create_dirs", True)
        
        if not template_path:
            raise ValueError("template_path or template is required")
        if not output_path:
            raise ValueError("output_path is required")
        
        # Determine if template_path is file or inline string
        # Check if file exists first (most reliable)
        is_file = False
        if os.path.exists(template_path):
            is_file = True
        elif not ('\n' in template_path or len(template_path) > 500):
            # Could be a relative path - check in template dir
            if os.path.exists(os.path.join(self.template_dir, template_path)):
                is_file = True
        
        if is_file:
            # Render from file
            result = await self._render_file({
                "template_path": template_path,
                "variables": variables
            })
        else:
            # Render inline template
            result = await self._render({
                "template": template_path,
                "variables": variables
            })
        
        content = result["content"]
        
        # Create output directory if needed
        output_file = Path(output_path)
        if create_dirs:
            output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        output_file.write_text(content, encoding='utf-8')
        
        return {
            "written": True,
            "path": str(output_file),
            "size": len(content),
            "template": template_path,
            "variables_used": list(variables.keys())
        }
    
    async def _render_directory(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render entire directory tree of templates.
        
        Args:
            template_dir: Source template directory
            output_dir: Output directory
            variables: Dictionary of variables for substitution
            pattern: File pattern to match (default: *)
            
        Returns:
            List of rendered files
        """
        template_dir = inputs.get("template_dir")
        output_dir = inputs.get("output_dir")
        variables = inputs.get("variables", {})
        pattern = inputs.get("pattern", "*")
        
        if not template_dir:
            raise ValueError("template_dir is required")
        if not output_dir:
            raise ValueError("output_dir is required")
        
        template_path = Path(template_dir)
        output_path = Path(output_dir)
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        
        rendered_files = []
        
        # Walk through template directory
        for template_file in template_path.rglob(pattern):
            if template_file.is_file():
                # Calculate relative path
                rel_path = template_file.relative_to(template_path)
                
                # Determine output path
                # Remove .j2 extension if present
                output_rel_path = str(rel_path)
                if output_rel_path.endswith('.j2'):
                    output_rel_path = output_rel_path[:-3]
                
                output_file_path = output_path / output_rel_path
                
                # Render template to file
                result = await self._render_to_file({
                    "template_path": str(template_file),
                    "output_path": str(output_file_path),
                    "variables": variables,
                    "create_dirs": True
                })
                
                rendered_files.append({
                    "source": str(template_file),
                    "output": str(output_file_path),
                    "size": result["size"]
                })
        
        return {
            "rendered_count": len(rendered_files),
            "files": rendered_files,
            "output_dir": str(output_path)
        }
