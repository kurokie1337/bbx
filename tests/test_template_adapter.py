# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
from blackbox.core.adapters.codegen.template import TemplateAdapter


@pytest.fixture
def template_adapter():
    """Create template adapter instance"""
    return TemplateAdapter()


@pytest.fixture
def templates_dir(tmp_path):
    """Create temporary templates directory"""
    templates = tmp_path / "templates"
    templates.mkdir()
    return templates


# ========== Basic Template Rendering ==========

@pytest.mark.asyncio
async def test_render_simple_string(template_adapter):
    """Test basic string template rendering"""
    result = await template_adapter.execute("render", {
        "template": "Hello {{name}}!",
        "variables": {"name": "World"}
    })
    
    assert result["content"] == "Hello World!"
    assert result["size"] == len("Hello World!")
    assert "name" in result["variables_used"]


@pytest.mark.asyncio
async def test_render_multiline_template(template_adapter):
    """Test multiline template with loops"""
    template = """
    Items:
    {% for item in items %}
    - {{item}}
    {% endfor %}
    """
    
    result = await template_adapter.execute("render", {
        "template": template,
        "variables": {"items": ["apple", "banana", "orange"]}
    })
    
    assert "- apple" in result["content"]
    assert "- banana" in result["content"]
    assert "- orange" in result["content"]


@pytest.mark.asyncio
async def test_render_with_conditionals(template_adapter):
    """Test template with conditional logic"""
    template = """
    {% if is_admin %}
    Admin Panel
    {% else %}
    User Panel
    {% endif %}
    """
    
    # Test admin=true
    result = await template_adapter.execute("render", {
        "template": template,
        "variables": {"is_admin": True}
    })
    assert "Admin Panel" in result["content"]
    
    # Test admin=false
    result = await template_adapter.execute("render", {
        "template": template,
        "variables": {"is_admin": False}
    })
    assert "User Panel" in result["content"]


@pytest.mark.asyncio
async def test_render_code_template(template_adapter):
    """Test rendering actual code (React component)"""
    template = """
    import React from 'react';
    
    export const {{component_name}} = () => {
      return (
        <div className="{{class_name}}">
          <h1>{{title}}</h1>
        </div>
      );
    };
    """
    
    result = await template_adapter.execute("render", {
        "template": template,
        "variables": {
            "component_name": "AnimeCard",
            "class_name": "anime-card",
            "title": "Anime Title"
        }
    })
    
    assert "export const AnimeCard" in result["content"]
    assert 'className="anime-card"' in result["content"]
    assert "<h1>Anime Title</h1>" in result["content"]


# ========== File-Based Templates ==========

@pytest.mark.asyncio
async def test_render_file_template(template_adapter, templates_dir):
    """Test rendering from template file"""
    # Create template file
    template_file = templates_dir / "greeting.txt.j2"
    template_file.write_text("Hello {{name}}!")
    
    # Create adapter with template directory
    adapter = TemplateAdapter(template_dir=str(templates_dir))
    
    result = await adapter.execute("render_file", {
        "template_path": "greeting.txt.j2",
        "variables": {"name": "Alice"}
    })
    
    assert result["content"] == "Hello Alice!"
    assert result["template"] == "greeting.txt.j2"


@pytest.mark.asyncio
async def test_render_file_absolute_path(template_adapter, tmp_path):
    """Test rendering from absolute path"""
    template_file = tmp_path / "test_template.txt"
    template_file.write_text("Value: {{value}}")
    
    result = await template_adapter.execute("render_file", {
        "template_path": str(template_file),
        "variables": {"value": 42}
    })
    
    assert result["content"] == "Value: 42"


@pytest.mark.asyncio
async def test_render_file_not_found(template_adapter):
    """Test error when template file doesn't exist"""
    with pytest.raises(FileNotFoundError):
        await template_adapter.execute("render_file", {
            "template_path": "nonexistent.txt",
            "variables": {}
        })


# ========== Render to File ==========

@pytest.mark.asyncio
async def test_render_to_file_inline(template_adapter, tmp_path):
    """Test rendering inline template to file"""
    output_file = tmp_path / "output.txt"
    
    result = await template_adapter.execute("render_to_file", {
        "template": "Generated: {{timestamp}}",
        "output_path": str(output_file),
        "variables": {"timestamp": "2025-01-01"}
    })
    
    assert result["written"] is True
    assert output_file.exists()
    assert output_file.read_text() == "Generated: 2025-01-01"


@pytest.mark.asyncio
async def test_render_to_file_from_template(template_adapter, templates_dir, tmp_path):
    """Test rendering template file to output file"""
    # Create template
    template_file = templates_dir / "component.tsx.j2"
    template_file.write_text("""
import React from 'react';

export const {{name}} = () => {
  return <div>{{name}}</div>;
};
""")
    
    # Create adapter with template directory
    adapter = TemplateAdapter(template_dir=str(templates_dir))
    
    output_file = tmp_path / "Component.tsx"
    
    result = await adapter.execute("render_to_file", {
        "template_path": "component.tsx.j2",
        "output_path": str(output_file),
        "variables": {"name": "MyComponent"}
    })
    
    assert result["written"] is True
    assert output_file.exists()
    content = output_file.read_text()
    assert "export const MyComponent" in content


@pytest.mark.asyncio
async def test_render_to_file_creates_dirs(template_adapter, tmp_path):
    """Test that render_to_file creates parent directories"""
    output_file = tmp_path / "deep" / "nested" / "dir" / "file.txt"
    
    result = await template_adapter.execute("render_to_file", {
        "template": "Content",
        "output_path": str(output_file),
        "variables": {},
        "create_dirs": True
    })
    
    assert result["written"] is True
    assert output_file.exists()
    assert output_file.parent.exists()


# ========== Directory Rendering ==========

@pytest.mark.asyncio
async def test_render_directory(template_adapter, tmp_path):
    """Test rendering entire directory of templates"""
    # Create template directory structure
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    
    (template_dir / "file1.txt.j2").write_text("Value: {{value}}")
    (template_dir / "file2.txt.j2").write_text("Name: {{name}}")
    
    subdir = template_dir / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt.j2").write_text("Count: {{count}}")
    
    # Render directory
    output_dir = tmp_path / "output"
    
    adapter = TemplateAdapter(template_dir=str(template_dir))
    result = await adapter.execute("render_directory", {
        "template_dir": str(template_dir),
        "output_dir": str(output_dir),
        "variables": {"value": 42, "name": "Test", "count": 3}
    })
    
    assert result["rendered_count"] == 3
    assert (output_dir / "file1.txt").exists()
    assert (output_dir / "file2.txt").exists()
    assert (output_dir / "subdir" / "file3.txt").exists()
    
    # Check content
    assert (output_dir / "file1.txt").read_text() == "Value: 42"
    assert (output_dir / "file2.txt").read_text() == "Name: Test"
    assert (output_dir / "subdir" / "file3.txt").read_text() == "Count: 3"


# ========== Error Handling ==========

@pytest.mark.asyncio
async def test_render_invalid_syntax(template_adapter):
    """Test error handling for invalid template syntax"""
    with pytest.raises(ValueError, match="Template syntax error"):
        await template_adapter.execute("render", {
            "template": "{{invalid",
            "variables": {}
        })


@pytest.mark.asyncio
async def test_render_missing_required_arg(template_adapter):
    """Test error when template is missing"""
    with pytest.raises(ValueError, match="template is required"):
        await template_adapter.execute("render", {
            "variables": {"name": "Test"}
        })


@pytest.mark.asyncio
async def test_render_unknown_method(template_adapter):
    """Test error for unknown method"""
    with pytest.raises(ValueError, match="Unknown codegen.template method"):
        await template_adapter.execute("invalid_method", {})


# ========== Real-World Use Cases ==========

@pytest.mark.asyncio
async def test_generate_react_component(template_adapter, tmp_path):
    """Test generating a complete React component"""
    template = """
import React from 'react';

interface {{name}}Props {
  {% for prop in props %}
  {{prop.name}}: {{prop.type}};
  {% endfor %}
}

export const {{name}}: React.FC<{{name}}Props> = ({
  {% for prop in props %}
  {{prop.name}},
  {% endfor %}
}) => {
  return (
    <div className="{{name | lower}}">
      {/* Component implementation */}
    </div>
  );
};
"""
    
    output_file = tmp_path / "AnimeCard.tsx"
    
    await template_adapter.execute("render_to_file", {
        "template": template,
        "output_path": str(output_file),
        "variables": {
            "name": "AnimeCard",
            "props": [
                {"name": "title", "type": "string"},
                {"name": "rating", "type": "number"},
                {"name": "onLike", "type": "() => void"}
            ]
        }
    })
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "interface AnimeCardProps" in content
    assert "title: string;" in content
    assert "rating: number;" in content


@pytest.mark.asyncio
async def test_generate_fastapi_endpoint(template_adapter, tmp_path):
    """Test generating FastAPI endpoint"""
    template = """
from fastapi import APIRouter, Depends
from {{module}}.models import {{model}}
from {{module}}.schemas import {{model}}Create, {{model}}Response

router = APIRouter()

@router.get("/{{plural}}", response_model=list[{{model}}Response])
async def list_{{plural}}():
    # Implementation here
    pass

@router.post("/{{plural}}", response_model={{model}}Response)
async def create_{{singular}}(data: {{model}}Create):
    # Implementation here
    pass
"""
    
    output_file = tmp_path / "anime.py"
    
    await template_adapter.execute("render_to_file", {
        "template": template,
        "output_path": str(output_file),
        "variables": {
            "module": "app",
            "model": "Anime",
            "singular": "anime",
            "plural": "anime"
        }
    })
    
    assert output_file.exists()
    content = output_file.read_text()
    assert "def list_anime():" in content
    assert "def create_anime(data: AnimeCreate):" in content
