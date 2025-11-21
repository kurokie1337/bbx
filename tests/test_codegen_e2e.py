# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
End-to-End Integration Test for Code Generation

This test validates the COMPLETE code generation pipeline:
1. Create directory structure
2. Generate files from templates
3. Write JSON/YAML config files
4. Generate actual working code (React + FastAPI)
5. Verify all files exist and content is correct

This is the ULTIMATE test - if this passes, code generation works!
"""

import pytest
import json
from blackbox.core.adapters.codegen.template import TemplateAdapter
from blackbox.core.adapters.codegen.fs import FileSystemGenAdapter


@pytest.fixture
def template_adapter():
    return TemplateAdapter()


@pytest.fixture
def fs_adapter():
    return FileSystemGenAdapter()


@pytest.mark.asyncio
async def test_e2e_generate_complete_project(template_adapter, fs_adapter, tmp_path):
    """
    END-TO-END TEST: Generate a complete full-stack application
    
    This simulates what app_generator.bbx will do:
    - Create React frontend
    - Create FastAPI backend
    - Generate configuration files
    - Generate README
    """
    project_name = "TestApp"
    project_path = tmp_path / project_name
    
    # ========== STEP 1: Create Directory Structure ==========
    print("\n📁 Step 1: Creating directory structure...")
    
    structure_result = await fs_adapter.execute("create_structure", {
        "base_path": str(project_path),
        "structure": {
            "frontend": {
                "src": {
                    "components": {},
                    "pages": {},
                    "hooks": {},
                    "utils": {}
                },
                "public": {}
            },
            "backend": {
                "app": {
                    "api": {},
                    "models": {},
                    "schemas": {}
                },
                "tests": {}
            },
            "docker": {},
            "k8s": {},
            "blackbox": {}
        }
    })
    
    assert structure_result["created_count"] > 0
    assert (project_path / "frontend" / "src" / "components").exists()
    assert (project_path / "backend" / "app" / "api").exists()
    print(f"✅ Created {structure_result['created_count']} directories")
    
    # ========== STEP 2: Generate Frontend Files ==========
    print("\n⚛️  Step 2: Generating React components...")
    
    # Generate React component
    component_template = """
import React from 'react';

interface {{name}}Props {
  title: string;
  onAction: () => void;
}

export const {{name}}: React.FC<{{name}}Props> = ({ title, onAction }) => {
  return (
    <div className="{{class_name}}">
      <h2>{title}</h2>
      <button onClick={onAction}>Click Me</button>
    </div>
  );
};
"""
    
    component_result = await template_adapter.execute("render_to_file", {
        "template": component_template,
        "output_path": str(project_path / "frontend" / "src" / "components" / "Card.tsx"),
        "variables": {
            "name": "Card",
            "class_name": "card-component"
        }
    })
    
    assert component_result["written"] is True
    card_file = project_path / "frontend" / "src" / "components" / "Card.tsx"
    assert card_file.exists()
    assert "export const Card" in card_file.read_text()
    print("✅ Generated Card.tsx")
    
    # Generate App.tsx
    app_template = """
import React from 'react';
import { Card } from './components/Card';

export const App: React.FC = () => {
  const handleAction = () => {
    console.log('Action triggered!');
  };

  return (
    <div className="app">
      <h1>{{app_name}}</h1>
      <Card title="Welcome" onAction={handleAction} />
    </div>
  );
};
"""
    
    await template_adapter.execute("render_to_file", {
        "template": app_template,
        "output_path": str(project_path / "frontend" / "src" / "App.tsx"),
        "variables": {"app_name": project_name}
    })
    
    assert (project_path / "frontend" / "src" / "App.tsx").exists()
    print("✅ Generated App.tsx")
    
    # ========== STEP 3: Generate Backend Files ==========
    print("\n🚀 Step 3: Generating FastAPI backend...")
    
    # Generate FastAPI model
    model_template = """
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class {{model_name}}(Base):
    __tablename__ = \"{{table_name}}\"
    
    id = Column(Integer, primary_key=True, index=True)
    {% for field in fields %}
    {{field.name}} = Column({{field.type}})
    {% endfor %}
    created_at = Column(DateTime, default=datetime.utcnow)
"""
    
    await template_adapter.execute("render_to_file", {
        "template": model_template,
        "output_path": str(project_path / "backend" / "app" / "models" / "item.py"),
        "variables": {
            "model_name": "Item",
            "table_name": "items",
            "fields": [
                {"name": "title", "type": "String"},
                {"name": "description", "type": "String"}
            ]
        }
    })
    
    model_file = project_path / "backend" / "app" / "models" / "item.py"
    assert model_file.exists()
    assert "class Item(Base):" in model_file.read_text()
    print("✅ Generated Item model")
    
    # Generate FastAPI endpoint
    endpoint_template = """
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()

@router.get("/{{plural}}")
async def list_{{plural}}():
    # TODO: Implement database query
    return []

@router.post("/{{plural}}")
async def create_{{singular}}(title: str, description: str):
    # TODO: Implement database insert
    return {"id": 1, "title": title, "description": description}

@router.get("/{{plural}}/{id}")
async def get_{{singular}}(id: int):
    # TODO: Implement database query
    return {"id": id, "title": "Sample"}
"""
    
    await template_adapter.execute("render_to_file", {
        "template": endpoint_template,
        "output_path": str(project_path / "backend" / "app" / "api" / "items.py"),
        "variables": {
            "singular": "item",
            "plural": "items"
        }
    })
    
    endpoint_file = project_path / "backend" / "app" / "api" / "items.py"
    assert endpoint_file.exists()
    assert "async def list_items():" in endpoint_file.read_text()
    print("✅ Generated items API endpoint")
    
    # ========== STEP 4: Generate Configuration Files ==========
    print("\n⚙️  Step 4: Generating configuration files...")
    
    # package.json
    package_json = {
        "name": project_name.lower(),
        "version": "1.0.0",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "test": "jest"
        },
        "dependencies": {
            "react": "^18.2.0",
            "react-dom": "^18.2.0"
        },
        "devDependencies": {
            "vite": "^5.0.0",
            "typescript": "^5.0.0"
        }
    }
    
    await fs_adapter.execute("write_json", {
        "path": str(project_path / "frontend" / "package.json"),
        "data": package_json
    })
    
    package_file = project_path / "frontend" / "package.json"
    assert package_file.exists()
    loaded_pkg = json.loads(package_file.read_text())
    assert loaded_pkg["name"] == project_name.lower()
    print("✅ Generated package.json")
    
    # requirements.txt
    await fs_adapter.execute("create_file", {
        "path": str(project_path / "backend" / "requirements.txt"),
        "content": "fastapi\nuvicorn\nsqlalchemy\nalembic\npydantic"
    })
    
    assert (project_path / "backend" / "requirements.txt").exists()
    print("✅ Generated requirements.txt")
    
    # docker-compose.yml
    docker_compose = {
        "version": "3.8",
        "services": {
            "frontend": {
                "build": "./frontend",
                "ports": ["3000:3000"]
            },
            "backend": {
                "build": "./backend",
                "ports": ["8000:8000"]
            },
            "db": {
                "image": "postgres:15",
                "environment": {
                    "POSTGRES_PASSWORD": "secret"
                }
            }
        }
    }
    
    await fs_adapter.execute("write_yaml", {
        "path": str(project_path / "docker" / "docker-compose.yml"),
        "data": docker_compose
    })
    
    compose_file = project_path / "docker" / "docker-compose.yml"
    assert compose_file.exists()
    print("✅ Generated docker-compose.yml")
    
    # ========== STEP 5: Generate README ==========
    print("\n📝 Step 5: Generating README...")
    
    readme_template = """
# {{project_name}}

Generated by BBX Code Generator! 🚀

## Project Structure

```
{{project_name}}/
├── frontend/      # React frontend
├── backend/       # FastAPI backend
├── docker/        # Docker configuration
└── blackbox/      # BBX orchestration workflows
```

## Quick Start

```bash
# Install dependencies
cd frontend && npm install
cd backend && pip install -r requirements.txt

# Run locally
docker-compose up
```

## Features

- ⚛️  React 18 frontend with TypeScript
- 🚀 FastAPI backend with SQLAlchemy
- 🐳 Docker containerization
- 📦 PostgreSQL database
- 🔧 One-command setup with BBX

## Generated Files

- Frontend: {{frontend_files}} components
- Backend: {{backend_files}} endpoints
- Config: {{config_files}} configuration files

---

**This entire project was generated in {{generation_time}} seconds!**
"""
    
    await template_adapter.execute("render_to_file", {
        "template": readme_template,
        "output_path": str(project_path / "README.md"),
        "variables": {
            "project_name": project_name,
            "frontend_files": 2,
            "backend_files": 1,
            "config_files": 3,
            "generation_time": 0.5
        }
    })
    
    readme_file = project_path / "README.md"
    assert readme_file.exists()
    readme_content = readme_file.read_text()
    assert f"# {project_name}" in readme_content
    assert "Generated by BBX" in readme_content
    print("✅ Generated README.md")
    
    # ========== STEP 6: Verify Complete Project ==========
    print("\n✅ Step 6: Verifying complete project...")
    
    # Count all generated files
    all_files = list(project_path.rglob('*'))
    file_count = sum(1 for f in all_files if f.is_file())
    dir_count = sum(1 for f in all_files if f.is_dir())
    
    print("\n🎉 PROJECT GENERATION COMPLETE!")
    print(f"📁 Directories: {dir_count}")
    print(f"📄 Files: {file_count}")
    print(f"📦 Total size: {sum(f.stat().st_size for f in all_files if f.is_file())} bytes")
    
    # Verify critical files exist
    critical_files = [
        "frontend/src/components/Card.tsx",
        "frontend/src/App.tsx",
        "frontend/package.json",
        "backend/app/models/item.py",
        "backend/app/api/items.py",
        "backend/requirements.txt",
        "docker/docker-compose.yml",
        "README.md"
    ]
    
    for file_path in critical_files:
        full_path = project_path / file_path
        assert full_path.exists(), f"Critical file missing: {file_path}"
        assert full_path.stat().st_size > 0, f"File is empty: {file_path}"
    
    print(f"\n✅ All {len(critical_files)} critical files verified!")
    
    # Verify content quality
    assert "React.FC" in (project_path / "frontend" / "src" / "App.tsx").read_text()
    assert "Column" in (project_path / "backend" / "app" / "models" / "item.py").read_text()
    assert "fastapi" in (project_path / "backend" / "requirements.txt").read_text()
    
    print("\n🎉 END-TO-END TEST PASSED! Code generation pipeline is FULLY FUNCTIONAL!")
    print("=" * 80)
    
    # Final assertions
    assert file_count >= 8
    assert dir_count >= 10
    assert readme_file.stat().st_size > 200


@pytest.mark.asyncio
async def test_e2e_template_and_fs_integration(template_adapter, fs_adapter, tmp_path):
    """
    Test integration between template and fs adapters
    """
    # Create structure
    await fs_adapter.execute("create_structure", {
        "base_path": str(tmp_path),
        "structure": ["output/"]
    })
    
    # Generate file with template
    template = "Hello {{name}}! Your score is {{score}}."
    result = await template_adapter.execute("render_to_file", {
        "template": template,
        "output_path": str(tmp_path / "output" / "result.txt"),
        "variables": {"name": "Alice", "score": 100}
    })
    
    assert result["written"] is True
    assert (tmp_path / "output" / "result.txt").exists()
    
    content = (tmp_path / "output" / "result.txt").read_text()
    assert content == "Hello Alice! Your score is 100."
