# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import json
import yaml as yaml_lib
from blackbox.core.adapters.codegen.fs import FileSystemGenAdapter


@pytest.fixture
def fs_adapter():
    """Create file system gen adapter instance"""
    return FileSystemGenAdapter()


# ========== Single File Creation ==========

@pytest.mark.asyncio
async def test_create_file_basic(fs_adapter, tmp_path):
    """Test basic file creation"""
    file_path = tmp_path / "test.txt"
    
    result = await fs_adapter.execute("create_file", {
        "path": str(file_path),
        "content": "Hello World"
    })
    
    assert result["created"] is True
    assert file_path.exists()
    assert file_path.read_text() == "Hello World"


@pytest.mark.asyncio
async def test_create_file_with_dirs(fs_adapter, tmp_path):
    """Test file creation with automatic directory creation"""
    file_path = tmp_path / "deep" / "nested" / "file.txt"
    
    result = await fs_adapter.execute("create_file", {
        "path": str(file_path),
        "content": "Content",
        "create_dirs": True
    })
    
    assert result["created"] is True
    assert file_path.exists()
    assert file_path.parent.exists()


@pytest.mark.asyncio
async def test_create_file_overwrite_protection(fs_adapter, tmp_path):
    """Test that overwrite protection works"""
    file_path = tmp_path / "existing.txt"
    file_path.write_text("Original")
    
    # Should fail without overwrite=true
    with pytest.raises(FileExistsError):
        await fs_adapter.execute("create_file", {
            "path": str(file_path),
            "content": "New",
            "overwrite": False
        })
    
    # Original content preserved
    assert file_path.read_text() == "Original"
    
    # Should succeed with overwrite=true
    result = await fs_adapter.execute("create_file", {
        "path": str(file_path),
        "content": "New",
        "overwrite": True
    })
    
    assert result["created"] is True
    assert file_path.read_text() == "New"


@pytest.mark.asyncio
async def test_create_file_encoding(fs_adapter, tmp_path):
    """Test file creation with different encodings"""
    file_path = tmp_path / "unicode.txt"
    content = "Привет мир! 🚀"
    
    result = await fs_adapter.execute("create_file", {
        "path": str(file_path),
        "content": content,
        "encoding": "utf-8"
    })
    
    assert result["created"] is True
    assert file_path.read_text(encoding='utf-8') == content


# ========== Multiple Files Creation ==========

@pytest.mark.asyncio
async def test_create_files_batch(fs_adapter, tmp_path):
    """Test creating multiple files at once"""
    result = await fs_adapter.execute("create_files", {
        "files": [
            {"path": "file1.txt", "content": "Content 1"},
            {"path": "file2.txt", "content": "Content 2"},
            {"path": "subdir/file3.txt", "content": "Content 3"}
        ],
        "base_dir": str(tmp_path)
    })
    
    assert result["created_count"] == 3
    assert (tmp_path / "file1.txt").exists()
    assert (tmp_path / "file2.txt").exists()
    assert (tmp_path / "subdir" / "file3.txt").exists()


@pytest.mark.asyncio
async def test_create_files_with_absolute_paths(fs_adapter, tmp_path):
    """Test creating files with mix of absolute and relative paths"""
    result = await fs_adapter.execute("create_files", {
        "files": [
            {"path": "relative.txt", "content": "Relative"},
            {"path": str(tmp_path / "absolute.txt"), "content": "Absolute"}
        ],
        "base_dir": str(tmp_path)
    })
    
    assert result["created_count"] == 2
    assert (tmp_path / "relative.txt").exists()
    assert (tmp_path / "absolute.txt").exists()


# ========== Directory Creation ==========

@pytest.mark.asyncio
async def test_create_dir_simple(fs_adapter, tmp_path):
    """Test simple directory creation"""
    dir_path = tmp_path / "new_dir"
    
    result = await fs_adapter.execute("create_dir", {
        "path": str(dir_path)
    })
    
    assert result["created"] is True
    assert dir_path.exists()
    assert dir_path.is_dir()


@pytest.mark.asyncio
async def test_create_dir_with_parents(fs_adapter, tmp_path):
    """Test creating nested directories"""
    dir_path = tmp_path / "level1" / "level2" / "level3"
    
    result = await fs_adapter.execute("create_dir", {
        "path": str(dir_path),
        "parents": True
    })
    
    assert result["created"] is True
    assert dir_path.exists()


# ========== Directory Structure Creation ==========

@pytest.mark.asyncio
async def test_create_structure_from_list(fs_adapter, tmp_path):
    """Test creating directory structure from list"""
    result = await fs_adapter.execute("create_structure", {
        "base_path": str(tmp_path),
        "structure": [
            "src/",
            "src/components/",
            "src/utils/",
            "tests/",
            "docs/"
        ]
    })
    
    assert result["created_count"] == 5
    assert (tmp_path / "src").exists()
    assert (tmp_path / "src" / "components").exists()
    assert (tmp_path / "src" / "utils").exists()
    assert (tmp_path / "tests").exists()
    assert (tmp_path / "docs").exists()


@pytest.mark.asyncio
async def test_create_structure_from_dict(fs_adapter, tmp_path):
    """Test creating directory structure from nested dict"""
    result = await fs_adapter.execute("create_structure", {
        "base_path": str(tmp_path),
        "structure": {
            "frontend": {
                "src": {
                    "components": {},
                    "pages": {},
                    "hooks": {}
                },
                "public": {}
            },
            "backend": {
                "app": {
                    "api": {},
                    "models": {}
                }
            }
        }
    })
    
    assert result["created_count"] > 0
    assert (tmp_path / "frontend" / "src" / "components").exists()
    assert (tmp_path / "frontend" / "src" / "pages").exists()
    assert (tmp_path / "backend" / "app" / "api").exists()


# ========== JSON Writing ==========

@pytest.mark.asyncio
async def test_write_json(fs_adapter, tmp_path):
    """Test writing data to JSON file"""
    file_path = tmp_path / "config.json"
    data = {
        "name": "TestApp",
        "version": "1.0.0",
        "dependencies": ["react", "fastapi"]
    }
    
    result = await fs_adapter.execute("write_json", {
        "path": str(file_path),
        "data": data,
        "indent": 2
    })
    
    assert result["written"] is True
    assert file_path.exists()
    
    # Verify content
    loaded = json.loads(file_path.read_text())
    assert loaded["name"] == "TestApp"
    assert loaded["version"] == "1.0.0"
    assert len(loaded["dependencies"]) == 2


@pytest.mark.asyncio
async def test_write_json_complex_data(fs_adapter, tmp_path):
    """Test writing complex nested JSON"""
    file_path = tmp_path / "complex.json"
    data = {
        "users": [
            {"id": 1, "name": "Alice", "admin": True},
            {"id": 2, "name": "Bob", "admin": False}
        ],
        "settings": {
            "theme": "dark",
            "language": "en"
        }
    }
    
    result = await fs_adapter.execute("write_json", {
        "path": str(file_path),
        "data": data
    })
    
    assert result["written"] is True
    loaded = json.loads(file_path.read_text())
    assert len(loaded["users"]) == 2
    assert loaded["users"][0]["admin"] is True


# ========== YAML Writing ==========

@pytest.mark.asyncio
async def test_write_yaml(fs_adapter, tmp_path):
    """Test writing data to YAML file"""
    file_path = tmp_path / "config.yaml"
    data = {
        "name": "TestApp",
        "version": "1.0.0",
        "services": ["frontend", "backend", "database"]
    }
    
    result = await fs_adapter.execute("write_yaml", {
        "path": str(file_path),
        "data": data
    })
    
    assert result["written"] is True
    assert file_path.exists()
    
    # Verify content
    loaded = yaml_lib.safe_load(file_path.read_text())
    assert loaded["name"] == "TestApp"
    assert "backend" in loaded["services"]


# ========== Template Copying ==========

@pytest.mark.asyncio
async def test_copy_template(fs_adapter, tmp_path):
    """Test copying template directory"""
    # Create source template
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    (template_dir / "file1.txt").write_text("Content 1")
    (template_dir / "file2.txt").write_text("Content 2")
    
    subdir = template_dir / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("Content 3")
    
    # Copy template
    dest_dir = tmp_path / "output"
    
    result = await fs_adapter.execute("copy_template", {
        "src": str(template_dir),
        "dst": str(dest_dir)
    })
    
    assert result["copied"] is True
    assert result["file_count"] == 3
    assert (dest_dir / "file1.txt").exists()
    assert (dest_dir / "file2.txt").exists()
    assert (dest_dir / "subdir" / "file3.txt").exists()


@pytest.mark.asyncio
async def test_copy_template_overwrite_protection(fs_adapter, tmp_path):
    """Test that copy template protects against overwrites"""
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    
    source.mkdir()
    (source / "file.txt").write_text("Source")
    
    dest.mkdir()
    (dest / "existing.txt").write_text("Existing")
    
    # Should fail without overwrite
    with pytest.raises(FileExistsError):
        await fs_adapter.execute("copy_template", {
            "src": str(source),
            "dst": str(dest),
            "overwrite": False
        })


# ========== Error Handling ==========

@pytest.mark.asyncio
async def test_create_file_missing_path(fs_adapter):
    """Test error when path is missing"""
    with pytest.raises(ValueError, match="path is required"):
        await fs_adapter.execute("create_file", {
            "content": "Content"
        })


@pytest.mark.asyncio
async def test_write_json_invalid_data(fs_adapter, tmp_path):
    """Test error when JSON data is not serializable"""
    file_path = tmp_path / "bad.json"
    
    # Create non-serializable object
    class NonSerializable:
        pass
    
    with pytest.raises(ValueError, match="JSON serialization error"):
        await fs_adapter.execute("write_json", {
            "path": str(file_path),
            "data": {"obj": NonSerializable()}
        })


@pytest.mark.asyncio
async def test_copy_template_source_not_found(fs_adapter, tmp_path):
    """Test error when source template doesn't exist"""
    with pytest.raises(FileNotFoundError):
        await fs_adapter.execute("copy_template", {
            "src": "/nonexistent/path",
            "dst": str(tmp_path / "dest")
        })


# ========== Real-World Use Cases ==========

@pytest.mark.asyncio
async def test_generate_react_project_structure(fs_adapter, tmp_path):
    """Test generating a complete React project structure"""
    project_path = tmp_path / "my_react_app"
    
    # Create directory structure
    await fs_adapter.execute("create_structure", {
        "base_path": str(project_path),
        "structure": {
            "src": {
                "components": {},
                "pages": {},
                "hooks": {},
                "utils": {},
                "styles": {}
            },
            "public": {},
            "tests": {}
        }
    })
    
    # Create package.json
    await fs_adapter.execute("write_json", {
        "path": str(project_path / "package.json"),
        "data": {
            "name": "my-react-app",
            "version": "1.0.0",
            "dependencies": {
                "react": "^18.0.0",
                "react-dom": "^18.0.0"
            }
        }
    })
    
    # Create README
    await fs_adapter.execute("create_file", {
        "path": str(project_path / "README.md"),
        "content": "# My React App\n\nGenerated by BBX!"
    })
    
    # Verify structure
    assert (project_path / "src" / "components").exists()
    assert (project_path / "package.json").exists()
    assert (project_path / "README.md").exists()


@pytest.mark.asyncio
async def test_generate_fastapi_project(fs_adapter, tmp_path):
    """Test generating FastAPI project structure"""
    project_path = tmp_path / "my_api"
    
    # Create structure
    await fs_adapter.execute("create_structure", {
        "base_path": str(project_path),
        "structure": ["app/", "app/api/", "app/models/", "app/schemas/", "tests/"]
    })
    
    # Create requirements.txt
    await fs_adapter.execute("create_file", {
        "path": str(project_path / "requirements.txt"),
        "content": "fastapi\nuvicorn\nsqlalchemy\nalembic"
    })
    
    # Create main.py
    await fs_adapter.execute("create_file", {
        "path": str(project_path / "app" / "main.py"),
        "content": """from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
"""
    })
    
    assert (project_path / "app" / "main.py").exists()
    assert (project_path / "requirements.txt").exists()
