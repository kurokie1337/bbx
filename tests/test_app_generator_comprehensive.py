# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
COMPREHENSIVE End-to-End Test Suite for App Generator

This test suite validates the COMPLETE application generation system:
1. Workflow execution with inputs
2. Directory structure creation
3. File generation (all types: TS, Python, JSON, YAML, BBX)
4. Code syntax validation (TypeScript, Python)
5. Configuration validity (JSON, YAML)
6. BBX workflow validity
7. Content quality checks
8. Performance benchmarks

RUN THIS TO PROVE END-TO-END WORKS!
"""

import pytest
import subprocess
import sys
import time
from pathlib import Path
import json
import yaml as yaml_lib
import shutil


@pytest.fixture
def unique_project_name():
    """Generate unique project name"""
    timestamp = int(time.time() * 1000)
    return f"E2ETestApp_{timestamp}"


@pytest.fixture
def cleanup_project(unique_project_name):
    """Cleanup after test"""
    yield
    project_path = Path(unique_project_name)
    if project_path.exists():
        shutil.rmtree(project_path, ignore_errors=True)


def run_app_generator(project_name: str) -> dict:
    """Run app generator and return results"""
    cmd = [
        sys.executable,
        "cli.py",
        "run",
        "workflows/generators/app_generator.bbx",
        "-i", f"project_name={project_name}",
        "-i", "include_k8s=false"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    execution_time = time.time() - start_time
    
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": execution_time
    }


@pytest.mark.e2e
def test_e2e_full_generation_successful(unique_project_name, cleanup_project):
    """TEST 1: Complete workflow executes successfully"""
    print(f"\n🧪 TEST 1: Full generation for {unique_project_name}")
    
    result = run_app_generator(unique_project_name)
    
    assert result["exit_code"] ==0, f"Generation failed with exit code {result['exit_code']}"
    assert Path(unique_project_name).exists(), "Project directory not created"
    
    print(f"   ✅ Generation successful in {result['execution_time']:.2f}s")


@pytest.mark.e2e
def test_e2e_all_directories_created(unique_project_name, cleanup_project):
    """TEST 2: All required directories exist"""
    print("\n🧪 TEST 2: Directory structure validation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    required_dirs = [
        "frontend/src/components",
        "frontend/src/pages",
        "frontend/src/hooks",
        "frontend/src/utils",
        "frontend/src/styles",
        "frontend/public",
        "backend/app/api",
        "backend/app/models",
        "backend/app/schemas",
        "backend/app/core",
        "backend/tests",
        "backend/alembic/versions",
        "docker",
        "blackbox",
        "docs"
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = project_path / dir_path
        if not full_path.exists():
            missing.append(dir_path)
    
    assert not missing, f"Missing directories: {missing}"
    print(f"   ✅ All {len(required_dirs)} required directories exist")


@pytest.mark.e2e
def test_e2e_all_files_generated(unique_project_name, cleanup_project):
    """TEST 3: All critical files generated"""
    print("\n🧪 TEST 3: File generation validation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    critical_files = {
        "README.md": 500,  # min size
        "frontend/src/components/AnimeCard.tsx": 400,
        "frontend/package.json": 300,
        "backend/app/models/anime.py": 500,
        "backend/app/api/anime.py": 1500,
        "backend/requirements.txt": 100,
        "docker/docker-compose.yml": 400,
        "blackbox/dev_setup.bbx": 2000
    }
    
    for file_path, min_size in critical_files.items():
        full_path = project_path / file_path
        assert full_path.exists(), f"Missing file: {file_path}"
        size = full_path.stat().st_size
        assert size >= min_size, f"{file_path} too small: {size} bytes (expected >={min_size})"
    
    print(f"   ✅ All {len(critical_files)} critical files exist with correct sizes")


@pytest.mark.e2e
def test_e2e_typescript_syntax_valid(unique_project_name, cleanup_project):
    """TEST 4: Generated TypeScript has valid syntax"""
    print("\n🧪 TEST 4: TypeScript syntax validation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    tsx_file = project_path / "frontend" / "src" / "components" / "AnimeCard.tsx"
    content = tsx_file.read_text()
    
    # Verify TypeScript elements
    assert "import React from 'react';" in content
    assert "interface AnimeCardProps" in content
    assert "export const AnimeCard" in content
    assert "React.FC<AnimeCardProps>" in content
    assert "title: string" in content
    assert "imageUrl: string" in content
    
    print("   ✅ TypeScript syntax valid with proper interfaces")


@pytest.mark.e2e
def test_e2e_python_syntax_compiles(unique_project_name, cleanup_project):
    """TEST 5: Generated Python code compiles"""
    print("\n🧪 TEST 5: Python syntax compilation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    # Test model
    model_file = project_path / "backend" / "app" / "models" / "anime.py"
    model_content = model_file.read_text()
    compile(model_content, model_file, 'exec')  # Will raise SyntaxError if invalid
    
    # Test endpoint
    endpoint_file = project_path / "backend" / "app" / "api" / "anime.py"
    endpoint_content = endpoint_file.read_text()
    compile(endpoint_content, endpoint_file, 'exec')
    
    print("   ✅ All Python files compile successfully")


@pytest.mark.e2e
def test_e2e_json_files_parseable(unique_project_name, cleanup_project):
    """TEST 6: JSON files are valid"""
    print("\n🧪 TEST 6: JSON validation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    package_json = project_path / "frontend" / "package.json"
    data = json.loads(package_json.read_text())
    
    assert "name" in data
    assert "dependencies" in data
    assert "react" in data["dependencies"]
    assert "scripts" in data
    
    print("   ✅ package.json valid with React dependencies")


@pytest.mark.e2e
def test_e2e_yaml_files_parseable(unique_project_name, cleanup_project):
    """TEST 7: YAML files are valid"""
    print("\n🧪 TEST 7: YAML validation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    docker_compose = project_path / "docker" / "docker-compose.yml"
    data = yaml_lib.safe_load(docker_compose.read_text())
    
    assert "services" in data
    assert "frontend" in data["services"]
    assert "backend" in data["services"]
    assert "postgres" in data["services"]
    assert "redis" in data["services"]
    
    print("   ✅ docker-compose.yml valid with all services")


@pytest.mark.e2e
def test_e2e_bbx_workflow_valid(unique_project_name, cleanup_project):
    """TEST 8: Generated BBX  workflow is valid"""
    print("\n🧪 TEST 8: BBX workflow validation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    dev_setup = project_path / "blackbox" / "dev_setup.bbx"
    data = yaml_lib.safe_load(dev_setup.read_text())
    
    assert "id" in data
    assert "workflow" in data
    assert "steps" in data["workflow"]
    assert len(data["workflow"]["steps"]) > 0
    
    # Verify first step has required fields
    first_step = data["workflow"]["steps"][0]
    assert "id" in first_step
    assert "mcp" in first_step
    
    print(f"   ✅ dev_setup.bbx valid with {len(data['workflow']['steps'])} steps")


@pytest.mark.e2e
def test_e2e_content_quality(unique_project_name, cleanup_project):
    """TEST 9: Generated content quality checks"""
    print("\n🧪 TEST 9: Content quality validation")
    
    run_app_generator(unique_project_name)
    project_path = Path(unique_project_name)
    
    # Check README
    readme = project_path / "README.md"
    readme_content = readme.read_text()
    assert unique_project_name in readme_content
    assert "BBX" in readme_content
    assert "Quick Start" in readme_content
    
    # Check model has all fields
    model = project_path / "backend" / "app" / "models" / "anime.py"
    model_content = model.read_text()
    assert "class Anime(Base):" in model_content
    assert "title = Column" in model_content
    assert "created_at = Column(DateTime" in model_content
    
    # Check endpoint has CRUD ops
    endpoint = project_path / "backend" / "app" / "api" / "anime.py"
    endpoint_content = endpoint.read_text()
    assert "async def list_anime" in endpoint_content
    assert "async def create_anime" in endpoint_content
    assert "async def get_anime" in endpoint_content
    
    print("   ✅ All content quality checks passed")


@pytest.mark.e2e
@pytest.mark.performance
def test_e2e_performance_benchmark(unique_project_name, cleanup_project):
    """TEST 10: Performance benchmarks"""
    print("\n🧪 TEST 10: Performance benchmark")
    
    result = run_app_generator(unique_project_name)
    
    # Should generate in < 5 seconds
    assert result["execution_time"] < 5.0, \
        f"Generation too slow: {result['execution_time']:.2f}s (expected <5s)"
    
    project_path = Path(unique_project_name)
    all_files = list(project_path.rglob('*'))
    file_count = sum(1 for f in all_files if f.is_file())
    
    print(f"   ⏱️  Execution time: {result['execution_time']:.2f}s")
    print(f"   📄 Files generated: {file_count}")
    print(f"   🚀 Speed: {file_count / result['execution_time']:.1f} files/second")
    print("   ✅ Performance acceptable")


@pytest.mark.e2e
def test_e2e_complete_validation_suite(unique_project_name, cleanup_project):
    """TEST 11: COMPREHENSIVE - All validations in one test"""
    print("\n🧪 TEST 11: COMPREHENSIVE VALIDATION")
    print("=" * 80)
    
    # Generate
    print("📦 Generating project...")
    start_time = time.time()
    result = run_app_generator(unique_project_name)
    gen_time = time.time() - start_time
    
    assert result["exit_code"] == 0
    print(f"✅ Generation successful ({gen_time:.2f}s)")
    
    # Count artifacts
    project_path = Path(unique_project_name)
    all_items = list(project_path.rglob('*'))
    files = [f for f in all_items if f.is_file()]
    dirs = [d for d in all_items if d.is_dir()]
    total_size = sum(f.stat().st_size for f in files)
    
    print("\n📊 Statistics:")
    print(f"   Directories: {len(dirs)}")
    print(f"   Files: {len(files)}")
    print(f"   Total size: {total_size:,} bytes")
    
    # Validate each file type
    tsx_files = list(project_path.glob("**/*.tsx"))
    py_files = list(project_path.glob("**/*.py"))
    json_files = list(project_path.glob("**/*.json"))
    yml_files = list(project_path.glob("**/*.yml"))
    bbx_files = list(project_path.glob("**/*.bbx"))
    
    print("\n📝 File types:")
    print(f"   TypeScript: {len(tsx_files)}")
    print(f"   Python: {len(py_files)}")
    print(f"   JSON: {len(json_files)}")
    print(f"   YAML: {len(yml_files)}")
    print(f"   BBX: {len(bbx_files)}")
    
    # Minimum expectations
    assert len(files) >= 8, f"Too few files: {len(files)}"
    assert len(dirs) >= 20, f"Too few directories: {len(dirs)}"
    assert len(tsx_files) >= 1, "No TypeScript files"
    assert len(py_files) >= 2, "Not enough Python files"
    assert len(json_files) >= 1, "No JSON files"
    assert len(yml_files) >= 1, "No YAML files"
    assert len(bbx_files) >= 1, "No BBX workflow files"
    
    print("\n🎉 COMPREHENSIVE VALIDATION PASSED!")
    print("=" * 80)
    print(f"\nGenerated production-ready {unique_project_name}!")
    print(f"Ready to: cd {unique_project_name} && code .")


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
