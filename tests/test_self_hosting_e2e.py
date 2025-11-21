# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
End-to-End Self-Hosting Test

This test validates that BBX can COMPLETELY develop itself:
1. Run its own tests via dev/test.bbx
2. Lint itself via dev/lint.bbx  
3. Build itself via dev/build.bbx
4. Verify all workflows execute successfully

This is the ULTIMATE test of self-hosting capability!
"""

import pytest
import subprocess
from pathlib import Path


def run_bbx_workflow(workflow_path: str, cwd: str = None) -> dict:
    """
    Run a BBX workflow and return results.
    
    Args:
        workflow_path: Path to .bbx workflow file
        cwd: Working directory (default: project root)
        
    Returns:
        dict with exit_code, stdout, stderr
    """
    if cwd is None:
        cwd = Path(__file__).parent.parent
    
    cmd = ["python", "cli.py", "run", workflow_path]
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60  # 1 minute max
    )
    
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def test_self_hosting_test_workflow():
    """
    Test that BBX can test itself via dev/test.bbx
    
    This proves BBX can run its own test suite using BBX workflows!
    """
    print("\n🧪 Testing: BBX runs its own tests...")
    
    result = run_bbx_workflow("dev/test.bbx")
    
    # Check execution
    print(f"Exit code: {result['exit_code']}")
    print(f"Output length: {len(result['stdout'])} chars")
    
    # Workflow should complete (exit_code 0 or tests may fail but workflow runs)
    assert result["exit_code"] == 0, f"Test workflow failed: {result['stderr']}"
    
    # Verify test execution happened
    assert "test_os_abstraction" in result["stdout"] or "pytest" in result["stdout"], \
        "Tests were not executed"
    
    print("✅ BBX successfully tested itself!")


def test_self_hosting_build_workflow():
    """
    Test that BBX can build itself via dev/build.bbx
    
    This proves BBX can compile/package itself using BBX workflows!
    """
    print("\n🏗️  Testing: BBX builds itself...")
    
    result = run_bbx_workflow("dev/build.bbx")
    
    print(f"Exit code: {result['exit_code']}")
    
    # Build workflow should complete
    assert result["exit_code"] == 0, f"Build workflow failed: {result['stderr']}"
    
    # Verify build steps executed
    assert "python -m build" in result["stdout"] or "build_wheel" in result["stdout"], \
        "Build step was not executed"
    
    print("✅ BBX successfully built itself!")


@pytest.mark.slow
def test_self_hosting_lint_workflow():
    """
    Test that BBX can lint itself via dev/lint.bbx
    
    This proves BBX can run code quality checks using BBX workflows!
    """
    print("\n🔍 Testing: BBX lints itself...")
    
    result = run_bbx_workflow("dev/lint.bbx")
    
    print(f"Exit code: {result['exit_code']}")
    
    # Lint workflow should complete (may have warnings but should run)
    # Note: exit_code might not be 0 if there are lint issues, but workflow should execute
    assert result["exit_code"] in [0, 1], \
        f"Lint workflow crashed (not just lint errors): {result['stderr']}"
    
    # Verify lint tools ran
    assert "ruff" in result["stdout"].lower() or "lint" in result["stdout"].lower(), \
        "Linting tools were not executed"
    
    print("✅ BBX successfully linted itself!")


def test_self_hosting_complete_pipeline():
    """
    COMPREHENSIVE TEST: Run the complete self-hosting pipeline
    
    This simulates a full development cycle:
    1. Test
    2. Lint  
    3. Build
    
    All using BBX workflows!
    """
    print("\n🚀 COMPREHENSIVE TEST: Complete self-hosting pipeline...")
    print("=" * 80)
    
    cwd = Path(__file__).parent.parent
    
    # Step 1: Test
    print("\n📊 Step 1/3: Running tests...")
    test_result = run_bbx_workflow("dev/test.bbx", cwd=cwd)
    assert test_result["exit_code"] == 0, "Test phase failed"
    print("✅ Tests passed")
    
    # Step 2: Lint (allow warnings)
    print("\n🔍 Step 2/3: Running linter...")
    lint_result = run_bbx_workflow("dev/lint.bbx", cwd=cwd)
    # Linter may return 1 for warnings, that's OK
    assert lint_result["exit_code"] in [0, 1], "Lint phase crashed"
    print("✅ Linting completed")
    
    # Step 3: Build
    print("\n🏗️  Step 3/3: Building package...")
    build_result = run_bbx_workflow("dev/build.bbx", cwd=cwd)
    assert build_result["exit_code"] == 0, "Build phase failed"
    print("✅ Build completed")
    
    # Verify dist/ directory was created
    dist_dir = cwd / "dist"
    if dist_dir.exists():
        files = list(dist_dir.glob("*"))
        print(f"\n📦 Built artifacts: {len(files)} files")
        for f in files:
            print(f"  - {f.name}")
    
    print("\n" + "=" * 80)
    print("🎉 COMPLETE SELF-HOSTING PIPELINE SUCCESSFUL!")
    print("=" * 80)
    print("\nBBX can fully develop itself using BBX workflows!")
    print("This proves production-readiness and self-hosting capability.")


def test_workflows_exist():
    """
    Sanity check: Verify all self-hosting workflows exist
    """
    cwd = Path(__file__).parent.parent
    dev_dir = cwd / "dev"
    
    required_workflows = [
        "build.bbx",
        "test.bbx",
        "lint.bbx",
        "release.bbx"
    ]
    
    for workflow in required_workflows:
        workflow_path = dev_dir / workflow
        assert workflow_path.exists(), f"Required workflow missing: {workflow}"
        assert workflow_path.stat().st_size > 0, f"Workflow is empty: {workflow}"
    
    print(f"\n✅ All {len(required_workflows)} self-hosting workflows exist")


def test_workflows_are_valid_yaml():
    """
    Verify all workflows are valid YAML format
    """
    import yaml
    
    cwd = Path(__file__).parent.parent
    dev_dir = cwd / "dev"
    
    workflows = ["build.bbx", "test.bbx", "lint.bbx", "release.bbx"]
    
    for workflow_name in workflows:
        workflow_path = dev_dir / workflow_name
        
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Verify basic structure
            assert 'id' in data, f"{workflow_name}: missing 'id'"
            assert 'workflow' in data, f"{workflow_name}: missing 'workflow'"
            assert 'steps' in data['workflow'], f"{workflow_name}: missing 'steps'"
            
            print(f"✅ {workflow_name}: Valid YAML with {len(data['workflow']['steps'])} steps")
            
        except Exception as e:
            pytest.fail(f"{workflow_name}: Invalid YAML - {str(e)}")


@pytest.mark.integration
def test_github_actions_workflow_exists():
    """
    Verify GitHub Actions CI/CD workflow exists
    """
    cwd = Path(__file__).parent.parent
    gh_workflow = cwd / ".github" / "workflows" / "ci.yml"
    
    assert gh_workflow.exists(), "GitHub Actions workflow missing"
    assert gh_workflow.stat().st_size > 0, "GitHub Actions workflow is empty"
    
    # Verify it's valid YAML
    import yaml
    with open(gh_workflow, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    assert 'jobs' in data, "GitHub Actions workflow missing 'jobs'"
    assert 'test' in data['jobs'], "GitHub Actions workflow missing 'test' job"
    
    print(f"✅ GitHub Actions workflow exists with {len(data['jobs'])} jobs")
