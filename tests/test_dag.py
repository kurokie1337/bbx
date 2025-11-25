# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

import pytest
from blackbox.core.dag import should_use_dag, DAGError, create_dag

def test_simple_dag():
    """Test basic DAG creation"""
    steps = [
        {"id": "step1", "mcp": "http", "method": "get"},
        {"id": "step2", "mcp": "http", "method": "get", "depends_on": ["step1"]},
    ]
    
    dag = create_dag(steps)
    levels = dag.get_execution_levels()
    
    assert len(levels) == 2
    assert levels[0] == ["step1"]
    assert levels[1] == ["step2"]

def test_parallel_dag():
    """Test parallel execution levels"""
    steps = [
        {"id": "step1", "mcp": "http", "method": "get", "parallel": True},
        {"id": "step2", "mcp": "http", "method": "get", "parallel": True},
        {"id": "step3", "mcp": "http", "method": "get", "depends_on": ["step1", "step2"]},
    ]
    
    dag = create_dag(steps)
    levels = dag.get_execution_levels()
    
    assert len(levels) == 2
    assert set(levels[0]) == {"step1", "step2"}  # Can run in parallel
    assert levels[1] == ["step3"]

def test_dag_cycle_detection():
    """Test that cycles are detected"""
    steps = [
        {"id": "step1", "mcp": "http", "method": "get", "depends_on": ["step2"]},
        {"id": "step2", "mcp": "http", "method": "get", "depends_on": ["step1"]},
    ]
    
    with pytest.raises(DAGError, match="Cycle detected"):
        create_dag(steps)

def test_dag_unknown_dependency():
    """Test that unknown dependencies are detected"""
    steps = [
        {"id": "step1", "mcp": "http", "method": "get", "depends_on": ["unknown_step"]},
    ]
    
    with pytest.raises(DAGError, match="unknown step"):
        create_dag(steps)

def test_should_use_dag():
    """Test DAG detection logic"""
    # No dependencies or parallel flags
    steps1 = [
        {"id": "step1", "mcp": "http", "method": "get"},
    ]
    assert not should_use_dag(steps1)
    
    # Has dependencies
    steps2 = [
        {"id": "step1", "mcp": "http", "method": "get"},
        {"id": "step2", "mcp": "http", "method": "get", "depends_on": ["step1"]},
    ]
    assert should_use_dag(steps2)
    
    # Has parallel flag
    steps3 = [
        {"id": "step1", "mcp": "http", "method": "get", "parallel": True},
    ]
    assert should_use_dag(steps3)

def test_complex_dag():
    """Test complex DAG with multiple levels"""
    steps = [
        {"id": "fetch_user", "mcp": "http", "method": "get", "parallel": True},
        {"id": "fetch_posts", "mcp": "http", "method": "get", "parallel": True},
        {"id": "fetch_comments", "mcp": "http", "method": "get", "parallel": True},
        {"id": "merge_data", "mcp": "transform", "method": "merge", "depends_on": ["fetch_user", "fetch_posts"]},
        {"id": "final_output", "mcp": "logger", "method": "info", "depends_on": ["merge_data", "fetch_comments"]},
    ]
    
    dag = create_dag(steps)
    levels = dag.get_execution_levels()
    
    assert len(levels) == 3
    assert set(levels[0]) == {"fetch_user", "fetch_posts", "fetch_comments"}  # Level 1: all parallel
    assert levels[1] == ["merge_data"]  # Level 2: depends on user and posts
    assert levels[2] == ["final_output"]  # Level 3: depends on merge and comments
