# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Complex End-to-End Integration Tests for BBX v2

These tests verify the complete v2 system behavior including:
- Full workflow execution through runtime
- Component integration (Ring, Hooks, Flow, Quotas, Snapshots)
- Error handling and recovery
- Performance under load
- State management and persistence
- Policy enforcement modes
"""

import asyncio
import json
import pytest
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


def run_cli(args: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run CLI command and return exit code, stdout, stderr."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli.py")] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


class TestComplexWorkflowExecution:
    """Test complex workflow execution scenarios"""

    def test_hello_world_full_cycle(self):
        """Test simplest workflow runs completely"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0
        assert "BBX 2.0 Runtime started successfully" in stderr
        assert "Hello from Blackbox!" in stderr  # Skip emoji check
        assert "BBX 2.0 Runtime stopped" in stderr
        assert "[+] Step: greet" in stdout
        assert "success" in stdout

    def test_parallel_workflow_execution(self):
        """Test parallel workflow execution with AgentRing"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/parallel_demo.bbx"])
        assert code == 0
        assert "DAG parallel execution with AgentRing" in stderr
        assert "Level 1:" in stderr  # Parallel levels detected

    def test_transform_demo_all_steps_succeed(self):
        """Test transform demo with data transformations"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/transform_demo.bbx"])
        assert code == 0
        assert "Transform Demo Starting" in stderr
        assert "Transform Demo Complete" in stderr

        # Verify transform operations worked
        assert "[+] Step: merge_data" in stdout
        assert "[+] Step: filter_numbers" in stdout
        assert "[+] Step: sum_numbers" in stdout
        assert "[+] Step: extract_field" in stdout

    def test_bbx_only_demo_file_operations(self):
        """Test BBX-only demo with file operations"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/bbx_only_demo.bbx"])
        assert code == 0

        # Verify file operations
        assert "[+] Step: create_output_dir" in stdout
        assert "[+] Step: write_json_output" in stdout
        assert "[+] Step: write_backup" in stdout
        assert "[+] Step: create_report" in stdout
        assert "[+] Step: list_output" in stdout

        # Verify state operations
        assert "[+] Step: save_summary" in stdout


class TestV2ComponentIntegration:
    """Test integration of all v2 components"""

    def test_all_kernel_components_start(self):
        """Verify all kernel components start correctly"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0

        # Kernel components
        assert "[KERNEL] AgentRing started" in stderr
        assert "[KERNEL] BBX Hooks started" in stderr
        assert "[KERNEL] ContextTiering started" in stderr
        assert "[KERNEL] FlowIntegrity started" in stderr
        assert "[KERNEL] AgentQuotas started" in stderr
        assert "[KERNEL] StateSnapshots started" in stderr

    def test_nt_kernel_components_start(self):
        """Verify NT Kernel components start correctly"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0

        # NT Kernel components
        assert "[NT KERNEL] BBX Executive started" in stderr
        assert "Executive-managed components" in stderr

    def test_distro_components_start(self):
        """Verify distribution layer components start correctly"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0

        # Distribution layer
        assert "[DISTRO] PolicyEngine started" in stderr
        assert "[DISTRO] NetworkFabric started" in stderr
        assert "[DISTRO] AgentBundles started" in stderr

    def test_filter_stack_loaded(self):
        """Verify filter stack is properly loaded"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0

        # Verify filters
        assert "Filter loaded: AuditFilter" in stderr
        assert "Filter loaded: MetricsFilter" in stderr
        assert "Filter loaded: SecurityFilter" in stderr
        assert "Filter loaded: QuotaFilter" in stderr
        assert "Filter loaded: RateLimitFilter" in stderr
        assert "Filter loaded: CacheFilter" in stderr

    def test_builtin_hooks_registered(self):
        """Verify built-in hooks are registered"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0

        assert "Registered hook: builtin_metrics" in stderr
        assert "Registered hook: builtin_flow_integrity" in stderr


class TestFlowIntegrityBehavior:
    """Test flow integrity system behavior"""

    def test_flow_integrity_warnings_in_permissive_mode(self):
        """Verify flow integrity logs warnings but doesn't block in permissive mode"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/bbx_only_demo.bbx"])
        assert code == 0

        # Should have warnings but continue
        assert "FlowIntegrity violation" in stderr
        assert "permissive" in stderr

        # All steps should succeed
        success_count = stdout.count("[+] Step:")
        assert success_count >= 10, f"Expected 10+ successful steps, got {success_count}"


class TestAgentRingOperations:
    """Test AgentRing high-performance batch operations"""

    def test_ring_benchmark_performance(self):
        """Test AgentRing benchmark meets performance targets"""
        code, stdout, stderr = run_cli(
            ["v2", "ring", "benchmark", "-n", "100", "-b", "20", "-w", "4"],
            timeout=120
        )
        assert code == 0
        assert "Benchmark Results" in stdout
        assert "Throughput:" in stdout
        assert "Successful:" in stdout and "100" in stdout

    def test_ring_stats_show_operations(self):
        """Test ring stats command"""
        code, stdout, stderr = run_cli(["v2", "ring", "stats"])
        assert code == 0
        assert "AgentRing Statistics" in stdout
        assert "Operations:" in stdout
        assert "Workers:" in stdout


class TestResourceQuotas:
    """Test resource quota management"""

    def test_quotas_stats_show_hierarchy(self):
        """Test quotas stats shows quota hierarchy"""
        code, stdout, stderr = run_cli(["v2", "quotas", "stats"])
        assert code == 0
        assert "Agent Quotas Statistics" in stdout
        assert "Root Group:" in stdout

    def test_quotas_with_json_output(self):
        """Test quotas with JSON output format"""
        code, stdout, stderr = run_cli(["v2", "quotas", "stats", "-f", "json"])
        assert code == 0
        # Should contain JSON-like structure
        assert "{" in stdout or "root" in stdout.lower()


class TestStateSnapshots:
    """Test state snapshot system"""

    def test_workflow_uses_state_snapshots(self):
        """Test that workflows use state snapshots for state management"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/bbx_only_demo.bbx"])
        assert code == 0

        # state.set step should succeed
        assert "[+] Step: save_summary" in stdout
        assert "bbx_only_demo_result" in stdout


class TestMemoryManagement:
    """Test working set memory management"""

    def test_memory_stats(self):
        """Test memory stats command"""
        code, stdout, stderr = run_cli(["v2", "memory", "stats"])
        assert code == 0
        assert "Working Set Statistics" in stdout
        assert "Memory:" in stdout
        assert "Pages:" in stdout


class TestExecutiveSubsystem:
    """Test BBX Executive subsystem"""

    def test_executive_status_shows_all_managers(self):
        """Test executive status shows all managed components"""
        code, stdout, stderr = run_cli(["v2", "executive", "status"])
        assert code == 0
        assert "BBX Executive Status" in stdout
        assert "Kernel:" in stdout

    def test_objects_list_shows_namespace(self):
        """Test objects list command"""
        code, stdout, stderr = run_cli(["v2", "objects", "list"])
        assert code == 0
        assert "Object Namespace:" in stdout

    def test_registry_list(self):
        """Test registry list command"""
        code, stdout, stderr = run_cli(["v2", "registry", "list"])
        assert code == 0
        assert "Registry:" in stdout


class TestPolicyEngine:
    """Test policy engine"""

    def test_policy_list(self):
        """Test policy list command"""
        code, stdout, stderr = run_cli(["v2", "policy", "list"])
        assert code == 0
        assert "Policies" in stdout

    def test_policy_stats(self):
        """Test policy stats command"""
        code, stdout, stderr = run_cli(["v2", "policy", "stats"])
        assert code == 0
        assert "Policy Engine Statistics" in stdout


class TestAAL:
    """Test Agent Abstraction Layer"""

    def test_aal_stats(self):
        """Test AAL stats command"""
        code, stdout, stderr = run_cli(["v2", "aal", "stats"])
        assert code == 0
        assert "AAL Statistics" in stdout


class TestCleanShutdown:
    """Test clean shutdown of all components"""

    def test_all_components_stop_cleanly(self):
        """Verify all components stop cleanly after workflow"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0

        # Verify clean shutdown
        assert "AgentRing stopped" in stderr
        assert "ContextTiering stopped" in stderr
        assert "BBX Executive stopped" in stderr
        assert "BBX 2.0 Runtime stopped" in stderr

        # Verify filters unloaded
        assert "Filter unloaded: AuditFilter" in stderr
        assert "Filter unloaded: MetricsFilter" in stderr
        assert "Filter unloaded: SecurityFilter" in stderr


class TestConcurrentWorkloads:
    """Test concurrent workload handling"""

    def test_parallel_steps_execute_concurrently(self):
        """Test that parallel steps execute via AgentRing"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/bbx_only_demo.bbx"])
        assert code == 0

        # Should see [RING] prefix for parallel operations
        assert "[RING]" in stderr

    def test_dag_parallel_execution_enabled(self):
        """Test DAG parallel execution is detected"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/bbx_only_demo.bbx"])
        assert code == 0
        assert "DAG parallel execution with AgentRing" in stderr


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_missing_workflow_file(self):
        """Test handling of missing workflow file"""
        code, stdout, stderr = run_cli(["v2", "run", "nonexistent.bbx"])
        assert code != 0

    def test_workflow_with_external_api_failures(self):
        """Test workflow handles external API failures gracefully"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/parallel_demo.bbx"])
        # The workflow should complete even if http.get fails
        assert "BBX 2.0 Runtime stopped" in stderr


class TestStatusCommands:
    """Test v2 status commands"""

    def test_v2_status_shows_all_components(self):
        """Test v2 status shows comprehensive component list"""
        code, stdout, stderr = run_cli(["v2", "status"])
        assert code == 0
        assert "BBX 2.0 System Status" in stdout
        assert "[+]" in stdout  # Active components

    def test_v2_status_component_count(self):
        """Test v2 status shows sufficient active components"""
        code, stdout, stderr = run_cli(["v2", "status"])
        assert code == 0

        active_count = stdout.count("[+]")
        assert active_count >= 5, f"Expected 5+ active components, got {active_count}"


class TestContextTiering:
    """Test context tiering system"""

    def test_context_stats(self):
        """Test context tiering stats"""
        code, stdout, stderr = run_cli(["v2", "context", "stats"])
        assert code == 0
        assert "Context Tiering Statistics" in stdout
        assert "Items by Tier:" in stdout
        assert "HOT:" in stdout
        assert "WARM:" in stdout


class TestHooksSystem:
    """Test hooks system"""

    def test_hooks_list(self):
        """Test hooks list command"""
        code, stdout, stderr = run_cli(["v2", "hooks", "list"])
        assert code == 0
        assert "hooks" in stdout.lower() or "No hooks registered" in stdout

    def test_hooks_stats(self):
        """Test hooks stats command"""
        code, stdout, stderr = run_cli(["v2", "hooks", "stats"])
        assert code == 0
        assert "Hooks Statistics" in stdout


class TestFiltersSystem:
    """Test filter stack system"""

    def test_filters_list(self):
        """Test filters list command"""
        code, stdout, stderr = run_cli(["v2", "filters", "list"])
        assert code == 0
        assert "filter" in stdout.lower() or "No filters" in stdout


class TestConfigSystem:
    """Test declarative config system"""

    def test_config_show(self):
        """Test config show command"""
        code, stdout, stderr = run_cli(["v2", "config", "show"])
        assert code == 0
        assert "config" in stdout.lower() or "No configuration" in stdout

    def test_generation_list(self):
        """Test generation list command"""
        code, stdout, stderr = run_cli(["v2", "generation", "list"])
        assert code == 0
        assert "generation" in stdout.lower() or "No generations" in stdout


class TestWorkflowOutputIntegrity:
    """Test workflow output data integrity"""

    def test_transform_output_values_correct(self):
        """Verify transform operations produce correct values"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/transform_demo.bbx"])
        assert code == 0

        # Verify merge produced combined dict
        assert "merge_data" in stdout
        assert "name" in stdout.lower() or "John" in stdout

        # Verify filter produced correct values
        assert "filter_numbers" in stdout
        # Filter of [1-10] where x > 5 = [6,7,8,9,10]
        assert "[6, 7, 8, 9, 10]" in stdout

        # Verify reduce/sum produced correct value
        assert "sum_numbers" in stdout
        # Sum of [1-10] = 55
        assert "55" in stdout

    def test_file_operations_create_files(self):
        """Verify file operations create actual files"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/bbx_only_demo.bbx"])
        assert code == 0

        # Check output indicates files were created
        assert "output" in stdout
        assert "result.json" in stdout
        assert "backup.b64" in stdout
        assert "report.md" in stdout


class TestStepTimingMetrics:
    """Test step timing and metrics"""

    def test_steps_report_timing(self):
        """Verify steps report execution timing"""
        code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
        assert code == 0
        assert "completed in" in stderr
        assert "ms" in stderr


class TestSystemStress:
    """Stress tests for the v2 system"""

    def test_ring_handles_burst_load(self):
        """Test AgentRing handles burst of operations"""
        code, stdout, stderr = run_cli(
            ["v2", "ring", "benchmark", "-n", "200", "-b", "50", "-w", "4"],
            timeout=180
        )
        assert code == 0
        assert "Successful:" in stdout and "200" in stdout

    def test_multiple_sequential_workflows(self):
        """Test running multiple workflows sequentially"""
        for _ in range(3):
            code, stdout, stderr = run_cli(["v2", "run", "examples/hello_world.bbx"])
            assert code == 0
            assert "BBX 2.0 Runtime stopped" in stderr
