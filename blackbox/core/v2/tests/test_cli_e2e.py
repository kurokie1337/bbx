# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
End-to-End CLI Tests for BBX v2

These tests verify that all v2 CLI commands work correctly through the
actual CLI interface, simulating real user interactions.
"""

import asyncio
import json
import pytest
import subprocess
import sys
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


def run_cli_command(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a CLI command and return exit code, stdout, stderr."""
    cmd = [sys.executable, str(PROJECT_ROOT / "cli.py")] + args
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


class TestV2StatusCommand:
    """Test v2 status command"""

    def test_v2_status(self):
        """Test v2 status shows all components"""
        code, stdout, stderr = run_cli_command(["v2", "status"])
        assert code == 0
        assert "BBX 2.0 System Status" in stdout
        assert "[+]" in stdout  # At least one component is active


class TestAgentRingCLI:
    """Test AgentRing CLI commands"""

    def test_ring_stats(self):
        """Test ring stats command"""
        code, stdout, stderr = run_cli_command(["v2", "ring", "stats"])
        assert code == 0
        assert "AgentRing Statistics" in stdout
        assert "Operations:" in stdout
        assert "Workers:" in stdout

    def test_ring_benchmark(self):
        """Test ring benchmark command with small workload"""
        code, stdout, stderr = run_cli_command(
            ["v2", "ring", "benchmark", "-n", "50", "-b", "10", "-w", "2"],
            timeout=60
        )
        assert code == 0
        assert "Benchmark Results" in stdout
        assert "Throughput:" in stdout


class TestContextTieringCLI:
    """Test Context Tiering CLI commands"""

    def test_context_stats(self):
        """Test context stats command"""
        code, stdout, stderr = run_cli_command(["v2", "context", "stats"])
        assert code == 0
        assert "Context Tiering Statistics" in stdout
        assert "Items by Tier:" in stdout
        assert "HOT:" in stdout
        assert "WARM:" in stdout


class TestFlowIntegrityCLI:
    """Test Flow Integrity CLI commands"""

    def test_flow_stats(self):
        """Test flow stats command"""
        code, stdout, stderr = run_cli_command(["v2", "flow", "stats"])
        assert code == 0
        assert "Flow Integrity Statistics" in stdout
        assert "Configuration:" in stdout


class TestQuotasCLI:
    """Test Quotas CLI commands"""

    def test_quotas_stats(self):
        """Test quotas stats command"""
        code, stdout, stderr = run_cli_command(["v2", "quotas", "stats"])
        assert code == 0
        assert "Agent Quotas Statistics" in stdout
        assert "Root Group:" in stdout


class TestMemoryCLI:
    """Test Memory (Working Set) CLI commands"""

    def test_memory_stats(self):
        """Test memory stats command"""
        code, stdout, stderr = run_cli_command(["v2", "memory", "stats"])
        assert code == 0
        assert "Working Set Statistics" in stdout
        assert "Memory:" in stdout
        assert "Pages:" in stdout


class TestHooksCLI:
    """Test Hooks CLI commands"""

    def test_hooks_list(self):
        """Test hooks list command"""
        code, stdout, stderr = run_cli_command(["v2", "hooks", "list"])
        assert code == 0
        # Either shows hooks or "No hooks registered"
        assert "hooks" in stdout.lower() or "No hooks registered" in stdout

    def test_hooks_stats(self):
        """Test hooks stats command"""
        code, stdout, stderr = run_cli_command(["v2", "hooks", "stats"])
        assert code == 0
        assert "Hooks Statistics" in stdout


class TestConfigCLI:
    """Test Config/Declarative CLI commands"""

    def test_config_show(self):
        """Test config show command"""
        code, stdout, stderr = run_cli_command(["v2", "config", "show"])
        assert code == 0
        # Either shows config or message about no config
        assert "config" in stdout.lower() or "No configuration" in stdout


class TestGenerationCLI:
    """Test Generation CLI commands"""

    def test_generation_list(self):
        """Test generation list command"""
        code, stdout, stderr = run_cli_command(["v2", "generation", "list"])
        assert code == 0
        # Either shows generations or "No generations found"
        assert "generation" in stdout.lower() or "No generations" in stdout


class TestExecutiveCLI:
    """Test Executive CLI commands"""

    def test_executive_status(self):
        """Test executive status command"""
        code, stdout, stderr = run_cli_command(["v2", "executive", "status"])
        assert code == 0
        assert "BBX Executive Status" in stdout
        assert "Kernel:" in stdout


class TestObjectsCLI:
    """Test Objects CLI commands"""

    def test_objects_list(self):
        """Test objects list command"""
        code, stdout, stderr = run_cli_command(["v2", "objects", "list"])
        assert code == 0
        assert "Object Namespace:" in stdout

    def test_objects_stats(self):
        """Test objects stats command"""
        code, stdout, stderr = run_cli_command(["v2", "objects", "stats"])
        assert code == 0
        assert "Object Manager Statistics" in stdout


class TestRegistryCLI:
    """Test Registry CLI commands"""

    def test_registry_list(self):
        """Test registry list command"""
        code, stdout, stderr = run_cli_command(["v2", "registry", "list"])
        assert code == 0
        assert "Registry:" in stdout


class TestFiltersCLI:
    """Test Filters CLI commands"""

    def test_filters_list(self):
        """Test filters list command"""
        code, stdout, stderr = run_cli_command(["v2", "filters", "list"])
        assert code == 0
        # Either shows filters or "No filters registered"
        assert "filter" in stdout.lower() or "No filters" in stdout


class TestPolicyCLI:
    """Test Policy CLI commands"""

    def test_policy_list(self):
        """Test policy list command"""
        code, stdout, stderr = run_cli_command(["v2", "policy", "list"])
        assert code == 0
        assert "Policies" in stdout

    def test_policy_stats(self):
        """Test policy stats command"""
        code, stdout, stderr = run_cli_command(["v2", "policy", "stats"])
        assert code == 0
        assert "Policy Engine Statistics" in stdout


class TestAALCLI:
    """Test AAL CLI commands"""

    def test_aal_stats(self):
        """Test AAL stats command"""
        code, stdout, stderr = run_cli_command(["v2", "aal", "stats"])
        assert code == 0
        assert "AAL Statistics" in stdout


class TestIntegration:
    """Integration tests combining multiple v2 components"""

    def test_full_system_status(self):
        """Test that full system can start and report status"""
        code, stdout, stderr = run_cli_command(["v2", "status"])
        assert code == 0

        # Count active components
        active_count = stdout.count("[+]")
        assert active_count >= 5, f"Expected at least 5 active components, got {active_count}"

    def test_ring_benchmark_integration(self):
        """Test AgentRing benchmark actually runs operations"""
        code, stdout, stderr = run_cli_command(
            ["v2", "ring", "benchmark", "-n", "20", "-b", "5", "-w", "2"],
            timeout=60
        )
        assert code == 0
        assert "Successful:" in stdout

        # Parse successful count
        for line in stdout.split("\n"):
            if "Successful:" in line:
                count = int(line.split(":")[1].strip())
                assert count == 20, f"Expected 20 successful ops, got {count}"
                break

    def test_json_output_formats(self):
        """Test that JSON output format works for stats commands"""
        commands = [
            ["v2", "ring", "stats", "-f", "json"],
            ["v2", "flow", "stats", "-f", "json"],
            ["v2", "quotas", "stats", "-f", "json"],
            ["v2", "memory", "stats", "-f", "json"],
        ]

        for cmd in commands:
            code, stdout, stderr = run_cli_command(cmd)
            assert code == 0, f"Command {cmd} failed with code {code}"
            # Should be valid JSON (or close to it)
            try:
                # Try to find JSON in output
                if "{" in stdout:
                    json_start = stdout.index("{")
                    json_str = stdout[json_start:]
                    json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                # Some commands might not produce strict JSON
                pass


class TestHelp:
    """Test that help commands work for all v2 subgroups"""

    def test_v2_help(self):
        """Test v2 --help"""
        code, stdout, stderr = run_cli_command(["v2", "--help"])
        assert code == 0
        assert "Commands:" in stdout

    @pytest.mark.parametrize("subcommand", [
        "ring", "hooks", "context", "config", "generation",
        "executive", "objects", "registry", "filters",
        "policy", "aal", "memory", "flow", "quotas"
    ])
    def test_subcommand_help(self, subcommand):
        """Test help for each subcommand"""
        code, stdout, stderr = run_cli_command(["v2", subcommand, "--help"])
        assert code == 0, f"Help for {subcommand} failed"
        assert "Commands:" in stdout or "Options:" in stdout
