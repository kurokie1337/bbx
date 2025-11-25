# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""Tests for BBX Adapters"""

import pytest
import tempfile
import os
from pathlib import Path


# =============================================================================
# Logger Adapter Tests
# =============================================================================

class TestLoggerAdapter:
    """Tests for LoggerAdapter"""

    @pytest.fixture
    def adapter(self):
        from blackbox.core.adapters.logger import LoggerAdapter
        return LoggerAdapter()

    @pytest.mark.asyncio
    async def test_log_info(self, adapter):
        """Test info logging"""
        result = await adapter.execute("info", {"message": "Test message"})
        assert result["status"] == "logged"
        assert result["level"] == "info"
        assert result["message"] == "Test message"

    @pytest.mark.asyncio
    async def test_log_warning(self, adapter):
        """Test warning logging"""
        result = await adapter.execute("warning", {"message": "Warning!"})
        assert result["status"] == "logged"
        assert result["level"] == "warning"

    @pytest.mark.asyncio
    async def test_log_error(self, adapter):
        """Test error logging"""
        result = await adapter.execute("error", {"message": "Error occurred"})
        assert result["status"] == "logged"
        assert result["level"] == "error"

    @pytest.mark.asyncio
    async def test_log_debug(self, adapter):
        """Test debug logging"""
        result = await adapter.execute("debug", {"message": "Debug info"})
        assert result["status"] == "logged"
        assert result["level"] == "debug"

    @pytest.mark.asyncio
    async def test_unknown_method(self, adapter):
        """Test unknown method raises error"""
        with pytest.raises(ValueError, match="Unknown logger method"):
            await adapter.execute("unknown", {})


# =============================================================================
# Transform Adapter Tests
# =============================================================================

class TestTransformAdapter:
    """Tests for TransformAdapter"""

    @pytest.fixture
    def adapter(self):
        from blackbox.core.adapters.transform import TransformAdapter
        return TransformAdapter()

    @pytest.mark.asyncio
    async def test_merge_dicts(self, adapter):
        """Test merging dictionaries"""
        result = await adapter.execute("merge", {
            "objects": [
                {"a": 1, "b": 2},
                {"c": 3, "d": 4}
            ]
        })
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    @pytest.mark.asyncio
    async def test_merge_multiple_dicts(self, adapter):
        """Test merging multiple dictionaries"""
        result = await adapter.execute("merge", {
            "objects": [
                {"name": "John"},
                {"age": 30},
                {"city": "NYC"}
            ]
        })
        assert result == {"name": "John", "age": 30, "city": "NYC"}

    @pytest.mark.asyncio
    async def test_filter_greater_than(self, adapter):
        """Test filtering with greater than"""
        result = await adapter.execute("filter", {
            "array": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "condition": "> 5"
        })
        assert result == [6, 7, 8, 9, 10]

    @pytest.mark.asyncio
    async def test_filter_less_than(self, adapter):
        """Test filtering with less than"""
        result = await adapter.execute("filter", {
            "array": [1, 2, 3, 4, 5],
            "condition": "< 3"
        })
        assert result == [1, 2]

    @pytest.mark.asyncio
    async def test_map_field_extraction(self, adapter):
        """Test mapping with field extraction"""
        result = await adapter.execute("map", {
            "array": [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 25}
            ],
            "field": "name"
        })
        assert result == ["John", "Jane"]

    @pytest.mark.asyncio
    async def test_reduce_sum(self, adapter):
        """Test reduce with sum"""
        result = await adapter.execute("reduce", {
            "array": [1, 2, 3, 4, 5],
            "operation": "sum"
        })
        assert result == 15

    @pytest.mark.asyncio
    async def test_reduce_count(self, adapter):
        """Test reduce with count"""
        result = await adapter.execute("reduce", {
            "array": [1, 2, 3, 4, 5],
            "operation": "count"
        })
        assert result == 5

    @pytest.mark.asyncio
    async def test_reduce_min(self, adapter):
        """Test reduce with min"""
        result = await adapter.execute("reduce", {
            "array": [5, 2, 8, 1, 9],
            "operation": "min"
        })
        assert result == 1

    @pytest.mark.asyncio
    async def test_reduce_max(self, adapter):
        """Test reduce with max"""
        result = await adapter.execute("reduce", {
            "array": [5, 2, 8, 1, 9],
            "operation": "max"
        })
        assert result == 9

    @pytest.mark.asyncio
    async def test_reduce_average(self, adapter):
        """Test reduce with average"""
        result = await adapter.execute("reduce", {
            "array": [10, 20, 30],
            "operation": "average"
        })
        assert result == 20

    @pytest.mark.asyncio
    async def test_extract_simple(self, adapter):
        """Test simple field extraction"""
        result = await adapter.execute("extract", {
            "object": {"name": "John", "age": 30},
            "path": "name"
        })
        assert result == "John"

    @pytest.mark.asyncio
    async def test_extract_nested(self, adapter):
        """Test nested field extraction"""
        result = await adapter.execute("extract", {
            "object": {"user": {"profile": {"name": "Alice"}}},
            "path": "user.profile.name"
        })
        assert result == "Alice"

    @pytest.mark.asyncio
    async def test_format_json(self, adapter):
        """Test JSON formatting"""
        result = await adapter.execute("format", {
            "data": {"name": "Bob"},
            "format": "json"
        })
        assert "name" in result
        assert "Bob" in result


# =============================================================================
# Storage Adapter Tests
# =============================================================================

class TestStorageAdapter:
    """Tests for StorageAdapter"""

    @pytest.fixture
    def adapter(self):
        from blackbox.core.adapters.storage import StorageAdapter
        return StorageAdapter()

    @pytest.mark.asyncio
    async def test_kv_set_and_get(self, adapter):
        """Test setting and getting a value"""
        await adapter.execute("kv.set", {"key": "test_key_1", "value": "test_value"})
        result = await adapter.execute("kv.get", {"key": "test_key_1"})
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_kv_get_nonexistent(self, adapter):
        """Test getting nonexistent key"""
        result = await adapter.execute("kv.get", {"key": "nonexistent_key_xyz"})
        assert result is None

    @pytest.mark.asyncio
    async def test_kv_get_with_default(self, adapter):
        """Test getting nonexistent key with default"""
        result = await adapter.execute("kv.get", {"key": "nonexistent_abc", "default": "fallback"})
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_kv_delete(self, adapter):
        """Test deleting a key"""
        await adapter.execute("kv.set", {"key": "to_delete_2", "value": "temp"})
        await adapter.execute("kv.delete", {"key": "to_delete_2"})
        result = await adapter.execute("kv.get", {"key": "to_delete_2"})
        assert result is None

    @pytest.mark.asyncio
    async def test_kv_set_complex_value(self, adapter):
        """Test storing complex JSON value"""
        complex_value = {"nested": {"data": [1, 2, 3]}, "flag": True}
        await adapter.execute("kv.set", {"key": "complex_3", "value": complex_value})
        result = await adapter.execute("kv.get", {"key": "complex_3"})
        assert result == complex_value


# =============================================================================
# Python Adapter Tests
# =============================================================================

class TestPythonAdapter:
    """Tests for PythonAdapter"""

    @pytest.fixture
    def adapter(self):
        from blackbox.core.adapters.python import PythonAdapter
        return PythonAdapter()

    @pytest.mark.asyncio
    async def test_simple_script(self, adapter):
        """Test simple Python script"""
        result = await adapter.execute("script", {
            "script": "print('Hello from Python!')"
        })
        assert result["success"] is True
        stdout = result.get("metadata", {}).get("stdout", "")
        assert "Hello from Python!" in stdout

    @pytest.mark.asyncio
    async def test_script_with_calculation(self, adapter):
        """Test script with calculation"""
        result = await adapter.execute("script", {
            "script": "x = 2 + 2\nprint(x)"
        })
        assert result["success"] is True
        stdout = result.get("metadata", {}).get("stdout", "")
        assert "4" in stdout

    @pytest.mark.asyncio
    async def test_script_with_json(self, adapter):
        """Test script using json module"""
        result = await adapter.execute("script", {
            "script": "import json\ndata = {'a': 1}\nprint(json.dumps(data))"
        })
        assert result["success"] is True


# =============================================================================
# Registry Tests
# =============================================================================

class TestRegistry:
    """Tests for MCPRegistry"""

    def test_get_logger_adapter(self):
        """Test getting logger adapter"""
        from blackbox.core.registry import MCPRegistry
        registry = MCPRegistry()
        adapter = registry.get_adapter("logger")
        assert adapter is not None

    def test_get_bbx_logger_adapter(self):
        """Test getting adapter with bbx. prefix"""
        from blackbox.core.registry import MCPRegistry
        registry = MCPRegistry()
        adapter = registry.get_adapter("bbx.logger")
        assert adapter is not None

    def test_get_http_adapter(self):
        """Test getting HTTP adapter"""
        from blackbox.core.registry import MCPRegistry
        registry = MCPRegistry()
        adapter = registry.get_adapter("http")
        assert adapter is not None

    def test_get_transform_adapter(self):
        """Test getting transform adapter"""
        from blackbox.core.registry import MCPRegistry
        registry = MCPRegistry()
        adapter = registry.get_adapter("transform")
        assert adapter is not None

    def test_get_docker_adapter(self):
        """Test getting docker adapter"""
        from blackbox.core.registry import MCPRegistry
        registry = MCPRegistry()
        adapter = registry.get_adapter("docker")
        assert adapter is not None

    def test_get_unknown_adapter(self):
        """Test getting unknown adapter returns None"""
        from blackbox.core.registry import MCPRegistry
        registry = MCPRegistry()
        adapter = registry.get_adapter("nonexistent_adapter")
        assert adapter is None

    def test_list_adapters(self):
        """Test listing all adapters"""
        from blackbox.core.registry import MCPRegistry
        registry = MCPRegistry()
        adapters = registry.list_adapters()
        assert "logger" in adapters
        assert "http" in adapters
        assert "transform" in adapters
        assert "docker" in adapters
        assert "system" in adapters


# =============================================================================
# Parser Tests
# =============================================================================

class TestBBXParser:
    """Tests for BBX v6 Parser"""

    def test_parse_duration_seconds(self):
        """Test parsing seconds"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        assert BBXv6Parser.parse_duration("5s") == 5000

    def test_parse_duration_milliseconds(self):
        """Test parsing milliseconds"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        assert BBXv6Parser.parse_duration("500ms") == 500

    def test_parse_duration_minutes(self):
        """Test parsing minutes"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        assert BBXv6Parser.parse_duration("2m") == 120000

    def test_parse_duration_hours(self):
        """Test parsing hours"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        assert BBXv6Parser.parse_duration("1h") == 3600000

    def test_parse_duration_number(self):
        """Test parsing plain number"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        assert BBXv6Parser.parse_duration(5000) == 5000

    def test_detect_version_v6_dict_steps(self):
        """Test detecting v6 format with dict steps"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        data = {
            "id": "test",
            "steps": {
                "step1": {"use": "logger.info", "args": {"message": "test"}}
            }
        }
        assert BBXv6Parser.detect_version(data) == "6.0"

    def test_detect_version_v6_use_field(self):
        """Test detecting v6 format by use field"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        data = {
            "id": "test",
            "steps": [
                {"id": "step1", "use": "logger.info", "args": {"message": "test"}}
            ]
        }
        assert BBXv6Parser.detect_version(data) == "6.0"

    def test_detect_version_v5(self):
        """Test detecting v5 format"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        data = {
            "workflow": {
                "id": "test",
                "steps": [
                    {"id": "step1", "mcp": "logger", "method": "info"}
                ]
            }
        }
        assert BBXv6Parser.detect_version(data) == "5.0"

    def test_parse_step_v6_to_v5(self):
        """Test converting v6 step to v5"""
        from blackbox.core.parsers.v6 import BBXv6Parser
        v6_step = {
            "use": "http.get",
            "args": {"url": "https://example.com"},
            "timeout": "10s"
        }
        v5_step = BBXv6Parser.parse_step("fetch", v6_step)

        assert v5_step["id"] == "fetch"
        assert v5_step["mcp"] == "http"
        assert v5_step["method"] == "get"
        assert v5_step["inputs"]["url"] == "https://example.com"
        assert v5_step["timeout"] == 10000


# =============================================================================
# Workflow Runtime Tests
# =============================================================================

class TestWorkflowRuntime:
    """Tests for workflow runtime"""

    @pytest.mark.asyncio
    async def test_run_simple_workflow(self):
        """Test running a simple workflow"""
        from blackbox.core.runtime import run_file
        import tempfile
        import os

        workflow_content = """
id: test_workflow
name: Test Workflow
steps:
  log_test:
    use: logger.info
    args:
      message: "Test passed!"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False, encoding='utf-8') as f:
            f.write(workflow_content)
            temp_path = f.name

        try:
            results = await run_file(temp_path)
            assert "log_test" in results
            assert results["log_test"]["status"] == "success"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_run_workflow_with_inputs(self):
        """Test running workflow with inputs"""
        from blackbox.core.runtime import run_file
        import tempfile
        import os

        workflow_content = """
id: input_test
steps:
  log_input:
    use: logger.info
    args:
      message: "Hello, ${inputs.name}!"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False, encoding='utf-8') as f:
            f.write(workflow_content)
            temp_path = f.name

        try:
            results = await run_file(temp_path, inputs={"name": "World"})
            assert "log_input" in results
            assert results["log_input"]["status"] == "success"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_run_workflow_with_depends_on(self):
        """Test workflow with dependencies (DAG)"""
        from blackbox.core.runtime import run_file
        import tempfile
        import os

        workflow_content = """
id: deps_test
steps:
  step1:
    use: logger.info
    args:
      message: "Step 1"
  step2:
    use: logger.info
    depends_on: [step1]
    args:
      message: "Step 2 after Step 1"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False, encoding='utf-8') as f:
            f.write(workflow_content)
            temp_path = f.name

        try:
            results = await run_file(temp_path)
            assert results["step1"]["status"] == "success"
            assert results["step2"]["status"] == "success"
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_run_transform_workflow(self):
        """Test workflow with transform operations"""
        from blackbox.core.runtime import run_file
        import tempfile
        import os

        workflow_content = """
id: transform_test
steps:
  merge:
    use: transform.merge
    args:
      objects:
        - name: "John"
        - age: 30
  log_result:
    use: logger.info
    depends_on: [merge]
    args:
      message: "Merged data"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bbx', delete=False, encoding='utf-8') as f:
            f.write(workflow_content)
            temp_path = f.name

        try:
            results = await run_file(temp_path)
            assert results["merge"]["status"] == "success"
            assert results["merge"]["output"] == {"name": "John", "age": 30}
        finally:
            os.unlink(temp_path)
