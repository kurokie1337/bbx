
import asyncio
import json
import unittest
from unittest.mock import MagicMock, patch

from blackbox.core.adapters.claude_hooks import ClaudeHooksAdapter

class TestClaudeHooksAdapter(unittest.TestCase):
    @patch("blackbox.core.adapters.claude_hooks.BBXRuntimeV2")
    @patch("blackbox.core.adapters.claude_hooks.Path.exists")
    def test_handle_event_allow(self, mock_exists, mock_runtime_cls):
        # Instantiate adapter here so it uses the mocked runtime
        adapter = ClaudeHooksAdapter()
        
        # Mock exists
        mock_exists.return_value = True
        
        # Mock runtime
        mock_runtime = mock_runtime_cls.return_value
        
        # Mock execute_file result
        # The adapter expects results['hook_response']['output'] or similar
        async def mock_execute(*args, **kwargs):
            return {
                "hook_response": {
                    "status": "success",
                    "output": {
                        "success": True,
                        "metadata": {
                            "stdout": json.dumps({
                                "decision": "allow",
                                "reason": "Test allow"
                            }),
                            "stderr": ""
                        }
                    }
                }
            }
        mock_runtime.execute_file.side_effect = mock_execute
        
        # Run async test
        async def run_test():
            event_data = {"hook_event_name": "PreToolUse", "tool_name": "Read"}
            result = await adapter.handle_event(event_data, workflow_path="dummy.yaml")
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result["decision"], "allow")
        self.assertEqual(result["reason"], "Test allow")

    @patch("blackbox.core.adapters.claude_hooks.BBXRuntimeV2")
    @patch("blackbox.core.adapters.claude_hooks.Path.exists")
    def test_handle_event_deny(self, mock_exists, mock_runtime_cls):
        adapter = ClaudeHooksAdapter()
        
        # Mock exists
        mock_exists.return_value = True
        
        # Mock runtime
        mock_runtime = mock_runtime_cls.return_value
        
        async def mock_execute(*args, **kwargs):
            return {
                "hook_response": {
                    "status": "success",
                    "output": {
                        "success": True,
                        "metadata": {
                            "stdout": json.dumps({
                                "decision": "deny",
                                "reason": "Test deny"
                            }),
                            "stderr": ""
                        }
                    }
                }
            }
        mock_runtime.execute_file.side_effect = mock_execute
        
        async def run_test():
            event_data = {"hook_event_name": "PreToolUse", "tool_name": "Rm"}
            result = await adapter.handle_event(event_data, workflow_path="dummy.yaml")
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result["decision"], "deny")
        self.assertEqual(result["reason"], "Test deny")

if __name__ == "__main__":
    unittest.main()
