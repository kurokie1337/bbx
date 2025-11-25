# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Python Adapter

Allows executing inline Python scripts with access to workflow variables.
"""

import io
import sys
import traceback
from typing import Any, Dict

from ..base_adapter import BaseAdapter, AdapterResponse, AdapterErrorType


class VariablesWrapper:
    """Wrapper for context variables to provide .get() and .set() API"""

    def __init__(self, context_variables: Dict[str, Any]):
        self._vars = context_variables

    def get(self, key: str, default: Any = None) -> Any:
        """Get a variable value"""
        return self._vars.get(key, default)

    def set(self, key: str, value: Any):
        """Set a variable value"""
        self._vars[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._vars[key]

    def __setitem__(self, key: str, value: Any):
        self._vars[key] = value


class PythonAdapter(BaseAdapter):
    """
    Adapter for executing Python code.
    """

    def __init__(self):
        super().__init__("python")
        self.context = None
        self.register_method("script", self.run_script)

    def set_context(self, context):
        """Inject workflow context"""
        self.context = context

    async def run_script(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a Python script.
        
        Inputs:
            script: The Python code to execute
        """
        script = inputs.get("script")
        if not script:
            return AdapterResponse.error_response("Script required").to_dict()

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # Prepare environment
        # We inject 'variables' wrapper and 'print' that writes to our capture
        
        variables = VariablesWrapper(self.context.variables if self.context else {})
        
        # Custom print function to capture output
        def custom_print(*args, **kwargs):
            # Use file=stdout_capture by default unless specified
            if "file" not in kwargs:
                kwargs["file"] = stdout_capture
            print(*args, **kwargs)

        # Execution scope
        scope = {
            "variables": variables,
            "print": custom_print,
            "sys": sys,
            "os": __import__("os"),
            "json": __import__("json"),
        }

        try:
            # Execute script
            exec(script, scope)
            
            stdout_val = stdout_capture.getvalue()
            stderr_val = stderr_capture.getvalue()
            
            return AdapterResponse.success_response(
                stdout=stdout_val,
                stderr=stderr_val
            ).to_dict()

        except Exception as e:
            traceback.print_exc(file=stderr_capture)
            return AdapterResponse.error_response(
                error=str(e),
                error_type=AdapterErrorType.EXECUTION_ERROR,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            ).to_dict()
