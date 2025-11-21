# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
BBX Base Adapter Architecture

Provides unified base classes for all BBX adapters with:
- Standard interface
- Error handling
- Logging integration
- Input validation
- Response standardization
"""

import logging
import subprocess
import json
import shutil
from typing import Dict, Any, Optional, List, Union, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    ValidationError = None


# Configure module logger
logger = logging.getLogger("bbx.adapters")


class AdapterErrorType(Enum):
    """Standard error types for adapters"""
    VALIDATION_ERROR = "validation_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT_ERROR = "timeout_error"
    NOT_FOUND_ERROR = "not_found_error"
    PERMISSION_ERROR = "permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class AdapterResponse:
    """Standardized adapter response format"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    error_type: Optional[AdapterErrorType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "success": self.success,
            "data": self.data,
        }
        if self.error:
            result["error"] = self.error
            result["error_type"] = self.error_type.value if self.error_type else None
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def success_response(cls, data: Any = None, **metadata) -> "AdapterResponse":
        """Create success response"""
        return cls(success=True, data=data, metadata=metadata)

    @classmethod
    def error_response(
        cls,
        error: str,
        error_type: AdapterErrorType = AdapterErrorType.EXECUTION_ERROR,
        **metadata
    ) -> "AdapterResponse":
        """Create error response"""
        return cls(success=False, error=error, error_type=error_type, metadata=metadata)


class MCPAdapter(ABC):
    """
    Base class for all BBX adapters (MCP-compatible).

    All adapters must inherit from this class and implement execute().
    Provides standard interface for workflow runtime integration.
    """

    def __init__(self, adapter_name: str = None):
        self.adapter_name = adapter_name or self.__class__.__name__
        self.logger = logging.getLogger(f"bbx.adapters.{self.adapter_name}")

    @abstractmethod
    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute adapter method with inputs.

        Args:
            method: Method name to execute
            inputs: Input parameters as dictionary

        Returns:
            Result data (preferably AdapterResponse)

        Raises:
            ValueError: If method not found
            Exception: On execution errors
        """
        pass

    def log_execution(self, method: str, inputs: Dict[str, Any]):
        """Log method execution"""
        self.logger.debug(f"Executing {self.adapter_name}.{method}", extra={"inputs": inputs})

    def log_success(self, method: str, result: Any):
        """Log successful execution"""
        self.logger.info(f"{self.adapter_name}.{method} completed successfully")

    def log_error(self, method: str, error: Exception):
        """Log execution error"""
        self.logger.error(
            f"{self.adapter_name}.{method} failed: {error}",
            exc_info=True
        )

    def validate_inputs(
        self,
        inputs: Dict[str, Any],
        schema: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Validate inputs using Pydantic schema (if available)

        Args:
            inputs: Input dictionary to validate
            schema: Pydantic model class for validation

        Returns:
            Validated inputs (possibly with defaults applied)

        Raises:
            ValueError: If validation fails or schema not available
        """
        if schema is None:
            # No schema, return inputs as-is
            return inputs

        if not PYDANTIC_AVAILABLE:
            self.logger.warning("Pydantic not available, skipping validation")
            return inputs

        try:
            # Validate using Pydantic
            model = schema(**inputs)
            return model.dict(exclude_none=True)
        except ValidationError as e:
            # Convert Pydantic errors to readable message
            errors = e.errors()
            error_messages = []
            for error in errors:
                field = ".".join(str(loc) for loc in error['loc'])
                msg = error['msg']
                error_messages.append(f"{field}: {msg}")

            raise ValueError(
                f"Input validation failed: {'; '.join(error_messages)}"
            )


class BaseAdapter(MCPAdapter):
    """
    Enhanced base adapter with common functionality.

    Provides:
    - Method routing
    - Standard error handling
    - Response formatting
    """

    def __init__(self, adapter_name: str = None):
        super().__init__(adapter_name)
        self._methods: Dict[str, callable] = {}

    def register_method(self, name: str, handler: callable):
        """Register a method handler"""
        self._methods[name] = handler

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute registered method"""
        self.log_execution(method, inputs)

        handler = self._methods.get(method)
        if not handler:
            error = f"Unknown method: {method}"
            self.log_error(method, ValueError(error))
            return AdapterResponse.error_response(
                error=error,
                error_type=AdapterErrorType.NOT_FOUND_ERROR
            ).to_dict()

        try:
            result = await handler(inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            return AdapterResponse.error_response(
                error=str(e),
                error_type=AdapterErrorType.EXECUTION_ERROR
            ).to_dict()


class CLIAdapter(BaseAdapter):
    """
    Base adapter for CLI tool integrations.

    Eliminates code duplication across AWS, GCP, Azure, Docker, Kubernetes, etc.
    Provides standard patterns for:
    - CLI tool detection
    - Command execution
    - Output parsing
    - Error handling
    """

    def __init__(
        self,
        adapter_name: str,
        cli_tool: str,
        version_args: List[str] = None,
        required: bool = True
    ):
        super().__init__(adapter_name)
        self.cli_tool = cli_tool
        self.version_args = version_args or ["--version"]
        self.required = required
        self.cli_available = False

        # Check if CLI tool is available
        self._check_cli_tool()

    def _check_cli_tool(self) -> bool:
        """
        Check if CLI tool is installed and available.

        Returns:
            True if available, False otherwise
        """
        if not shutil.which(self.cli_tool):
            if self.required:
                self.logger.error(
                    f"{self.cli_tool} not found in PATH. "
                    f"Install {self.cli_tool} to use {self.adapter_name}"
                )
            else:
                self.logger.warning(
                    f"{self.cli_tool} not found. Some features may not work."
                )
            self.cli_available = False
            return False

        try:
            result = subprocess.run(
                [self.cli_tool] + self.version_args,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()
                self.logger.info(f"{self.cli_tool} available: {version}")
                self.cli_available = True
                return True
            else:
                self.logger.warning(f"{self.cli_tool} check failed: {result.stderr}")
                self.cli_available = False
                return False

        except Exception as e:
            self.logger.error(f"Error checking {self.cli_tool}: {e}")
            self.cli_available = False
            return False

    def run_command(
        self,
        *args: str,
        timeout: int = 300,
        output_format: str = "json",
        check: bool = False,
        env: Dict[str, str] = None
    ) -> AdapterResponse:
        """
        Execute CLI command with standard error handling.

        Args:
            *args: Command arguments
            timeout: Timeout in seconds
            output_format: Expected output format ('json', 'text', 'yaml')
            check: Raise exception on non-zero exit code
            env: Additional environment variables

        Returns:
            AdapterResponse with command results
        """
        if not self.cli_available:
            return AdapterResponse.error_response(
                error=f"{self.cli_tool} is not available",
                error_type=AdapterErrorType.DEPENDENCY_ERROR
            )

        cmd = [self.cli_tool] + list(args)

        # Add output format flag if JSON expected
        if output_format == "json" and "--output" not in args:
            cmd.extend(["--output", "json"])

        self.logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=check
            )

            # Parse output based on format
            output_data = None
            if result.stdout:
                if output_format == "json":
                    try:
                        output_data = json.loads(result.stdout)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON output: {e}")
                        output_data = {"raw": result.stdout}
                elif output_format == "text":
                    output_data = result.stdout.strip()
                else:
                    output_data = result.stdout

            if result.returncode == 0:
                return AdapterResponse.success_response(
                    data=output_data,
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr
                )
            else:
                return AdapterResponse.error_response(
                    error=result.stderr.strip() or f"Command failed with exit code {result.returncode}",
                    error_type=AdapterErrorType.EXECUTION_ERROR,
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr
                )

        except subprocess.TimeoutExpired as e:
            return AdapterResponse.error_response(
                error=f"Command timed out after {timeout}s",
                error_type=AdapterErrorType.TIMEOUT_ERROR
            )
        except FileNotFoundError as e:
            return AdapterResponse.error_response(
                error=f"{self.cli_tool} not found: {e}",
                error_type=AdapterErrorType.DEPENDENCY_ERROR
            )
        except Exception as e:
            return AdapterResponse.error_response(
                error=str(e),
                error_type=AdapterErrorType.EXECUTION_ERROR
            )

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute CLI adapter method"""
        # For CLI adapters, execute is implemented by subclasses
        # This base implementation provides the run_command utility
        raise NotImplementedError(
            f"{self.adapter_name} must implement execute() method"
        )


# Export all base classes
__all__ = [
    "MCPAdapter",
    "BaseAdapter",
    "CLIAdapter",
    "AdapterResponse",
    "AdapterErrorType"
]
