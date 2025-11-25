# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Base Adapter Architecture

Provides unified base classes for all BBX adapters with:
- Standard interface
- Error handling
- Logging integration
- Input validation
- Response standardization
"""

import json
import logging
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

try:
    from pydantic import BaseModel, ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = type(object)  # type: ignore
    ValidationError = type(Exception)  # type: ignore


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
        **metadata,
    ) -> "AdapterResponse":
        """Create error response"""
        return cls(success=False, error=error, error_type=error_type, metadata=metadata)


class MCPAdapter(ABC):
    """
    Base class for all BBX adapters (MCP-compatible).

    All adapters must inherit from this class and implement execute().
    Provides standard interface for workflow runtime integration.
    """

    def __init__(self, adapter_name: Optional[str] = None):
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

    def log_execution(self, method: str, inputs: Dict[str, Any]):
        """Log method execution"""
        self.logger.debug(
            f"Executing {self.adapter_name}.{method}", extra={"inputs": inputs}
        )

    def log_success(self, method: str, result: Any):
        """Log successful execution"""
        self.logger.info(f"{self.adapter_name}.{method} completed successfully")

    def log_error(self, method: str, error: Exception):
        """Log execution error"""
        self.logger.error(
            f"{self.adapter_name}.{method} failed: {error}", exc_info=True
        )

    def validate_inputs(
        self, inputs: Dict[str, Any], schema: Optional[Type[BaseModel]] = None
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
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                error_messages.append(f"{field}: {msg}")

            raise ValueError(f"Input validation failed: {'; '.join(error_messages)}")


class BaseAdapter(MCPAdapter):
    """
    Enhanced base adapter with common functionality.

    Provides:
    - Method routing
    - Standard error handling
    - Response formatting
    """

    def __init__(self, adapter_name: Optional[str] = None):
        super().__init__(adapter_name)
        self._methods: Dict[str, Callable] = {}

    def register_method(self, name: str, handler: Callable):
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
                error=error, error_type=AdapterErrorType.NOT_FOUND_ERROR
            ).to_dict()

        try:
            result = await handler(inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            return AdapterResponse.error_response(
                error=str(e), error_type=AdapterErrorType.EXECUTION_ERROR
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
        version_args: Optional[List[str]] = None,
        required: bool = True,
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
                timeout=10,
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
        env: Optional[Dict[str, str]] = None,
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
                error_type=AdapterErrorType.DEPENDENCY_ERROR,
            )

        cmd = [self.cli_tool] + list(args)

        # Add output format flag if JSON expected
        # Note: Docker CLI doesn't support --output for all commands
        if output_format == "json" and "--output" not in args:
            # Skip --output for docker commands that don't support it
            if self.cli_tool == "docker" and args and args[0] in ["pull", "push", "tag", "login", "logout"]:
                pass  # These commands don't support --output
            else:
                cmd.extend(["--output", "json"])

        self.logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=check,
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
                    stderr=result.stderr,
                )
            else:
                return AdapterResponse.error_response(
                    error=result.stderr.strip()
                    or f"Command failed with exit code {result.returncode}",
                    error_type=AdapterErrorType.EXECUTION_ERROR,
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

        except subprocess.TimeoutExpired:
            return AdapterResponse.error_response(
                error=f"Command timed out after {timeout}s",
                error_type=AdapterErrorType.TIMEOUT_ERROR,
            )
        except FileNotFoundError as e:
            return AdapterResponse.error_response(
                error=f"{self.cli_tool} not found: {e}",
                error_type=AdapterErrorType.DEPENDENCY_ERROR,
            )
        except Exception as e:
            return AdapterResponse.error_response(
                error=str(e), error_type=AdapterErrorType.EXECUTION_ERROR
            )

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute CLI adapter method"""
        # For CLI adapters, execute is implemented by subclasses
        # This base implementation provides the run_command utility
        raise NotImplementedError(
            f"{self.adapter_name} must implement execute() method"
        )


class DockerizedAdapter(MCPAdapter):
    """
    Base adapter for running commands inside Docker containers.

    Provides Docker execution capabilities for adapters that need
    containerized environments (Universal Adapter, Terraform, GCP, etc.)
    """

    def __init__(
        self,
        adapter_name: str,
        docker_image: Optional[str] = None,
        cli_tool: Optional[str] = None,  # For compatibility with CLIAdapter calls
        required: bool = True,  # For compatibility with CLIAdapter calls
    ):
        super().__init__(adapter_name)
        self.docker_image = docker_image
        self.cli_tool = cli_tool  # Store but may not use
        self.required = required  # Store but may not use

    def _image_exists(self, image: str) -> bool:
        """Check if Docker image exists locally"""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _pull_image(self, image: str) -> bool:
        """Pull Docker image if not exists"""
        if self._image_exists(image):
            return True

        self.logger.info(f"Pulling Docker image {image}... (this may take a while)")
        try:
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                timeout=600
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Failed to pull image {image}: {e}")
            return False

    def run_command(
        self,
        *args: str,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        volumes: Optional[Dict[str, str]] = None,
        output_format: str = "text",
    ) -> AdapterResponse:
        """
        Run command inside Docker container.

        Args:
            *args: Command and arguments to run
            working_dir: Working directory inside container
            env: Environment variables
            volumes: Volume mounts (host_path: container_path)
            output_format: Output format

        Returns:
            AdapterResponse with command results
        """
        if not self.docker_image:
            return AdapterResponse.error_response(
                error="No Docker image specified",
                error_type=AdapterErrorType.CONFIGURATION_ERROR,
            )

        # Ensure image is available
        if not self._image_exists(self.docker_image):
            if not self._pull_image(self.docker_image):
                return AdapterResponse.error_response(
                    error=f"Failed to pull Docker image {self.docker_image}",
                    error_type=AdapterErrorType.DEPENDENCY_ERROR,
                )

        # Build docker run command
        docker_cmd = ["docker", "run", "--rm"]

        # Add volume mounts
        if volumes:
            for host_path, container_path in volumes.items():
                docker_cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Add environment variables
        if env:
            for key, value in env.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

        # Set working directory
        if working_dir:
            docker_cmd.extend(["-w", working_dir])

        # Add user mapping (Linux only)
        if platform.system() != "Windows":
            try:
                import os

                if hasattr(os, "getuid") and hasattr(os, "getgid"):
                    docker_cmd.extend(["-u", f"{os.getuid()}:{os.getgid()}"])  # type: ignore
            except Exception:
                pass

        # Add image and command
        docker_cmd.append(self.docker_image)
        docker_cmd.extend(args)

        # Execute
        try:
            result = subprocess.run(
                docker_cmd, capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                return AdapterResponse.success_response(
                    data={
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": 0,
                    }
                )
            else:
                return AdapterResponse.error_response(
                    error=f"Command failed: {result.stderr}",
                    error_type=AdapterErrorType.EXECUTION_ERROR,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                )
        except subprocess.TimeoutExpired:
            return AdapterResponse.error_response(
                error="Command timed out", error_type=AdapterErrorType.TIMEOUT_ERROR
            )
        except Exception as e:
            return AdapterResponse.error_response(
                error=str(e), error_type=AdapterErrorType.EXECUTION_ERROR
            )

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute method - must be implemented by subclasses"""
        raise NotImplementedError(
            f"{self.adapter_name} must implement execute() method"
        )


# Export all base classes
__all__ = [
    "MCPAdapter",
    "BaseAdapter",
    "CLIAdapter",
    "DockerizedAdapter",
    "AdapterResponse",
    "AdapterErrorType",
]
