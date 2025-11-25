# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
BBX Exception Hierarchy

Comprehensive exception handling for BBX workflows.

Exception Hierarchy:
    BBXError (base)
    ├── ConfigError
    ├── ValidationError
    ├── WorkflowError
    │   ├── WorkflowParseError
    │   ├── WorkflowExecutionError
    │   └── WorkflowTimeoutError
    ├── AdapterError
    │   ├── AdapterNotFoundError
    │   ├── AdapterExecutionError
    │   ├── AdapterTimeoutError
    │   └── AdapterValidationError
    ├── ExpressionError
    ├── DAGError
    │   ├── DAGCycleError
    │   └── DAGValidationError
    └── ResourceError
        ├── NetworkError
        ├── FileSystemError
        └── PermissionError
"""

import traceback
from typing import Any, Dict, List, Optional

# ============================================================================
# Base Exceptions
# ============================================================================


class BBXError(Exception):
    """Base exception for all BBX errors"""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        result = {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }

        if self.cause:
            result["cause"] = {
                "type": self.cause.__class__.__name__,
                "message": str(self.cause),
            }

        return result

    def __str__(self):
        base = self.message
        if self.details:
            base += f" | Details: {self.details}"
        if self.cause:
            base += f" | Caused by: {self.cause}"
        return base


# ============================================================================
# Configuration Errors
# ============================================================================


class ConfigError(BBXError):
    """Configuration-related errors"""


class ConfigValidationError(ConfigError):
    """Configuration validation failed"""


class ConfigFileError(ConfigError):
    """Configuration file error"""


# ============================================================================
# Validation Errors
# ============================================================================


class ValidationError(BBXError):
    """Input validation errors"""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "field": self.field,
                "value": self.value,
                "errors": self.errors,
            }
        )
        return result


# ============================================================================
# Workflow Errors
# ============================================================================


class WorkflowError(BBXError):
    """Workflow-related errors"""

    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.workflow_id = workflow_id
        self.step_id = step_id

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "workflow_id": self.workflow_id,
                "step_id": self.step_id,
            }
        )
        return result


class WorkflowParseError(WorkflowError):
    """Failed to parse workflow file"""


class WorkflowExecutionError(WorkflowError):
    """Workflow execution failed"""


class WorkflowTimeoutError(WorkflowError):
    """Workflow execution timed out"""

    def __init__(self, message: str, timeout_ms: int, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_ms = timeout_ms

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["timeout_ms"] = self.timeout_ms
        return result


class WorkflowConditionError(WorkflowError):
    """Workflow condition evaluation failed"""


# ============================================================================
# Adapter Errors
# ============================================================================


class AdapterError(BBXError):
    """Adapter-related errors"""

    def __init__(
        self,
        message: str,
        adapter_name: Optional[str] = None,
        method: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.adapter_name = adapter_name
        self.method = method

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "adapter": self.adapter_name,
                "method": self.method,
            }
        )
        return result


class AdapterNotFoundError(AdapterError):
    """Adapter not found in registry"""


class AdapterExecutionError(AdapterError):
    """Adapter method execution failed"""


class AdapterValidationError(AdapterError):
    """Adapter input validation failed"""


class WorkflowValidationError(BBXError):
    """Workflow validation failed"""

    def __init__(self, message: str, errors: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["errors"] = self.errors
        return result


class AdapterTimeoutError(AdapterError):
    """Adapter execution timed out"""

    def __init__(self, message: str, timeout_ms: int, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout_ms = timeout_ms

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["timeout_ms"] = self.timeout_ms
        return result


class AdapterNotAvailableError(AdapterError):
    """Adapter dependencies not available"""

    def __init__(
        self, message: str, missing_dependencies: Optional[List[str]] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.missing_dependencies = missing_dependencies or []

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["missing_dependencies"] = self.missing_dependencies
        return result


# ============================================================================
# Expression Errors
# ============================================================================


class ExpressionError(BBXError):
    """Expression evaluation errors"""

    def __init__(self, message: str, expression: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.expression = expression

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["expression"] = self.expression
        return result


# ============================================================================
# DAG Errors
# ============================================================================


class DAGError(BBXError):
    """DAG-related errors"""


class DAGCycleError(DAGError):
    """Circular dependency detected in DAG"""

    def __init__(self, message: str, cycle: Optional[List[str]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.cycle = cycle or []

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["cycle"] = self.cycle
        return result


class DAGValidationError(DAGError):
    """DAG validation failed"""


# ============================================================================
# Resource Errors
# ============================================================================


class ResourceError(BBXError):
    """Resource access errors"""


class NetworkError(ResourceError):
    """Network-related errors"""

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.url = url
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "url": self.url,
                "status_code": self.status_code,
            }
        )
        return result


class FileSystemError(ResourceError):
    """File system errors"""

    def __init__(self, message: str, path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.path = path

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["path"] = self.path
        return result


class PermissionError(ResourceError):
    """Permission denied errors"""


# ============================================================================
# Error Handler
# ============================================================================


class ErrorHandler:
    """Centralized error handling and logging"""

    @staticmethod
    def handle_exception(
        error: Exception, context: Optional[Dict[str, Any]] = None, reraise: bool = True
    ) -> Dict[str, Any]:
        """
        Handle an exception with proper logging and formatting

        Args:
            error: The exception to handle
            context: Additional context information
            reraise: Whether to reraise the exception

        Returns:
            Error dictionary

        Raises:
            The original exception if reraise=True
        """
        import logging

        logger = logging.getLogger("bbx.error_handler")

        # Convert to BBXError if needed
        if isinstance(error, BBXError):
            bbx_error = error
        else:
            bbx_error = BBXError(message=str(error), cause=error, details=context or {})

        # Log error
        error_dict = bbx_error.to_dict()
        if context:
            error_dict["context"] = context

        logger.error(
            f"{error_dict['type']}: {error_dict['message']}",
            extra={"error_details": error_dict},
        )

        # Log stack trace for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Stack trace:\n{traceback.format_exc()}")

        if reraise:
            raise bbx_error

        return error_dict

    @staticmethod
    def wrap_exception(
        func,
        exception_class: type = BBXError,
        message: Optional[str] = None,
        **error_kwargs,
    ):
        """
        Decorator to wrap function exceptions

        Usage:
            @ErrorHandler.wrap_exception(AdapterError, adapter_name="my_adapter")
            def my_function():
                ...
        """

        def decorator(f):
            def wrapper(*args, **kwargs):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if isinstance(e, BBXError):
                        raise
                    error_message = message or f"Error in {f.__name__}: {e}"
                    raise exception_class(error_message, cause=e, **error_kwargs)

            return wrapper

        if callable(func):
            return decorator(func)
        return decorator


# ============================================================================
# Retry Decorator with Error Handling
# ============================================================================


def retry_on_error(
    max_retries: int = 3,
    delay_ms: int = 1000,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Retry decorator with exponential backoff

    Args:
        max_retries: Maximum number of retries
        delay_ms: Initial delay in milliseconds
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch

    Usage:
        @retry_on_error(max_retries=3, delay_ms=1000)
        async def my_function():
            ...
    """
    import asyncio
    import time
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            delay = delay_ms / 1000

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        delay *= backoff
                    else:
                        raise

            raise last_error

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            delay = delay_ms / 1000

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        raise

            raise last_error

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
