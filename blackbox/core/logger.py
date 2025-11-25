# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Production-grade logging system for BBX.

Provides structured logging with multiple output targets,
log rotation, and configurable log levels.
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class BBXLogger:
    """
    Centralized logging for BBX components.

    Features:
    - Console and file logging
    - Automatic log rotation
    - Structured log format with timestamps
    - Per-component log levels
    - Color-coded console output (if supported)
    """

    def __init__(
        self,
        name: str = "bbx",
        level: str = "INFO",
        log_dir: Optional[Path] = None,
        console_output: bool = True,
        file_output: bool = True
    ):
        self.name = name
        self.logger = logging.getLogger(name)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Set level
        self.logger.setLevel(self._parse_level(level))

        # Create formatters
        console_formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(name)s:%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(self._parse_level(level))
            self.logger.addHandler(console_handler)

        # Add file handler with rotation
        if file_output:
            if log_dir is None:
                log_dir = Path.home() / ".bbx" / "logs"

            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{name}.log"

            # Rotate after 10MB, keep 5 backup files
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets everything
            self.logger.addHandler(file_handler)

    def _parse_level(self, level: str) -> int:
        """Convert string level to logging constant"""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return levels.get(level.upper(), logging.INFO)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message"""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message"""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)

    def set_level(self, level: str):
        """Change log level dynamically"""
        self.logger.setLevel(self._parse_level(level))


# Global logger instances
_loggers = {}


def get_logger(name: str, level: Optional[str] = None) -> BBXLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name (usually component name)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        BBXLogger instance
    """
    if name not in _loggers:
        # Get level from environment or use default
        log_level = level or os.getenv("BBX_LOG_LEVEL", "INFO")

        # Check if we're in CI/test mode
        disable_file_logging = os.getenv("BBX_NO_FILE_LOGS", "false").lower() == "true"

        _loggers[name] = BBXLogger(
            name=name,
            level=log_level,
            file_output=not disable_file_logging
        )

    return _loggers[name]


# Convenience functions
def debug(message: str, component: str = "bbx"):
    """Quick debug log"""
    get_logger(component).debug(message)


def info(message: str, component: str = "bbx"):
    """Quick info log"""
    get_logger(component).info(message)


def warning(message: str, component: str = "bbx"):
    """Quick warning log"""
    get_logger(component).warning(message)


def error(message: str, component: str = "bbx", exc_info: bool = False):
    """Quick error log"""
    get_logger(component).error(message, exc_info=exc_info)


def critical(message: str, component: str = "bbx", exc_info: bool = False):
    """Quick critical log"""
    get_logger(component).critical(message, exc_info=exc_info)
