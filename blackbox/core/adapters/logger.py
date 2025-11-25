# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Logger MCP Adapter for Blackbox
Provides simple logging capabilities
"""

import logging
from typing import Any, Dict

from blackbox.core.base_adapter import MCPAdapter


class LoggerAdapter(MCPAdapter):
    """Simple logging adapter"""

    def __init__(self):
        super().__init__("logger")
        self.logger = logging.getLogger("blackbox.workflow")
        self.logger.setLevel(logging.INFO)

        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute logger method"""
        message = inputs.get("message", "")

        if method == "info":
            self.logger.info(message)
            return {"status": "logged", "level": "info", "message": message}

        elif method == "error":
            self.logger.error(message)
            return {"status": "logged", "level": "error", "message": message}

        elif method == "warning":
            self.logger.warning(message)
            return {"status": "logged", "level": "warning", "message": message}

        elif method == "debug":
            self.logger.debug(message)
            return {"status": "logged", "level": "debug", "message": message}

        else:
            raise ValueError(f"Unknown logger method: {method}")
