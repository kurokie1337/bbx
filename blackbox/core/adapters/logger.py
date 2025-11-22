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
