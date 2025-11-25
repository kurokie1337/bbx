# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

from typing import Any, Dict, List

from blackbox.core.registry import MCPRegistry


class BBXSchemaGenerator:
    """
    Generates JSON Schema for BBX workflow files.
    Enables IntelliSense in VS Code.
    """

    @staticmethod
    def generate() -> Dict[str, Any]:
        registry = MCPRegistry()

        # Base schema structure
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Blackbox Workflow",
            "description": "Configuration for Blackbox Workflow Engine",
            "type": "object",
            "required": ["steps"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique identifier for the workflow",
                },
                "name": {"type": "string", "description": "Human-readable name"},
                "version": {"type": "string", "description": "Workflow version"},
                "description": {
                    "type": "string",
                    "description": "Description of what the workflow does",
                },
                "steps": {
                    "type": "object",
                    "description": "Dictionary of workflow steps",
                    "patternProperties": {
                        "^[a-zA-Z0-9_-]+$": {"$ref": "#/definitions/step"}
                    },
                },
            },
            "definitions": {
                "step": {
                    "type": "object",
                    "required": ["use"],
                    "properties": {
                        "use": {
                            "type": "string",
                            "description": "Adapter and method to use (e.g., 'http.get')",
                            "enum": BBXSchemaGenerator._get_available_methods(registry),
                        },
                        "args": {
                            "type": "object",
                            "description": "Arguments for the adapter method",
                        },
                        "when": {
                            "type": "string",
                            "description": "Condition to execute this step",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of step IDs this step depends on",
                        },
                        "timeout": {
                            "type": ["string", "integer"],
                            "description": "Timeout duration (e.g., '5s', 5000)",
                        },
                        "retry": {
                            "type": "integer",
                            "description": "Number of retries",
                        },
                    },
                }
            },
        }

        return schema

    @staticmethod
    def _get_available_methods(registry: MCPRegistry) -> List[str]:
        """Get all available adapter methods for autocomplete."""
        methods = []

        # This is a bit manual because adapters don't currently expose their schema.
        # In a future version, adapters should define their own schema.

        # Standard adapters
        methods.extend(
            [
                "http.get",
                "http.post",
                "http.put",
                "http.delete",
                "logger.info",
                "logger.error",
                "logger.warning",
                "logger.debug",
                "transform.merge",
                "transform.filter",
                "transform.map",
                "transform.reduce",
                "transform.extract",
                "transform.format",
                "telegram.send",
                "browser.open",
                "browser.goto",
                "browser.click",
                "browser.type",
                "browser.text",
                "browser.screenshot",
                "browser.close",
                "system.shell",
                "system.fs.delete",
                "system.fs.list",
            ]
        )

        # MCP Bridge adapters (dynamic)
        # We can add common ones here
        mcp_servers = [
            "firebase",
            "github",
            "stripe",
            "mongodb",
            "notion",
            "linear",
            "neon",
            "redis",
            "supabase",
            "prisma",
        ]

        for server in mcp_servers:
            methods.append(f"{server}.*")

        return sorted(methods)
