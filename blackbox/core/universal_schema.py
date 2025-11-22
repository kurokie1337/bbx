# Copyright 2025 Ilya Makarov, Krasnoyarsk
# Licensed under the Apache License, Version 2.0

"""
BBX Universal Adapter - JSON Schema Validation
"""

from typing import Tuple, List

UNIVERSAL_DEFINITION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BBX Universal Adapter Definition",
    "type": "object",
    "required": ["id", "uses", "cmd"],
    "properties": {
        "id": {
            "type": "string",
            "pattern": "^[a-z0-9_]+$",
            "description": "Unique identifier for this adapter"
        },
        "uses": {
            "type": "string",
            "pattern": "^(docker://)?[a-z0-9/:.-]+$",
            "description": "Docker image to use (format: docker://image:tag or image:tag)"
        },
        "auth": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["kubeconfig", "aws_credentials", "gcp_credentials", "azure_credentials"]
                }
            },
            "required": ["type"]
        },
        "cmd": {
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                {
                    "type": "string"
                }
            ],
            "description": "Command template (supports Jinja2)"
        },
        "env": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Environment variables"
        },
        "volumes": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Volume mounts (host:container)"
        },
        "workdir": {
            "type": "string",
            "description": "Working directory inside container"
        },
        "output_parser": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["json", "text", "yaml"]
                },
                "query": {
                    "type": "string",
                    "description": "JMESPath query for JSON outputs"
                }
            },
            "required": ["type"]
        },
        "resources": {
            "type": "object",
            "properties": {
                "cpu": {
                    "type": ["string", "number"],
                    "description": "CPU limit (e.g., '2' or 0.5)"
                },
                "memory": {
                    "type": "string",
                    "pattern": "^[0-9]+[kKmMgG]?$",
                    "description": "Memory limit (e.g., '512m', '2g')"
                }
            }
        },
        "steps": {
            "type": "array",
            "description": "Multi-step workflow (executes sequentially)",
            "items": {
                "type": "object",
                "required": ["name", "cmd"],
                "properties": {
                    "name": {"type": "string"},
                    "cmd": {
                        "oneOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "string"}
                        ]
                    },
                    "continue_on_error": {
                        "type": "boolean",
                        "default": False
                    }
                }
            }
        }
    }
}

def validate_definition(definition: dict) -> Tuple[bool, List[str]]:
    """
    Validate a Universal Adapter definition against the schema.
    
    Returns:
        (is_valid, errors)
    """
    try:
        import jsonschema
        jsonschema.validate(definition, UNIVERSAL_DEFINITION_SCHEMA)
        return True, []
    except ImportError:
        # Fallback to manual validation if jsonschema not installed
        return _manual_validate(definition)
    except jsonschema.ValidationError as e:
        return False, [str(e)]

def _manual_validate(definition: dict) -> Tuple[bool, List[str]]:
    """Manual validation without jsonschema library."""
    errors = []
    
    # Required fields
    if "id" not in definition:
        errors.append("Missing required field: id")
    if "uses" not in definition:
        errors.append("Missing required field: uses")
    if "cmd" not in definition and "steps" not in definition:
        errors.append("Must have either 'cmd' or 'steps'")
    
    # ID format
    if "id" in definition:
        import re
        if not re.match(r'^[a-z0-9_]+$', definition["id"]):
            errors.append("id must be lowercase alphanumeric with underscores")
    
    # Auth type
    if "auth" in definition:
        valid_auth_types = ["kubeconfig", "aws_credentials", "gcp_credentials", "azure_credentials"]
        if definition["auth"].get("type") not in valid_auth_types:
            errors.append(f"Invalid auth type. Must be one of: {valid_auth_types}")
    
    # Output parser
    if "output_parser" in definition:
        valid_types = ["json", "text", "yaml"]
        if definition["output_parser"].get("type") not in valid_types:
            errors.append(f"Invalid output_parser type. Must be one of: {valid_types}")
    
    return len(errors) == 0, errors
