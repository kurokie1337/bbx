# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Transform adapter for data manipulation.
Provides methods to transform, filter, map, and reduce data.
"""

import json
from typing import Any, Dict, List

from blackbox.core.base_adapter import MCPAdapter


class TransformAdapter(MCPAdapter):
    """Adapter for transforming data within workflows."""

    def __init__(self):
        super().__init__("transform")

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute transform operation.

        Args:
            method: Transform method name
            inputs: Input parameters

        Returns:
            Transformed data
        """
        if method == "merge":
            return await self._merge(inputs)
        elif method == "filter":
            return await self._filter(inputs)
        elif method == "map":
            return await self._map(inputs)
        elif method == "reduce":
            return await self._reduce(inputs)
        elif method == "extract":
            return await self._extract(inputs)
        elif method == "format":
            return await self._format(inputs)
        else:
            raise ValueError(f"Unknown transform method: {method}")

    async def _merge(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple objects into one.

        Example:
            inputs:
                objects: [{"a": 1}, {"b": 2}, {"c": 3}]
            returns: {"a": 1, "b": 2, "c": 3}
        """
        objects = inputs.get("objects", [])
        result = {}
        for obj in objects:
            if isinstance(obj, dict):
                result.update(obj)
        return result

    async def _filter(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Filter array by condition.

        Example:
            inputs:
                array: [1, 2, 3, 4, 5]
                condition: "x > 2"
            returns: [3, 4, 5]
        """
        array = inputs.get("array", [])
        condition = inputs.get("condition", "true")
        field = inputs.get("field")

        result = []
        for item in array:
            # Simple condition evaluation
            context = {"x": item}
            if field and isinstance(item, dict):
                context["x"] = item.get(field)

            # Very basic condition check (for safety, limited operators)
            if self._eval_condition(condition, context):
                result.append(item)

        return result

    async def _map(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Map array values using a field extraction.

        Example:
            inputs:
                array: [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
                field: "name"
            returns: ["John", "Jane"]
        """
        array = inputs.get("array", [])
        field = inputs.get("field")

        if not field:
            return array

        result = []
        for item in array:
            if isinstance(item, dict):
                result.append(item.get(field))
            else:
                result.append(item)

        return result

    async def _reduce(self, inputs: Dict[str, Any]) -> Any:
        """
        Reduce array to a single value.

        Example:
            inputs:
                array: [1, 2, 3, 4, 5]
                operation: "sum"
            returns: 15
        """
        array = inputs.get("array", [])
        operation = inputs.get("operation", "sum")

        if operation == "sum":
            return sum([x for x in array if isinstance(x, (int, float))])
        elif operation == "count":
            return len(array)
        elif operation == "average":
            numbers = [x for x in array if isinstance(x, (int, float))]
            return sum(numbers) / len(numbers) if numbers else 0
        elif operation == "min":
            return min(array) if array else None
        elif operation == "max":
            return max(array) if array else None
        else:
            return array

    async def _extract(self, inputs: Dict[str, Any]) -> Any:
        """
        Extract field from object.

        Example:
            inputs:
                object: {"user": {"name": "John", "age": 30}}
                path: "user.name"
            returns: "John"
        """
        obj = inputs.get("object", {})
        path = inputs.get("path", "")

        if not path:
            return obj

        parts = path.split(".")
        current = obj

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    async def _format(self, inputs: Dict[str, Any]) -> str:
        """
        Format data as string.

        Example:
            inputs:
                data: {"name": "John", "age": 30}
                format: "json"
            returns: '{"name": "John", "age": 30}'
        """
        data = inputs.get("data")
        format_type = inputs.get("format", "json")

        if format_type == "json":
            return json.dumps(data, indent=2)
        elif format_type == "string":
            return str(data)
        else:
            return str(data)

    def _eval_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate simple conditions."""
        # Replace x with actual value
        x = context.get("x")
        if x is None:
            return False

        # Very simple condition evaluation
        if ">" in condition:
            parts = condition.split(">")
            if len(parts) == 2:
                try:
                    threshold = float(parts[1].strip())
                    return float(x) > threshold
                except Exception:
                    return False
        elif "<" in condition:
            parts = condition.split("<")
            if len(parts) == 2:
                try:
                    threshold = float(parts[1].strip())
                    return float(x) < threshold
                except Exception:
                    return False
        elif "==" in condition:
            parts = condition.split("==")
            if len(parts) == 2:
                return str(x) == parts[1].strip().strip('"').strip("'")

        return condition.lower() == "true"
