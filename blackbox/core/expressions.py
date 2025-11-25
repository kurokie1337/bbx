# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

"""
Safe expression evaluator for BBX workflows.
Replaces unsafe eval() with a controlled expression parser.

Supports:
- Variable access: step.fetch.status
- Comparisons: ==, !=, >, <, >=, <=
- Logical operators: and, or, not
- Ternary operator: condition ? true_value : false_value
- Null coalescence: value ?? default
- String/number literals
- Boolean literals: true, false, $true, $false
"""

from typing import Any, Dict


class ExpressionError(Exception):
    """Raised when expression evaluation fails"""


class SafeExpr:
    """Safe expression evaluator for BBX conditions"""

    # Supported operators (ordered by length - longest first for correct parsing)
    COMPARISON_OPS = [
        (">=", lambda a, b: a >= b),
        ("<=", lambda a, b: a <= b),
        ("==", lambda a, b: a == b),
        ("!=", lambda a, b: a != b),
        (">", lambda a, b: a > b),
        ("<", lambda a, b: a < b),
    ]

    @staticmethod
    def evaluate(expr: str, context: Dict[str, Any]) -> Any:
        """
        Safely evaluate an expression.

        Args:
            expr: Expression string (e.g., "step.fetch.status == 'success'")
            context: Variable context dictionary

        Returns:
            Result of evaluation (can be bool, str, int, float, None)

        Raises:
            ExpressionError: If expression is invalid or evaluation fails
        """
        expr = expr.strip()

        # Handle ternary operator: condition ? true_value : false_value
        if " ? " in expr and " : " in expr:
            parts = expr.split(" ? ", 1)
            if len(parts) == 2:
                condition = parts[0].strip()
                rest = parts[1]
                # Find the matching : for this ?
                # Need to handle nested ternaries properly
                true_false = rest.split(" : ", 1)
                if len(true_false) == 2:
                    true_val = true_false[0].strip()
                    false_val = true_false[1].strip()
                    condition_result = SafeExpr.evaluate(condition, context)
                    if condition_result:
                        return SafeExpr._resolve_value(true_val, context)
                    else:
                        return SafeExpr._resolve_value(false_val, context)

        # Handle null coalescence: value ?? default
        if " ?? " in expr:
            parts = expr.split(" ?? ", 1)
            if len(parts) == 2:
                lhs = SafeExpr._resolve_value(parts[0].strip(), context)
                if lhs is not None:
                    return lhs
                else:
                    return SafeExpr._resolve_value(parts[1].strip(), context)

        # Handle logical operators
        if " and " in expr:
            parts = expr.split(" and ")
            return all(SafeExpr.evaluate(p.strip(), context) for p in parts)

        if " or " in expr:
            parts = expr.split(" or ")
            return any(SafeExpr.evaluate(p.strip(), context) for p in parts)

        if expr.startswith("not "):
            return not SafeExpr.evaluate(expr[4:].strip(), context)

        # Handle comparison operations (check longest operators first)
        for op, op_func in SafeExpr.COMPARISON_OPS:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    lhs = SafeExpr._resolve_value(parts[0].strip(), context)
                    rhs = SafeExpr._resolve_value(parts[1].strip(), context)
                    return op_func(lhs, rhs)

        # If no operator, treat as value
        return SafeExpr._resolve_value(expr, context)

    @staticmethod
    def _resolve_value(value: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a value from string to actual value.

        Handles:
        - String literals: 'value' or "value"
        - Number literals: 123, 45.67
        - Boolean literals: true, false, True, False
        - Null/undefined: null, none, undefined, empty string
        - Variable access: step.fetch.data
        """
        value = value.strip()

        # Empty value or special undefined markers
        if not value or value in ("", "null", "none", "undefined"):
            return None

        # String literal
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            return value[1:-1]

        # Boolean literal (case-insensitive + support for $True/$False from Python)
        lower_value = value.lower()
        if lower_value in ("true", "$true"):
            return True
        if lower_value in ("false", "$false"):
            return False

        # Number literal
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # Variable access (e.g., step.fetch.status)
        if "." in value:
            parts = value.split(".")
            current: Any = context
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                    if current is None:
                        # Return None instead of raising error for graceful degradation
                        return None
                else:
                    # Return None instead of raising error
                    return None
            return current

        # Simple variable
        if value in context:
            return context[value]

        # If starts with $, it might be unresolved variable - return None gracefully
        if value.startswith("$"):
            return None

        # Undefined variable - return None for graceful degradation (used in ?? operator)
        # This allows expressions like: undefined_var ?? 'default'
        return None
