"""
Safe expression evaluator for BBX workflows.
Replaces unsafe eval() with a controlled expression parser.

Supports:
- Variable access: step.fetch.status
- Comparisons: ==, !=, >, <, >=, <=
- Logical operators: and, or, not
- String/number literals
"""

from typing import Any, Dict

class ExpressionError(Exception):
    """Raised when expression evaluation fails"""
    pass

class SafeExpr:
    """Safe expression evaluator for BBX conditions"""
    
    # Supported operators (ordered by length - longest first for correct parsing)
    COMPARISON_OPS = [
        ('>=', lambda a, b: a >= b),
        ('<=', lambda a, b: a <= b),
        ('==', lambda a, b: a == b),
        ('!=', lambda a, b: a != b),
        ('>', lambda a, b: a > b),
        ('<', lambda a, b: a < b),
    ]
    
    @staticmethod
    def evaluate(expr: str, context: Dict[str, Any]) -> bool:
        """
        Safely evaluate a boolean expression.
        
        Args:
            expr: Expression string (e.g., "step.fetch.status == 'success'")
            context: Variable context dictionary
            
        Returns:
            Boolean result of evaluation
            
        Raises:
            ExpressionError: If expression is invalid or evaluation fails
        """
        expr = expr.strip()
        
        # Handle logical operators
        if ' and ' in expr:
            parts = expr.split(' and ')
            return all(SafeExpr.evaluate(p.strip(), context) for p in parts)
        
        if ' or ' in expr:
            parts = expr.split(' or ')
            return any(SafeExpr.evaluate(p.strip(), context) for p in parts)
        
        if expr.startswith('not '):
            return not SafeExpr.evaluate(expr[4:].strip(), context)
        
        # Handle comparison operations (check longest operators first)
        for op, op_func in SafeExpr.COMPARISON_OPS:
            if op in expr:
                parts = expr.split(op, 1)
                if len(parts) == 2:
                    lhs = SafeExpr._resolve_value(parts[0].strip(), context)
                    rhs = SafeExpr._resolve_value(parts[1].strip(), context)
                    return op_func(lhs, rhs)
        
        # If no operator, treat as boolean value
        return SafeExpr._resolve_value(expr, context)
    
    @staticmethod
    def _resolve_value(value: str, context: Dict[str, Any]) -> Any:
        """
        Resolve a value from string to actual value.
        
        Handles:
        - String literals: 'value' or "value"
        - Number literals: 123, 45.67
        - Boolean literals: true, false
        - Variable access: step.fetch.data
        """
        value = value.strip()
        
        # String literal
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]
        
        # Boolean literal
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
        
        # Number literal
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Variable access (e.g., step.fetch.status)
        if '.' in value:
            parts = value.split('.')
            current = context
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                    if current is None:
                        raise ExpressionError(f"Variable '{value}' not found in context")
                else:
                    raise ExpressionError(f"Cannot access '{part}' on non-dict value")
            return current
        
        # Simple variable
        if value in context:
            return context[value]
        
        raise ExpressionError(f"Unknown value: '{value}'")
