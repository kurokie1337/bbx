import pytest
from blackbox.core.expressions import SafeExpr, ExpressionError

def test_simple_equality():
    ctx = {"step": {"fetch": {"status": "success"}}}
    assert SafeExpr.evaluate("step.fetch.status == 'success'", ctx) == True
    assert SafeExpr.evaluate("step.fetch.status == 'error'", ctx) == False

def test_comparisons():
    ctx = {"price": 100}
    assert SafeExpr.evaluate("price > 50", ctx) == True
    assert SafeExpr.evaluate("price < 50", ctx) == False
    assert SafeExpr.evaluate("price >= 100", ctx) == True
    assert SafeExpr.evaluate("price <= 100", ctx) == True

def test_logical_operators():
    ctx = {"a": True, "b": False}
    assert SafeExpr.evaluate("a and b", ctx) == False
    assert SafeExpr.evaluate("a or b", ctx) == True
    assert SafeExpr.evaluate("not b", ctx) == True

def test_nested_variables():
    ctx = {"step": {"fetch": {"data": {"price": 42}}}}
    assert SafeExpr.evaluate("step.fetch.data.price == 42", ctx) == True

def test_type_coercion():
    ctx = {"value": "123"}
    # String comparison
    assert SafeExpr.evaluate("value == '123'", ctx) == True

def test_error_on_unknown_variable():
    ctx = {}
    with pytest.raises(ExpressionError):
        SafeExpr.evaluate("unknown.var == 'test'", ctx)
