# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0

from blackbox.core.expressions import SafeExpr

def test_simple_equality():
    ctx = {"step": {"fetch": {"status": "success"}}}
    assert SafeExpr.evaluate("step.fetch.status == 'success'", ctx)
    assert not SafeExpr.evaluate("step.fetch.status == 'error'", ctx)

def test_comparisons():
    ctx = {"price": 100}
    assert SafeExpr.evaluate("price > 50", ctx)
    assert not SafeExpr.evaluate("price < 50", ctx)
    assert SafeExpr.evaluate("price >= 100", ctx)
    assert SafeExpr.evaluate("price <= 100", ctx)

def test_logical_operators():
    ctx = {"a": True, "b": False}
    assert not SafeExpr.evaluate("a and b", ctx)
    assert SafeExpr.evaluate("a or b", ctx)
    assert SafeExpr.evaluate("not b", ctx)

def test_nested_variables():
    ctx = {"step": {"fetch": {"data": {"price": 42}}}}
    assert SafeExpr.evaluate("step.fetch.data.price == 42", ctx)

def test_type_coercion():
    ctx = {"value": "123"}
    # String comparison
    assert SafeExpr.evaluate("value == '123'", ctx)

def test_error_on_unknown_variable():
    ctx = {}
    # Unknown variables now return False rather than raising exception (graceful degradation)
    result = SafeExpr.evaluate("unknown.var == 'test'", ctx)
    assert result is False
