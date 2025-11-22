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

import pytest
from blackbox.core.expressions import SafeExpr, ExpressionError

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
    with pytest.raises(ExpressionError):
        SafeExpr.evaluate("unknown.var == 'test'", ctx)
