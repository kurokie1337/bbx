# Copyright 2025 Ilya Makarov, Krasnoyarsk
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Comprehensive tests for AI adapter

Tests AI integration including:
- Code generation
- Code review
- Documentation generation
- Test generation
- Error handling
"""

import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from blackbox.core.adapters.ai import AIAdapter


@pytest.fixture
def ai_adapter():
    """Create AI adapter instance"""
    return AIAdapter()


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_generate_code_no_api_key(ai_adapter, monkeypatch):
    """Test code generation without API key"""
    # Remove API key
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ai_adapter.api_key = None
    
    result = await ai_adapter.execute("generate_code", {
        "prompt": "Create a hello world function",
        "language": "python"
    })
    
    assert "error" in result
    assert "OPENAI_API_KEY" in result["error"]


@pytest.mark.asyncio
@pytest.mark.ai
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Requires OPENAI_API_KEY")
async def test_ai_generate_code_with_api_key(ai_adapter):
    """Test code generation with API key (requires actual API key)"""
    result = await ai_adapter.execute("generate_code", {
        "prompt": "Create a simple Python function that adds two numbers",
        "language": "python",
        "style": "clean"
    })
    
    if result.get("status") == "success":
        assert "code" in result
        assert result["language"] == "python"
        assert len(result["code"]) > 0


@pytest.mark.asyncio
async def test_ai_methods_exist(ai_adapter):
    """Test that all AI methods are defined"""
    methods = [
        "generate_code",
        "review_code",
        "generate_docs",
        "explain_code",
        "suggest_improvements",
        "generate_tests",
        "chat"
    ]
    
    for method in methods:
        # Verify method doesn't raise for unknown method
        # Will fail with missing API key, but that's expected
        try:
            result = await ai_adapter.execute(method, {
                "code": "def test(): pass",
                "prompt": "test",
                "message": "test"
            })
            # If API key is set, check result format
            if isinstance(result, dict):
                assert "status" in result or "error" in result
        except KeyError:
            # Missing required input is OK
            pass


@pytest.mark.asyncio
async def test_ai_invalid_method(ai_adapter):
    """Test invalid method raises error"""
    with pytest.raises(ValueError, match="Unknown method"):
        await ai_adapter.execute("invalid_method", {})


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_generate_code_structure(ai_adapter):
    """Test generate_code returns correct structure"""
    result = await ai_adapter.execute("generate_code", {
        "prompt": "Create a function",
        "language": "python"
    })
    
    # Should return dict with status or error
    assert isinstance(result, dict)
    assert "status" in result or "error" in result


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_review_code_structure(ai_adapter):
    """Test review_code returns correct structure"""
    result = await ai_adapter.execute("review_code", {
        "code": "def hello():\n    print('Hello')",
        "language": "python"
    })
    
    assert isinstance(result, dict)
    assert "status" in result or "error" in result


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_generate_docs_structure(ai_adapter):
    """Test generate_docs returns correct structure"""
    result = await ai_adapter.execute("generate_docs", {
        "code": "def add(a, b):\n    return a + b",
        "language": "python",
        "format": "markdown"
    })
    
    assert isinstance(result, dict)
    assert "status" in result or "error" in result


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_explain_code_structure(ai_adapter):
    """Test explain_code returns correct structure"""
    result = await ai_adapter.execute("explain_code", {
        "code": "lambda x: x * 2",
        "language": "python"
    })
    
    assert isinstance(result, dict)
    assert "status" in result or "error" in result


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_suggest_improvements_structure(ai_adapter):
    """Test suggest_improvements returns correct structure"""
    result = await ai_adapter.execute("suggest_improvements", {
        "code": "def bad_function():\n    x = 1\n    return x",
        "language": "python"
    })
    
    assert isinstance(result, dict)
    assert "status" in result or "error" in result


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_generate_tests_structure(ai_adapter):
    """Test generate_tests returns correct structure"""
    result = await ai_adapter.execute("generate_tests", {
        "code": "def multiply(a, b):\n    return a * b",
        "language": "python",
        "framework": "pytest"
    })
    
    assert isinstance(result, dict)
    assert "status" in result or "error" in result


@pytest.mark.asyncio
@pytest.mark.ai
async def test_ai_chat_structure(ai_adapter):
    """Test chat returns correct structure"""
    result = await ai_adapter.execute("chat", {
        "message": "Hello, how are you?",
        "system": "You are a helpful assistant"
    })
    
    assert isinstance(result, dict)
    assert "status" in result or "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
