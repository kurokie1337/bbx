# Contributing to Blackbox Workflow Engine

First off, thank you for considering contributing to BBX! It's people like you that make BBX a great tool for the community.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and collaborative environment.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** - Summarize the issue
- **Steps to reproduce** - Detailed steps to reproduce the behavior
- **Expected behavior** - What you expected to happen
- **Actual behavior** - What actually happened
- **Environment** - OS, Python version, BBX version
- **Workflow file** - Minimal `.bbx` file that reproduces the issue

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Use case** - Why this enhancement would be useful
- **Proposed solution** - How you envision it working
- **Alternatives considered** - Other approaches you've thought about

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
3. **Ensure tests pass**: `pytest`
4. **Commit with clear messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/)
5. **Push to your fork** and submit a pull request

#### Pull Request Guidelines

- **One feature per PR** - Keep changes focused
- **Write tests** - Ensure your code is tested
- **Update docs** - Add or update documentation for your changes
- **Follow style** - Use existing code as a reference
- **Describe changes** - Write a clear PR description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/blackbox-workflow.git
cd blackbox-workflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
pytest

# Run with coverage
pytest --cov=blackbox --cov-report=html
```

## Code Style

- **Python**: Follow PEP 8
- **Imports**: Use absolute imports, group standard library, third-party, and local
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Add type hints to new code
- **Comments**: Write clear, concise comments for complex logic

### Example

```python
from typing import Dict, Any, Optional

def execute_workflow(
    workflow_path: str,
    inputs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute a BBX workflow file.

    Args:
        workflow_path: Path to the .bbx workflow file
        inputs: Optional workflow inputs

    Returns:
        Dictionary of step results

    Raises:
        WorkflowParseError: If workflow cannot be parsed
        WorkflowExecutionError: If execution fails
    """
    # Implementation
    pass
```

## Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for high coverage (>80%)
- Use pytest fixtures for common setup

```python
import pytest
from blackbox.core.runtime import run_file

def test_simple_workflow(tmp_path):
    """Test basic workflow execution"""
    workflow = tmp_path / "test.bbx"
    workflow.write_text("""
        workflow:
          id: test
          steps:
            - id: log
              mcp: bbx.logger
              method: info
              inputs:
                message: "Hello"
    """)

    result = await run_file(str(workflow))
    assert result["log"]["status"] == "success"
```

## Documentation

- Update README.md for user-facing changes
- Update docs/ for detailed documentation
- Add examples to workflows/ for new features
- Keep CHANGELOG.md updated

## Project Structure

```
blackbox-workflow/
├── blackbox/core/          # Core engine
│   ├── runtime.py          # Main execution engine
│   ├── dag.py              # DAG parallelization
│   ├── registry.py         # Adapter registry
│   ├── config.py           # Configuration
│   ├── exceptions.py       # Error handling
│   ├── schemas.py          # Input validation
│   ├── adapters/           # Built-in adapters
│   └── observability/      # Metrics, traces, logs
├── tests/                  # Test suite
├── docs/                   # Documentation
├── workflows/              # Example workflows
├── cli.py                  # CLI interface
└── api_server.py           # API server
```

## Adding a New Adapter

1. Create adapter file in `blackbox/core/adapters/`
2. Inherit from `MCPAdapter` or `CLIAdapter`
3. Implement `execute()` method
4. Add Pydantic schema to `schemas.py`
5. Register in `registry.py`
6. Write tests in `tests/`
7. Add documentation

```python
from blackbox.core.base_adapter import MCPAdapter, AdapterResponse

class MyAdapter(MCPAdapter):
    """My custom adapter"""

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute adapter method"""
        if method == "my_method":
            return await self._my_method(inputs)
        else:
            return AdapterResponse.error_response(
                error=f"Unknown method: {method}"
            ).to_dict()

    async def _my_method(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """My method implementation"""
        return AdapterResponse.success_response(
            data={"result": "success"}
        ).to_dict()
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Feel free to open an issue or discussion if you have questions!

---

**Thank you for contributing to BBX!**

*Built with ❤️ in Siberia*
