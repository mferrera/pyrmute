# Contributing to Pyrmute

Thank you for your interest in contributing to Pyrmute! This document provides
guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to
follow. Please be respectful and constructive in all interactions.

**Expected behavior:**

- Be respectful and inclusive
- Welcome newcomers
- Focus on what's best for the community
- Show empathy towards others

**Unacceptable behavior:**

- Harassment or discriminatory language
- Personal attacks
- Trolling or inflammatory comments
- Publishing private information

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```sh
   git clone https://github.com/YOUR_USERNAME/pyrmute.git
   cd pyrmute
   ```
3. **Add upstream remote**:
   ```sh
   git remote add upstream https://github.com/mferrera/pyrmute.git
   ```
4. **Create a branch** for your changes:
   ```sh
   git checkout -b feature/my-feature
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip or uv for package management
- Git

### Install Development Dependencies

```sh
# Using uv (recommended)
uv sync --all-groups

# Using pip
pip install . -e --group dev --group docs
```

This installs:

- The package in editable mode
- Development tools (ruff, mypy)
- Testing tools (pytest, pytest-cov)
- Documentation tools and packages

### Verify Installation

```sh
# Run tests
pytest

# Check code style
ruff format --check src/ tests/
ruff check src/ tests/

# Check types
mypy src/ tests/
```

## Testing

### Running Tests

```sh
# Run all tests
pytest

# Run with coverage
pytest --cov=src/ --cov-report=term-missing

# Run specific test file
pytest tests/test_registry.py

# Run specific test
pytest tests/test_registry.py::test_registry_initialization

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_model_manager"
```

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Follow existing test patterns
- Fully type annotate tests
- Use a [BLUF](https://en.wikipedia.org/wiki/BLUF_(communication)) docstring

**Test structure:**

```python
def test_feature_does_something() -> None:
    """Test that feature behaves correctly in normal conditions."""
    # Arrange
    registry = Registry()

    # Act
    result = registry.some_method()

    # Assert
    assert result == expected_value


def test_feature_handles_error() -> None:
    """Test that feature handles errors gracefully."""
    # Test error conditions
    with pytest.raises(SpecificError, match="error string or substring"):
        registry.failing_method()
```

**Test coverage targets:**

- Aim for 90%+ coverage
- 100% coverage for critical paths
- Test both success and error cases
- Test edge cases and boundary conditions

### Mocking

Use `unittest.mock` for external dependencies:

```python
from unittest.mock import Mock, patch

def test_with_mocked_dependency() -> None:
    """Tests with a mocked dependency."""
    manager = ModelManager()

    with patch("some.package") as mock_method:
        mock_method.return_value = Mock(
            value=1,
        )

        # Test code that uses some.package
        result = manager.get_schema("Model", "1.0.0")

        assert result["data"] == "value"
```

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture
def model_manager() -> ModelManager:
    """Create a fresh ModelManager for testing."""
    return ModelManager()


@pytest.fixture
def sample_schema() -> dict[str, Any]:
    """Sample JSON schema for testing."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    }
```

Use existing fixtures if possible.

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters
- **Quotes**: Double quotes for strings
- **Imports**: Organized with ruff isort
- **Type hints**: Required for all functions and tests

### Tools

**Ruff** - Linting:
```sh
ruff check src/ tests/
ruff check --fix src/ tests/  # Auto-fix issues
```

**mypy** - Type checking:
```sh
mypy src/
```

## Commit Messages

### Format

```
Brief description with a max 50 characters

Detailed explanation of what changed and why with a max 71 characters.

- Bullet points for multiple changes
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with latest upstream:
   ```sh
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```sh
   pytest
   ruff format --check src/ tests/
   ruff check src/ tests/
   mypy src/ tests/
   ```

3. **Update documentation** if needed:
   - Update docstrings
   - Update README.md
   - Add examples for new features
   - Update documentation in docs/

4. **Add tests** for new functionality

5. **Update CHANGELOG** if applicable

### Submitting

1. **Push your branch** to your fork:
   ```sh
   git push origin feature/my-feature
   ```

2. **Create a Pull Request** on GitHub

3. **Fill out the PR template**:
   - Description of changes
   - Related issues
   - Type of change (feature, fix, docs, etc.)
   - Testing performed
   - Checklist completion

### Review Process

1. **Automated checks** must pass:
   - Tests
   - Linting
   - Type checking

2. **Code review** by maintainers:
   - Typically within 2-3 days
   - Address feedback in new commits
   - Don't force-push after review starts

3. **Approval and merge**:
   - At least one approval required
   - Maintainer will merge
   - Branch will be deleted

## Reporting Bugs

### Before Reporting

1. **Check existing issues** - Your bug may already be reported
2. **Update to latest version** - Bug may already be fixed
3. **Minimal reproduction** - Create smallest example that shows the bug

## Documentation

### What to Document

- **All public APIs** - Complete docstrings
- **Configuration options** - Environment variables, parameters
- **Examples** - Real-world usage patterns

### Documentation Style

- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Keep examples runnable
- Update docs with code changes

## Questions?

- **GitHub Issues**: For bugs and features
- **GitHub Discussions**: For questions and discussions

## License

By contributing to Pyrmute, you agree that your contributions will be licensed
under the MIT License.

## Recognition

Contributors are recognized in:

- GitHub contributors list
- CHANGELOG.md for significant contributions
- Release notes

Thank you for contributing to Pyrmute! 🎉
