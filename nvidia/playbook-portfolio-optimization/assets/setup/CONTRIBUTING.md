# Contributing to NVIDIA Quantitative Portfolio Optimization developer example

Thank you for your interest in contributing to Quantitative Portfolio Optimization developer example! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Getting Help](#getting-help)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Ways to Contribute

There are many ways to contribute to Quantitative Portfolio Optimization developer example:

- **Report bugs**: If you find a bug, please open an issue with detailed information
- **Suggest enhancements**: Have an idea for a new feature? Let us know!
- **Improve documentation**: Help us make the docs clearer and more comprehensive
- **Submit code**: Fix bugs, implement features, or improve performance
- **Share examples**: Contribute notebooks, examples, or use cases

## Development Setup

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (for GPU-accelerated features)
- NVIDIA PyTorch container or equivalent CUDA environment

### Setting Up Your Development Environment

1. **Fork and Clone the Repository**

   ```bash
   git clone https://github.com/your-username/cufolio.git
   cd cufolio
   ```

2. **Install uv (if not already installed)**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   For Windows, use:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install Development Dependencies**

   ```bash
   uv sync --extra dev
   ```

   This automatically creates a virtual environment and installs the project in editable mode along with development tools like `black`, `isort`, `flake8`, and `pre-commit`.

4. **Set Up Pre-commit Hooks**

   ```bash
   uv run pre-commit install
   ```

   This will automatically run code formatting and linting checks before each commit.

### Docker Development (Recommended)

For a consistent development environment with all GPU dependencies:

```bash
docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/pytorch:25.08-py3
cd /workspace
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following specifications:

- **Line length**: 88 characters (Black default)
- **String quotes**: Use double quotes for strings
- **Import ordering**: Managed by `isort` with Black profile

### Code Formatting

All code must be formatted with:

- **Black**: For consistent code formatting
- **isort**: For import statement ordering

Run formatters before committing:

```bash
uv run black .
uv run isort .
```

Or let pre-commit hooks handle it automatically.

### Linting

We use `flake8` for linting. Run it with:

```bash
uv run flake8 src/ 
```

### Documentation

- **Docstrings**: Use Google-style docstrings for all public functions, classes, and modules
- **Type hints**: Include type hints for function parameters and return values
- **Comments**: Write clear comments explaining complex logic

Example:

```python
def optimize_portfolio(
    returns: np.ndarray,
    risk_measure: str = "cvar",
    confidence_level: float = 0.95
) -> dict:
    """
    Optimize portfolio allocation using specified risk measure.

    Args:
        returns: Historical returns data as numpy array
        risk_measure: Risk measure to use ('cvar' or 'variance')
        confidence_level: Confidence level for CVaR calculation

    Returns:
        Dictionary containing optimal weights and performance metrics

    Raises:
        ValueError: If risk_measure is not supported
    """
    # Implementation
    pass
```

## Testing

### Running Tests

We encourage comprehensive testing of all new features and bug fixes.

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_cvar_optimizer.py

# Run with coverage
uv run pytest --cov=src --cov-report=html
```

### Writing Tests

- Place test files in the appropriate `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both unit tests and integration tests
- Test edge cases and error conditions

### GPU Testing

For GPU-specific features, ensure tests can run on both CPU and GPU:

```python
import pytest
from cuml.common import has_cuda

@pytest.mark.skipif(not has_cuda(), reason="CUDA not available")
def test_gpu_optimization():
    # GPU-specific test
    pass
```

## Submitting Changes

### Branch Naming

Use descriptive branch names following this pattern:

- `feature/description` - for new features
- `bugfix/description` - for bug fixes
- `docs/description` - for documentation updates
- `refactor/description` - for code refactoring

### Commit Messages

Write clear, concise commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
Explain what changes were made and why.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
- Reference issues: "Fixes #123" or "Relates to #456"
```

### Pull Request Process

1. **Update your fork** with the latest changes from main:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks** before submitting:

   ```bash
   uv run black .
   uv run isort .
   uv run flake8 src/
   uv run pytest
   ```

3. **Create a Pull Request** with:
   - Clear title describing the change
   - Detailed description of what and why
   - Link to related issues
   - Screenshots or examples if applicable

4. **Address review feedback** promptly and professionally

5. **Ensure CI passes** - all automated checks must pass

### Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines (Black, isort, flake8)
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated (docstrings, README, etc.)
- [ ] Commit messages are clear and descriptive
- [ ] No merge conflicts with main branch
- [ ] Pre-commit hooks pass

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Exact steps to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: OS, Python version, CUDA version, GPU model
- **Code snippet**: Minimal reproducible example
- **Error messages**: Full error traceback

Use this template:

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
1. Step one
2. Step two
3. ...

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10]
- CUDA: [e.g., 12.2]
- GPU: [e.g., A100]
- NVIDIA Quantitative Portfolio Optimization developer example version: [e.g., 25.10]

**Additional Context**
Any other relevant information.
```

### Feature Requests

When suggesting features:

- Describe the problem it solves
- Explain the proposed solution
- Consider alternative approaches
- Note any breaking changes

## Getting Help

- **Issues**: Open an issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check the README and module-specific docs

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

```bash
$ git commit -s -m "Add cool feature."
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

## Developer Certificate of Origin
```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
```

```
Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
