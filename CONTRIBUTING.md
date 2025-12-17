# Contributing to CoPEM Framework

Thank you for your interest in contributing to the CoPEM (Consensus-Driven Predictive Energy Management) Framework!

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/PolyUDavid/CoPEM-Framework.git
   cd CoPEM-Framework
   ```

2. **Create Virtual Environment**
   ```bash
   conda create -n copem-dev python=3.10
   conda activate copem-dev
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

## Code Style

We follow PEP 8 guidelines with the following tools:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Before submitting a pull request:

```bash
# Format code
black copem/ experiments/ scripts/

# Check linting
flake8 copem/ experiments/ scripts/

# Type checking
mypy copem/
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request on GitHub

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests liberally

## Testing

All new features must include tests:

```python
# tests/test_your_feature.py
import pytest
from copem import YourFeature

def test_your_feature():
    feature = YourFeature()
    assert feature.works() == True
```

Run tests with coverage:

```bash
pytest --cov=copem tests/
```

## Documentation

Update documentation for any new features:

- Add docstrings to all functions and classes
- Update README.md if needed
- Add examples to `docs/` folder

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the code

## Code of Conduct

Be respectful and constructive in all interactions.

---

**Date**: December 15, 2025  
**Maintained by**: [Your Names]

