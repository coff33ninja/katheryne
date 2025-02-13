# Contributing to Katheryne

First off, thank you for considering contributing to Katheryne! It's people like you that make Katheryne such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

1. Check if the bug has already been reported in the Issues section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable
   - System information

### Suggesting Enhancements

1. Check existing issues/discussions
2. Create a new issue with:
   - Clear title and description
   - Use case and benefits
   - Implementation suggestions if any
   - Examples of similar features in other projects

### Pull Requests

1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Run tests:
   ```bash
   python python/run_tests.py
   npm test  # for Node.js components
   ```
5. Update documentation
6. Commit your changes:
   ```bash
   git commit -m "feat: add your feature description"
   ```
7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
8. Create a Pull Request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Katheryne.git
cd Katheryne
```

2. Install dependencies:
```bash
# Python
pip install -r python/requirements.txt
pip install -r python/requirements-dev.txt

# Node.js
cd node
npm install
npm install --only=dev
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

## Project Structure

- `python/` - Python components
  - `data/` - Data processing
  - `models/` - ML models
  - `training/` - Training scripts
  - `tests/` - Test files

- `node/` - Node.js components
  - `src/` - Source code
  - `dist/` - Compiled code
  - `tests/` - Test files

## Coding Standards

### Python

- Follow PEP 8
- Use type hints
- Document functions and classes
- Write unit tests
- Use f-strings for string formatting

Example:
```python
def process_data(data: Dict[str, Any]) -> List[str]:
    """Process input data and return results.
    
    Args:
        data: Input dictionary with data
        
    Returns:
        List of processed strings
    """
    results = []
    for key, value in data.items():
        results.append(f"{key}: {value}")
    return results
```

### TypeScript/JavaScript

- Use TypeScript for new code
- Follow ESLint configuration
- Document using JSDoc
- Write unit tests
- Use async/await

Example:
```typescript
/**
 * Process query and return response
 * @param query Query string
 * @param context Optional context
 * @returns Query response
 */
async function processQuery(
  query: string,
  context?: Record<string, any>
): Promise<QueryResponse> {
  try {
    const response = await api.process(query, context);
    return response;
  } catch (error) {
    throw new ProcessingError(error.message);
  }
}
```

## Testing

### Python Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_data.py

# Run with coverage
python -m pytest --cov=katheryne
```

### Node.js Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test -- tests/data.test.ts

# Run with coverage
npm run test:coverage
```

## Documentation

- Use Google style docstrings for Python
- Use JSDoc for TypeScript/JavaScript
- Keep README.md updated
- Update API documentation when changing interfaces
- Include examples for new features

## Git Workflow

1. Create feature branch from `develop`
2. Make changes and commit
3. Pull latest `develop`
4. Resolve conflicts if any
5. Push changes
6. Create Pull Request to `develop`

### Commit Messages

Follow Conventional Commits:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

Example: