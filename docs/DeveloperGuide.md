# Developer Guide

This guide provides detailed information for developers who want to understand, modify, or contribute to Katheryne.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Code Style](#code-style)

## Architecture Overview

### System Components

```plaintext
┌─────────────────┐     ┌─────────────────┐
│   Node.js API   │     │   Python ML     │
│     Client      │     │    Pipeline     │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  External APIs  │     │  ML Models &    │
│  Data Sources   │     │   Processing    │
└─────────────────┘     └─────────────────┘
```

### Key Components

1. **Node.js API Client**
   - TypeScript-based API wrapper
   - Type definitions for all data structures
   - Error handling and rate limiting

2. **Python ML Pipeline**
   - Data collection and preprocessing
   - Model training and evaluation
   - Embeddings generation

3. **ML Models**
   - LSTM-based assistant
   - Autoencoder for pattern recognition
   - Attention mechanism for queries

## Development Setup

### 1. Development Dependencies

```bash
# Node.js development
npm install --save-dev typescript @types/node jest @types/jest ts-jest

# Python development
pip install pytest pytest-cov black isort mypy
```

### 2. IDE Configuration

#### VSCode Settings
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "typescript.tsdk": "node_modules/typescript/lib",
  "editor.formatOnSave": true
}
```

### 3. Git Hooks

Create `.git/hooks/pre-commit`:
```bash
#!/bin/sh
npm run lint
npm run test
python -m pytest
```

## Contributing Guidelines

### 1. Branching Strategy

- `main`: Production-ready code
- `develop`: Development branch
- Feature branches: `feature/your-feature`
- Bug fixes: `fix/bug-description`

### 2. Pull Request Process

1. Create feature branch
2. Make changes
3. Run tests
4. Update documentation
5. Create PR
6. Code review
7. Merge

### 3. Commit Messages

Follow conventional commits: