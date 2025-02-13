# Installation Guide

This guide will walk you through the process of setting up Katheryne on your system.

## Prerequisites

### Node.js Requirements
- Node.js 14.x or higher
- npm 6.x or higher

### Python Requirements
- Python 3.8 or higher
- pip (Python package manager)

### System Requirements
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- CUDA-capable GPU (optional, for faster training)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/katheryne.git
cd katheryne
```

### 2. Node.js Setup

Install and build the Node.js API client:

```bash
cd node
npm install
npm run build
```

This will:
- Install all Node.js dependencies
- Build the TypeScript code
- Generate type definitions

### 3. Python Setup

Install Python dependencies:

```bash
cd python
pip install -r requirements.txt
```

Key Python packages installed:
- PyTorch (ML framework)
- tqdm (Progress bars)
- numpy (Numerical computations)
- pandas (Data processing)

### 4. Environment Configuration

1. Create environment files:
   ```bash
   cp .env.example .env
   ```

2. Configure environment variables:
   ```env
   # Training configuration
   EPOCHS=1
   BATCH_SIZE=32
   LEARNING_RATE=0.002

   # API configuration (if needed)
   API_KEY=your_api_key
   API_URL=https://api.example.com
   ```

### 5. Verify Installation

Run the verification script:
```bash
python python/verify_setup.py
```

This will check:
- Python package installation
- CUDA availability (if applicable)
- Data directory structure
- Environment configuration

## Directory Structure

After installation, your project structure should look like this:

```plaintext
Katheryne/
├── node/               # TypeScript/Node.js API client
│   ├── src/           # Source code
│   ├── dist/          # Compiled JavaScript
│   └── package.json   # Node.js dependencies
├── python/            # Python ML pipeline
│   ├── models/        # ML model definitions
│   ├── preprocessing/ # Data processing
│   └── requirements.txt
├── data/              # Created at runtime
│   ├── raw/          # Raw API data
│   ├── processed/    # Processed datasets
│   └── models/       # Trained models
└── docs/             # Documentation
```

## Common Issues

### Node.js Issues

1. **Module not found errors**
   ```bash
   npm install
   npm cache clean --force
   ```

2. **TypeScript build errors**
   ```bash
   npm run clean
   npm run build
   ```

### Python Issues

1. **PyTorch installation**
   - Visit [PyTorch website](https://pytorch.org) for specific installation commands
   - Choose the right CUDA version if using GPU

2. **Package conflicts**
   ```bash
   pip install -r requirements.txt --no-cache-dir
   ```

## Next Steps

After installation:
1. Read the [User Guide](UserGuide.md) for basic usage
2. Try training the AI assistant with `train.bat`
3. Explore the [API Reference](APIReference.md)

## Updating

To update Katheryne:

1. Pull latest changes:
   ```bash
   git pull origin main
   ```

2. Update dependencies:
   ```bash
   # Node.js
   cd node
   npm install

   # Python
   cd ../python
   pip install -r requirements.txt
   ```

3. Rebuild (if needed):
   ```bash
   cd ../node
   npm run build
   ```

## Support

If you encounter any issues:
1. Check [Troubleshooting](Troubleshooting.md)
2. Search GitHub issues
3. Create a new issue with:
   - System information
   - Error messages
   - Steps to reproduce