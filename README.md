# Katheryne - Genshin Impact AI Assistant

A Node.js and Python project for building an AI assistant that can answer queries about Genshin Impact game data. The project combines data collection, processing, and machine learning to create an intelligent assistant for game-related questions.

> "Ad Astra Abyssosque! Welcome to the Adventurers' Guild."

## Features

- Comprehensive game data collection and processing
- Training data generation for various query types
- Machine learning models for natural language understanding
- Interactive AI assistant for game-related queries
- Support for multiple query categories:
  - Character information and builds
  - Weapon recommendations
  - Artifact set bonuses and stats
  - Domain strategies
  - Team compositions and synergies

## Project Structure

```plaintext
Katheryne/
├── node/                # TypeScript/Node.js components
│   ├── src/            # Source code
│   │   ├── client.ts   # API client implementation
│   │   └── types.ts    # TypeScript type definitions
│   ├── dist/           # Compiled JavaScript
│   ├── package.json    # Node.js dependencies
│   └── tsconfig.json   # TypeScript configuration
├── python/             # Python ML components
│   ├── data/          # Data processing scripts
│   │   └── generate_training_data.py
│   ├── models/        # ML model implementations
│   │   └── assistant.py
│   ├── training/      # Training scripts
│   │   └── train.py
│   └── requirements.txt # Python dependencies
├── data/               # Data directory
│   ├── characters/    # Character data
│   ├── weapons/       # Weapon data
│   ├── artifacts/     # Artifact data
│   ├── domains/       # Domain data
│   └── teams/         # Team composition data
├── training_data/     # Generated training data
│   ├── training_data.json    # Training samples
│   └── dataset_summary.json  # Dataset statistics
└── docs/              # Documentation
    ├── data_format.md # Data format specifications
    ├── training.md    # Training instructions
    └── api.md         # API documentation
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 14+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Katheryne.git
cd Katheryne
```

2. Install Node.js dependencies:
```bash
cd node
npm install
npm run build
cd ..
```

3. Install Python dependencies:
```bash
cd python
pip install -r requirements.txt
cd ..
```

### Testing the Setup

1. Generate training data:
```bash
python python/generate_training_data.py
```

This will create:
- `training_data/training_data.json` - Training samples
- `training_data/dataset_summary.json` - Dataset statistics

2. Verify the generated data:
```bash
python python/verify_data.py
```

The script will show:
- Number of samples per category
- Data format validation results
- Example queries and responses

## Building Your Own Model

### 1. Data Preparation

1. Customize the data generation:
   - Edit `data/*.json` files to add your own game data
   - Modify `python/generate_training_data.py` to add new query types

2. Generate training data:
```bash
python python/generate_training_data.py
```

### 2. Model Training

1. Basic training:
```bash
python python/train.py
```

2. Advanced training options:
```bash
python python/train.py \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --model-type transformer \
  --save-dir models/custom
```

Training parameters:
- `epochs`: Number of training epochs (default: 5)
- `batch-size`: Batch size (default: 32)
- `learning-rate`: Learning rate (default: 0.001)
- `model-type`: Model architecture (transformer/lstm)
- `save-dir`: Model save directory

### 3. Model Evaluation

Test your trained model:
```bash
python python/evaluate.py --model-path models/custom
```

The evaluation will show:
- Accuracy metrics
- Example predictions
- Error analysis

### 4. Using the Model

1. Start the assistant:
```bash
python python/assistant.py --model-path models/custom
```

2. Ask questions: