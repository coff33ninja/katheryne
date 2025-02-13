# Katheryne - Genshin Impact Data Analysis

A Node.js and Python project for analyzing Genshin Impact game data using AI and machine learning. The project combines API data collection with advanced ML techniques to analyze game characters, artifacts, and weapons.

> "Ad Astra Abyssosque! Welcome to the Adventurers' Guild."

## First Lines Spoken

- **Traveler**: "Excuse me, are you Katheryne?"
- **Paimon**: "Wow, so many commissions!"
- **Katheryne**: "Ad Astra Abyssosque! Welcome to the Adventurers' Guild. How may I assist you today?"

## Features

- Data collection from multiple Genshin Impact APIs
- Advanced data processing and feature extraction
- Machine learning models for pattern recognition
- Embeddings generation for similarity search
- TypeScript/Node.js API client with full type support
- Python ML pipeline with autoencoder models
- Automatic data updates and model retraining
- Interactive AI assistant for game-related queries

## Project Structure

```plaintext
Katheryne/
├── node/               # TypeScript/Node.js API client
│   ├── src/           # Source code
│   │   ├── client.ts  # API client implementation
│   │   └── types.ts   # TypeScript type definitions
│   ├── dist/          # Compiled JavaScript
│   ├── package.json   # Node.js dependencies
│   └── tsconfig.json  # TypeScript configuration
├── python/            # Python ML pipeline
│   ├── scraper/       # API data collection
│   │   └── api_client.py
│   ├── preprocessing/ # Data processing
│   │   └── data_processor.py
│   ├── models/       # ML models
│   │   └── genshin_model.py
│   ├── main.py       # Main execution script
│   └── requirements.txt # Python dependencies
├── data/              # Data directory (created at runtime)
│   ├── raw/          # Raw JSON data from APIs
│   ├── processed/    # Processed datasets
│   └── models/       # Trained ML models
└── README.md         # Project documentation
```

## Installation

### Node.js API Client

```bash
cd node
npm install
npm run build
```

### Python ML Pipeline

```bash
cd python
pip install -r requirements.txt
```

## Usage

### Node.js API Client

```typescript
import { GenshinClient } from './src/client';

// Initialize client
const client = new GenshinClient();

// Get all characters
const characters = await client.getAllCharacters();
```

### Python ML Pipeline

```bash
python python/main.py
```

This will:
- Fetch data from Genshin Impact APIs
- Process and clean the data
- Train machine learning models
- Generate embeddings for similarity search

### Training the Assistant Model

The project includes a Genshin Impact assistant model that can answer queries about characters, weapons, and game mechanics. To train the model:

1. Using the Batch Script (Windows):
   ```bash
   # Simply run the batch file
   train.bat
   ```
   
   Or customize training parameters:
   ```bash
   set EPOCHS=5
   set BATCH_SIZE=64
   set LEARNING_RATE=0.002
   train.bat
   ```

2. Direct Python Execution:
   ```bash
   python python/train_assistant.py
   ```

Training Configuration:
- Default epochs: 1 (for testing)
- Batch size: 32 (optimized)
- Learning rate: 0.002
- Model architecture: LSTM with attention
- Sequence length: 64 tokens
- Embedding dimension: 128
- Hidden dimension: 256

The training script will:
1. Check for required dependencies (PyTorch, tqdm)
2. Load and process the training data
3. Train the model with progress bars
4. Save checkpoints in the `models/` directory
5. Run test queries after training

Note: Model checkpoints and cache files are ignored by git to keep the repository clean.

## Data Sources

- [genshin.dev](https://genshin.dev) - Primary API
- [genshin.jmp.blue](https://genshin.jmp.blue) - Secondary API

## License

MIT