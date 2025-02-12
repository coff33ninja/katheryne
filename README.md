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

## Data Sources

- [genshin.dev](https://genshin.dev) - Primary API
- [genshin.jmp.blue](https://genshin.jmp.blue) - Secondary API

## License

MIT
