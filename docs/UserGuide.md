# User Guide

This guide covers the basic usage of Katheryne, including the API client, ML pipeline, and AI assistant.

## Table of Contents
- [Using the API Client](#using-the-api-client)
- [Training the AI Assistant](#training-the-ai-assistant)
- [Data Processing](#data-processing)
- [Using the AI Assistant](#using-the-ai-assistant)
- [Common Use Cases](#common-use-cases)

## Using the API Client

### Basic Usage

```typescript
import { GenshinClient } from './src/client';

// Initialize client
const client = new GenshinClient();

// Get character data
const characters = await client.getAllCharacters();
const hutao = await client.getCharacter('hutao');

// Get weapon data
const weapons = await client.getAllWeapons();
const staffOfHoma = await client.getWeapon('staff-of-homa');
```

### Error Handling

```typescript
try {
  const character = await client.getCharacter('nonexistent');
} catch (error) {
  if (error.response?.status === 404) {
    console.error('Character not found');
  } else {
    console.error('API error:', error.message);
  }
}
```

## Training the AI Assistant

Katheryne provides two ways to initiate training for the AI assistant.

### Using the Batch Script (Windows)

1. Basic Training:
   ```bash
   train.bat
   ```

2. Custom Training Parameters:
   ```bash
   set EPOCHS=5
   set BATCH_SIZE=64
   set LEARNING_RATE=0.002
   train.bat
   ```

### Direct Python Execution

Run the training script directly:
```bash
python python/train_assistant.py
```

During training, you will see:
- Training progress with loss metrics
- Model checkpoints being saved
- Early stopping information when needed

The trained models will be saved in the `data/models/` directory.

## Data Processing

Before training, you need to generate the training data from raw game information:

```bash
python python/generate_training_data.py
```

This command creates:
- `training_data/training_data.json` – Training samples
- `training_data/dataset_summary.json` – Dataset overview

For character, artifact, and weapon embeddings, use:
```bash
python python/check_and_train.py
```

This processes game data and trains autoencoders for each data type.

## Using the AI Assistant

### Loading and Using a Trained Model

```python
from pathlib import Path
from python.models.genshin_assistant import GenshinAssistant, GenshinAssistantDataset
import torch

# Initialize dataset to get vocabulary
dataset = GenshinAssistantDataset(data_path=Path("data"))
vocab_size = dataset.vocab_size

# Initialize the assistant
assistant = GenshinAssistant(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=256
)

# Load trained model
checkpoint = torch.load("data/models/assistant_best.pt")
assistant.load_state_dict(checkpoint['model_state_dict'])
assistant.eval()

# Generate a response
response = assistant.generate_response(
    query="Tell me about Ganyu's abilities",
    max_length=64
)
print("AI Assistant Response:", response)
```

### Interactive Mode

For interactive usage, run:
```bash
python python/main.py
```

This starts a chat-like interface where you can directly interact with the assistant.

### Using the Autoencoder Models

To generate embeddings for game data:

```python
from python.models.genshin_model import GenshinAutoencoder
import torch

# Load trained autoencoder
model_path = "data/models/characters_autoencoder.pt"
model = GenshinAutoencoder(input_dim=feature_dim)
model.load_state_dict(torch.load(model_path))

# Generate embeddings
with torch.no_grad():
    embeddings = model.encode(features)
```

## Common Use Cases

### Character Information
- "What are Hu Tao's best artifacts?"
- "Tell me about Ganyu's abilities"
- "How should I build Raiden Shogun?"

### Team Building
- "Suggest a team for Abyss Floor 12"
- "What team works well with Ayaka?"
- "Best team comp for Hu Tao?"

### Equipment Advice
- "Is Staff of Homa good for Xiao?"
- "Best 4-star weapons for Eula?"
- "Which artifacts for support Zhongli?"

### Meta Analysis
- "Current meta teams for Spiral Abyss?"
- "Is Kazuha worth pulling?"
- "Compare Ayaka and Ganyu as DPS"

The assistant can handle both specific queries about game mechanics and broader strategic questions about team composition and meta analysis.