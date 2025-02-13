# User Guide

This guide covers the basic usage of Katheryne, including the API client, ML pipeline, and AI assistant.

## Table of Contents
- [Using the API Client](#using-the-api-client)
- [Training the AI Assistant](#training-the-ai-assistant)
- [Data Processing](#data-processing)
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

```bash
python python/train_assistant.py
```

### Training Progress

During training, you'll see: