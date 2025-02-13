# ML Models Documentation

Comprehensive documentation for Katheryne's machine learning models and training pipelines.

## Table of Contents
- [Assistant Model Architecture](#assistant-model-architecture)
- [Training Process](#training-process)
- [Autoencoder Model](#autoencoder-model)
- [Model Usage](#model-usage)
- [Performance Optimization](#performance-optimization)

## Assistant Model Architecture

The main assistant model is implemented in `genshin_assistant.py` and consists of an encoder-decoder architecture with attention mechanism.

### GenshinAssistant Model

```python
class GenshinAssistant(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Decoder with attention
        self.decoder = nn.LSTM(
            embedding_dim + hidden_dim * 2,
            hidden_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size)
        )
```

### Model Components

1. **Embedding Layer**
   - Converts token IDs to dense vectors
   - Dimension: vocab_size × embedding_dim
   - Dropout rate: 0.1

2. **Encoder**
   - Bidirectional LSTM
   - 2 layers with dropout
   - Processes input sequences

3. **Attention Mechanism**
   - Calculates attention weights
   - Helps focus on relevant parts of input
   - Uses tanh activation

4. **Decoder**
   - LSTM with attention context
   - 2 layers with dropout
   - Generates response tokens

## Training Process

### Data Preparation

```python
class GenshinAssistantDataset(Dataset):
    def __init__(self, data_path: Path, max_length: int = 64):
        """Initialize dataset from training data JSON."""
        self.max_length = max_length
        
        # Load training data
        with open(data_path / "training_data" / "training_data.json", "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
            
        # Create vocabulary
        self.vocab = self._create_vocabulary()
        
        # Convert to tensors
        self.query_tensors = [self._text_to_tensor(q) for q in self.queries]
        self.response_tensors = [self._text_to_tensor(r) for r in self.responses]
```

### Training Configuration

```python
# Default hyperparameters
config = {
    'epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.002,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'max_length': 64,
    'dropout': 0.1
}

trainer = GenshinAssistantTrainer(
    data_dir=data_dir,
    **config
)

trainer.train()
```

## Autoencoder Model

Katheryne includes an autoencoder model (implemented in `genshin_model.py`) for generating embeddings from processed Genshin Impact data.

### GenshinAutoencoder Architecture

```python
class GenshinAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Calculate dimensions
        hidden_dim = max(input_dim // 2, 2)  # At least 2 dimensions
        latent_dim = max(hidden_dim // 2, 1)  # At least 1 dimension
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Tanh for normalized data
        )
```

### Training with GenshinAITrainer

The `GenshinAITrainer` class manages training of autoencoders for different datasets:

```python
trainer = GenshinAITrainer(data_dir="/path/to/data")

# Train models for characters, artifacts, and weapons
trainer.train_models(epochs=100, batch_size=32)

# Generate embeddings for a specific dataset
embeddings = trainer.generate_embeddings("characters")
```

Key features:
- Loads processed data from parquet files
- Normalizes numeric features automatically
- Trains separate models for characters, artifacts, and weapons
- Saves trained models in the `data/models/` directory

## Model Usage

### Loading and Using the Assistant

```python
def load_assistant(checkpoint_path: str) -> GenshinAssistant:
    checkpoint = torch.load(checkpoint_path)
    model = GenshinAssistant(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Generate response
response = assistant.generate_response(
    query="Tell me about Hu Tao's abilities",
    max_length=64
)
```

### Using the Autoencoder

```python
# Load trained autoencoder
model_path = "data/models/characters_autoencoder.pt"
model = GenshinAutoencoder(input_dim=feature_dim)
model.load_state_dict(torch.load(model_path))

# Generate embeddings
with torch.no_grad():
    embeddings = model.encode(features)
```

## Performance Optimization

### Memory Optimization

1. **Gradient Accumulation**
   ```python
   accumulation_steps = 4
   optimizer.zero_grad()
   
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

2. **Mixed Precision Training**
   ```python
   scaler = torch.cuda.amp.GradScaler()
   
   with torch.cuda.amp.autocast():
       output = model(input)
       loss = criterion(output, target)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

### Speed Optimization

1. **DataLoader Configuration**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       num_workers=4,
       pin_memory=True
   )
   ```

2. **GPU Utilization**
   ```python
   if torch.cuda.is_available():
       model = model.cuda()
       torch.backends.cudnn.benchmark = True
   ```

### Best Practices

1. **Data Management**
   - Use proper train/val/test splits
   - Implement data augmentation
   - Handle class imbalance

2. **Training**
   - Start with small epochs
   - Use learning rate scheduling
   - Implement early stopping

3. **Model Deployment**
   - Export to ONNX format if needed
   - Quantize for production
   - Monitor inference time