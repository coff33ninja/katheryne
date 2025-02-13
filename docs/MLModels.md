# Machine Learning Models Documentation

## Overview

Katheryne uses several ML models to provide intelligent responses and recommendations:

1. **Main Assistant Model**
   - Transformer-based architecture
   - Trained on game-specific data
   - Handles natural language queries

2. **Character Autoencoder**
   - Learns character embeddings
   - Used for similarity matching
   - Enables character comparisons

3. **Equipment Autoencoder**
   - Processes weapon and artifact data
   - Learns equipment relationships
   - Aids in build recommendations

## Model Architectures

### 1. Assistant Model

```python
class GenshinAssistant(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )
        self.output = nn.Linear(embedding_dim, vocab_size)
```

Key Components:
- Embedding layer for token representation
- Transformer encoder for sequence processing
- Linear output layer for token prediction

### 2. Character Autoencoder

```python
class GenshinAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
```

Key Components:
- Encoder network for dimension reduction
- Decoder network for reconstruction
- ReLU activation for non-linearity

## Training Process

### 1. Data Preparation

```python
def prepare_training_data():
    # Load raw data
    raw_data = load_game_data()
    
    # Process character data
    character_features = extract_character_features(raw_data)
    
    # Process equipment data
    equipment_features = extract_equipment_features(raw_data)
    
    # Create training samples
    training_samples = generate_training_samples(
        character_features,
        equipment_features
    )
    
    return training_samples
```

### 2. Training Configuration

```python
training_config = {
    # Model parameters
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    
    # Training parameters
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 0.001,
    
    # Optimization
    "optimizer": "Adam",
    "weight_decay": 1e-4,
    "gradient_clip": 1.0,
    
    # Learning rate schedule
    "lr_schedule": "cosine",
    "warmup_steps": 1000,
    
    # Early stopping
    "patience": 3,
    "min_delta": 1e-4
}
```

### 3. Training Loop

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = get_scheduler(optimizer, config)
    
    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["gradient_clip"]
            )
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = compute_loss(model, batch)
                val_loss += loss.item()
        
        # Early stopping check
        if early_stopping(val_loss):
            break
```

## Model Evaluation

### 1. Metrics

```python
def evaluate_model(model, test_loader):
    metrics = {
        "perplexity": compute_perplexity(model, test_loader),
        "bleu_score": compute_bleu(model, test_loader),
        "response_accuracy": compute_accuracy(model, test_loader)
    }
    return metrics
```

### 2. Performance Analysis

```python
def analyze_performance(model, test_cases):
    results = []
    for case in test_cases:
        prediction = model.generate_response(case["query"])
        results.append({
            "query": case["query"],
            "prediction": prediction,
            "ground_truth": case["ground_truth"],
            "metrics": compute_metrics(prediction, case["ground_truth"])
        })
    return results
```

## Model Usage

### 1. Loading Models

```python
def load_models():
    # Load assistant model
    assistant = GenshinAssistant(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256
    )
    assistant.load_state_dict(
        torch.load("models/assistant_best.pt")
    )
    
    # Load autoencoders
    char_autoencoder = GenshinAutoencoder(
        input_dim=char_feature_dim
    )
    char_autoencoder.load_state_dict(
        torch.load("models/char_autoencoder.pt")
    )
    
    return assistant, char_autoencoder
```

### 2. Inference

```python
def generate_response(
    assistant: GenshinAssistant,
    query: str,
    max_length: int = 64
):
    # Preprocess query
    tokens = preprocess_text(query)
    
    # Generate response
    with torch.no_grad():
        response = assistant.generate(
            tokens,
            max_length=max_length
        )
    
    # Postprocess response
    return postprocess_text(response)
```

## Model Maintenance

### 1. Regular Updates

```python
def update_models():
    # Check for new game data
    if new_data_available():
        # Collect new data
        new_data = collect_new_data()
        
        # Update training dataset
        update_dataset(new_data)
        
        # Retrain models
        retrain_models()
```

### 2. Performance Monitoring

```python
def monitor_performance():
    # Track metrics over time
    metrics_history.append({
        "timestamp": current_time(),
        "metrics": evaluate_model(model, test_loader)
    })
    
    # Check for degradation
    if performance_degraded():
        trigger_retraining()
```

## Best Practices

1. **Model Deployment**
   - Use version control for models
   - Implement A/B testing
   - Monitor resource usage

2. **Error Handling**
   - Implement fallback responses
   - Log prediction errors
   - Monitor out-of-distribution queries

3. **Performance Optimization**
   - Use batch processing
   - Implement caching
   - Optimize model size

## Future Improvements

1. **Model Architecture**
   - Implement attention mechanisms
   - Add knowledge distillation
   - Explore multi-task learning

2. **Training Process**
   - Implement curriculum learning
   - Add data augmentation
   - Improve regularization

3. **User Experience**
   - Add confidence scores
   - Implement response diversity
   - Add explanation generation

## Contributing

See [Contributing Guide](DeveloperGuide.md#contributing) for information on:
- Adding new models
- Improving existing models
- Testing and validation