# ML Models Documentation

Comprehensive documentation for Katheryne's machine learning models and training pipelines.

## Table of Contents
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Model Usage](#model-usage)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Model Architecture

### LSTM Assistant Model

```python
class GenshinAssistant(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256):
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

5. **Output Layer**
   - Projects to vocabulary size
   - ReLU activation
   - Dropout for regularization

## Training Process

### Data Preparation

```python
class GenshinAssistantDataset(Dataset):
    def __init__(self, data_path: Path, max_length: int = 64):
        self.max_length = max_length
        
        # Load training data
        with open(data_path / "training_data.json", "r") as f:
            self.raw_data = json.load(f)
            
        # Create vocabulary
        self.vocab = self._create_vocabulary()
        
        # Convert to tensors
        self.query_tensors = [
            self._text_to_tensor(q) for q in self.queries
        ]
        self.response_tensors = [
            self._text_to_tensor(r) for r in self.responses
        ]
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

# Training loop
trainer = GenshinAssistantTrainer(
    data_dir=data_dir,
    **config
)

trainer.train()
```

### Training Steps

1. **Data Loading**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=True,
       collate_fn=self._collate_fn
   )
   ```

2. **Optimization**
   ```python
   optimizer = torch.optim.Adam(
       model.parameters(),
       lr=learning_rate
   )
   criterion = nn.CrossEntropyLoss(
       ignore_index=dataset.vocab["<PAD>"]
   )
   ```

3. **Training Loop**
   ```python
   for epoch in range(epochs):
       for batch in dataloader:
           # Forward pass
           outputs = model(queries, responses)
           loss = criterion(outputs, targets)
           
           # Backward pass
           optimizer.zero_grad()
           loss.backward()
           torch.nn.utils.clip_grad_norm_(
               model.parameters(),
               1.0
           )
           optimizer.step()
   ```

4. **Checkpointing**
   ```python
   if avg_loss < best_loss:
       torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'loss': avg_loss
       }, 'models/assistant_best.pt')
   ```

## Model Usage

### Loading a Trained Model

```python
def load_model(checkpoint_path: str) -> GenshinAssistant:
    checkpoint = torch.load(checkpoint_path)
    model = GenshinAssistant(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
```

### Generating Responses

```python
def generate_response(
    model: GenshinAssistant,
    query: str,
    max_length: int = 64
) -> str:
    model.eval()
    with torch.no_grad():
        # Convert query to tensor
        query_tensor = tokenize(query)
        
        # Generate response
        response = model.generate(
            query_tensor,
            max_length=max_length
        )
        
        # Convert to text
        return detokenize(response)
```

### Example Usage

```python
# Load model
model = load_model('models/assistant_best.pt')

# Generate response
query = "Tell me about Hu Tao's abilities"
response = generate_response(model, query)
print(response)
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

1. **DataLoader Optimization**
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

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision

2. **Slow Training**
   - Check GPU utilization
   - Increase number of workers
   - Use pin_memory=True

3. **Poor Performance**
   - Check learning rate
   - Increase model capacity
   - Add more training data

### Debugging Tools

1. **Memory Profiling**
   ```python
   from torch.utils.profiler import profile

   with profile(activities=[
       ProfilerActivity.CPU,
       ProfilerActivity.CUDA],
       profile_memory=True) as prof:
       model(input)
   print(prof.key_averages().table())
   ```

2. **Loss Analysis**
   ```python
   import matplotlib.pyplot as plt

   plt.plot(train_losses)
   plt.plot(val_losses)
   plt.title('Model Loss')
   plt.show()
   ```

### Model Evaluation

1. **Metrics**
   ```python
   def calculate_metrics(model, test_loader):
       model.eval()
       total_loss = 0
       correct = 0
       total = 0
       
       with torch.no_grad():
           for batch in test_loader:
               outputs = model(batch)
               loss = criterion(outputs, targets)
               total_loss += loss.item()
               
               # Calculate accuracy
               pred = outputs.argmax(dim=1)
               correct += (pred == targets).sum().item()
               total += targets.size(0)
               
       return {
           'loss': total_loss / len(test_loader),
           'accuracy': correct / total
       }
   ```

2. **Validation**
   ```python
   def validate_model(model, val_loader):
       metrics = calculate_metrics(model, val_loader)
       print(f"Validation Loss: {metrics['loss']:.4f}")
       print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
   ```

## Best Practices

1. **Data Management**
   - Use proper train/val/test splits
   - Implement data augmentation
   - Handle class imbalance

2. **Training**
   - Start with small epochs
   - Use learning rate scheduling
   - Implement early stopping

3. **Model Deployment**
   - Export to ONNX format
   - Quantize if needed
   - Monitor inference time

4. **Maintenance**
   - Regular model updates
   - Performance monitoring
   - Data quality checks