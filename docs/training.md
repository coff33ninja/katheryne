# Training Guide

This guide explains how to train the Katheryne AI assistant model.

## Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

Required packages:
```bash
pip install torch transformers datasets tqdm numpy
```

## Data Preparation

1. Generate training data:
```bash
python python/generate_training_data.py
```

2. Verify the data:
```bash
python python/verify_data.py
```

3. (Optional) Add custom data:
   - Edit JSON files in `data/`
   - Run data generation again

## Training Process

### 1. Basic Training

```bash
python python/train.py
```

Default parameters:
- Epochs: 5
- Batch size: 32
- Learning rate: 0.001
- Model: Transformer
- Save directory: models/default

### 2. Advanced Training

```bash
python python/train.py \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --model-type transformer \
  --save-dir models/custom \
  --warmup-steps 1000 \
  --gradient-accumulation 2 \
  --fp16
```

Parameters:
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate
- `--model-type`: Model architecture (transformer/lstm)
- `--save-dir`: Model save directory
- `--warmup-steps`: Learning rate warmup steps
- `--gradient-accumulation`: Gradient accumulation steps
- `--fp16`: Enable mixed precision training

### 3. Distributed Training

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  python/train.py \
  --distributed
```

### 4. Custom Training Loop

Create a custom training script:

```python
from katheryne.training import Trainer
from katheryne.models import GenshinAssistant
from katheryne.data import DataLoader

# Initialize model
model = GenshinAssistant(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=12
)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    optimizer='adam',
    scheduler='linear',
    max_epochs=10
)

# Train
trainer.train()
```

## Model Evaluation

### 1. Basic Evaluation

```bash
python python/evaluate.py --model-path models/custom
```

Metrics:
- Accuracy
- F1 Score
- BLEU Score
- Response Quality

### 2. Advanced Evaluation

```bash
python python/evaluate.py \
  --model-path models/custom \
  --test-file data/custom_test.json \
  --metrics accuracy f1 bleu quality \
  --output-dir eval_results
```

### 3. Interactive Testing

```bash
python python/interactive.py --model-path models/custom
```

## Hyperparameter Tuning

1. Using Optuna:
```bash
python python/tune.py \
  --n-trials 100 \
  --study-name genshin-assistant \
  --storage sqlite:///studies.db
```

2. Manual grid search:
```bash
python python/grid_search.py \
  --param-grid params.json \
  --output-dir tuning_results
```

## Model Export

1. Export to ONNX:
```bash
python python/export.py \
  --model-path models/custom \
  --format onnx \
  --output model.onnx
```

2. Export to TorchScript:
```bash
python python/export.py \
  --model-path models/custom \
  --format torchscript \
  --output model.pt
```

## Troubleshooting

Common issues and solutions:

1. Out of Memory (OOM):
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training

2. Slow Training:
   - Check GPU utilization
   - Increase batch size
   - Enable mixed precision
   - Use distributed training

3. Poor Performance:
   - Check data quality
   - Adjust learning rate
   - Increase model size
   - Add more training data

## Best Practices

1. Data Quality:
   - Validate input data
   - Balance dataset
   - Clean and preprocess text

2. Training:
   - Start with small models
   - Use learning rate scheduling
   - Monitor training metrics
   - Save checkpoints regularly

3. Evaluation:
   - Use multiple metrics
   - Test on diverse queries
   - Compare with baselines
   - Analyze error cases

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Training Tips and Tricks](https://huggingface.co/docs/transformers/training)