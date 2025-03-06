## Katheryne Genshin Assistant: Complete Guide

Here's a comprehensive guide to set up, train, and interact with the Katheryne Genshin Assistant model:

### 1. Setting Up the Environment

First, let's set up the Python environment:

```sh
# Create and activate a virtual environment
python -m venv genshin_env

# On Windows
genshin_env\Scripts\activate

# On macOS/Linux
source genshin_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Downloading API Data

The project uses Genshin Impact data from various sources. Here's how to download it:

```sh
# Run the data downloader script
python python/data_downloader.py
```

This script will:
- Fetch character, weapon, artifact, and other game data from Genshin APIs
- Save raw data to the `data/raw/` directory
- Process and organize the data for model training

### 3. Generating Training Data

Before training, you need to generate training data:

```sh
# Generate training examples
python python/generate_training_data.py
```

This will:
- Create query-response pairs for training
- Save the training data to `training_data/training_data.json`
- Generate a summary of the dataset in `training_data/dataset_summary.json`

### 4. Training the Model

You can train the model using the provided script:

```sh
# On Windows
train.bat

# OR
python train.py

# On macOS/Linux
python train.py
```

The training process:
- Automatically detects your hardware capabilities
- Uses the heavy model on GPU systems, light model on CPU-only systems
- Saves checkpoints during training
- Saves the best model to `models/assistant_best.pt`
- Saves the vocabulary to `models/assistant_vocab.json`

Training parameters can be adjusted in `python/train_assistant.py` if needed.

### 5. Interacting with the Model

#### A. Using the Command Line

You can test the model directly from the command line:

```sh
python python/test_model.py --query "Tell me about Raiden Shogun"
```

#### B. Using the API Server

For a more robust solution, you can run the API server:

```sh
python python/api_server.py
```

This starts a FastAPI server on port 8000. You can then make requests:

```sh
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"text\":\"What are the best artifacts for Hu Tao?\"}"
```

#### C. Using the GUI (if available)

If you prefer a graphical interface:

```sh
python python/katheryne_gui.py
```

### 6. Advanced Usage

#### Checking Hardware and Training Status

To check your hardware capabilities and whether training is needed:

```sh
python python/check_and_train.py
```

#### Analyzing Characters and Building Teams

The project includes analysis tools:

```sh
# Analyze a specific character
python python/analyzer/character_analyzer.py --character "Hu Tao"

# Get team recommendations
python python/analyzer/team_builder.py --character "Hu Tao" --role "Main DPS"
```

### 7. Troubleshooting

If you encounter issues:
- **Model not found**: Ensure you've completed the training step
- **CUDA errors**: Try using the light model with `--failover=True`
- **Memory issues**: Reduce batch size in training parameters
- **API errors**: Check your internet connection for data downloading

### 8. Model Architecture Notes

The model has two configurations:

#### Heavy: Better performance, requires more computational resources
- Bidirectional LSTM encoder with 2 layers
- Attention-based decoder with 2 layers

#### Light: Runs on less powerful hardware
- Unidirectional LSTM encoder with 1 layer
- Simple decoder without attention

The system automatically selects the appropriate configuration based on your hardware.

---

This guide should help you get started with downloading data, training the model, and interacting with it through various interfaces!
