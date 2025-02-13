# API Reference

## Table of Contents
- [GenshinAssistant](#genshinassistant)
- [GenshinAssistantDataset](#genshinassistantdataset)
- [GenshinTrainer](#genshintrainer)
- [Utilities](#utilities)
- [Constants](#constants)

## GenshinAssistant

The main assistant class for generating responses to Genshin Impact related queries.

### Constructor

```python
def __init__(
    self,
    vocab_size: int = 10000,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
```

#### Parameters
- `vocab_size` (int, optional): Size of the vocabulary. Defaults to 10000.
- `embedding_dim` (int, optional): Dimension of embeddings. Defaults to 256.
- `hidden_dim` (int, optional): Hidden layer dimension. Defaults to 512.
- `num_layers` (int, optional): Number of transformer layers. Defaults to 4.
- `num_heads` (int, optional): Number of attention heads. Defaults to 8.
- `dropout` (float, optional): Dropout rate. Defaults to 0.1.
- `device` (str, optional): Device to run the model on. Defaults to CUDA if available.

### Methods

#### generate_response

```python
def generate_response(
    self,
    query: str,
    max_length: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
```

Generates a response for the given query.

##### Parameters
- `query` (str): Input query text
- `max_length` (int, optional): Maximum response length. Defaults to 64.
- `temperature` (float, optional): Sampling temperature. Defaults to 0.7.
- `top_p` (float, optional): Nucleus sampling parameter. Defaults to 0.9.
- `top_k` (int, optional): Top-k sampling parameter. Defaults to 50.

##### Returns
- str: Generated response text

##### Raises
- ValueError: If query is empty
- RuntimeError: If model is not initialized

#### load_state_dict

```python
def load_state_dict(
    self,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True
) -> None:
```

Loads model parameters from a state dictionary.

##### Parameters
- `state_dict` (Dict[str, torch.Tensor]): Model state dictionary
- `strict` (bool, optional): Strict loading mode. Defaults to True.

##### Raises
- RuntimeError: If state dict is incompatible

#### save_state_dict

```python
def save_state_dict(
    self,
    path: str
) -> None:
```

Saves model parameters to a file.

##### Parameters
- `path` (str): Path to save the state dictionary

##### Raises
- IOError: If saving fails

## GenshinAssistantDataset

Dataset class for training the assistant model.

### Constructor

```python
def __init__(
    self,
    data_path: str,
    max_length: int = 64,
    vocab_size: int = 10000
) -> None:
```

#### Parameters
- `data_path` (str): Path to data directory
- `max_length` (int, optional): Maximum sequence length. Defaults to 64.
- `vocab_size` (int, optional): Size of vocabulary. Defaults to 10000.

### Methods

#### load_data

```python
def load_data(self) -> None:
```

Loads training data from disk.

##### Raises
- FileNotFoundError: If data files are missing
- ValueError: If data format is invalid

#### get_item

```python
def __getitem__(
    self,
    idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
```

Gets a single training example.

##### Parameters
- `idx` (int): Index of the example

##### Returns
- Tuple[torch.Tensor, torch.Tensor]: Input and target tensors

##### Raises
- IndexError: If index is out of range

## GenshinTrainer

Trainer class for the assistant model.

### Constructor

```python
def __init__(
    self,
    model: GenshinAssistant,
    dataset: GenshinAssistantDataset,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    num_workers: int = 4
) -> None:
```

#### Parameters
- `model` (GenshinAssistant): Model to train
- `dataset` (GenshinAssistantDataset): Training dataset
- `learning_rate` (float, optional): Learning rate. Defaults to 1e-4.
- `batch_size` (int, optional): Batch size. Defaults to 32.
- `num_workers` (int, optional): Number of data loading workers. Defaults to 4.

### Methods

#### train

```python
def train(
    self,
    epochs: int,
    validation_split: float = 0.1,
    early_stopping_patience: int = 3
) -> Dict[str, List[float]]:
```

Trains the model.

##### Parameters
- `epochs` (int): Number of training epochs
- `validation_split` (float, optional): Validation data fraction. Defaults to 0.1.
- `early_stopping_patience` (int, optional): Early stopping patience. Defaults to 3.

##### Returns
- Dict[str, List[float]]: Training history

##### Raises
- RuntimeError: If training fails

#### evaluate

```python
def evaluate(
    self,
    test_dataset: Optional[GenshinAssistantDataset] = None
) -> Dict[str, float]:
```

Evaluates the model.

##### Parameters
- `test_dataset` (Optional[GenshinAssistantDataset]): Test dataset. Uses validation set if None.

##### Returns
- Dict[str, float]: Evaluation metrics

## Utilities

### Text Processing

#### tokenize

```python
def tokenize(
    text: str,
    max_length: int = 64
) -> List[str]:
```

Tokenizes input text.

##### Parameters
- `text` (str): Input text
- `max_length` (int, optional): Maximum sequence length. Defaults to 64.

##### Returns
- List[str]: List of tokens

#### detokenize

```python
def detokenize(
    tokens: List[str]
) -> str:
```

Converts tokens back to text.

##### Parameters
- `tokens` (List[str]): List of tokens

##### Returns
- str: Reconstructed text

### Model Utils

#### load_model

```python
def load_model(
    path: str,
    device: Optional[str] = None
) -> GenshinAssistant:
```

Loads a model from disk.

##### Parameters
- `path` (str): Path to model file
- `device` (Optional[str]): Device to load model to

##### Returns
- GenshinAssistant: Loaded model

##### Raises
- FileNotFoundError: If model file not found
- RuntimeError: If loading fails

#### save_model

```python
def save_model(
    model: GenshinAssistant,
    path: str
) -> None:
```

Saves a model to disk.

##### Parameters
- `model` (GenshinAssistant): Model to save
- `path` (str): Path to save to

##### Raises
- IOError: If saving fails

## Constants

### Model Constants

```python
MAX_SEQUENCE_LENGTH = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
```

### Training Constants

```python
TRAIN_VAL_SPLIT = 0.1
EARLY_STOPPING_PATIENCE = 3
MAX_EPOCHS = 100
```

### Generation Constants

```python
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
```

### Special Tokens

```python
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
```

## Error Types

### AssistantError

Base class for assistant-related errors.

### TokenizationError

Raised when text tokenization fails.

### GenerationError

Raised when response generation fails.

### ModelError

Raised when model operations fail.

## Type Hints

```python
from typing import Dict, List, Optional, Tuple, Union

Tensor = torch.Tensor
StateDict = Dict[str, Tensor]
BatchType = Tuple[Tensor, Tensor]
MetricsType = Dict[str, float]
HistoryType = Dict[str, List[float]]
```

For more detailed examples and usage patterns, see the [Examples](examples.md) document.