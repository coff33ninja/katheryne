# API Documentation

This document describes the API endpoints and interfaces for the Katheryne project.

## Python API

### Assistant Class

```python
from katheryne.assistant import GenshinAssistant

assistant = GenshinAssistant(model_path="models/custom")
```

#### Methods

1. Query Processing
```python
response = assistant.process_query(
    query: str,
    context: Optional[Dict] = None
) -> Dict[str, Any]
```

2. Batch Processing
```python
responses = assistant.process_batch(
    queries: List[str],
    contexts: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]
```

3. Interactive Mode
```python
assistant.start_interactive_session(
    prompt: str = "How can I help you?",
    history_size: int = 10
) -> None
```

4. Model Management
```python
assistant.load_model(path: str) -> None
assistant.save_model(path: str) -> None
assistant.update_model(new_data: Dict[str, Any]) -> None
```

### Data Manager

```python
from katheryne.data import DataManager

data_manager = DataManager()
```

#### Methods

1. Data Loading
```python
data = data_manager.load_data(
    data_type: str,  # 'characters', 'weapons', 'artifacts', etc.
    file_path: Optional[str] = None
) -> Dict[str, Any]
```

2. Data Validation
```python
is_valid = data_manager.validate_data(
    data: Dict[str, Any],
    schema_type: str
) -> bool
```

3. Data Export
```python
data_manager.export_data(
    data: Dict[str, Any],
    format: str = 'json',
    output_path: str
) -> None
```

### Training Manager

```python
from katheryne.training import TrainingManager

trainer = TrainingManager(
    model=model,
    config=training_config
)
```

#### Methods

1. Training Control
```python
trainer.train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    checkpoint_dir: str
) -> Dict[str, Any]
```

2. Evaluation
```python
metrics = trainer.evaluate(
    test_data: Dict[str, Any],
    metrics: List[str] = ['accuracy', 'f1', 'bleu']
) -> Dict[str, float]
```

## Node.js API

### Client Class

```typescript
import { GenshinClient } from 'katheryne';

const client = new GenshinClient({
  modelPath: 'models/custom',
  apiKey: 'your-api-key'
});
```

#### Methods

1. Query Processing
```typescript
interface QueryResponse {
  text: string;
  data: Record<string, any>;
  confidence: number;
  sources: string[];
}

async function processQuery(
  query: string,
  context?: Record<string, any>
): Promise<QueryResponse>
```

2. Batch Processing
```typescript
async function processBatch(
  queries: string[],
  contexts?: Record<string, any>[]
): Promise<QueryResponse[]>
```

3. Data Management
```typescript
async function loadData(
  dataType: 'characters' | 'weapons' | 'artifacts' | 'domains' | 'teams'
): Promise<Record<string, any>>

async function updateData(
  dataType: string,
  data: Record<string, any>
): Promise<void>
```

### WebSocket API

```typescript
import { GenshinWebSocket } from 'katheryne';

const ws = new GenshinWebSocket({
  url: 'ws://localhost:8080',
  apiKey: 'your-api-key'
});
```

#### Events

1. Connection Events
```typescript
ws.on('connect', () => {
  console.log('Connected to server');
});

ws.on('disconnect', () => {
  console.log('Disconnected from server');
});
```

2. Message Events
```typescript
ws.on('message', (data: QueryResponse) => {
  console.log('Received response:', data);
});

ws.on('error', (error: Error) => {
  console.error('Error:', error);
});
```

#### Methods

1. Send Query
```typescript
ws.sendQuery(query: string, context?: Record<string, any>): void
```

2. Connection Control
```typescript
ws.connect(): Promise<void>
ws.disconnect(): Promise<void>
```

## REST API

### Endpoints

1. Query Processing