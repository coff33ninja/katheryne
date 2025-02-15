from pathlib import Path
import os
import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from torch.utils.data import DataLoader
from python.models.decoders import HardwareManager
from python.models.tokenizer import GenshinTokenizer
from python.models.genshin_model import GenshinModel
import torch.nn.functional as F
from tqdm.auto import tqdm  # Changed from 'from tqdm import tqdm'
from tqdm.auto import tqdm  # Changed from 'from tqdm import tqdm'

class GenshinAssistantTrainer:
    def __init__(self, dataset, batch_size, learning_rate, epochs, device, config):
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        # Initialize tokenizer first
        self.tokenizer = GenshinTokenizer()
        self.vocab_size = len(self.tokenizer.token2idx)
        
        # Create model and other components
        self.dataloader = self.create_dataloader()
        self.model = self.create_model()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.token2idx[self.tokenizer.pad_token])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def generate_response(self, query: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate a response for the given query."""
        self.model.eval()
        
        try:
            # Tokenize and encode the query
            input_ids = self.tokenizer.encode(query).unsqueeze(0).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature
                )
            
            # Decode the generated tokens
            response = self.tokenizer.decode(output_ids[0])
            return response
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def create_dataloader(self):
        # Implement the method to create and return a dataloader
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def create_model(self):
        from python.models.genshin_model import GenshinModel
        return GenshinModel(tokenizer=self.tokenizer)

    def train_step(self, inputs, targets):
        """Perform a single training step."""
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        outputs, _ = self.model(inputs)
        loss = self.criterion(outputs.view(-1, self.vocab_size), targets.view(-1))
        
        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()



def load_dataset(project_root, tokenizer):
    """Create a dataset by loading data from files and injecting it dynamically."""
    from torch.utils.data import TensorDataset
    import torch
    import json
    from pathlib import Path
    
    # Load data from both locations
    data_paths = [
        Path(project_root) / 'data',
        Path(project_root) / 'models' / 'data',
        Path(project_root) / 'training_data'  # Add training_data folder
    ]
    
    conversations = []
    
    # Load data from each path
    for data_path in data_paths:
        if data_path.exists():
            # Load JSON files
            for json_file in data_path.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Handle different JSON structures
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    # Extract question-answer pairs using tokenizer methods
                                    q = item.get('question', item.get('input', ''))
                                    a = item.get('answer', item.get('response', ''))
                                    if q and a:
                                        # Use tokenizer instance methods properly
                                        q_tokens = tokenizer.tokenize(q)
                                        a_tokens = tokenizer.tokenize(a)
                                        if q_tokens and a_tokens:
                                            conversations.append((q, a))
                        
                        elif isinstance(data, dict):
                            # Handle nested structures
                            def extract_qa_pairs(d):
                                pairs = []
                                if 'question' in d and 'answer' in d:
                                    q = d['question']
                                    a = d['answer']
                                    if q and a:
                                        q_tokens = tokenizer.tokenize(q)
                                        a_tokens = tokenizer.tokenize(a)
                                        if q_tokens and a_tokens:
                                            pairs.append((q, a))
                                elif 'input' in d and 'response' in d:
                                    q = d['input']
                                    a = d['response']
                                    if q and a:
                                        q_tokens = tokenizer.tokenize(q)
                                        a_tokens = tokenizer.tokenize(a)
                                        if q_tokens and a_tokens:
                                            pairs.append((q, a))
                                
                                # Check nested structures
                                for value in d.values():
                                    if isinstance(value, dict):
                                        pairs.extend(extract_qa_pairs(value))
                                    elif isinstance(value, list):
                                        for item in value:
                                            if isinstance(item, dict):
                                                pairs.extend(extract_qa_pairs(item))
                                return pairs
                            
                            conversations.extend(extract_qa_pairs(data))
                            
                except Exception as e:
                    print(f"Error loading {json_file}: {str(e)}")
    
    # Try to load training data from training_data folder
    training_data_path = Path(project_root) / 'training_data' / 'training_data.json'
    if training_data_path.exists():
        try:
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
                if isinstance(training_data, list):
                    for item in training_data:
                        if isinstance(item, dict):
                            q = item.get('question', item.get('input', ''))
                            a = item.get('answer', item.get('response', ''))
                            if q and a:
                                q_tokens = tokenizer.tokenize(q)
                                a_tokens = tokenizer.tokenize(a)
                                if q_tokens and a_tokens:
                                    conversations.append((q, a))
        except Exception as e:
            print(f"Error loading training data: {str(e)}")
    
    print(f"\nLoaded {len(conversations)} conversations for training")
    
    if not conversations:
        print("No conversations found, using default examples for testing")
        conversations = [
            ("Tell me about Hu Tao", "Hu Tao is the 77th Director of the Wangsheng Funeral Parlor in Liyue. She is a Pyro character who wields a polearm."),
            ("What is a Vision?", "A Vision is a gem granted by the Archons to those with great ambitions. It allows its wielder to channel elemental power."),
            ("Compare Amber and Lisa", "Amber is a Pyro bow user and Mondstadt's Outrider, while Lisa is an Electro catalyst user and the Librarian of the Knights of Favonius."),
            ("Tell me about artifacts", "Artifacts are equipment items that provide stat bonuses. They come in sets of 5: Flower, Feather, Sands, Goblet, and Circlet."),
            ("What is Resin used for?", "Resin is used to claim rewards from domains, ley lines, and weekly bosses. It regenerates over time and is capped at 160."),
        ]
    
    # Prepare input and target sequences
    input_sequences = []
    target_sequences = []
    
    for question, answer in conversations:
        # Encode question and answer
        q_tokens = tokenizer.encode(question)
        a_tokens = tokenizer.encode(answer)
        
        input_sequences.append(q_tokens)
        target_sequences.append(a_tokens)
    
    # Pad sequences to the same length
    max_len = max(max(len(seq) for seq in input_sequences),
                 max(len(seq) for seq in target_sequences))
    
    def pad_sequence(seq, max_len, pad_token):
        if len(seq) < max_len:
            padding = torch.full((max_len - len(seq),), pad_token, dtype=torch.long)
            seq = torch.cat([seq, padding])
        return seq[:max_len]
    
    pad_token_id = tokenizer.token2idx[tokenizer.pad_token]
    input_tensors = torch.stack([pad_sequence(seq, max_len, pad_token_id) 
                               for seq in input_sequences])
    target_tensors = torch.stack([pad_sequence(seq, max_len, pad_token_id) 
                                for seq in target_sequences])
    
    return TensorDataset(input_tensors, target_tensors)

def main():
    # Configuration
    config = {
        'vocab_size': 2000,
        'embedding_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.0005,  # Reduced learning rate for finer tuning
        'batch_size': 32,
        'epochs': 100,  # Increased epochs
        'temperature': 0.7,
        'resin_weight': 2.0  # Added weight for Resin-related training examples
    }
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create tokenizer and dataset
    tokenizer = GenshinTokenizer()
    dataset = load_dataset(project_root, tokenizer)
    
    # Create data loader
    train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Initialize model and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GenshinTokenizer()
    model = GenshinModel(
        tokenizer=tokenizer,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # Initialize trainer
    trainer = GenshinAssistantTrainer(
        dataset=dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        epochs=config['epochs'],
        device=device,
        config=config
    )
    
    # Training loop
    print(f"\nStarting training with {config['epochs']} epochs, batch size {config['batch_size']}, learning rate {config['learning_rate']}")
    
    progress_bar = tqdm(range(config['epochs']), desc='Training Progress')
    
    for epoch in progress_bar:
        epoch_loss = 0
        batch_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(batch_progress):
            loss = trainer.train_step(inputs, targets)
            epoch_loss += loss
            
            batch_progress.set_postfix({'Loss': f'{loss:.4f}', 'Total Loss': f'{epoch_loss/(batch_idx+1):.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        progress_bar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})
        print(f"\nEpoch [{epoch+1}/{config['epochs']}], Average Loss: {avg_loss:.4f}\n")
    
    # Test the model
    print("\nTesting the trained model with simple queries:\n")
    test_queries = [
        "Tell me about Hu Tao",
        "What is a Vision?",
        "Compare Amber and Lisa",
        "Tell me about artifacts",
        "What is Resin used for?"
    ]
    
    print("Generating responses:\n")
    for query in test_queries:
        print(f"Q: {query}")
        try:
            response = trainer.generate_response(query, temperature=config['temperature'])
            print(f"A: {response}\n")
        except Exception as e:
            print(f"Error generating response: {str(e)}\n")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
