import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pathlib import Path
import pandas as pd
import numpy as np

class GenshinDataset(Dataset):
    """Dataset for Genshin Impact data."""
    
    def __init__(self, data_dir: Path, dataset_type: str):
        """Initialize dataset from parquet file."""
        parquet_file = data_dir / "processed" / f"{dataset_type}.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(f"No processed data found for {dataset_type}")
            
        self.data = pd.read_parquet(parquet_file)
        if self.data.empty:
            raise ValueError(f"Empty dataset for {dataset_type}")
            
        # Select only numeric columns
        numeric_data = self.data.select_dtypes(include=['int64', 'float64'])
        if numeric_data.empty:
            raise ValueError(f"No numeric features found in {dataset_type}")
            
        # Convert to numpy first, then to tensor
        numpy_data = numeric_data.values
        
        # Normalize the data
        mean = numpy_data.mean(axis=0)
        std = numpy_data.std(axis=0) + 1e-8
        normalized_data = (numpy_data - mean) / std
        
        # Convert to tensor
        self.features = torch.FloatTensor(normalized_data)
        self.feature_dim = self.features.shape[1]
        
        print(f"Dataset {dataset_type}: {len(self.features)} samples, {self.feature_dim} features")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class GenshinAutoencoder(nn.Module):
    """Autoencoder for Genshin Impact data."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        # Ensure minimum dimensions
        if input_dim < 1:
            raise ValueError("Input dimension must be at least 1")
            
        # Calculate hidden dimensions
        hidden_dim = max(input_dim // 2, 2)  # At least 2 dimensions
        latent_dim = max(hidden_dim // 2, 1)  # At least 1 dimension
        
        print(f"Model dimensions: input={input_dim}, hidden={hidden_dim}, latent={latent_dim}")
        
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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

class GenshinAITrainer:
    """Trainer for Genshin Impact AI models."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        print(f"Using device: {self.device}")

    def train_models(self, epochs: int = 100, batch_size: int = 32):
        """Train models for each dataset."""
        for dataset_name in ['characters', 'artifacts', 'weapons']:
            try:
                print(f"\nTraining model for {dataset_name}...")
                
                # Prepare data
                dataset = GenshinDataset(self.data_dir, dataset_name)
                if dataset.feature_dim == 0:
                    print(f"Skipping {dataset_name}: No features available")
                    continue
                    
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                # Initialize model
                model = GenshinAutoencoder(dataset.feature_dim).to(self.device)
                
                # Training
                optimizer = optim.Adam(model.parameters())
                criterion = nn.MSELoss()
                
                for epoch in range(epochs):
                    total_loss = 0
                    for batch in dataloader:
                        batch = batch.to(self.device)
                        
                        # Forward pass
                        output = model(batch)
                        loss = criterion(output, batch)
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                    
                    if (epoch + 1) % 10 == 0:
                        avg_loss = total_loss/len(dataloader)
                        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                
                self.models[dataset_name] = model
                
                # Save model
                model_path = self.models_dir / f"{dataset_name}_autoencoder.pt"
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
                
            except Exception as e:
                print(f"Skipping {dataset_name}: {str(e)}")
                continue

    def generate_embeddings(self, dataset_name: str) -> torch.Tensor:
        """Generate embeddings for a dataset."""
        try:
            # Load data
            dataset = GenshinDataset(self.data_dir, dataset_name)
            if dataset.feature_dim == 0:
                print(f"No features available for {dataset_name}")
                return torch.tensor([])
            
            # Get or load model
            if dataset_name not in self.models:
                model_path = self.models_dir / f"{dataset_name}_autoencoder.pt"
                if not model_path.exists():
                    raise FileNotFoundError(f"No trained model found for {dataset_name}")
                    
                model = GenshinAutoencoder(dataset.feature_dim).to(self.device)
                model.load_state_dict(torch.load(model_path))
                self.models[dataset_name] = model
            
            # Generate embeddings
            model = self.models[dataset_name]
            model.eval()
            with torch.no_grad():
                embeddings = model.encode(dataset.features.to(self.device))
                print(f"Generated embeddings shape: {embeddings.shape}")
            
            return embeddings.cpu()
            
        except Exception as e:
            print(f"Error generating embeddings for {dataset_name}: {e}")
            return torch.tensor([])

import torch.nn as nn

class GenshinModel(nn.Module):
    def __init__(self, tokenizer, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.1):
        vocab_size = len(tokenizer.token2idx)
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM layers
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and layer norm
        lstm_out = self.layer_norm1(lstm_out + attn_out)
        
        # Output layers
        out = self.fc1(lstm_out)
        out = self.layer_norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out, hidden
    
    def generate(self, input_seq, max_length=100, temperature=0.7):
        """Generate text given an input sequence."""
        self.eval()
        
        with torch.no_grad():
            current_seq = input_seq
            hidden = None
            
            # Generate one token at a time
            for _ in range(max_length):
                # Get predictions
                output, hidden = self(current_seq, hidden)
                
                # Get predictions for next token
                next_token_logits = output[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_token = torch.multinomial(next_token_probs, 1)
                
                # Append to input sequence
                current_seq = torch.cat([current_seq, next_token], dim=1)
                
                # Stop if we predict the end token
                if next_token.item() == 3:  # EOS token
                    break
            
            return current_seq