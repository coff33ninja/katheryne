import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


class GenshinAssistantDataset(Dataset):
    """Dataset for training the Genshin Impact assistant."""

    def __init__(self, data_path: Path, max_length: int = 64):  # Reduced from 128
        """Initialize dataset from training data JSON."""
        self.max_length = max_length

        with open(
            data_path / "training_data" / "training_data.json", "r", encoding="utf-8"
        ) as f:
            self.raw_data = json.load(f)

        # Process queries and responses
        self.queries = []
        self.responses = []
        self.query_types = []

        for item in self.raw_data:
            self.queries.append(item["query"])
            self.responses.append(item["response"])
            self.query_types.append(item["type"])

        # Create vocabulary
        self.vocab = self._create_vocabulary()
        self.vocab_size = len(self.vocab)

        # Convert text to tensors
        print("Converting text to tensors...")
        self.query_tensors = [self._text_to_tensor(q) for q in tqdm(self.queries)]
        self.response_tensors = [self._text_to_tensor(r) for r in tqdm(self.responses)]

        print(
            f"Dataset loaded: {len(self.queries)} samples, vocabulary size: {self.vocab_size}"
        )

    def _create_vocabulary(self) -> Dict[str, int]:
        """Create vocabulary from all text data."""
        vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        word_set = set()

        print("Building vocabulary...")
        # Add words from queries and responses
        for text in tqdm(self.queries + self.responses):
            words = text.lower().split()
            word_set.update(words)

        # Add words to vocabulary
        for i, word in enumerate(tqdm(sorted(word_set))):
            vocab[word] = i + 4  # Start after special tokens

        return vocab

    def _text_to_tensor(self, text: str, max_length: int = None) -> torch.Tensor:
        """Convert text to tensor using vocabulary."""
        if max_length is None:
            max_length = self.max_length

        words = text.lower().split()
        indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]

        # Add start and end tokens
        indices = [self.vocab["<START>"]] + indices + [self.vocab["<END>"]]

        # Pad or truncate
        if len(indices) < max_length:
            indices += [self.vocab["<PAD>"]] * (max_length - len(indices))
        else:
            indices = indices[: max_length - 1] + [self.vocab["<END>"]]

        return torch.LongTensor(indices)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return (
            self.query_tensors[idx],
            self.response_tensors[idx],
            self.query_types[idx],
        )


class GenshinAssistant(nn.Module):
    """Neural network model for the Genshin Impact assistant."""

    def __init__(
        self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)

        # Encoder
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # Decoder
        self.decoder = nn.LSTM(
            embedding_dim + hidden_dim * 2,  # Add context vector to input
            hidden_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size),
        )

    def attention_weights(self, decoder_hidden, encoder_outputs):
        """Calculate attention weights."""
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Reshape decoder hidden state to match encoder outputs
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Ensure dimensions match before concatenation
        if decoder_hidden.size(-1) != encoder_outputs.size(-1):
            # Project decoder hidden state to match encoder output dimension
            decoder_hidden = decoder_hidden.view(batch_size, seq_len, -1)
            if decoder_hidden.size(-1) > encoder_outputs.size(-1):
                decoder_hidden = decoder_hidden[:, :, :encoder_outputs.size(-1)]
            else:
                # Pad with zeros if needed
                pad_size = encoder_outputs.size(-1) - decoder_hidden.size(-1)
                padding = torch.zeros(batch_size, seq_len, pad_size, device=decoder_hidden.device)
                decoder_hidden = torch.cat([decoder_hidden, padding], dim=-1)
        
        # Concatenate decoder hidden state and encoder outputs
        attention_input = torch.cat((decoder_hidden, encoder_outputs), dim=2)
        
        # Calculate attention scores
        attention_weights = self.attention(attention_input)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        return attention_weights

    def forward(self, src, tgt):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        # Embeddings with dropout
        src_embedded = self.embedding_dropout(self.embedding(src))
        tgt_embedded = self.embedding_dropout(self.embedding(tgt))

        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)

        # Prepare decoder hidden state
        hidden = hidden.view(
            2, 2, batch_size, -1
        )  # [num_layers, num_directions, batch, hidden]
        hidden = hidden.transpose(
            1, 2
        ).contiguous()  # [num_layers, batch, num_directions, hidden]
        hidden = hidden.view(
            2, batch_size, -1
        )  # [num_layers, batch, hidden*num_directions]

        cell = cell.view(2, 2, batch_size, -1)
        cell = cell.transpose(1, 2).contiguous()
        cell = cell.view(2, batch_size, -1)

        # Initialize decoder input
        decoder_outputs = []
        decoder_hidden = hidden
        decoder_cell = cell

        # Teacher forcing: use target tokens as input
        for t in range(
            tgt_len - 1
        ):  # -1 because we don't need to generate after last token
            # Calculate attention weights
            attn_weights = self.attention_weights(decoder_hidden[-1], encoder_outputs)

            # Calculate context vector
            context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)

            # Concatenate embedding and context vector
            decoder_input = torch.cat([tgt_embedded[:, t : t + 1, :], context], dim=2)

            # Decode one step
            output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_input, (decoder_hidden, decoder_cell)
            )

            # Project to vocabulary
            output = self.output_layer(output)
            decoder_outputs.append(output)

        # Stack all decoder outputs
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs


class GenshinAssistantTrainer:
    """Trainer for the Genshin Impact assistant."""

    def __init__(self, data_dir: Path, embedding_dim: int = 128, hidden_dim: int = 256):
        self.data_dir = Path(data_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create models directory if it doesn't exist
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Load dataset
        self.dataset = GenshinAssistantDataset(self.data_dir)

        # Initialize model
        self.model = GenshinAssistant(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Save vocabulary
        vocab_path = self.models_dir / "assistant_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset.vocab, f, ensure_ascii=False, indent=2)

    def train(
        self, epochs: int = 1, batch_size: int = 32, learning_rate: float = 0.002
    ):
        """Train the assistant model."""
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.dataset.vocab["<PAD>"])

        best_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            self.model.train()

            # Create progress bar for batches
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch_idx, (queries, responses, _) in enumerate(pbar):
                queries = queries.to(self.device)
                responses = responses.to(self.device)

                # Forward pass
                outputs = self.model(queries, responses)

                # Reshape outputs and targets for loss calculation
                outputs = outputs.view(-1, self.dataset.vocab_size)
                targets = responses[:, 1:].contiguous().view(-1)  # Skip start token

                # Calculate loss
                loss = criterion(outputs, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, avg_loss, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, avg_loss)

    def _collate_fn(self, batch):
        """Custom collate function for DataLoader."""
        queries, responses, types = zip(*batch)
        return torch.stack(queries), torch.stack(responses), types

    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save a model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "loss": loss,
        }

        if is_best:
            checkpoint_path = self.models_dir / "assistant_best.pt"
        else:
            checkpoint_path = self.models_dir / f"assistant_checkpoint_{epoch+1}.pt"

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def generate_response(self, query: str, max_length: int = 64) -> str:
        """Generate a response for a given query."""
        self.model.eval()

        # Convert query to tensor
        query_tensor = self.dataset._text_to_tensor(query).unsqueeze(0).to(self.device)

        # Initialize with start token
        current_token = torch.LongTensor([[self.dataset.vocab["<START>"]]]).to(self.device)

        # Generate response
        response = [self.dataset.vocab["<START>"]]

        with torch.no_grad():
            # Encode query
            src_embedded = self.model.embedding_dropout(self.model.embedding(query_tensor))
            encoder_outputs, (hidden, cell) = self.model.encoder(src_embedded)

            # Prepare decoder states
            decoder_hidden = hidden
            decoder_cell = cell

            # Generate tokens
            for _ in range(max_length):
                # Get current token embedding
                token_embedded = self.model.embedding_dropout(
                    self.model.embedding(current_token)
                )

                # Calculate attention
                attn_weights = self.model.attention_weights(
                    decoder_hidden[-1], encoder_outputs
                )
                context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)

                # Prepare decoder input
                decoder_input = torch.cat([token_embedded, context], dim=2)

                # Generate next token
                output, (decoder_hidden, decoder_cell) = self.model.decoder(
                    decoder_input, (decoder_hidden, decoder_cell)
                )

                # Project to vocabulary
                output = self.model.output_layer(output)

                # Get next token
                next_token = output.argmax(dim=2)
                current_token = next_token

                # Add token to response
                token_idx = next_token.item()
                response.append(token_idx)

                # Stop if end token or max length
                if token_idx == self.dataset.vocab["<END>"]:
                    break

        # Convert response indices to words
        idx_to_word = {v: k for k, v in self.dataset.vocab.items()}
        response_words = [idx_to_word[idx] for idx in response[1:-1]]  # Skip start and end tokens

        return " ".join(response_words)
