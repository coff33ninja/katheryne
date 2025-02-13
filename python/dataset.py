import json
import os
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

class GenshinAssistantDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        try:
            self.vocab = self.load_vocab()
            self.data = self.load_data()
            self.vocab_size = len(self.vocab)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dataset: {str(e)}")

    def _text_to_tensor(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Convert text to tensor representation with proper tokenization and padding.

        Args:
            text (str): Input text to convert
            max_length (int): Maximum sequence length, including special tokens

        Returns:
            torch.Tensor: Tensor of token indices
        """
        # Basic tokenization (you might want to use a proper tokenizer like BERT's)
        tokens = ["<BOS>"]
        # Split on word boundaries and keep punctuation
        for word in text.lower().replace("'", " '").replace('"', ' "').split():
            # Remove punctuation attached to words
            word = word.strip(".,!?:;()")
            if word:
                tokens.append(word)
        tokens.append("<EOS>")

        # Truncate if too long (account for <BOS> and <EOS>)
        if len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ["<EOS>"]

        # Convert to indices
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Pad if necessary
        padding_length = max_length - len(indices)
        if padding_length > 0:
            indices.extend([self.vocab["<PAD>"]] * padding_length)

        return torch.tensor(indices, dtype=torch.long)

    def load_vocab(self) -> Dict[str, int]:
        """Load vocabulary from a JSON file.
        
        Returns:
            Dict[str, int]: Vocabulary mapping tokens to indices
            
        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
            json.JSONDecodeError: If vocabulary file is not valid JSON
        """
        vocab_path = self.data_dir / "assistant_vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
            
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            
        # Verify required special tokens
        required_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
        missing_tokens = required_tokens - set(vocab.keys())
        if missing_tokens:
            raise ValueError(f"Vocabulary is missing required special tokens: {missing_tokens}")
            
        return vocab

    def load_data(self) -> List[Any]:
        """Load dataset from files.
        
        Returns:
            List[Any]: List of data points
            
        Raises:
            NotImplementedError: If the method is not implemented
        """
        raise NotImplementedError("load_data() method must be implemented")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        if not 0 <= idx < len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        return self.data[idx]