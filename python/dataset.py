import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
from torch.utils.data import Dataset


class GenshinAssistantDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")

        # Initialize vocabulary with special tokens
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

        try:
            # First load the data
            self.data = self.load_data()
            # Then build vocabulary from the data
            self.build_vocab()
            self.vocab_size = len(self.vocab)
            print(
                f"Loaded dataset with {len(self.data)} samples and vocabulary size {self.vocab_size}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize dataset: {str(e)}")

    def build_vocab(self):
        """Build vocabulary from training data"""
        print("Building vocabulary from training data...")
        # Collect all unique words from queries and responses
        word_set = set()
        for query, response, _ in self.data:
            # Add words from query
            words = query.lower().split()
            word_set.update(words)

            # Add words from response (which is a JSON string)
            try:
                response_dict = json.loads(response)

                # Recursively extract all string values from the response dictionary
                def extract_strings(obj):
                    if isinstance(obj, str):
                        words = obj.lower().split()
                        word_set.update(words)
                    elif isinstance(obj, dict):
                        for value in obj.values():
                            extract_strings(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_strings(item)

                extract_strings(response_dict)
            except json.JSONDecodeError:
                # If response is not JSON, treat it as plain text
                words = response.lower().split()
                word_set.update(words)

        # Add words to vocabulary with indices starting after special tokens
        for word in sorted(word_set):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        # Save vocabulary
        vocab_path = self.data_dir / "models"
        vocab_path.mkdir(exist_ok=True)
        vocab_file = vocab_path / "assistant_vocab.json"
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"Saved vocabulary with {len(self.vocab)} tokens to {vocab_file}")

    def _text_to_tensor(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Convert text to tensor representation with proper tokenization and padding."""
        # Basic tokenization
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
            tokens = tokens[: max_length - 1] + ["<EOS>"]

        # Convert to indices
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Pad if necessary
        padding_length = max_length - len(indices)
        if padding_length > 0:
            indices.extend([self.vocab["<PAD>"]] * padding_length)

        return torch.tensor(indices, dtype=torch.long)

    def load_data(self) -> List[Tuple[str, str, str]]:
        """Load training data from JSON file."""
        data_path = self.data_dir / "training_data" / "training_data.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Training data file not found: {data_path}")

        print(f"Loading training data from {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        processed_data = []
        for item in raw_data:
            query = item["query"]
            # Convert response dict to string
            response = json.dumps(item["response"], ensure_ascii=False)
            query_type = item["type"]
            processed_data.append((query, response, query_type))

        print(f"Loaded {len(processed_data)} training samples")
        return processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if not 0 <= idx < len(self.data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}"
            )

        query, response, query_type = self.data[idx]
        query_tensor = self._text_to_tensor(query)
        response_tensor = self._text_to_tensor(response)

        return query_tensor, response_tensor, query_type
