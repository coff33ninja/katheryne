import torch
from typing import List, Dict, Optional
import json
from pathlib import Path
import re

class GenshinTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        # Initialize with special tokens
        self.token2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        self.sep_token = '<SEP>'
        self.start_token = '<BOS>'  # Alias for bos_token
        self.end_token = '<EOS>'    # Alias for eos_token
        
        # Load vocabulary from data files
        self._load_dynamic_vocab()
        
    def _load_dynamic_vocab(self):
        """Load vocabulary from JSON files and autoencoder data."""
        import json
        import torch
        from pathlib import Path
        import os
        
        # Get project root
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        
        # Paths to check for data
        data_paths = [
            project_root / 'data',
            project_root / 'models' / 'data'
        ]
        
        # Set to store unique tokens
        vocab_set = set()
        
        # Load from JSON files
        for data_path in data_paths:
            if data_path.exists():
                for json_file in data_path.glob('*.json'):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # Process different JSON structures
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict):
                                        # Add all text values to vocabulary
                                        for value in item.values():
                                            if isinstance(value, str):
                                                words = self._tokenize_text(value)
                                                vocab_set.update(words)
                            
                            elif isinstance(data, dict):
                                # Handle nested structures
                                def process_dict(d):
                                    for value in d.values():
                                        if isinstance(value, str):
                                            words = self._tokenize_text(value)
                                            vocab_set.update(words)
                                        elif isinstance(value, dict):
                                            process_dict(value)
                                        elif isinstance(value, list):
                                            for item in value:
                                                if isinstance(item, dict):
                                                    process_dict(item)
                                                elif isinstance(item, str):
                                                    words = self._tokenize_text(item)
                                                    vocab_set.update(words)
                                
                                process_dict(data)
                                
                    except Exception as e:
                        print(f"Error loading {json_file}: {str(e)}")
        
        # Try to load autoencoder data
        autoencoder_path = project_root / 'models' / 'autoencoder.pt'
        if autoencoder_path.exists():
            try:
                autoencoder_data = torch.load(autoencoder_path)
                if 'vocab' in autoencoder_data:
                    vocab_set.update(autoencoder_data['vocab'])
            except Exception as e:
                print(f"Error loading autoencoder data: {str(e)}")
        
        # Add tokens to vocabulary (up to vocab_size limit)
        available_slots = self.vocab_size - len(self.token2idx)
        vocab_list = list(vocab_set)
        print(f"Found {len(vocab_list)} unique tokens in data files")
        
        for token in vocab_list[:available_slots]:
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        
        print(f"Loaded {len(self.token2idx)} tokens into vocabulary")
        print("Sample tokens:", list(self.token2idx.keys())[:20])
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words for vocabulary building."""
        if not isinstance(text, str):
            return []
            
        # Basic text normalization
        text = text.lower().strip()
        if not text:
            return []
            
        # Remove punctuation and split into words
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Handle special cases and compound words
        tokens = []
        for word in words:
            # Split hyphenated words
            if '-' in word:
                tokens.extend(word.split('-'))
            else:
                tokens.append(word)
                
        return tokens

    def tokenize(self, text):
        """Tokenize text into subwords."""
        if not isinstance(text, str):
            return [self.unk_token]
        
        # Normalize text
        text = text.strip()
        if not text:
            return [self.unk_token]
        
        # Handle special cases
        text = text.replace("'s", " 's")  # Handle possessives
        text = text.replace(".", " .")
        text = text.replace(",", " ,")
        text = text.replace("!", " !")
        text = text.replace("?", " ?")
        text = text.replace("(", " ( ")
        text = text.replace(")", " ) ")
        
        # Split into words
        words = text.split()
        
        # Handle compound words and phrases
        tokens = []
        i = 0
        while i < len(words):
            # Try to match longer sequences first
            matched = False
            for j in range(3, 0, -1):  # Try 3-word, 2-word, then 1-word combinations
                if i + j <= len(words):
                    compound = " ".join(words[i:i+j])
                    if compound in self.token2idx:
                        tokens.append(compound)
                        i += j
                        matched = True
                        break
            
            if not matched:
                # If no compound match, try the single word
                word = words[i]
                if word in self.token2idx:
                    tokens.append(word)
                else:
                    # Try lowercase version
                    word_lower = word.lower()
                    if word_lower in self.token2idx:
                        tokens.append(word_lower)
                    else:
                        tokens.append(self.unk_token)
                i += 1
        
        return tokens
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to token indices."""
        tokens = self.tokenize(text)
        return torch.tensor([self.token2idx.get(token, self.token2idx[self.unk_token]) 
                           for token in tokens], dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Convert token indices back to text."""
        tokens = [self.idx2token.get(idx.item(), self.unk_token) 
                 for idx in token_ids if idx.item() in self.idx2token]
        
        # Remove special tokens
        tokens = [token for token in tokens 
                 if token not in [self.start_token, self.end_token, self.pad_token]]
        
        # Basic detokenization
        text = ' '.join(tokens)
        text = re.sub(r' ([.,!?()])', r'\1', text)  # Remove spaces before punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text
    
    def save(self, path: str):
        """Save tokenizer vocabulary to file."""
        vocab_data = {
            'token2idx': self.token2idx,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'GenshinTokenizer':
        """Load tokenizer vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        tokenizer = cls(vocab_size=vocab_data['vocab_size'])
        tokenizer.token2idx = vocab_data['token2idx']
        tokenizer.idx2token = {int(idx): token for token, idx 
                             in tokenizer.token2idx.items()}
        return tokenizer