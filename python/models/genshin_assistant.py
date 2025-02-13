import torch
import torch.nn as nn
from typing import Dict, Tuple
from pathlib import Path
import json
from tqdm import tqdm
from .encoders import HeavyEncoder, LightEncoder
from .decoders import HeavyDecoder, LightDecoder

class GenshinAssistant(nn.Module):
    """Neural network model for the Genshin Impact assistant.
       Supports both heavy and failover (light) architectures.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, failover: bool = False):
        """
        If failover==True then a simpler (light) encoder/decoder will be used.
        Otherwise the default heavy architecture is built.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        self.failover = failover

        if not failover:
            # Use heavy (default) components
            self.encoder = HeavyEncoder(embedding_dim, hidden_dim)
            self.decoder = HeavyDecoder(embedding_dim, hidden_dim, vocab_size)
        else:
            # Use light (failover) components
            self.encoder = LightEncoder(embedding_dim, hidden_dim)
            self.decoder = LightDecoder(embedding_dim, hidden_dim, vocab_size)

    def forward(self, src, tgt):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # Embed the source and target
        src_embedded = self.embedding_dropout(self.embedding(src))
        tgt_embedded = self.embedding_dropout(self.embedding(tgt))
        
        # Encode the source text
        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)
        
        if not self.failover:
            # For heavy mode, reshape the encoder hidden states for the decoder
            hidden = hidden.view(2, 2, batch_size, -1)\
                         .transpose(1, 2).contiguous()\
                         .view(2, batch_size, -1)
            cell = cell.view(2, 2, batch_size, -1)\
                       .transpose(1, 2).contiguous()\
                       .view(2, batch_size, -1)
            decoder_hidden = hidden
            decoder_cell = cell
        else:
            # In light mode (single layer, unidirectional) no reshaping needed
            decoder_hidden = hidden
            decoder_cell = cell
            # Save encoder's final hidden state as context
            encoder_final_hidden = hidden[-1]

        outputs = []
        # Teacher-forcing: feed target embeddings step-by-step
        for t in range(tgt_len - 1):  # exclude final token
            current_embedding = tgt_embedded[:, t:t+1, :]
            if not self.failover:
                # Heavy mode: use attention at each step
                output, (decoder_hidden, decoder_cell) = self.decoder(
                    current_embedding,
                    (decoder_hidden, decoder_cell),
                    encoder_outputs
                )
            else:
                # Light mode: use static encoder final hidden context
                output, (decoder_hidden, decoder_cell) = self.decoder(
                    current_embedding,
                    (decoder_hidden, decoder_cell),
                    encoder_final_hidden
                )
            outputs.append(output)
            
        outputs = torch.cat(outputs, dim=1)
        return outputs