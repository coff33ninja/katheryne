import torch
import torch.nn as nn

class HeavyEncoder(nn.Module):
    """A full-featured encoder using a bidirectional, two-layer LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
    
    def forward(self, x):
        return self.lstm(x)  # returns (output, (hidden, cell))


class LightEncoder(nn.Module):
    """A simpler encoder using a single-layer, unidirectional LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
    
    def forward(self, x):
        return self.lstm(x)