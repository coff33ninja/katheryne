import torch
import torch.nn as nn

class HeavyDecoder(nn.Module):
    """A full-featured decoder that uses attention."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        # The decoder input will be the word embedding concatenated with context from the encoder
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim * 2,  # context from bidirectional encoder
            hidden_dim * 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, vocab_size),
        )

    def compute_attention(self, decoder_hidden, encoder_outputs):
        batch_size, seq_len, enc_dim = encoder_outputs.size()
        # Expand decoder hidden state
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # Handle dimension mismatches
        if decoder_hidden_expanded.size(-1) != encoder_outputs.size(-1):
            if decoder_hidden_expanded.size(-1) > encoder_outputs.size(-1):
                decoder_hidden_expanded = decoder_hidden_expanded[:, :, :encoder_outputs.size(-1)]
            else:
                pad_size = encoder_outputs.size(-1) - decoder_hidden_expanded.size(-1)
                padding = torch.zeros(batch_size, seq_len, pad_size, device=decoder_hidden.device)
                decoder_hidden_expanded = torch.cat([decoder_hidden_expanded, padding], dim=-1)
        attn_input = torch.cat((decoder_hidden_expanded, encoder_outputs), dim=2)
        attn_scores = self.attention(attn_input)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights

    def forward(self, decoder_input, decoder_states, encoder_outputs):
        hidden, cell = decoder_states
        # Compute attention weight from last layer's hidden state
        attn_weights = self.compute_attention(hidden[-1], encoder_outputs)
        # Compute context vector
        context = torch.bmm(attn_weights.transpose(1,2), encoder_outputs)
        # Concatenate input embedding with context
        lstm_input = torch.cat([decoder_input, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.output_layer(output)  # project to vocabulary space
        return output, (hidden, cell)


class LightDecoder(nn.Module):
    """A simpler decoder that does not use a full attention mechanism.
       Instead, it uses the final hidden state of the encoder as context.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        # For light mode the decoder input is the concatenation of the word embedding and context vector
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, vocab_size),
        )
    
    def forward(self, decoder_input, decoder_states, encoder_final_hidden):
        # Use the static encoder final hidden as context for every decoder step
        context = encoder_final_hidden.unsqueeze(1)
        # Concatenate decoder_input (word embedding) with context
        lstm_input = torch.cat([decoder_input, context], dim=2)
        hidden, cell = decoder_states
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        output = self.output_layer(output)
        return output, (hidden, cell)