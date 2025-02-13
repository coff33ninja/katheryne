import torch
import torch.nn as nn
import torch.nn.functional as F


class HeavyEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )

    def forward(self, src):
        return self.lstm(src)


class LightEncoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

    def forward(self, src):
        return self.lstm(src)


class HeavyDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def compute_attention(self, decoder_hidden, encoder_outputs):
        attention_weights = torch.bmm(
            encoder_outputs, decoder_hidden.unsqueeze(2)
        ).squeeze(2)
        return F.softmax(attention_weights, dim=1).unsqueeze(1)

    def forward(self, tgt, hidden, encoder_outputs):
        attention_weights = self.compute_attention(hidden[0][-1], encoder_outputs)
        context = torch.bmm(attention_weights, encoder_outputs)
        lstm_input = torch.cat([tgt, context], dim=2)
        output, hidden = self.lstm(lstm_input, hidden)
        prediction = self.output(output)
        return prediction, hidden


class LightDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, hidden, encoder_final_hidden):
        output, hidden = self.lstm(tgt, hidden)
        prediction = self.output(output)
        return prediction, hidden


class GenshinAssistant(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        failover: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        self.failover = failover

        if not failover:
            self.encoder = HeavyEncoder(embedding_dim, hidden_dim)
            self.decoder = HeavyDecoder(embedding_dim, hidden_dim, vocab_size)
        else:
            self.encoder = LightEncoder(embedding_dim, hidden_dim)
            self.decoder = LightDecoder(embedding_dim, hidden_dim, vocab_size)

    def forward(self, src, tgt):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)

        src_embedded = self.embedding_dropout(self.embedding(src))
        tgt_embedded = self.embedding_dropout(self.embedding(tgt))

        encoder_outputs, (hidden, cell) = self.encoder(src_embedded)

        if not self.failover:
            hidden = (
                hidden.view(2, 2, batch_size, -1)
                .transpose(1, 2)
                .contiguous()
                .view(2, batch_size, -1)
            )
            cell = (
                cell.view(2, 2, batch_size, -1)
                .transpose(1, 2)
                .contiguous()
                .view(2, batch_size, -1)
            )
            decoder_hidden = hidden
            decoder_cell = cell
        else:
            decoder_hidden = hidden
            decoder_cell = cell
            encoder_final_hidden = hidden[-1]

        outputs = []
        for t in range(tgt_len - 1):
            current_embedding = tgt_embedded[:, t : t + 1, :]
            if not self.failover:
                output, (decoder_hidden, decoder_cell) = self.decoder(
                    current_embedding, (decoder_hidden, decoder_cell), encoder_outputs
                )
            else:
                output, (decoder_hidden, decoder_cell) = self.decoder(
                    current_embedding,
                    (decoder_hidden, decoder_cell),
                    encoder_final_hidden,
                )
            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return outputs
