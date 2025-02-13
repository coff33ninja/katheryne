import torch
import torch.nn as nn
import torch.nn.functional as F

class HardwareManager:
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return torch.device('cuda')  # ROCm uses CUDA device type
        else:
            return torch.device('cpu')

    @staticmethod
    def optimize_for_device(model, device):
        if device.type == 'cuda':
            model = model.cuda()
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
        elif device.type == 'cpu':
            if torch.backends.mkl.is_available():
                torch.set_num_threads(torch.get_num_threads())
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model)
            except ImportError:
                pass
        return model

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=2):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Latent to hidden
        self.latent_layer = nn.Linear(latent_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Hardware optimization
        self.device = HardwareManager.get_device()
        self.to(self.device)
        self = HardwareManager.optimize_for_device(self, self.device)

    def forward(self, z):
        # Move input to correct device
        z = z.to(self.device)

        # Latent to hidden
        x = F.relu(self.latent_layer(z))

        # Hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Output layer
        output = torch.sigmoid(self.output_layer(x))

        return output

    def decode(self, z):
        return self.forward(z)

class HeavyDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=2):
        super(HeavyDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, (hidden, cell)

class LightDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LightDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, (hidden, cell)
