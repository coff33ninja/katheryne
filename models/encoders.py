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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])

        # Latent space projections
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        # Hardware optimization
        self.device = HardwareManager.get_device()
        self.to(self.device)
        self = HardwareManager.optimize_for_device(self, self.device)

    def forward(self, x):
        # Move input to correct device
        x = x.to(self.device)

        # Input layer with activation
        x = F.relu(self.input_layer(x))

        # Hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        # Latent space parameters
        mean = self.mean(x)
        log_var = self.log_var(x)

        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

class HeavyEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=2):
        super(HeavyEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)

class LightEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LightEncoder, self).__init__()
        self.embedding = nn.Embedding(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, (hidden, cell)
