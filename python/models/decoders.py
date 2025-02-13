import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import HardwareManager


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


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_layers)

        # Hardware optimization
        self.device = HardwareManager.get_device()
        self.to(self.device)
        self = HardwareManager.optimize_for_device(self, self.device)

    def forward(self, x):
        # Move input to correct device
        x = x.to(self.device)

        # Encode
        mean, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mean, log_var)

        # Decode
        reconstruction = self.decoder(z)

        return reconstruction, mean, log_var

    def get_loss(self, x, reconstruction, mean, log_var):
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy(reconstruction, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return recon_loss + kl_loss


def create_model(input_dim, hidden_dim=256, latent_dim=32, num_layers=2):
    """
    Factory function to create an AutoEncoder with optimal hardware settings
    """
    model = AutoEncoder(input_dim, hidden_dim, latent_dim, num_layers)
    device = HardwareManager.get_device()
    model = HardwareManager.optimize_for_device(model, device)

    print(f"Model created on {device}")
    print(f"Hardware info: {Encoder.get_hardware_info()}")

    return model
