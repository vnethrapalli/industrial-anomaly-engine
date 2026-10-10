# src/models/autoencoder.py
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Custom Convolutional Autoencoder for Reconstruction-based Anomaly Detection."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def compute_reconstruction_loss(
    reconstructions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Computes Mean Squared Error reconstruction loss."""
    criterion = nn.MSELoss()
    return criterion(reconstructions, targets)