# src/models/autoencoder.py
import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch_msssim import SSIM


_SSIM_MODULE = SSIM(data_range=1.0, size_average=True, channel=3)


class ConvAutoencoder(nn.Module):
    """Custom Convolutional Autoencoder for Reconstruction-based Anomaly Detection."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 32):
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
    reconstructions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.85,
) -> torch.Tensor:
    """Compute a weighted combination of MSE and structural similarity loss."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Loss alpha must be between 0 and 1, got {alpha}.")

    mse_loss = F.mse_loss(reconstructions, targets)
    ssim_loss = 1.0 - _SSIM_MODULE(reconstructions, targets)
    return (alpha * mse_loss) + ((1.0 - alpha) * ssim_loss)