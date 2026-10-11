import argparse
import torch

from src.dataset import get_mvtec_dataloaders
from src.models import get_model
from src.models.autoencoder import ConvAutoencoder, compute_reconstruction_loss


def train_custom_autoencoder(
    model: ConvAutoencoder, datamodule, epochs: int = 10, lr: float = 1e-3, save_path: str = "autoencoder_weights.pth"
):
    """Native PyTorch training loop for ConvAutoencoder using a pre-initialized datamodule."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = datamodule.train_dataloader()

    print(f"Training ConvAutoencoder on device: {device}")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            images = batch.image if hasattr(batch, "image") else batch["image"]
            images = images.to(device)

            optimizer.zero_grad()
            reconstructions = model(images)
            loss = compute_reconstruction_loss(reconstructions, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


def run_pipeline(model_name: str = "autoencoder", category: str = "bottle", datamodule=None, epochs: int = 10, save_path: str = "autoencoder_weights.pth"):
    """Router function that handles model selection and training without redundant dataloading."""
    
    # Lazy-load datamodule only if called directly as a standalone script
    if datamodule is None:
        from src.dataset import get_mvtec_dataloaders
        datamodule = get_mvtec_dataloaders(category=category, model_name=model_name)

    # Instantiate model based on selection (expandable for future models)
    if model_name == "autoencoder":
        model = ConvAutoencoder()
        train_custom_autoencoder(model, datamodule, epochs=epochs, save_path=save_path)
    # Future model routing example:
    # elif model_name == "patchcore":
    #     model = get_patchcore_model(...)
    #     train_patchcore(model, datamodule, ...)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")
        