import argparse
import torch

from src.dataset import get_mvtec_dataloaders
from src.models import get_model
from src.models.autoencoder import ConvAutoencoder, compute_reconstruction_loss


def train_custom_autoencoder(
    model: ConvAutoencoder, datamodule, epochs: int = 10, lr: float = 1e-3
):
    """Native PyTorch training loop for ConvAutoencoder."""
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

    # Save custom PyTorch model weights
    torch.save(model.state_dict(), "autoencoder_weights.pth")
    print("Autoencoder weights saved to autoencoder_weights.pth")


def run_pipeline(model_name: str = "patchcore", category: str = "bottle"):
    # 1. Load Data
    datamodule = get_mvtec_dataloaders(category=category, model_name=model_name)

    # 2. Get Model
    model = get_model(model_name)

    # 3. Route execution path
    if isinstance(model, ConvAutoencoder):
        train_custom_autoencoder(model, datamodule)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Expecting ConvAutoencoder"
        )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Industrial Edge Vision Training Engine"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="autoencoder",
        choices=["autoencoder"],
    )
    parser.add_argument("--category", type=str, default="bottle")
    args = parser.parse_args()

    run_pipeline(model_name=args.model, category=args.category)