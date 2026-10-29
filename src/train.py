import argparse
from copy import deepcopy
import torch

from src.dataset import get_mvtec_dataloaders
from src.models import get_model
from src.models.autoencoder import ConvAutoencoder, compute_reconstruction_loss
from src.models.patchcore import PatchCoreMemoryBank


def train_custom_autoencoder(
    model: ConvAutoencoder,
    datamodule,
    epochs: int = 10,
    lr: float = 1e-3,
    save_path: str = "autoencoder_weights.pth",
    device: str = "auto",
    selection: str = "best",
    loss_alpha: float = 0.85,
):
    """Native PyTorch training loop for ConvAutoencoder using a pre-initialized datamodule."""
    requested_device = device.lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Unsupported training device '{device}'. Use 'auto', 'cpu', or 'cuda'."
        )

    if requested_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested for training but unavailable; falling back to CPU.")
        selected_device = "cpu"
    else:
        selected_device = requested_device

    torch_device = torch.device(selected_device)
    model = model.to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    if selection == "best":
        selected_epoch = None
    elif selection == "epoch":
        selected_epoch = epochs
    else:
        raise ValueError("Model selection must be 'best' or 'epoch'.")

    print(f"Training ConvAutoencoder on device: {torch_device}")
    train_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    best_state = None
    selected_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch.image if hasattr(batch, "image") else batch["image"]
            images = images.to(torch_device)

            optimizer.zero_grad()
            reconstructions = model(images)
            loss = compute_reconstruction_loss(
                reconstructions, images, alpha=loss_alpha
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        model.eval()
        validation_total_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch.image if hasattr(batch, "image") else batch["image"]
                images = images.to(torch_device)
                reconstructions = model(images)
                validation_total_loss += compute_reconstruction_loss(
                    reconstructions, images, alpha=loss_alpha
                ).item()

        avg_validation_loss = validation_total_loss / len(val_loader)
        train_losses.append(avg_loss)
        validation_losses.append(avg_validation_loss)
        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {avg_loss:.6f} - "
            f"Validation Loss: {avg_validation_loss:.6f}"
        )

        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            best_state = deepcopy(model.state_dict())
        if selected_epoch == epoch + 1:
            selected_state = deepcopy(model.state_dict())

    if best_state is None or (selected_epoch is not None and selected_state is None):
        raise RuntimeError("Training did not produce a model checkpoint.")

    model.load_state_dict(best_state if selected_epoch is None else selected_state)
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    return {
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "best_epoch": validation_losses.index(best_validation_loss) + 1,
    }


def prepare_patchcore(
    model: PatchCoreMemoryBank,
    datamodule,
    device: str = "auto",
):
    """Build the PatchCore coreset bank without running a learned training loop."""
    requested_device = device.lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            f"Unsupported PatchCore device '{device}'. Use 'auto', 'cpu', or 'cuda'."
        )
    if requested_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested for PatchCore preparation but unavailable; falling back to CPU.")
        selected_device = "cpu"
    else:
        selected_device = requested_device

    model.fit(datamodule.train_dataloader(), device=selected_device)
    return {"train_losses": [], "validation_losses": [], "best_epoch": 0}


def run_pipeline(
    model_name: str = "autoencoder",
    category: str = "bottle",
    datamodule=None,
    epochs: int = 10,
    lr: float = 1e-3,
    save_path: str = "autoencoder_weights.pth",
    device: str = "auto",
    selection: str = "best",
    loss_alpha: float = 0.85,
):
    """Router function that handles model selection and training without redundant dataloading."""
    
    # Lazy-load datamodule only if called directly as a standalone script
    if datamodule is None:
        from src.dataset import get_mvtec_dataloaders
        datamodule = get_mvtec_dataloaders(category=category, model_name=model_name)

    # Instantiate model based on selection (expandable for future models)
    if model_name == "autoencoder":
        model = ConvAutoencoder()
        return train_custom_autoencoder(
            model,
            datamodule,
            epochs=epochs,
            lr=lr,
            save_path=save_path,
            device=device,
            selection=selection,
            loss_alpha=loss_alpha,
        )
    elif model_name == "patchcore":
        model = PatchCoreMemoryBank()
        return prepare_patchcore(model, datamodule, device=device)
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")
        