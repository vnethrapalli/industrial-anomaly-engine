from src.models.autoencoder import ConvAutoencoder


def get_model(model_name: str, **kwargs):
    """Factory function to instantiate anomaly detection models."""
    model_name = model_name.lower()

    if model_name == "autoencoder":
        return ConvAutoencoder(**kwargs)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: ['autoencoder']"
        )