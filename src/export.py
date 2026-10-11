import torch
from src.models.autoencoder import ConvAutoencoder

def export_to_onnx(
    weights_path: str = "autoencoder_weights.pth",
    onnx_output_path: str = "autoencoder.onnx",
):
    """Loads trained PyTorch ConvAutoencoder weights and exports the graph to ONNX."""
    print(f"Loading weights from {weights_path}...")
    
    # 1. Instantiate model architecture
    model = ConvAutoencoder()
    
    # 2. Load trained weights safely
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    # 3. Switch to evaluation mode (crucial for disabling dropout/batchnorm updates)
    model.eval()

    # 4. Create a dummy input matching your training dimensions [Batch, Channels, Height, Width]
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Exporting model to ONNX format: {onnx_output_path}...")
    
    # 5. Execute PyTorch native ONNX export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=17,
        input_names=["input_image"],
        output_names=["reconstructed_image"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "reconstructed_image": {0: "batch_size"},
        },
    )
    print(f"Successfully exported ONNX model to {onnx_output_path}!")


if __name__ == "__main__":
    export_to_onnx()