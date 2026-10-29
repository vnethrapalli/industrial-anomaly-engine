import json
from pathlib import Path

import joblib
import numpy as np
import torch
from src.models.autoencoder import ConvAutoencoder
from src.models.patchcore import PatchCoreMemoryBank

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


def export_patchcore_to_onnx(
    model: PatchCoreMemoryBank,
    onnx_output_path: str,
    metadata_output_path: str,
    input_size: tuple[int, int] = (224, 224),
    memory_bank_path: str | None = None,
    index_path: str | None = None,
):
    """Export the PatchCore feature extractor and describe its serving components."""
    model.feature_extractor.eval().cpu()
    dummy_input = torch.randn(1, 3, *input_size, dtype=torch.float32)
    torch.onnx.export(
        model.feature_extractor,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=17,
        input_names=["input_image"],
        output_names=["patch_features"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "patch_features": {0: "batch_size"},
        },
    )

    with torch.no_grad():
        feature_shape = tuple(model.feature_extractor(dummy_input).shape)
    metadata = {
        "model": "patchcore",
        "backbone": model.backbone_name,
        "input_shape": [1, 3, input_size[0], input_size[1]],
        "feature_shape": list(feature_shape),
        "preprocessing": {
            "scale": "input pixels must be float32 in [0, 1]",
            "normalization": "ImageNet mean/std applied inside feature extractor",
        },
        "coreset_sampling_ratio": model.coreset_sampling_ratio,
        "memory_bank": Path(memory_bank_path).name if memory_bank_path else None,
        "nearest_neighbors_index": Path(index_path).name if index_path else None,
        "nearest_neighbors": {
            "metric": "euclidean",
            "n_neighbors": 1,
            "execution": "CPU via scikit-learn",
        },
    }
    with Path(metadata_output_path).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def save_patchcore_components(
    model: PatchCoreMemoryBank,
    memory_bank_path: str,
    index_path: str,
):
    """Persist the sampled bank and fitted nearest-neighbor index for later serving."""
    if model.nn_engine is None or len(model.memory_bank) == 0:
        raise RuntimeError("PatchCore must be fitted before its artifacts can be saved.")
    if str(memory_bank_path).endswith(".pt"):
        torch.save(model.memory_bank, memory_bank_path)
    else:
        np.save(memory_bank_path, model.memory_bank)
    joblib.dump(model.nn_engine, index_path)


if __name__ == "__main__":
    export_to_onnx()