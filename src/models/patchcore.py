import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.neighbors import NearestNeighbors
import numpy as np


class PatchCoreFeatureExtractor(nn.Module):
    """Frozen ResNet18 feature extractor used by PatchCore and ONNX export."""

    def __init__(self, backbone_name: str = "resnet18"):
        super().__init__()
        if backbone_name.lower() != "resnet18":
            raise ValueError("PatchCore currently supports only the 'resnet18' backbone.")

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.avg_pool_kernel_size = 3
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.image_mean) / self.image_std
        features = self.backbone(x)
        return F.avg_pool2d(
            features,
            kernel_size=self.avg_pool_kernel_size,
            stride=1,
            padding=1,
        )


class PatchCoreMemoryBank(nn.Module):
    """Lightweight PatchCore implementation using a frozen ResNet backbone."""

    def __init__(self, backbone_name: str = "resnet18", coreset_sampling_ratio: float = 0.1):
        super().__init__()
        if not 0.0 < coreset_sampling_ratio <= 1.0:
            raise ValueError("coreset_sampling_ratio must be greater than 0 and at most 1.")

        self.coreset_sampling_ratio = coreset_sampling_ratio

        self.backbone_name = backbone_name.lower()
        self.feature_extractor = PatchCoreFeatureExtractor(self.backbone_name)
        self.memory_bank = np.empty((0, 256), dtype=np.float32)
        self.nn_engine = None

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts and pools spatial patch embeddings from intermediate layers."""
        with torch.no_grad():
            features = self.feature_extractor(x)

        # Reshape from (B, C, H, W) to (B * H * W, C)
        b, c, h, w = features.shape
        patches = features.permute(0, 2, 3, 1).reshape(-1, c)
        return patches

    @staticmethod
    def _batch_images(batch) -> torch.Tensor:
        if hasattr(batch, "image"):
            return batch.image
        if isinstance(batch, dict):
            return batch["image"]
        if isinstance(batch, (tuple, list)):
            return batch[0]
        raise TypeError("PatchCore batches must expose an image tensor.")

    def fit(self, dataloader, device: str = "cpu"):
        """Passes normal training images through the backbone to populate the memory bank."""
        torch_device = torch.device(device)
        self.feature_extractor.to(torch_device).eval()

        print("Extracting patch embeddings from normal training images...")
        patch_list = []

        for batch in dataloader:
            batch_images = self._batch_images(batch).to(torch_device)
            patches = self._extract_patches(batch_images)
            patch_list.append(patches.cpu())

        if not patch_list:
            raise RuntimeError("Cannot build a PatchCore memory bank from an empty dataloader.")

        # Concatenate all patch embeddings
        all_patches = torch.cat(patch_list, dim=0).numpy()

        # Sample before evaluation so all quantization targets use the same bank.
        num_samples = max(1, int(np.ceil(len(all_patches) * self.coreset_sampling_ratio)))
        num_samples = min(num_samples, len(all_patches))
        indices = np.random.choice(len(all_patches), num_samples, replace=False)
        self.memory_bank = np.asarray(all_patches[indices], dtype=np.float32)

        print(f"Memory bank built successfully! Total patch embeddings stored: {len(self.memory_bank)}")

        # Initialize Nearest Neighbors lookup engine
        self.nn_engine = NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean')
        self.nn_engine.fit(self.memory_bank)

    def score_features(self, features: np.ndarray, image_size: tuple[int, int]):
        """Score an ONNX feature-map output using the fitted CPU memory bank."""
        if self.nn_engine is None:
            raise RuntimeError("PatchCore memory bank has not been fitted.")
        if features.ndim != 4:
            raise ValueError(f"Expected feature maps with four dimensions, got {features.shape}.")

        batch_size, _, feature_height, feature_width = features.shape
        patches = np.transpose(features, (0, 2, 3, 1)).reshape(-1, features.shape[1])
        distances, _ = self.nn_engine.kneighbors(patches)
        patch_scores = distances.reshape(batch_size, feature_height, feature_width)
        anomaly_map = F.interpolate(
            torch.from_numpy(patch_scores).unsqueeze(1),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1).numpy()
        image_scores = patch_scores.reshape(batch_size, -1).max(axis=1)
        return image_scores, anomaly_map

    def predict(self, x: torch.Tensor):
        """Computes patch-level anomaly scores and upsampled heatmaps for a test image."""
        device = next(self.feature_extractor.parameters()).device
        x = x.to(device)
        features = self.feature_extractor(x).cpu().numpy()
        return self.score_features(features, image_size=(x.shape[2], x.shape[3]))