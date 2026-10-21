# src/dataset.py
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from anomalib.data import MVTecAD
from anomalib.data.utils import concatenate_datasets
from huggingface_hub import snapshot_download
import numpy as np
import torch
from torch.utils.data import DataLoader

DATASET_ROOT = "./datasets/MVTecAD"
HF_DATASET_REPO = "foersben/mvtec-ad"


def _ensure_mvtec_category(category: str) -> None:
    """Download the requested MVTec category when it is not cached locally."""
    category_root = f"{DATASET_ROOT}/{category}"
    required_directories = (
        f"{category_root}/train",
        f"{category_root}/test",
        f"{category_root}/ground_truth",
    )
    if all(os.path.isdir(path) for path in required_directories):
        return

    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[f"{category}/**"],
        local_dir=DATASET_ROOT,
    )

    if not all(os.path.isdir(path) for path in required_directories):
        raise RuntimeError(
            f"Hugging Face dataset did not provide the expected files for category "
            f"'{category}' under {DATASET_ROOT}."
        )


class CombinedMVTecDataModule:
    """Combine per-category MVTec splits while preserving their collate function."""

    def __init__(self, datamodules, batch_size: int):
        self.train_data = concatenate_datasets(
            [datamodule.train_data for datamodule in datamodules]
        )
        self.val_data = concatenate_datasets(
            [datamodule.val_data for datamodule in datamodules]
        )
        self.test_data = concatenate_datasets(
            [datamodule.test_data for datamodule in datamodules]
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_data.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_data.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_data.collate_fn,
        )


def get_mvtec_dataloaders(
    category: str | list[str] = "bottle",
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    model_name: str = None,
    validation_split: float = 0.2,
    seed: int = 42,
):
    """Downloads category data and constructs MVTec AD DataModule with Albumentations transforms."""
    # 1. Enforce batch size constraint if EfficientAD is specified
    if model_name in ["efficient_ad", "efficientad"]:
        batch_size = 1

    # 2. Define Albumentations pipeline
    pipeline = A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            ToTensorV2(),
        ]
    )

    # 3. Callable wrapper converting positional call img -> pipeline(image=img)
    def transform(image, mask=None):
        if hasattr(image, "convert"):  # PIL Image -> NumPy
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()

        # If mask is present and valid, pass to Albumentations as numpy array
        if mask is not None:
            if hasattr(mask, "convert"):
                mask = np.array(mask)
            elif isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            augmented = pipeline(image=image, mask=mask)
            return augmented["image"], augmented.get("mask", mask)

        # Training set images (mask is None)
        augmented = pipeline(image=image)
        return augmented["image"], None

    categories = [category] if isinstance(category, str) else category
    if not categories:
        raise ValueError("At least one MVTec category is required.")

    datamodules = []
    for category_name in categories:
        _ensure_mvtec_category(category_name)
        datamodule = MVTecAD(
            root=DATASET_ROOT,
            category=category_name,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            test_split_mode="from_dir",
            val_split_mode="from_train",
            val_split_ratio=validation_split,
            seed=seed,
            augmentations=transform,
        )
        datamodule.setup()
        datamodules.append(datamodule)

    if len(datamodules) == 1:
        return datamodules[0]
    return CombinedMVTecDataModule(datamodules, batch_size=batch_size)