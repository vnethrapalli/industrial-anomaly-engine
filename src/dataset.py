# src/dataset.py
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from anomalib.data import MVTecAD
from huggingface_hub import snapshot_download
import numpy as np
import torch

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


def get_mvtec_dataloaders(
    category: str = "bottle",
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
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
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

    _ensure_mvtec_category(category)

    # 4. Instantiate DataModule
    datamodule = MVTecAD(
        root=DATASET_ROOT,
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        test_split_mode="from_dir",
        val_split_mode="from_train",
        val_split_ratio=validation_split,
        seed=seed,
        augmentations=transform,
    )

    datamodule.setup()
    return datamodule