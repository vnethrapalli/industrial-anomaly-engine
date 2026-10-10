# src/dataset.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from anomalib.data import MVTecAD
import numpy as np
import torch


def get_mvtec_dataloaders(
    category: str = "bottle",
    batch_size: int = 32,
    image_size: tuple[int, int] = (224, 224),
    model_name: str = None,
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

    # 4. Instantiate DataModule
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category=category,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        augmentations=transform,
    )

    datamodule.prepare_data()
    datamodule.setup()
    return datamodule