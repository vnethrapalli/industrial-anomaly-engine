import matplotlib.pyplot as plt


def show_sample_and_mask(batch, index: int = 0):
    """Visualizer for Anomalib ImageBatch samples and ground-truth masks."""
    img = batch.image[index].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())  # Rescale to [0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"Sample #{index} (Label: {batch.gt_label[index].item()})")

    if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
        axes[1].imshow(batch.gt_mask[index].squeeze().cpu().numpy(), cmap="gray")
        axes[1].set_title("Ground Truth Mask")

    for ax in axes:
        ax.axis("off")
    plt.show()