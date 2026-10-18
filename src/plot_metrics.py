# src/plot_metrics.py
import matplotlib.pyplot as plt


def plot_train_validation_loss(history, output_path):
    """Plot training and validation reconstruction loss by epoch."""
    if not history["train_losses"]:
        return

    epochs = range(1, len(history["train_losses"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_losses"], marker="o", label="Train")
    ax.plot(
        epochs,
        history["validation_losses"],
        marker="o",
        label="Validation",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {output_path}")


def plot_latency_vs_size(results, output_path, percentile=50):
    """Plot latency by quantization level for each execution provider."""
    if not results:
        return

    format_order = {
        "FP32": 0,
        "FP16": 1,
        "INT8 (Dynamic)": 2,
    }
    formats = list(dict.fromkeys(result["format"] for result in results))
    formats = sorted(
        formats,
        key=lambda model_format: format_order.get(model_format, len(format_order)),
    )
    devices = list(dict.fromkeys(result["device"] for result in results))
    device_labels = {
        "cpu": "CPU",
        "cuda": "T4 GPU (CUDA)",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    x_values = range(len(formats))
    for device in devices:
        device_results = {
            result["format"]: result
            for result in results
            if result["device"] == device
        }
        latencies = [
            device_results[model_format][f"latency_p{percentile}"]
            for model_format in formats
            if model_format in device_results
        ]
        device_formats = [
            model_format for model_format in formats if model_format in device_results
        ]
        ax.plot(
            [formats.index(model_format) for model_format in device_formats],
            latencies,
            marker="o",
            linewidth=2,
            label=device_labels.get(device.lower(), device.upper()),
        )

    ax.set_xticks(list(x_values), formats)
    ax.set_xlabel("Quantization Level")
    ax.set_ylabel(f"p{percentile} Latency (ms)")
    ax.set_title("Inference Latency by Quantization Level")
    ax.legend(title="Execution Provider")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved benchmark plot to {output_path}")