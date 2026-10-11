# src/plot_metrics.py
import matplotlib.pyplot as plt


def plot_latency_vs_size(results, output_path, percentile=50):
    """Plot latency by quantization level for each execution provider."""
    if not results:
        return

    formats = list(dict.fromkeys(result["format"] for result in results))
    devices = list(dict.fromkeys(result["device"] for result in results))

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
            label=device.upper(),
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