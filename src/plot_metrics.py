# src/plot_metrics.py
import matplotlib.pyplot as plt

def plot_latency_vs_size(results):
    formats = [r["format"] for r in results]
    latencies = [r["latency_p50"] for r in results]
    sizes = [r["size_mb"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Model Format')
    ax1.set_ylabel('p50 Latency (ms)', color=color)
    ax1.bar(formats, latencies, color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Model Size (MB)', color=color)
    ax2.plot(formats, sizes, color=color, marker='o', linewidth=2, label="Size (MB)")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Performance Trade-off: FP32 vs INT8 Quantization")
    plt.savefig("quantization_benchmark.png", dpi=300, bbox_inches='tight')
    print("Saved benchmark plot to quantization_benchmark.png")