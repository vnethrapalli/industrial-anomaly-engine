# experiment.py
import argparse
import time
from pathlib import Path
import yaml
import torch
import numpy as np

from src.dataset import get_mvtec_dataloaders
from src.export import export_to_onnx
from src.plot_metrics import plot_latency_vs_size


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def benchmark_onnx_model(
    onnx_path: str, num_runs: int = 100, device: str = "cpu"
):
    """Measure inference latency using the requested ONNX Runtime provider."""
    import onnxruntime as ort

    providers = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }
    try:
        provider = providers[device.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported benchmark device '{device}'. Use 'cpu' or 'cuda'."
        ) from exc
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            f"ONNX Runtime provider '{provider}' is not available. "
            "Install the matching ONNX Runtime package before benchmarking it."
        )

    session = ort.InferenceSession(onnx_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    latencies = []
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    # Timed runs
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0) # convert to ms
        
    latencies = np.array(latencies)
    size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
    
    return {
        "size_mb": round(size_mb, 4),
        "latency_p50": round(np.percentile(latencies, 50), 2),
        "latency_p99": round(np.percentile(latencies, 99), 2),
    }


def quantize_onnx_dynamic(input_onnx_path: str, output_onnx_path: str):
    """Applies Post-Training Dynamic INT8 Quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"Applying dynamic INT8 quantization to {input_onnx_path}...")
    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_onnx_path,
        weight_type=QuantType.QUInt8
    )
    print(f"Saved quantized model to {output_onnx_path}")


def get_artifact_path(experiment_name: str, artifact_name: str) -> Path:
    """Return the path for an artifact in the named experiment run."""
    artifact_directory = Path("experiment_artifacts") / experiment_name
    artifact_directory.mkdir(parents=True, exist_ok=True)
    return artifact_directory / Path(artifact_name).name


def run_experiment(config_path: str):
    config = load_config(config_path)
    experiment_name = config["experiment_name"]
    artifact_directory = Path("experiment_artifacts") / experiment_name
    artifact_directory.mkdir(parents=True, exist_ok=True)
    print(f"Running Experiment: {experiment_name}")
    print(f"Saving artifacts to {artifact_directory}")
    
    # 1. Load Data
    datamodule = get_mvtec_dataloaders(
        category=config["category"], 
        batch_size=config["training"]["batch_size"]
    )
    
    results = []
    benchmarking = config.get("benchmarking", {})
    benchmark_devices = benchmarking.get("devices", ["cpu"])
    benchmark_iterations = config.get("evaluation", {}).get(
        "benchmark_iterations", 100
    )
    
    # 2. Iterate through configured models
    for model_cfg in config["models"]:
        if not model_cfg["enabled"]:
            continue
            
        print(f"\n--- Processing Model: {model_cfg['name']} ---")
        
        # Train / load baseline weights
        weights_path = get_artifact_path(
            experiment_name, model_cfg["weights_path"]
        )
        if not weights_path.exists():
            print(f"Trained weights not found for {model_cfg['name']}. Invoking training pipeline...")
            from src.train import run_pipeline
            
            epochs = config.get("training", {}).get("epochs", 10)
            training_config = config.get("training", {})
            
            # Pass the already-loaded datamodule directly into the pipeline router
            run_pipeline(
                model_name=model_cfg["name"],
                category=config["category"],
                datamodule=datamodule,
                epochs=epochs,
                lr=training_config.get("learning_rate", 1e-3),
                device=training_config.get("device", "auto"),
                save_path=str(weights_path)
            )

        # Export to FP32 ONNX first
        base_onnx = get_artifact_path(experiment_name, "autoencoder_fp32.onnx")
        export_to_onnx(
            weights_path=str(weights_path), onnx_output_path=str(base_onnx)
        )
        
        # 3. Handle Quantization Targets & Benchmarking
        for q_target in config["quantization_targets"]:
            if not q_target["enabled"]:
                continue
                
            fmt = q_target["format"]
            model_path = base_onnx
            if fmt == "fp32":
                format_name = "FP32"
            elif fmt == "int8_dynamic":
                quant_onnx = get_artifact_path(
                    experiment_name, "autoencoder_int8.onnx"
                )
                quantize_onnx_dynamic(str(base_onnx), str(quant_onnx))
                model_path = quant_onnx
                format_name = "INT8 (Dynamic)"
            else:
                raise ValueError(f"Unsupported quantization format '{fmt}'.")

            for device in benchmark_devices:
                metrics = benchmark_onnx_model(
                    str(model_path),
                    num_runs=benchmark_iterations,
                    device=device,
                )
                results.append(
                    {
                        "model": model_cfg["name"],
                        "format": format_name,
                        "device": device.lower(),
                        **metrics,
                    }
                )

    plot_latency_vs_size(
        results,
        output_path=get_artifact_path(
            experiment_name, "quantization_benchmark.png"
        ),
    )

    print("\n=== EXPERIMENT SUMMARY ===")
    for res in results:
        print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven ML Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    
    run_experiment(args.config)