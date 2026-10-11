# experiment.py
import argparse
import os
import time
from pathlib import Path
import yaml
import torch
import numpy as np

from src.dataset import get_mvtec_dataloaders
from src.train import run_pipeline
from src.export import export_to_onnx


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def benchmark_onnx_model(onnx_path: str, num_runs: int = 100):
    """Measures p50 and p99 inference latency using ONNX Runtime."""
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
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
        "size_mb": round(size_mb, 2),
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


def run_experiment(config_path: str):
    config = load_config(config_path)
    print(f"Running Experiment: {config['experiment_name']}")
    
    # 1. Load Data
    datamodule = get_mvtec_dataloaders(
        category=config["category"], 
        batch_size=config["training"]["batch_size"]
    )
    
    results = []
    
    # 2. Iterate through configured models
    for model_cfg in config["models"]:
        if not model_cfg["enabled"]:
            continue
            
        print(f"\n--- Processing Model: {model_cfg['name']} ---")
        
        # Train / load baseline weights
        weights_path = model_cfg["weights_path"]
        if not os.path.exists(weights_path):
            print(f"Trained weights not found for {model_cfg['name']}. Invoking training pipeline...")
            from src.train import run_pipeline
            
            epochs = config.get("training", {}).get("epochs", 10)
            
            # Pass the already-loaded datamodule directly into the pipeline router
            run_pipeline(
                model_name=model_cfg["name"],
                category=config["category"],
                datamodule=datamodule,
                epochs=epochs,
                save_path=weights_path
            )
            
        # Export to FP32 ONNX first
        base_onnx = "autoencoder_fp32.onnx"
        export_to_onnx(weights_path=weights_path, onnx_output_path=base_onnx)
        
        # 3. Handle Quantization Targets & Benchmarking
        for q_target in config["quantization_targets"]:
            if not q_target["enabled"]:
                continue
                
            fmt = q_target["format"]
            if fmt == "fp32":
                metrics = benchmark_onnx_model(base_onnx)
                results.append({"model": model_cfg["name"], "format": "FP32", **metrics})
                
            elif fmt == "int8_dynamic":
                quant_onnx = "autoencoder_int8.onnx"
                quantize_onnx_dynamic(base_onnx, quant_onnx)
                metrics = benchmark_onnx_model(quant_onnx)
                results.append({"model": model_cfg["name"], "format": "INT8 (Dynamic)", **metrics})

    print("\n=== EXPERIMENT SUMMARY ===")
    for res in results:
        print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven ML Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    
    run_experiment(args.config)