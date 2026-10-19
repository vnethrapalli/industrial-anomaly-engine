# experiment.py
import argparse
import json
import random
import time
from pathlib import Path
import yaml
import torch
import numpy as np

from src.dataset import get_mvtec_dataloaders
from src.export import export_to_onnx
from src.plot_metrics import plot_latency_vs_size, plot_train_validation_loss


def set_reproducibility(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch sources of experiment randomness."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    input_dtypes = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }
    try:
        input_dtype = input_dtypes[model_input.type]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported ONNX input type '{model_input.type}' for '{input_name}'."
        ) from exc

    dummy_input = np.random.randn(1, 3, 224, 224).astype(input_dtype)
    
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


def _batch_field(batch, field: str):
    try:
        return getattr(batch, field)
    except (AttributeError, KeyError):
        return batch[field]


def _batch_label(batch):
    """Return the Anomalib ground-truth label across batch schema versions."""
    for field in ("gt_label", "label", "anomaly_label"):
        try:
            return _batch_field(batch, field)
        except (AttributeError, KeyError, TypeError):
            continue
    raise RuntimeError(
        "Test batch does not expose a ground-truth label field; "
        "expected gt_label, label, or anomaly_label."
    )


def _collect_onnx_scores(
    session,
    input_name: str,
    dataloader,
    include_labels: bool,
    input_dtype: np.dtype = np.float32,
):
    scores = []
    labels = []
    for batch in dataloader:
        images = _batch_field(batch, "image")
        model_input = images.detach().cpu().numpy().astype(input_dtype, copy=False)
        reconstruction = np.asarray(
            session.run(None, {input_name: model_input})[0], dtype=np.float32
        )
        input_for_error = model_input.astype(np.float32, copy=False)
        scores.extend(
            np.mean(
                np.square(input_for_error - reconstruction), axis=(1, 2, 3)
            ).tolist()
        )
        if include_labels:
            batch_labels = _batch_label(batch)
            labels.extend(batch_labels.detach().cpu().numpy().astype(int).tolist())

    if not scores:
        raise RuntimeError("No images were available for ONNX evaluation.")
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return float("nan")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _select_evaluation_provider(device: str = "auto") -> str:
    """Use CUDA for evaluation when requested or available, otherwise CPU."""
    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    if device.lower() in {"auto", "cuda"} and "CUDAExecutionProvider" in available_providers:
        return "CUDAExecutionProvider"
    if device.lower() == "cuda":
        print("CUDA evaluation requested but unavailable; falling back to CPU.")
    return "CPUExecutionProvider"


def evaluate_onnx_anomaly_detection(
    onnx_path: str,
    datamodule,
    threshold_percentile: float,
    device: str = "auto",
):
    """Calibrate on normal validation data and evaluate image labels on test data."""
    import onnxruntime as ort

    provider = _select_evaluation_provider(device)
    session = ort.InferenceSession(onnx_path, providers=[provider])
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    input_dtypes = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
    }
    try:
        input_dtype = input_dtypes[model_input.type]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported ONNX input type '{model_input.type}' for '{input_name}'."
        ) from exc
    validation_scores, _ = _collect_onnx_scores(
        session,
        input_name,
        datamodule.val_dataloader(),
        include_labels=False,
        input_dtype=input_dtype,
    )
    test_scores, test_labels = _collect_onnx_scores(
        session,
        input_name,
        datamodule.test_dataloader(),
        include_labels=True,
        input_dtype=input_dtype,
    )
    threshold = float(np.percentile(validation_scores, threshold_percentile))
    predictions = test_scores >= threshold
    true_positive = int(np.sum(predictions & (test_labels == 1)))
    true_negative = int(np.sum(~predictions & (test_labels == 0)))
    false_positive = int(np.sum(predictions & (test_labels == 0)))
    false_negative = int(np.sum(~predictions & (test_labels == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "threshold_percentile": threshold_percentile,
        "anomaly_threshold": threshold,
        "evaluation_provider": provider,
        "image_accuracy": float(np.mean(predictions == test_labels)),
        "image_precision": precision,
        "image_recall": recall,
        "image_f1": f1,
        "image_auroc": _binary_auroc(test_scores, test_labels),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def evaluate_quantized_models(
    model_path: str,
    datamodule,
    threshold_percentile: float,
    device: str = "auto",
):
    """Calibrate a model on normal validation data and evaluate its test labels."""
    return evaluate_onnx_anomaly_detection(
        model_path,
        datamodule,
        threshold_percentile=threshold_percentile,
        device=device,
    )


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


def convert_onnx_to_fp16(input_onnx_path: str, output_onnx_path: str):
    """Convert an ONNX model to FP16 once and save it to the artifact directory."""
    import onnx
    from onnxconverter_common import float16

    print(f"Converting {input_onnx_path} to FP16...")
    model = onnx.load(input_onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_onnx_path)
    print(f"Saved FP16 model to {output_onnx_path}")


def get_artifact_path(experiment_name: str, artifact_name: str) -> Path:
    """Return the path for an artifact in the named experiment run."""
    artifact_directory = Path("experiment_artifacts") / experiment_name
    artifact_directory.mkdir(parents=True, exist_ok=True)
    return artifact_directory / Path(artifact_name).name


def get_selected_weights_path(experiment_name: str, model_cfg: dict, epochs: int):
    """Return the single weights artifact path for a model selection policy."""
    base_path = Path(model_cfg["weights_path"])
    selection = str(model_cfg.get("selection", "best"))
    artifact_directory = Path("experiment_artifacts") / experiment_name

    if selection == "best":
        matching_paths = sorted(
            artifact_directory.glob(f"{base_path.stem}_best_epoch_*_of_*.pth")
        )
        if len(matching_paths) > 1:
            raise RuntimeError(
                f"Expected at most one selected weights file for {model_cfg['name']}, "
                f"found {len(matching_paths)}."
            )
        return matching_paths[0] if matching_paths else None

    if selection != "epoch":
        raise ValueError(
            f"Invalid model selection '{selection}' for {model_cfg['name']}. "
            "Use 'best' or 'epoch'."
        )
    selected_epoch = epochs
    return get_artifact_path(
        experiment_name, f"{base_path.stem}_epoch_{selected_epoch}_of_{epochs}.pth"
    )


def remove_selected_weights(
    experiment_name: str, model_cfg: dict
):
    """Remove previously generated selection artifacts for one architecture."""
    artifact_directory = Path("experiment_artifacts") / experiment_name
    base_stem = Path(model_cfg["weights_path"]).stem
    for path in artifact_directory.glob(f"{base_stem}_best_epoch_*_of_*.pth"):
        path.unlink()
    for path in artifact_directory.glob(f"{base_stem}_epoch_*_of_*.pth"):
        path.unlink()


def run_experiment(config_path: str):
    config = load_config(config_path)
    set_reproducibility(int(config.get("seed", 42)))
    experiment_name = config["experiment_name"]
    artifact_directory = Path("experiment_artifacts") / experiment_name
    artifact_directory.mkdir(parents=True, exist_ok=True)
    print(f"Running Experiment: {experiment_name}")
    print(f"Saving artifacts to {artifact_directory}")
    
    # 1. Load Data
    training_config = config.get("training", {})
    datamodule = get_mvtec_dataloaders(
        category=config["category"], 
        batch_size=training_config["batch_size"],
        validation_split=training_config.get("validation_split", 0.2),
        seed=config.get("seed", 42),
    )
    
    results = []
    benchmarking = config.get("benchmarking", {})
    benchmark_devices = benchmarking.get("devices", ["cpu"])
    benchmark_iterations = config.get("evaluation", {}).get(
        "benchmark_iterations", 100
    )
    threshold_percentile = float(
        config.get("evaluation", {}).get("anomaly_threshold_percentile", 95.0)
    )
    evaluation_device = str(config.get("evaluation", {}).get("device", "auto"))
    detection_results = []
    
    # 2. Iterate through configured models
    for model_cfg in config["models"]:
        if not model_cfg["enabled"]:
            continue
            
        print(f"\n--- Processing Model: {model_cfg['name']} ---")
        
        # Train / load baseline weights
        epochs = config.get("training", {}).get("epochs", 10)
        selection = str(model_cfg.get("selection", "best"))
        selected_weights_path = get_selected_weights_path(
            experiment_name, model_cfg, epochs
        )
        if selected_weights_path is None or not selected_weights_path.exists():
            print(f"Trained weights not found for {model_cfg['name']}. Invoking training pipeline...")
            from src.train import run_pipeline
            
            training_config = config.get("training", {})
            output_stem = Path(model_cfg["weights_path"]).stem
            remove_selected_weights(experiment_name, model_cfg)
            if selection == "best":
                output_name = f"{output_stem}_best_epoch_pending_of_{epochs}.tmp"
            else:
                output_name = f"{output_stem}_epoch_{epochs}_of_{epochs}.pth"
            pending_path = get_artifact_path(experiment_name, output_name)
            
            # Pass the already-loaded datamodule directly into the pipeline router
            history = run_pipeline(
                model_name=model_cfg["name"],
                category=config["category"],
                datamodule=datamodule,
                epochs=epochs,
                lr=training_config.get("learning_rate", 1e-3),
                device=training_config.get("device", "auto"),
                save_path=str(pending_path),
                selection=selection,
            )
            if selection == "best":
                best_epoch = history["best_epoch"]
                selected_weights_path = get_artifact_path(
                    experiment_name,
                    f"{output_stem}_best_epoch_{best_epoch}_of_{epochs}.pth",
                )
                pending_path.rename(selected_weights_path)
            else:
                selected_weights_path = pending_path
            plot_train_validation_loss(
                history,
                get_artifact_path(
                    experiment_name, f"{output_stem}_loss.png"
                ),
            )
        else:
            print(f"Using existing selected weights: {selected_weights_path}")

        # Export to FP32 ONNX first
        model_stem = model_cfg["name"]
        base_onnx = get_artifact_path(
            experiment_name, f"{model_stem}_fp32.onnx"
        )
        export_to_onnx(
            weights_path=str(selected_weights_path), onnx_output_path=str(base_onnx)
        )

        # 3. Handle Quantization Targets & Benchmarking
        for q_target in config["quantization_targets"]:
            if not q_target["enabled"]:
                continue
                
            fmt = q_target["format"]
            model_path = base_onnx
            if fmt == "fp32":
                format_name = "FP32"
            elif fmt == "fp16":
                fp16_onnx = get_artifact_path(
                    experiment_name, f"{model_stem}_fp16.onnx"
                )
                if (
                    not fp16_onnx.exists()
                    or fp16_onnx.stat().st_mtime < base_onnx.stat().st_mtime
                ):
                    convert_onnx_to_fp16(str(base_onnx), str(fp16_onnx))
                model_path = fp16_onnx
                format_name = "FP16"
            elif fmt == "int8_dynamic":
                quant_onnx = get_artifact_path(
                    experiment_name, f"{model_stem}_int8.onnx"
                )
                if (
                    not quant_onnx.exists()
                    or quant_onnx.stat().st_mtime < base_onnx.stat().st_mtime
                ):
                    quantize_onnx_dynamic(str(base_onnx), str(quant_onnx))
                model_path = quant_onnx
                format_name = "INT8 (Dynamic)"
            else:
                raise ValueError(f"Unsupported quantization format '{fmt}'.")

            detection_metrics = evaluate_quantized_models(
                str(model_path),
                datamodule,
                threshold_percentile=threshold_percentile,
                device=evaluation_device,
            )
            detection_results.append(
                {
                    "model": model_cfg["name"],
                    "format": format_name,
                    **detection_metrics,
                }
            )

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

    evaluation_path = get_artifact_path(
        experiment_name, "anomaly_detection_metrics.json"
    )
    with evaluation_path.open("w", encoding="utf-8") as file:
        json.dump(detection_results, file, indent=2)
    print(f"Saved anomaly detection metrics to {evaluation_path}")

    print("\n=== IMAGE-LEVEL TEST EVALUATION ===")
    for result in detection_results:
        print(result)

    print("\n=== EXPERIMENT SUMMARY ===")
    for res in results:
        print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config-driven ML Experiment Runner")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()
    
    run_experiment(args.config)