# industrial-anomaly-engine

An experiment pipeline for training a convolutional autoencoder on an MVTec category,
exporting it to ONNX, and comparing FP32, FP16, and dynamic INT8 inference.

## Project flow

1. `src/experiment.py` loads `configs/experiment.yaml` and the MVTec dataloaders.
2. The configured model is trained, or existing selected weights are reused.
3. The weights are exported to a baseline FP32 ONNX model.
4. FP16 conversion and dynamic INT8 quantization create additional ONNX artifacts.
5. Each enabled format is benchmarked on the configured CPU and CUDA providers.
6. Model size and latency metrics are printed and saved with the benchmark plot.

Run the default experiment with:

```bash
python -m src.experiment --config configs/experiment.yaml
```

## Entrypoints

- `src/experiment.py`: end-to-end training, export, quantization, and benchmarking.
- `src/train.py`: model training pipeline used by the experiment runner.
- `src/export.py`: standalone PyTorch-to-ONNX export helper.
- `notebooks/colab_experiment.ipynb`: interactive Colab workflow.

## Insights and findings

- This autoencoder is small, so ONNX graph and quantization bookkeeping overhead
  outweighs the reduced numeric representation. As a result, the FP16 and INT8
  artifacts can be larger than the FP32 artifact. The storage savings from
  quantization are more likely to appear with larger models.
- FP32 and FP16 inference latency was very similar in these runs. Dynamic INT8
  inference was slightly slower, likely because runtime quantization and conversion
  overhead offsets the cheaper INT8 arithmetic for this small model.
- These findings are model- and provider-dependent; larger models or different
  hardware may show clearer FP16/INT8 size and latency benefits.
