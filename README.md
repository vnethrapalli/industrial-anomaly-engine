# industrial-anomaly-engine

An experiment pipeline for evaluating a convolutional autoencoder and PatchCore on
MVTec categories, exporting model artifacts, and comparing FP32, FP16, and dynamic
INT8 inference.

The detector uses an unsupervised reconstruction approach: it learns normal
examples from defect-free training images and flags large reconstruction errors
at inference time.

## Project flow

1. `src/experiment.py` loads `configs/experiment.yaml` and the MVTec dataloaders.
2. Autoencoder weights are trained or reused. PatchCore builds or reuses one coreset
  memory bank from the normal training images and does not require epoch training.
3. Each configured model produces a baseline FP32 ONNX artifact.
4. FP16 conversion and dynamic INT8 quantization create additional feature or
  reconstruction ONNX artifacts.
5. Each enabled format is evaluated and benchmarked on the configured providers.
6. Shared metrics and latency plots are saved under
  `experiment_artifacts/{experiment_name}/`, while each model's artifacts are
  isolated under its own subdirectory.

PatchCore exports its ResNet18 feature extractor as ONNX, its sampled memory bank as
`experiment_artifacts/{experiment_name}/patchcore/patchcore_memory_bank.npy`, and
its scikit-learn nearest-neighbor index as
`experiment_artifacts/{experiment_name}/patchcore/patchcore_nn_index.joblib`.
Autoencoder artifacts are stored under the corresponding `autoencoder/` directory.
PatchCore FP32, FP16, and INT8 measurements include
feature extraction, host transfer, and nearest-neighbor scoring. The nearest-neighbor
stage remains CPU-based and is included in CUDA end-to-end timings. These artifacts
are also consumed by the PatchCore serving endpoints described below.

The model uses fixed `224x224` RGB inputs. Images are resized to that square
without preserving aspect ratio, so unusual source aspect ratios may distort
features and affect anomaly quality and performance. API region coordinates are
mapped back to the uploaded image dimensions.

## Dataset attribution and license

This project uses the [MVTec AD dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

> Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
> “MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,”
> ICCV 2019. https://doi.org/10.1109/ICCV.2019.00982

MVTec AD is released under the [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
It is intended for non-commercial use only. The dataset is not included in this
repository; users are responsible for complying with its license and attribution
requirements.

On the first experiment run, the configured category is downloaded from the
[Hugging Face mirror](https://huggingface.co/datasets/foersben/mvtec-ad) into
`datasets/MVTecAD/`; later runs reuse the local copy.

Run the default experiment with:

```bash
python -m src.experiment --config configs/experiment.yaml
```

## Entrypoints

- `src/experiment.py`: end-to-end training, export, quantization, and benchmarking.
- `src/train.py`: model training pipeline used by the experiment runner.
- `src/export.py`: standalone PyTorch-to-ONNX export helper.
- `notebooks/colab_experiment.ipynb`: interactive Colab workflow.

## Serving the models

The existing `POST /predict` and `POST /predict/overlay` endpoints serve the
autoencoder. PatchCore is available through `POST /predict/patchcore` and
`POST /predict/patchcore/overlay`. The PatchCore response includes its feature-distance
score, thresholded regions, and a base64-encoded overlay using the same image resizing
and region-coordinate behavior as the autoencoder endpoint.

To enable PatchCore serving, place `patchcore_fp32.onnx`,
`patchcore_memory_bank.npy`, and `patchcore_nn_index.joblib` in `app/`, or set
`PATCHCORE_MODEL_PATH`, `PATCHCORE_MEMORY_BANK_PATH`, and `PATCHCORE_INDEX_PATH` to
their locations. The PatchCore image threshold defaults to the calibrated FP32 value
from the included experiment (`1.8178941`) and can be overridden with
`PATCHCORE_ANOMALY_SCORE_THRESHOLD`. Its patch-map region threshold can be overridden
with `PATCHCORE_ANOMALY_PIXEL_THRESHOLD`.

## Run the demo locally

Run the training/export workflow in Colab, then place the selected
`autoencoder_fp16.onnx` artifact in `app/`. Install the small app dependency set
and start FastAPI from the repository root:

```bash
python -m pip install -r app/requirements.txt
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>, expand any of `POST /predict` or
`POST /predict/patchcore`, choose **Try it out**, and upload a PNG or JPEG image
in the `file` field. The response includes the
reconstruction score, detected regions in the original image coordinates, and a
base64-encoded highlighted PNG overlay. Use `POST /predict/overlay` to receive
the highlighted result directly as a PNG response. The model must be present
before the server starts; set `MODEL_PATH` if it has another name or location.

For the reproducibility image only, build from the repository root after placing
the model in `app/`:

```bash
docker build -f app/Dockerfile -t industrial-anomaly-api .
```

## Insights and findings

- PatchCore improved FP32 image AUROC from `0.641` to `0.799` and F1 from `0.164`
  to `0.777` in the evaluated run.
- That improvement costs roughly `26.5 MB / 47 ms` CUDA p50 for PatchCore versus
  `0.0045 MB / 1.46 ms` for the autoencoder. PatchCore FP16 and INT8 preserved
  quality, but did not reduce latency or artifact size in this run.
- PatchCore timing includes CUDA feature extraction followed by host transfer and
  CPU scikit-learn nearest-neighbor lookup, so mixed-device execution contributes to
  its latency pattern.
- Both binary evaluations use a 95th-percentile threshold from normal validation
  scores: about `0.004` reconstruction MSE for the autoencoder versus `1.818`
  feature distance for PatchCore. AUROC is threshold-independent; F1 and other
  binary metrics use the selected threshold.
