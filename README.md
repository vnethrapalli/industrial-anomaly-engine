# industrial-anomaly-engine

An experiment pipeline for training a convolutional autoencoder on an MVTec category,
exporting it to ONNX, and comparing FP32, FP16, and dynamic INT8 inference.

The detector uses an unsupervised reconstruction approach: it learns normal
examples from defect-free training images and flags large reconstruction errors
at inference time.

## Project flow

1. `src/experiment.py` loads `configs/experiment.yaml` and the MVTec dataloaders.
2. The configured model is trained, or existing selected weights are reused.
3. The weights are exported to a baseline FP32 ONNX model.
4. FP16 conversion and dynamic INT8 quantization create additional ONNX artifacts.
5. Each enabled format is benchmarked on the configured CPU and CUDA providers.
6. Model size and latency metrics are printed and saved with the benchmark plot.

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

## Run the demo locally

Run the training/export workflow in Colab, then place the selected
`autoencoder_fp16.onnx` artifact in `app/`. Install the small app dependency set
and start FastAPI from the repository root:

```bash
python -m pip install -r app/requirements.txt
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/docs>, expand `POST /predict`, choose **Try it out**,
and upload a PNG or JPEG image in the `file` field. The response includes the
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

- This autoencoder is small, so ONNX graph and quantization bookkeeping overhead
  outweighs the reduced numeric representation. As a result, the FP16 and INT8
  artifacts can be larger than the FP32 artifact. The storage savings from
  quantization are more likely to appear with larger models.
- FP32 and FP16 inference latency was very similar in these runs. Dynamic INT8
  inference was slightly slower, likely because runtime quantization and conversion
  overhead offsets the cheaper INT8 arithmetic for this small model.
- These findings are model- and provider-dependent; larger models or different
  hardware may show clearer FP16/INT8 size and latency benefits.
- The autoencoder uses a hybrid reconstruction loss weighted 85% toward MSE and
  15% toward SSIM; although this improves structural sharpness, global
  autoencoders can dilute the pixel-level error from small localized defects,
  motivating feature-bank memory architectures such as PatchCore.
- The reported image-level AUROC for the autoencoder was 0.64145.
