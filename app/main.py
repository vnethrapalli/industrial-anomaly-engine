import base64
import binascii
import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageDraw

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_DIR / "autoencoder_fp32.onnx"))
SCORE_THRESHOLD = float(os.getenv("ANOMALY_SCORE_THRESHOLD", "0.01"))
PIXEL_THRESHOLD = float(os.getenv("ANOMALY_PIXEL_THRESHOLD", "0.01"))
MIN_REGION_AREA = int(os.getenv("MIN_REGION_AREA", "16"))

app = FastAPI(title="Visual Anomaly Detection API", version="1.0")

session: ort.InferenceSession | None = None
input_name: str | None = None
input_dtype = np.float32
model_error: str | None = None

try:
    loaded_session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    model_input = loaded_session.get_inputs()[0]
    loaded_input_name = model_input.name
    input_type = model_input.type
    if input_type == "tensor(float16)":
        input_dtype = np.float16
    elif input_type != "tensor(float)":
        raise RuntimeError(f"Unsupported ONNX input type: {input_type}")
    session = loaded_session
    input_name = loaded_input_name
    print(f"Loaded ONNX model from {MODEL_PATH}")
except (OSError, RuntimeError, ValueError) as exc:
    model_error = str(exc)
    print(f"Warning: model could not be loaded from {MODEL_PATH}: {exc}")


def _find_regions(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Return connected regions as (left, top, right, bottom, area)."""
    visited = np.zeros(mask.shape, dtype=bool)
    regions = []
    height, width = mask.shape

    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        points = []

        while stack:
            current_y, current_x = stack.pop()
            points.append((current_y, current_x))
            for next_y, next_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))

        if len(points) >= MIN_REGION_AREA:
            ys, xs = zip(*points)
            regions.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(points)))

    return sorted(regions, key=lambda region: region[4], reverse=True)


def _encode_overlay(image: Image.Image, mask: np.ndarray, regions: list[tuple[int, ...]]) -> str:
    overlay = image.convert("RGBA")
    highlight = Image.new("RGBA", overlay.size, (220, 38, 38, 0))
    highlight.putalpha(Image.fromarray((mask * 110).astype(np.uint8), mode="L"))
    overlay = Image.alpha_composite(overlay, highlight)
    draw = ImageDraw.Draw(overlay)
    for left, top, right, bottom, _ in regions:
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(220, 38, 38, 255), width=2)

    buffer = io.BytesIO()
    overlay.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _prepare_model_input(image: Image.Image) -> np.ndarray:
    """Match src.dataset.py: resize RGB pixels and convert to NCHW [0, 1]."""
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    model_input = np.transpose(image_array, (2, 0, 1))[None, ...]
    return model_input.astype(input_dtype, copy=False)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy" if session is not None else "unavailable",
        "model": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Return the reconstruction score, anomalous regions, and a highlighted overlay."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    if session is None or input_name is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable: {model_error or 'unknown model loading error'}",
        )

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
            )
        original = Image.open(io.BytesIO(contents)).convert("RGB")
        original_width, original_height = original.size
        model_input = _prepare_model_input(original)
        reconstruction = np.asarray(
            session.run(None, {input_name: model_input})[0],
            dtype=np.float32,
        )
        if reconstruction.shape != model_input.shape:
            raise RuntimeError(
                f"Model output shape {reconstruction.shape} does not match input "
                f"shape {model_input.shape}."
            )
        if not np.isfinite(reconstruction).all():
            raise RuntimeError("Model returned non-finite values.")

        pixel_error = np.mean(
            np.square(model_input.astype(np.float32) - reconstruction), axis=1
        )[0]
        reconstruction_mse = float(np.mean(pixel_error))
        resized_mask = pixel_error >= PIXEL_THRESHOLD
        resized_regions = _find_regions(resized_mask)
        scale_x = original_width / IMAGE_SIZE[0]
        scale_y = original_height / IMAGE_SIZE[1]
        region_details = [
            {
                "x": round(left * scale_x),
                "y": round(top * scale_y),
                "width": round((right - left) * scale_x),
                "height": round((bottom - top) * scale_y),
                "area": area,
            }
            for left, top, right, bottom, area in resized_regions
        ]
        anomaly_mask = np.asarray(
            Image.fromarray((resized_mask * 255).astype(np.uint8), mode="L").resize(
                original.size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) // 255
        regions = [
            (
                round(left * scale_x),
                round(top * scale_y),
                round(right * scale_x),
                round(bottom * scale_y),
                area,
            )
            for left, top, right, bottom, area in resized_regions
        ]

        return {
            "filename": file.filename,
            "reconstruction_mse": reconstruction_mse,
            "anomaly_detected": reconstruction_mse >= SCORE_THRESHOLD,
            "regions": region_details,
            "overlay_image": _encode_overlay(original, anomaly_mask, regions),
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc


@app.post(
    "/predict/overlay",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def predict_overlay(file: UploadFile = File(...)) -> Response:
    """Return the highlighted prediction overlay as a viewable PNG response."""
    result = await predict(file)
    data_url = result["overlay_image"]
    _, separator, encoded_image = data_url.partition(",")
    if not separator:
        raise HTTPException(status_code=500, detail="Invalid overlay image payload.")

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Invalid overlay image payload.") from exc

    return Response(
        content=image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="anomaly-overlay.png"'},
    )
