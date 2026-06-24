from __future__ import annotations

import io
import os
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
TFLITE_MODEL_PATH = BASE_DIR / "waste_classifier_fp16.tflite"
DEFAULT_MODEL_PATH = BASE_DIR / "waste_classifier.keras"
LEGACY_MODEL_PATH = BASE_DIR / "best_model.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Non-Recyclable", "Recyclable"]
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]


def _load_allowed_origins() -> List[str]:
    raw_origins = os.getenv("CORS_ORIGINS")
    if not raw_origins:
        return DEFAULT_ORIGINS

    cleaned = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return cleaned or DEFAULT_ORIGINS


def _resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if candidate.exists():
            return candidate

    if TFLITE_MODEL_PATH.exists():
        return TFLITE_MODEL_PATH

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH

    raise FileNotFoundError(
        "Could not find waste_classifier_fp16.tflite, waste_classifier.keras, or best_model.keras in backend/."
    )


MODEL_PATH = _resolve_model_path()


def _load_model(path: Path):
    if path.suffix == ".tflite":
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return {
            "type": "tflite",
            "interpreter": interpreter,
            "input_details": input_details,
            "output_details": output_details,
        }

    keras_model = tf.keras.models.load_model(path)
    return {"type": "keras", "model": keras_model}


LOADED_MODEL = _load_model(MODEL_PATH)


def _predict(processed_image: np.ndarray) -> np.ndarray:
    if LOADED_MODEL["type"] == "keras":
        return LOADED_MODEL["model"].predict(processed_image, verbose=0)

    interpreter = LOADED_MODEL["interpreter"]
    input_details = LOADED_MODEL["input_details"]
    output_details = LOADED_MODEL["output_details"]

    # TFLite input tensors can be float16 or float32 depending on conversion settings.
    model_input = processed_image.astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], model_input)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    return np.asarray(prediction, dtype=np.float32)

app = FastAPI(title="WasteWise API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_load_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.get("/")
def home() -> dict[str, object]:
    return {
        "message": "WasteWise API - Waste Classification Service",
        "version": "1.0.0",
        "model": MODEL_PATH.name,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Could not process the uploaded image.") from exc

    prediction = _predict(processed_image)

    if prediction.ndim == 2 and prediction.shape[-1] == 1:
        recyclable_probability = float(prediction[0][0])
        predicted_class = int(recyclable_probability >= 0.5)
        confidence = recyclable_probability if predicted_class == 1 else 1.0 - recyclable_probability
    else:
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

    predicted_class = max(0, min(predicted_class, len(CLASS_NAMES) - 1))
    prediction_label = CLASS_NAMES[predicted_class]

    return {
        "prediction": prediction_label,
        "confidence": round(confidence * 100, 2),
    }
