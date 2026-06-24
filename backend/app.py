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

    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    if LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH

    raise FileNotFoundError(
        "Could not find waste_classifier.keras in backend/ or best_model.keras in the project root."
    )


MODEL_PATH = _resolve_model_path()
model = tf.keras.models.load_model(MODEL_PATH)

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

    prediction = model.predict(processed_image, verbose=0)

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
