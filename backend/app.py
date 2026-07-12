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
MODEL_PATH = BASE_DIR / "waste_classifier.keras"
TFLITE_MODEL_PATH = BASE_DIR / "waste_classifier_dynamic_range.tflite"
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

LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3}|172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?::\d+)?$"


def _load_allowed_origins() -> List[str]:
    raw_origins = os.getenv("CORS_ORIGINS")
    if not raw_origins:
        return DEFAULT_ORIGINS

    cleaned = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return cleaned or DEFAULT_ORIGINS


def _load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path)


def _load_tflite_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter


def _load_model_safely():
    if TFLITE_MODEL_PATH.exists():
        try:
            return {
                "backend": "tflite",
                "model": _load_tflite_model(TFLITE_MODEL_PATH),
                "path": TFLITE_MODEL_PATH,
                "error": None,
            }
        except Exception as exc:
            tflite_error = str(exc)
    else:
        tflite_error = None

    try:
        return {
            "backend": "keras",
            "model": _load_model(MODEL_PATH),
            "path": MODEL_PATH,
            "error": None,
        }
    except Exception as exc:
        error_message = str(exc)
        if tflite_error:
            error_message = f"TFLite load failed: {tflite_error}; Keras load failed: {error_message}"
        return {"backend": None, "model": None, "path": MODEL_PATH, "error": error_message}


MODEL_RUNTIME = _load_model_safely()


def _predict(processed_image: np.ndarray) -> np.ndarray:
    if MODEL_RUNTIME["backend"] == "tflite":
        interpreter = MODEL_RUNTIME["model"]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], processed_image.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]["index"])

    return MODEL_RUNTIME["model"].predict(processed_image, verbose=0)

app = FastAPI(title="WasteWise API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
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
        "model": MODEL_RUNTIME["path"].name,
        "backend": MODEL_RUNTIME["backend"],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "model_loaded": MODEL_RUNTIME["model"] is not None,
        "backend": MODEL_RUNTIME["backend"],
        "model": MODEL_RUNTIME["path"].name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, float | str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    if MODEL_RUNTIME["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model could not be loaded: {MODEL_RUNTIME['error']}",
        )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
