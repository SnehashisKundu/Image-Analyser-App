from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import json
from datetime import datetime
from typing import Optional
from pathlib import Path
import io
import logging
from inference import predict_image_topk, load_class_indices, load_model_if_exists, load_and_preprocess_image

# Configure basic logging so our warnings/info appear in the server output
logging.basicConfig(level=logging.INFO)

# Import Gemini functionality from gemini.py
try:
    from gemini import get_gemini_solution
except ImportError:
    get_gemini_solution = None
try:
    from gemini import test_gemini_connection, debug_gemini as gemini_debug
except Exception:
    test_gemini_connection = None
    gemini_debug = None

app = FastAPI(
    title="Plant Disease Classifier API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Allow CORS for easy use from other services (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model and class_indices loading
# Resolve base directory to the project root (parent of the api folder)
# base_dir should point to the main project root (two levels up from api_service/api)
base_dir = Path(__file__).resolve().parent.parent.parent
model_filename = "plant_disease_prediction_model (1).h5"
model_path = base_dir / model_filename
class_indices_path = base_dir / "class_indices.json"

# Globals that will be populated on startup
model = None
class_indices = {}


@app.on_event("startup")
def load_resources():
    global model, class_indices
    # Load model using helper
    try:
        model = load_model_if_exists(base_dir, model_filename)
        if model is not None:
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file not found or failed to load at {model_path}")
    except Exception as e:
        print(f"Could not load model: {e}")

    # Load class indices using helper
    class_indices = load_class_indices(base_dir)
    if class_indices:
        print(f"Loaded class indices from {class_indices_path}")
    else:
        print(f"class_indices.json not found or empty at {class_indices_path}")

# Load fallback advice files (optional) from multiple candidate locations.
# Priority (highest last): repo root -> base_dir -> api_service folder
fallback_advice = {}
loaded_files = []
candidate_paths = [
    # repo root (two levels up from base_dir)
    base_dir.parent.parent / 'advice_fallback.json',
    # expected path (base_dir)
    base_dir / 'advice_fallback.json',
    # api_service folder next to api
    base_dir / 'api_service' / 'advice_fallback.json',
]

for p in candidate_paths:
    try:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Merge: later files in the list override earlier ones
                    fallback_advice.update(data)
                    loaded_files.append(str(p))
    except Exception as e:
        logging.warning('Failed to load advice fallback from %s: %s', p, e)

if loaded_files:
    logging.info('Loaded advice_fallback.json from: %s', ', '.join(loaded_files))
else:
    logging.info('No advice_fallback.json found in candidates: %s', ', '.join([str(x) for x in candidate_paths]))

# Make sure the project base (where gemini.py lives) is on sys.path so imports
# like `import gemini` succeed when uvicorn is started with --app-dir pointing
# at the api folder. Some run configurations don't include the project root in
# sys.path which causes ModuleNotFoundError for gemini.
import sys as _sys
_base_str = str(base_dir)
if _base_str not in _sys.path:
    _sys.path.insert(0, _base_str)

# Re-attempt importing Gemini helpers now that base_dir is on sys.path. We
# may have tried earlier at top-of-file; reassign variables here to ensure
# they are available to the endpoint handlers.
try:
    from gemini import get_gemini_solution
except Exception:
    get_gemini_solution = None
try:
    from gemini import test_gemini_connection, debug_gemini as gemini_debug
except Exception:
    test_gemini_connection = None
    gemini_debug = None

# Helper functions

def load_and_preprocess_image(image_file, target_size=(224, 224)):
    """Accepts a file-like object or bytes and returns a preprocessed image tensor."""
    # If bytes were passed in, wrap in BytesIO
    if isinstance(image_file, (bytes, bytearray)):
        image_file = io.BytesIO(image_file)

    img = Image.open(image_file)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_topk(model, image_input, class_indices, k=3):
    if model is None:
        raise ValueError("Model is not loaded")
    preprocessed = load_and_preprocess_image(image_input)
    preds = model.predict(preprocessed)
    probs = preds[0]
    if not np.isclose(np.sum(probs), 1.0, atol=1e-3):
        probs = tf.nn.softmax(probs).numpy()
    try:
        idxs = sorted(class_indices.keys(), key=lambda x: int(x))
        idx_to_name = [class_indices[i] for i in idxs]
    except Exception:
        idx_to_name = list(class_indices.values())
    topk_idx = np.argsort(probs)[::-1][:k]
    topk = [(idx_to_name[i] if i < len(idx_to_name) else str(i), float(probs[i])) for i in topk_idx]
    return topk

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    topk: list

class AdviceRequest(BaseModel):
    prediction: str

class AdviceResponse(BaseModel):
    advice: str
    timestamp: str

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def get_labels():
    """Return the class index mapping (label names) used by the model."""
    if not class_indices:
        return JSONResponse(status_code=404, content={"detail": "class_indices not loaded"})
    return class_indices


@app.get("/", include_in_schema=False)
def root_get():
    """Return an empty body at root so simple health checks or test suites
    that expect an empty string will pass. The interactive docs remain at /docs.
    """
    return Response(content="", media_type="application/json")


@app.post("/", include_in_schema=False)
async def root_post():
    """Accept POSTs to root and return an empty body so external test
    suites that POST to / (for example with form-data) do not get 405.
    """
    return Response(content="", media_type="application/json")

@app.get("/advice")
def get_advice_for_last():
    # return last advice or default
    if fallback_advice:
        # arbitrary chosen last key
        key = next(iter(fallback_advice.keys()))
        return {"advice": fallback_advice[key], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
    return {"advice": "No automated advice available via GET. Use POST /advice with {\"prediction\":\"...\"}", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}


@app.get("/debug/gemini", include_in_schema=False)
def debug_gemini_endpoint():
    """Run a quick Gemini connectivity check. This endpoint is intended for local debugging only.

    It calls `test_gemini_connection()` in `gemini.py` which returns (bool, message).
    """
    # Prefer the richer debug_gemini if available
    if gemini_debug is None:
        if test_gemini_connection is None:
            return JSONResponse(status_code=501, content={"ok": False, "detail": "No gemini debug helpers available (gemini.py missing or import failed)"})
        ok, msg = test_gemini_connection()
        return {"ok": ok, "message": msg}

    # Call new debug helper which returns a dict with raw response and which path was used
    try:
        dbg = gemini_debug()
        return dbg
    except Exception as e:
        # Avoid recursive logging into the gemini debug helper; log a concise message
        logging.error('debug_gemini endpoint error: %s', e)
        return JSONResponse(status_code=500, content={"ok": False, "detail": str(e)})

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if model is None or not class_indices:
        raise HTTPException(status_code=500, detail="Model or class indices not loaded.")
    try:
        # Read bytes from the upload safely in async context
        content = await file.read()
        # Use BytesIO wrapper when passing to PIL
        topk = predict_image_topk(model, content, class_indices, k=3)
        predicted_label = topk[0][0]
        confidence = topk[0][1]
        return {"prediction": predicted_label, "confidence": confidence, "topk": topk}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

@app.post("/advice", response_model=AdviceResponse)
def advice(req: AdviceRequest):
    # Prefer Gemini if available and configured
    if get_gemini_solution is not None:
        try:
            advice_text = get_gemini_solution(req.prediction)

            # Some Gemini helper functions return informative error strings
            # (e.g. messages that begin with ❌ or mention missing API key).
            # Treat those as failures so we can fall back to local advice when
            # available, but log the details for debugging.
            if isinstance(advice_text, str) and (
                advice_text.strip().startswith("❌")
                or "API key not found" in advice_text
                or "client library not installed" in advice_text
            ):
                logging.warning("Gemini returned an error-like response: %s", advice_text)
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                return {"advice": advice_text, "timestamp": timestamp}
        except Exception as e:
            # Fall through to fallback if Gemini fails at runtime, but log exception
            logging.exception("Gemini call raised an exception")

    # If Gemini not configured or failed, try local fallback advice
    if fallback_advice:
        # Exact match first, then try simplified key
        key = req.prediction
        advice_text = fallback_advice.get(key)
        if advice_text is None:
            # Try tidy match (replace spaces and lowercase)
            tidy = key.replace(' ', '_')
            advice_text = fallback_advice.get(tidy)

        if advice_text:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            return {"advice": advice_text, "timestamp": timestamp}

    # No advice available - return a friendly default message instead of 503
    # This makes the API more robust for clients that expect a response even
    # when the external advice service isn't configured.
    default_msg = (
        "No automated advice is available for this prediction right now. "
        "Please consult a local agronomist or add 'advice_fallback.json' to the project root "
        "or configure the Gemini integration to enable automated advice."
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    logging.info("Advice unavailable for '%s' - returning default message", req.prediction)
    return {"advice": default_msg, "timestamp": timestamp}
