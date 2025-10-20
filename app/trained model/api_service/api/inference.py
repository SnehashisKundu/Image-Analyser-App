from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
import json
from pathlib import Path

# Base directory for model and class indices - caller should set or compute as needed
BASE_DIR = Path(__file__).resolve().parent.parent

def load_and_preprocess_image(image_file, target_size=(224, 224)):
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

def load_class_indices(base_dir: str | Path):
    p = Path(base_dir) / 'class_indices.json'
    if not p.exists():
        return {}
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def load_model_if_exists(base_dir: str | Path, model_filename: str):
    p = Path(base_dir) / model_filename
    if not p.exists():
        return None
    try:
        return tf.keras.models.load_model(str(p))
    except Exception:
        return None
